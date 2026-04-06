import type {
  ProviderSession,
  ProviderSessionStartInput,
  ProviderSendTurnInput,
  ThreadId,
  ApprovalRequestId,
  ProviderApprovalDecision,
  ProviderUserInputAnswers,
  ProviderRuntimeEvent,
  CanonicalItemType,
  CanonicalRequestType,
  ThreadTokenUsageSnapshot,
} from "@t3tools/contracts";
import {
  EventId,
  RuntimeItemId,
  RuntimeRequestId,
  TurnId as TurnIdBrand,
} from "@t3tools/contracts";
import { Effect, Layer, Option, Queue, Stream } from "effect";
import { GeminiAcpAdapter, type GeminiAcpAdapterShape } from "../Services/GeminiAcpAdapter";
import type { ProviderThreadSnapshot } from "../Services/ProviderAdapter";
import {
  ProviderAdapterProcessError,
  ProviderAdapterRequestError,
  ProviderAdapterSessionClosedError,
  ProviderAdapterSessionNotFoundError,
  ProviderAdapterValidationError,
  type ProviderAdapterError,
} from "../Errors";
import { ServerSettingsService } from "../../serverSettings";
import {
  ClientSideConnection,
  ndJsonStream,
  type Client,
  type SessionNotification,
  type RequestPermissionRequest,
  type RequestPermissionResponse,
  type ToolKind,
  PROTOCOL_VERSION,
} from "@agentclientprotocol/sdk";
import { spawn, type ChildProcess as NodeChildProcess } from "node:child_process";
import { readFile } from "node:fs/promises";
import { Readable, Writable } from "node:stream";

const PROVIDER = "geminiAcp" as const;
const DEFAULT_GEMINI_CONTEXT_WINDOW = 1_000_000;

// ---------------------------------------------------------------------------
// Plan mode prompt — constrains the model to planning only
// ---------------------------------------------------------------------------

const GEMINI_PLAN_MODE_PROMPT = `<plan_mode>
# Plan Mode (Conversational)

You are in **Plan Mode**. This is a conversational collaboration mode — you chat with the user to build a great plan together before anyone implements anything.

## Mode rules (strict)

1. You are in Plan Mode until the user explicitly switches you out of it.
2. Plan Mode is NOT changed by user intent, tone, or imperative language. If the user asks you to execute or implement while in Plan Mode, treat it as a request to **plan the execution**, not perform it.
3. You can and should answer any questions the user has — about the plan, about the codebase, about tradeoffs, about anything relevant. You are a collaborator, not just a plan generator.

## How to collaborate

Work in roughly three phases, chatting your way to a great plan:

**Phase 1 — Ground in the environment.** Explore the codebase first, ask the user second. Read files, search code, inspect types and configs. Eliminate unknowns by discovering facts rather than guessing. Only ask the user about things you genuinely cannot figure out from the repo.

**Phase 2 — Clarify intent.** Once you understand the environment, clarify what the user actually wants: goals, success criteria, scope, constraints, and preferences. Ask questions — do not guess on high-impact ambiguities.

**Phase 3 — Refine the plan.** Iterate on the implementation details until the plan is decision-complete: approach, interfaces, data flow, edge cases, testing, and rollout. A great plan is detailed enough that another engineer or agent can implement it without making further decisions.

You do NOT need to present a plan on every turn. If the user asks a question, just answer it. If they want to discuss tradeoffs, discuss them. Only write the plan when you have something concrete and complete to present or update.

## Allowed actions (non-mutating, plan-improving)

- Reading, searching, and inspecting files, configs, schemas, types, and docs
- Static analysis, repo exploration, and inspection
- Running tests, builds, or checks that do not edit repo-tracked files

## Forbidden actions (mutating, plan-executing)

- Editing, writing, or creating project files
- Running formatters, linters, or codegen that rewrite files
- Applying patches, migrations, or any changes to repo-tracked files
- Any command whose purpose is to carry out the plan rather than refine it

When in doubt: if the action would be described as "doing the work" rather than "planning the work," do not do it.

## Presenting a plan

When your plan is ready (or meaningfully updated after refinement), write it to a markdown file. The plan should include:

- A clear title
- A brief summary
- Implementation steps with enough detail to be actionable
- Key files and changes involved

Do not ask "should I proceed?" — the user can switch out of Plan Mode to request implementation, or stay in Plan Mode to keep refining.
</plan_mode>`;

// ---------------------------------------------------------------------------
// Session context
// ---------------------------------------------------------------------------

interface PendingApproval {
  readonly requestType: CanonicalRequestType;
  readonly detail: string;
  resolve: (response: RequestPermissionResponse) => void;
}

interface GeminiAcpSessionContext {
  readonly session: ProviderSession;
  readonly child: NodeChildProcess;
  readonly connection: ClientSideConnection;
  readonly acpSessionId: string;
  readonly pendingApprovals: Map<string, PendingApproval>;
  turnState:
    | {
        turnId: string;
        startedAt: string;
        reasoningItemEmitted?: boolean;
      }
    | undefined;
  promptReject: ((err: Error) => void) | undefined;
  stopped: boolean;
  /** Current ACP mode — tracks whether we're in plan mode for event routing. */
  currentAcpMode: string;
  /**
   * Incremented on each tool call boundary within a turn.
   * Used to assign distinct itemIds to assistant text segments so that
   * pre-tool and post-tool text become separate messages.
   */
  assistantMessageSegment: number;
  /** Cumulative input tokens across all turns — used to approximate context window usage. */
  cumulativeInputTokens: number;
  /** Cumulative output tokens across all turns. */
  cumulativeOutputTokens: number;
  /** Cumulative reasoning/thought tokens across all turns. */
  cumulativeReasoningTokens: number;
  /** Last known context window size from a usage_update notification. */
  lastKnownMaxTokens: number | undefined;
  /** Tracks toolCallIds already seen, so tool_call_update can create segment boundaries for new tools. */
  seenToolCallIds: Set<string>;
  /** Set when a usage_update notification arrives during the current turn.
   *  Prevents the prompt-response handler from double-counting tokens that
   *  are already reflected in the absolute values reported by usage_update. */
  turnReceivedUsageUpdate: boolean;
  /** ACP messageIds from completed turns — used to skip replayed content.
   *  The Gemini ACP re-sends all historical messages on each prompt() call;
   *  we deduplicate by tracking messageIds that have already been streamed. */
  completedMessageIds: Set<string>;
  /** ACP messageIds seen in the current (active) turn. Moved to
   *  completedMessageIds when the next turn starts. */
  currentTurnMessageIds: Set<string>;
  /** Set to true once the first non-replayed agent_message_chunk is processed
   *  in the current turn. Until this is true, usage_update emissions are
   *  deferred — replayed usage_update notifications carry stale values that
   *  cause the context counter to flicker/reset. */
  turnHasFreshContent: boolean;
  /** Set to true when `user_message_chunk` is received during a turn.
   *  This is a definitive replay signal — during normal turns the user's
   *  message is sent via `prompt()`, never as a notification.  While active,
   *  `agent_message_chunk` is suppressed until enough time passes after the
   *  last `user_message_chunk` (indicating the replay finished and real
   *  generation started). */
  replayActive: boolean;
  /** `Date.now()` of the most recent `user_message_chunk` notification.
   *  Used together with `replayActive` to detect the replay→generation
   *  boundary. */
  lastReplaySignalTime: number;
}

// ---------------------------------------------------------------------------
// Helpers — event construction
// ---------------------------------------------------------------------------

let eventCounter = 0;
function nextEventId(): string {
  return `evt-gemini-${Date.now()}-${++eventCounter}`;
}

function makeEventBase(
  ctx: GeminiAcpSessionContext,
): Omit<ProviderRuntimeEvent, "type" | "payload"> {
  return {
    eventId: EventId.makeUnsafe(nextEventId()),
    provider: PROVIDER,
    threadId: ctx.session.threadId,
    createdAt: new Date().toISOString(),
    ...(ctx.turnState ? { turnId: TurnIdBrand.makeUnsafe(ctx.turnState.turnId) } : {}),
    providerRefs: {},
  };
}

// ---------------------------------------------------------------------------
// Helpers — ACP tool kind → canonical types
// ---------------------------------------------------------------------------

function mapToolKindToItemType(kind: ToolKind | undefined | null): CanonicalItemType {
  switch (kind) {
    case "read":
    case "search":
      return "file_change";
    case "edit":
    case "delete":
    case "move":
      return "file_change";
    case "execute":
      return "command_execution";
    case "fetch":
      return "web_search";
    case "think":
      return "reasoning";
    default:
      return "unknown";
  }
}

function isFileModifyingKind(kind: ToolKind | undefined | null): boolean {
  return kind === "edit" || kind === "delete" || kind === "move";
}

function buildTokenUsageSnapshot(
  ctx: GeminiAcpSessionContext,
  turnInputTokens?: number,
  turnOutputTokens?: number,
  turnReasoningTokens?: number,
): ThreadTokenUsageSnapshot {
  const usedTokens = ctx.cumulativeInputTokens + ctx.cumulativeOutputTokens;
  const maxTokens = ctx.lastKnownMaxTokens ?? DEFAULT_GEMINI_CONTEXT_WINDOW;
  const totalProcessedTokens =
    ctx.cumulativeInputTokens + ctx.cumulativeOutputTokens + ctx.cumulativeReasoningTokens;
  return {
    usedTokens,
    ...(totalProcessedTokens > usedTokens ? { totalProcessedTokens } : {}),
    maxTokens,
    ...(turnInputTokens !== undefined ? { lastInputTokens: turnInputTokens } : {}),
    ...(turnOutputTokens !== undefined ? { lastOutputTokens: turnOutputTokens } : {}),
    ...(turnReasoningTokens !== undefined
      ? { lastReasoningOutputTokens: turnReasoningTokens }
      : {}),
    ...(turnInputTokens !== undefined || turnOutputTokens !== undefined
      ? { lastUsedTokens: (turnInputTokens ?? 0) + (turnOutputTokens ?? 0) }
      : {}),
  } as ThreadTokenUsageSnapshot;
}

function mapToolKindToRequestType(kind: ToolKind | undefined | null): CanonicalRequestType {
  switch (kind) {
    case "read":
    case "search":
      return "file_read_approval";
    case "edit":
    case "delete":
    case "move":
      return "file_change_approval";
    case "execute":
      return "exec_command_approval";
    default:
      // Default to command approval so the UI renders interactive approve/deny buttons.
      // ACP permission requests are always actionable — "unknown" would silently skip the UI.
      return "command_execution_approval";
  }
}

// ---------------------------------------------------------------------------
// Helpers — plan file detection
// ---------------------------------------------------------------------------

interface PlanFileSource {
  /** Structured `locations` from `ToolCall` / `ToolCallUpdate`. */
  locations?: ReadonlyArray<{ path: string }> | null | undefined;
  /** Human-readable title (fallback when locations are unavailable). */
  title?: string | null | undefined;
}

/** Regex that extracts a `.md` file path from an ACP tool title. */
const PLAN_TITLE_RE = /(\S+\.md)\s*$/;

/**
 * Try to read a plan markdown file referenced by a tool call.
 * Prefers the structured `locations` array; falls back to parsing the title.
 * Returns the trimmed file content, or `undefined` if nothing matched or
 * the file is unreadable / empty.
 */
async function tryReadPlanFile(source: PlanFileSource): Promise<string | undefined> {
  const candidates: string[] = [];
  if (source.locations) {
    for (const loc of source.locations) {
      if (loc.path.endsWith(".md")) {
        candidates.push(loc.path);
      }
    }
  }
  if (candidates.length === 0 && source.title) {
    const match = source.title.match(PLAN_TITLE_RE);
    if (match?.[1]) {
      candidates.push(match[1]);
    }
  }

  for (const candidate of candidates) {
    try {
      const content = await readFile(candidate, "utf-8");
      const trimmed = content.trim();
      if (trimmed.length > 0) return trimmed;
    } catch {
      // File unreadable — try next candidate.
    }
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// Helpers — error mapping
// ---------------------------------------------------------------------------

function toMessage(cause: unknown, fallback: string): string {
  if (cause instanceof Error) return cause.message;
  if (typeof cause === "string") return cause;
  // ACP SDK rejects with JSON-RPC error objects ({ code, message, data })
  if (typeof cause === "object" && cause !== null) {
    const obj = cause as Record<string, unknown>;
    if (typeof obj.message === "string") return obj.message;
    const str = String(cause);
    if (str !== "[object Object]") return str;
    try {
      return JSON.stringify(cause);
    } catch {
      // fall through
    }
  }
  return fallback;
}

function toRequestError(threadId: ThreadId, method: string, cause: unknown): ProviderAdapterError {
  const message = toMessage(cause, "").toLowerCase();
  if (message.includes("session not found") || message.includes("unknown session")) {
    return new ProviderAdapterSessionNotFoundError({
      provider: PROVIDER,
      threadId,
    });
  }
  if (message.includes("session is closed") || message.includes("connection closed")) {
    return new ProviderAdapterSessionClosedError({
      provider: PROVIDER,
      threadId,
    });
  }
  return new ProviderAdapterRequestError({
    provider: PROVIDER,
    method,
    detail: toMessage(cause, `ACP ${method} request failed.`),
    cause,
  });
}

// ---------------------------------------------------------------------------
// Helpers — cleanup a single session (internal, does not throw)
// ---------------------------------------------------------------------------

function stopSessionInternal(ctx: GeminiAcpSessionContext, emitExitEvent: boolean): void {
  if (ctx.stopped) return;
  ctx.stopped = true;

  for (const [, pending] of ctx.pendingApprovals) {
    pending.resolve({ outcome: { outcome: "cancelled" } });
  }
  ctx.pendingApprovals.clear();

  try {
    ctx.child.kill();
  } catch {
    // Process may already have exited
  }

  if (emitExitEvent) {
    // Note: we cannot emit here because the queue may be shut down.
    // Caller is responsible for emitting events when needed.
  }
}

// ---------------------------------------------------------------------------
// Layer implementation
// ---------------------------------------------------------------------------

const makeGeminiAcpAdapter = Effect.gen(function* () {
  const settingsService = yield* ServerSettingsService;

  const runtimeEventQueue = yield* Queue.unbounded<ProviderRuntimeEvent>();
  const sessions = new Map<string, GeminiAcpSessionContext>();

  const emit = (...events: ReadonlyArray<ProviderRuntimeEvent>) =>
    Queue.offerAll(runtimeEventQueue, events).pipe(Effect.runSync);

  // ------------------------------------------------------------------
  // ACP Client handler factory — called per-session
  // ------------------------------------------------------------------

  function makeAcpClient(ctx: GeminiAcpSessionContext): Client {
    return {
      async sessionUpdate(notification: SessionNotification): Promise<void> {
        if (ctx.stopped) return;
        const update = notification.update;
        const base = makeEventBase(ctx);

        switch (update.sessionUpdate) {
          case "agent_message_chunk": {
            // ----- Replay deduplication (layered) -----
            // The Gemini ACP replays the full conversation history as
            // sessionUpdate notifications on each prompt() call.
            //
            // Layer 0 — no active turn:
            // Between turns (after the previous turn's prompt resolved and
            // before sendTurn sets the next turnState), any agent_message_chunk
            // is either a late replay or a leftover from the previous prompt.
            // Fresh model output can only arrive while a turn is active.
            if (!ctx.turnState) {
              break; // No active turn — definitively replay/leftover.
            }
            // Layer 1 — messageId (if available):
            const chunkMessageId = (update as { messageId?: string | null }).messageId;
            if (chunkMessageId && ctx.completedMessageIds.has(chunkMessageId)) {
              break; // Known replay — skip.
            }
            // Layer 2 — user_message_chunk timing:
            // user_message_chunk NEVER appears during normal turns (user
            // input goes through prompt()).  If we've seen one recently,
            // the ACP is replaying history.  Skip agent chunks that arrive
            // within 2 s of the last user_message_chunk — real model
            // generation always has a longer processing gap.
            if (ctx.replayActive && Date.now() - ctx.lastReplaySignalTime < 2000) {
              break; // Still within replay window — skip.
            }
            // If we passed both checks, replay is over.
            ctx.replayActive = false;

            if (chunkMessageId) {
              ctx.currentTurnMessageIds.add(chunkMessageId);
            }
            ctx.turnHasFreshContent = true;

            const text =
              update.content && "text" in update.content ? update.content.text : undefined;
            if (text) {
              // agent_message_chunk is always the model's visible response.
              // Reasoning/thinking is delivered via agent_thought_chunk instead.
              // Use a segment-scoped itemId so that text before and after tool calls
              // becomes separate assistant messages instead of being concatenated.
              const segmentItemId = ctx.turnState
                ? RuntimeItemId.makeUnsafe(
                    `msg-seg-${ctx.turnState.turnId}-${ctx.assistantMessageSegment}`,
                  )
                : undefined;
              emit({
                ...base,
                ...(segmentItemId ? { itemId: segmentItemId } : {}),
                type: "content.delta",
                payload: { streamKind: "assistant_text", delta: text },
              } satisfies ProviderRuntimeEvent);
            }
            break;
          }
          case "agent_thought_chunk": {
            // Gemini ACP sends agent_thought_chunk notifications that contain
            // internal conversation recaps rather than genuine reasoning steps.
            // Streaming these as visible reasoning_text causes the full
            // conversation to be re-printed as a "Reasoning" work-log entry.
            // Suppress until Gemini exposes opt-in thinking support.
            break;
          }
          case "tool_call": {
            if (!ctx.turnState) break; // No active turn — replay/leftover.
            // Each tool call creates a boundary — assistant text after this
            // point should be a separate message from text before it.
            ctx.seenToolCallIds.add(update.toolCallId);
            ctx.assistantMessageSegment++;

            const itemId = RuntimeItemId.makeUnsafe(update.toolCallId);
            const itemType = mapToolKindToItemType(update.kind);
            const title = update.title ?? undefined;
            if (update.status === "completed" || update.status === "failed") {
              emit({
                ...base,
                itemId,
                type: "item.completed",
                payload: { itemType, status: "completed", title },
              } satisfies ProviderRuntimeEvent);
              // Emit turn.diff.updated for file-modifying tools so that
              // CheckpointReactor captures a git checkpoint (same as Codex).
              if (update.status === "completed" && isFileModifyingKind(update.kind)) {
                emit({
                  ...base,
                  itemId,
                  type: "turn.diff.updated",
                  payload: { unifiedDiff: "" },
                } satisfies ProviderRuntimeEvent);
              }
              // In plan mode, detect completed plan file writes and emit a
              // proposed plan.  This covers auto-approved tools that bypass
              // requestPermission.  Gated to file-modifying kinds to avoid
              // capturing reads of arbitrary .md files as plans.
              if (
                update.status === "completed" &&
                ctx.currentAcpMode === "plan" &&
                isFileModifyingKind(update.kind)
              ) {
                const planMarkdown = await tryReadPlanFile({ locations: update.locations, title });
                if (planMarkdown) {
                  emit({
                    ...base,
                    type: "turn.proposed.completed",
                    payload: { planMarkdown },
                  } satisfies ProviderRuntimeEvent);
                }
              }
            } else {
              emit({
                ...base,
                itemId,
                type: "item.started",
                payload: { itemType, status: "inProgress", title },
              } satisfies ProviderRuntimeEvent);
            }
            break;
          }
          case "tool_call_update": {
            if (!ctx.turnState) break; // No active turn — replay/leftover.
            // If this is a tool call we haven't seen yet (i.e. no prior
            // tool_call event), create a segment boundary so that pre-tool
            // and post-tool assistant text become separate messages.
            if (update.toolCallId && !ctx.seenToolCallIds.has(update.toolCallId)) {
              ctx.seenToolCallIds.add(update.toolCallId);
              ctx.assistantMessageSegment++;
            }
            const itemId = update.toolCallId
              ? RuntimeItemId.makeUnsafe(update.toolCallId)
              : undefined;
            const title = update.title ?? undefined;
            if (update.status === "completed" || update.status === "failed") {
              emit({
                ...base,
                ...(itemId ? { itemId } : {}),
                type: "item.completed",
                payload: {
                  itemType: mapToolKindToItemType(update.kind),
                  status: "completed",
                  title,
                },
              } satisfies ProviderRuntimeEvent);
              // Emit turn.diff.updated for file-modifying tools so that
              // CheckpointReactor captures a git checkpoint (same as Codex).
              if (update.status === "completed" && isFileModifyingKind(update.kind)) {
                emit({
                  ...base,
                  ...(itemId ? { itemId } : {}),
                  type: "turn.diff.updated",
                  payload: { unifiedDiff: "" },
                } satisfies ProviderRuntimeEvent);
              }
              // In plan mode, detect completed plan file writes and emit a
              // proposed plan (same as the tool_call handler above).
              if (
                update.status === "completed" &&
                ctx.currentAcpMode === "plan" &&
                isFileModifyingKind(update.kind)
              ) {
                const planMarkdown = await tryReadPlanFile({ locations: update.locations, title });
                if (planMarkdown) {
                  emit({
                    ...base,
                    type: "turn.proposed.completed",
                    payload: { planMarkdown },
                  } satisfies ProviderRuntimeEvent);
                }
              }
            } else {
              emit({
                ...base,
                ...(itemId ? { itemId } : {}),
                type: "item.updated",
                payload: {
                  itemType: mapToolKindToItemType(update.kind),
                  status: "inProgress",
                  title,
                },
              } satisfies ProviderRuntimeEvent);
            }
            break;
          }
          case "plan": {
            // Plan has `entries` array, emit a summary as plan_text
            const entries = update.entries;
            if (entries && entries.length > 0) {
              const text = entries.map((e) => `- [${e.status}] ${e.content}`).join("\n");
              emit({
                ...base,
                type: "content.delta",
                payload: {
                  streamKind: "plan_text",
                  delta: text,
                },
              } satisfies ProviderRuntimeEvent);
            }
            break;
          }
          case "current_mode_update": {
            // Track mode changes from the agent side
            const modeId = (update as { currentModeId?: string }).currentModeId;
            if (modeId) {
              ctx.currentAcpMode = modeId;
            }
            break;
          }
          case "usage_update": {
            // Context window update from the agent — has `size` (total) and `used` (in-context).
            // These are absolute values that supersede any incremental accounting.
            const usageUpdate = update as { size: number; used: number };
            ctx.lastKnownMaxTokens = usageUpdate.size;
            ctx.cumulativeInputTokens = usageUpdate.used;
            ctx.cumulativeOutputTokens = 0;
            ctx.cumulativeReasoningTokens = 0;
            ctx.turnReceivedUsageUpdate = true;
            // Only emit when replay is over (fresh content has been seen).
            // During replay the ACP re-sends historical usage_update
            // notifications with stale values that cause the counter to
            // flicker/reset.  The prompt-response handler will emit final
            // usage regardless, so nothing is lost.
            if (ctx.turnHasFreshContent || !ctx.turnState) {
              emit({
                ...base,
                type: "thread.token-usage.updated",
                payload: {
                  usage: {
                    usedTokens: usageUpdate.used,
                    maxTokens: usageUpdate.size,
                  } as ThreadTokenUsageSnapshot,
                },
              } satisfies ProviderRuntimeEvent);
            }
            break;
          }
          case "user_message_chunk": {
            // user_message_chunk during a turn is a definitive replay signal.
            // Normal user messages go through prompt(), never as a
            // notification.  Any user_message_chunk we see is the ACP
            // replaying conversation history.
            ctx.replayActive = true;
            ctx.lastReplaySignalTime = Date.now();
            const chunkMessageId = (update as { messageId?: string | null }).messageId;
            if (chunkMessageId) {
              ctx.currentTurnMessageIds.add(chunkMessageId);
            }
            break;
          }
          default:
            // Ignore unknown session update types (available_commands_update, etc.)
            break;
        }
      },

      async requestPermission(
        request: RequestPermissionRequest,
      ): Promise<RequestPermissionResponse> {
        const base = makeEventBase(ctx);
        const toolCallUpdate = request.toolCall;
        const requestType = mapToolKindToRequestType(toolCallUpdate?.kind);
        const detail = toolCallUpdate?.title ?? "Tool approval requested";

        // Intercept plan approvals: if we are in plan mode and the tool
        // references a .md file, read its content and emit it as a proposed
        // plan (like Claude's ExitPlanMode) instead of showing a bare
        // approval panel.  Only active in plan mode to avoid capturing
        // regular .md file edits as plans.
        const planMarkdown =
          ctx.currentAcpMode === "plan"
            ? await tryReadPlanFile({
                locations: toolCallUpdate?.locations,
                title: detail,
              })
            : undefined;
        if (planMarkdown) {
          emit({
            ...base,
            type: "turn.proposed.completed",
            payload: { planMarkdown },
          } satisfies ProviderRuntimeEvent);

          // Approve the plan file write so the agent sees its plan as
          // successfully presented.  Cancelling would tell the agent the
          // plan was rejected, which breaks refinement on subsequent turns
          // (the agent refuses to re-write or enters a degraded state).
          const allowOption = request.options.find((opt) => opt.kind === "allow_once");
          return {
            outcome: {
              outcome: "selected",
              optionId: allowOption?.optionId ?? "proceed_once",
            },
          };
        }

        const requestId = `gemini-perm-${Date.now()}-${++eventCounter}`;

        // Create the Promise first so `resolve` is captured before emitting
        // the event (which may trigger respondToRequest synchronously).
        const promise = new Promise<RequestPermissionResponse>((resolve) => {
          ctx.pendingApprovals.set(requestId, { requestType, detail, resolve });
        });

        emit({
          ...base,
          requestId: RuntimeRequestId.makeUnsafe(requestId),
          type: "request.opened",
          payload: { requestType, detail, args: request },
        } satisfies ProviderRuntimeEvent);

        return promise;
      },
    };
  }

  // ------------------------------------------------------------------
  // Process spawning
  // ------------------------------------------------------------------

  function spawnAcpProcess(binaryPath: string): {
    child: NodeChildProcess;
    connection: ClientSideConnection;
    clientRef: { value: Client | undefined };
  } {
    const child = spawn(binaryPath, ["--acp"], {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env },
    });

    const stdout = Readable.toWeb(child.stdout!) as ReadableStream<Uint8Array>;
    const stdin = Writable.toWeb(child.stdin!) as WritableStream<Uint8Array>;

    const stream = ndJsonStream(stdin, stdout);
    const clientRef: { value: Client | undefined } = { value: undefined };

    const connection = new ClientSideConnection(() => {
      const proxy: Client = {
        sessionUpdate: (params) => clientRef.value?.sessionUpdate(params) ?? Promise.resolve(),
        requestPermission: (params) => {
          if (clientRef.value?.requestPermission) {
            return clientRef.value.requestPermission(params);
          }
          return Promise.resolve({
            outcome: { outcome: "cancelled" as const },
          });
        },
      };
      return proxy;
    }, stream);

    child.stderr?.on("data", (chunk: Buffer) => {
      const text = chunk.toString().trim();
      if (text) {
        console.error(`[gemini-acp stderr] ${text}`);
      }
    });

    return { child, connection, clientRef };
  }

  // ------------------------------------------------------------------
  // Adapter method: startSession
  // ------------------------------------------------------------------

  // Helper: extract ACP session ID from a resumeCursor if present
  function extractAcpSessionId(cursor: unknown): string | undefined {
    if (
      cursor &&
      typeof cursor === "object" &&
      "acpSessionId" in cursor &&
      typeof (cursor as Record<string, unknown>).acpSessionId === "string"
    ) {
      return (cursor as Record<string, unknown>).acpSessionId as string;
    }
    return undefined;
  }

  const startSession: GeminiAcpAdapterShape["startSession"] = (input: ProviderSessionStartInput) =>
    Effect.gen(function* () {
      if (input.provider !== undefined && input.provider !== PROVIDER) {
        return yield* new ProviderAdapterValidationError({
          provider: PROVIDER,
          operation: "startSession",
          issue: `Expected provider "${PROVIDER}", got "${input.provider}".`,
        });
      }

      const geminiSettings = yield* settingsService.getSettings.pipe(
        Effect.map((settings) => settings.providers.geminiAcp),
        Effect.mapError(
          (cause) =>
            new ProviderAdapterProcessError({
              provider: PROVIDER,
              threadId: input.threadId,
              detail: "Failed to read Gemini ACP settings.",
              cause,
            }),
        ),
      );

      const { child, connection, clientRef } = yield* Effect.try({
        try: () => spawnAcpProcess(geminiSettings.binaryPath),
        catch: (cause) =>
          new ProviderAdapterProcessError({
            provider: PROVIDER,
            threadId: input.threadId,
            detail: toMessage(cause, "Failed to spawn gemini process."),
            cause,
          }),
      });

      // Initialize the ACP connection
      const initResponse = yield* Effect.tryPromise({
        try: () =>
          connection.initialize({
            protocolVersion: PROTOCOL_VERSION,
            clientCapabilities: {},
            clientInfo: {
              name: "t3code",
              version: "0.1.0",
            },
          }),
        catch: (cause) =>
          new ProviderAdapterProcessError({
            provider: PROVIDER,
            threadId: input.threadId,
            detail: toMessage(cause, "ACP initialize failed."),
            cause,
          }),
      });

      // Authenticate — use the first available auth method
      const authMethods = initResponse.authMethods ?? [];
      const firstAuth = authMethods[0];
      if (firstAuth) {
        yield* Effect.tryPromise({
          try: () =>
            connection.authenticate({
              methodId: firstAuth.id,
            }),
          catch: (cause) =>
            new ProviderAdapterProcessError({
              provider: PROVIDER,
              threadId: input.threadId,
              detail: toMessage(
                cause,
                'Gemini authentication failed. Run "gemini login" and try again.',
              ),
              cause,
            }),
        });
      }

      // Try to resume an existing ACP session if we have a resumeCursor,
      // otherwise create a new one.
      const previousAcpSessionId = extractAcpSessionId(input.resumeCursor);
      let acpSessionId: string;

      if (previousAcpSessionId) {
        // Attempt session/load to resume the previous conversation
        const loadResult = yield* Effect.tryPromise(() =>
          connection.loadSession({
            sessionId: previousAcpSessionId,
            cwd: input.cwd ?? process.cwd(),
            mcpServers: [],
          }),
        ).pipe(Effect.option);

        if (Option.isSome(loadResult)) {
          acpSessionId = previousAcpSessionId;
        } else {
          // loadSession failed (session expired or not found) — create fresh
          const newSessionResponse = yield* Effect.tryPromise({
            try: () =>
              connection.newSession({
                cwd: input.cwd ?? process.cwd(),
                mcpServers: [],
              }),
            catch: (cause) =>
              new ProviderAdapterProcessError({
                provider: PROVIDER,
                threadId: input.threadId,
                detail: toMessage(cause, "Failed to create ACP session."),
                cause,
              }),
          });
          acpSessionId = newSessionResponse.sessionId;
        }
      } else {
        const newSessionResponse = yield* Effect.tryPromise({
          try: () =>
            connection.newSession({
              cwd: input.cwd ?? process.cwd(),
              mcpServers: [],
            }),
          catch: (cause) =>
            new ProviderAdapterProcessError({
              provider: PROVIDER,
              threadId: input.threadId,
              detail: toMessage(cause, "Failed to create ACP session."),
              cause,
            }),
        });
        acpSessionId = newSessionResponse.sessionId;
      }

      const now = new Date().toISOString();
      const providerSession: ProviderSession = {
        provider: PROVIDER,
        status: "ready" as const,
        runtimeMode: input.runtimeMode,
        threadId: input.threadId,
        resumeCursor: { threadId: input.threadId, acpSessionId },
        createdAt: now,
        updatedAt: now,
      };

      const ctx: GeminiAcpSessionContext = {
        session: providerSession,
        child,
        connection,
        acpSessionId,
        pendingApprovals: new Map(),
        turnState: undefined,
        promptReject: undefined,
        stopped: false,
        currentAcpMode: "default",
        assistantMessageSegment: 0,
        seenToolCallIds: new Set(),
        cumulativeInputTokens: 0,
        cumulativeOutputTokens: 0,
        cumulativeReasoningTokens: 0,
        lastKnownMaxTokens: undefined,
        turnReceivedUsageUpdate: false,
        completedMessageIds: new Set(),
        currentTurnMessageIds: new Set(),
        turnHasFreshContent: false,
        replayActive: false,
        lastReplaySignalTime: 0,
      };

      // Wire the ACP Client callbacks to this session context
      clientRef.value = makeAcpClient(ctx);

      // Handle process exit
      child.on("exit", (code, signal) => {
        if (!ctx.stopped) {
          ctx.stopped = true;
          emit({
            ...makeEventBase(ctx),
            type: "session.exited",
            payload: {
              reason: signal ? `Process killed by ${signal}` : `Process exited with code ${code}`,
              recoverable: false,
              exitKind: code === 0 ? "graceful" : "error",
            },
          } satisfies ProviderRuntimeEvent);

          ctx.promptReject?.(new Error("Gemini ACP process exited unexpectedly."));
        }
      });

      sessions.set(input.threadId, ctx);

      // Sync ACP approval mode based on runtimeMode
      const initialAcpMode = resolveAcpModeId(undefined, input.runtimeMode);
      if (initialAcpMode !== ctx.currentAcpMode) {
        const modeResult = yield* Effect.tryPromise(() =>
          connection.setSessionMode({
            sessionId: acpSessionId,
            modeId: initialAcpMode,
          }),
        ).pipe(Effect.option);
        if (Option.isSome(modeResult)) {
          ctx.currentAcpMode = initialAcpMode;
        }
      }

      emit({
        ...makeEventBase(ctx),
        type: "session.started",
        payload: {},
      } satisfies ProviderRuntimeEvent);

      return providerSession;
    });

  // ------------------------------------------------------------------
  // Adapter method: sendTurn
  // ------------------------------------------------------------------

  // Resolve the desired ACP session mode from T3Code's interactionMode and runtimeMode.
  // Plan mode overrides everything; otherwise runtimeMode determines approval behavior.
  function resolveAcpModeId(interactionMode: string | undefined, runtimeMode: string): string {
    if (interactionMode === "plan") return "plan";
    if (runtimeMode === "full-access") return "autoEdit";
    return "default";
  }

  const sendTurn: GeminiAcpAdapterShape["sendTurn"] = (input: ProviderSendTurnInput) =>
    Effect.gen(function* () {
      const ctx = sessions.get(input.threadId);
      if (!ctx) {
        return yield* new ProviderAdapterSessionNotFoundError({
          provider: PROVIDER,
          threadId: input.threadId,
        });
      }
      if (ctx.stopped) {
        return yield* new ProviderAdapterSessionClosedError({
          provider: PROVIDER,
          threadId: input.threadId,
        });
      }

      // Sync interaction mode + runtime mode to ACP session mode before sending the turn
      const desiredAcpMode = resolveAcpModeId(input.interactionMode, ctx.session.runtimeMode);
      if (desiredAcpMode !== ctx.currentAcpMode) {
        const result = yield* Effect.tryPromise(() =>
          ctx.connection.setSessionMode({
            sessionId: ctx.acpSessionId,
            modeId: desiredAcpMode,
          }),
        ).pipe(Effect.option);
        if (Option.isSome(result)) {
          ctx.currentAcpMode = desiredAcpMode;
        }
      }

      const turnId = `turn-${Date.now()}-${++eventCounter}`;
      const now = new Date().toISOString();
      ctx.turnState = { turnId, startedAt: now };
      ctx.assistantMessageSegment = 0;
      ctx.seenToolCallIds.clear();
      ctx.turnReceivedUsageUpdate = false;

      // Finalize previous turn's messageIds so replayed content is skipped.
      for (const id of ctx.currentTurnMessageIds) {
        ctx.completedMessageIds.add(id);
      }
      ctx.currentTurnMessageIds.clear();
      ctx.turnHasFreshContent = false;
      ctx.replayActive = false;
      ctx.lastReplaySignalTime = 0;

      emit({
        ...makeEventBase(ctx),
        type: "turn.started",
        payload: {},
      } satisfies ProviderRuntimeEvent);

      // Build ACP prompt content blocks
      const promptBlocks: Array<{ type: "text"; text: string }> = [];

      // In plan mode, prepend instructions that constrain the model to planning
      // only.  The ACP session mode controls tool *approval* policy but does not
      // instruct the model itself — without this the model will eventually start
      // executing changes even while the session is in plan mode.
      if (desiredAcpMode === "plan") {
        promptBlocks.push({ type: "text", text: GEMINI_PLAN_MODE_PROMPT });
      }

      if (input.input) {
        promptBlocks.push({ type: "text", text: input.input });
      }

      // Fire the ACP prompt in the background — it returns when the turn completes
      const promptPromise = ctx.connection
        .prompt({
          sessionId: ctx.acpSessionId,
          prompt: promptBlocks.length > 0 ? promptBlocks : [{ type: "text", text: "" }],
        })
        .then((response) => {
          const state =
            response.stopReason === "cancelled" ? ("cancelled" as const) : ("completed" as const);

          // Extract token usage from the prompt response.
          // Gemini CLI returns usage in _meta.quota.token_count; the ACP SDK
          // also defines a standard `usage` field — we check both.
          const acpUsage = response.usage as
            | {
                inputTokens?: number;
                outputTokens?: number;
                thoughtTokens?: number;
                totalTokens?: number;
              }
            | null
            | undefined;
          const meta = response._meta as Record<string, unknown> | undefined;
          const quota = meta?.quota as
            | {
                token_count?: { input_tokens?: number; output_tokens?: number };
              }
            | undefined;

          const turnInputTokens =
            acpUsage?.inputTokens ?? quota?.token_count?.input_tokens ?? undefined;
          const turnOutputTokens =
            acpUsage?.outputTokens ?? quota?.token_count?.output_tokens ?? undefined;
          const turnReasoningTokens = acpUsage?.thoughtTokens ?? undefined;

          if (turnInputTokens !== undefined || turnOutputTokens !== undefined) {
            // Only update cumulative counters from the prompt response when
            // no usage_update notification was received during this turn.
            // usage_update provides authoritative absolute values; adding
            // turn tokens on top would double-count and cause the context
            // counter to "reset" when the next usage_update corrects it.
            if (!ctx.turnReceivedUsageUpdate) {
              ctx.cumulativeInputTokens += turnInputTokens ?? 0;
              ctx.cumulativeOutputTokens += turnOutputTokens ?? 0;
              ctx.cumulativeReasoningTokens += turnReasoningTokens ?? 0;
            }

            emit({
              ...makeEventBase(ctx),
              type: "thread.token-usage.updated",
              payload: {
                usage: buildTokenUsageSnapshot(
                  ctx,
                  turnInputTokens,
                  turnOutputTokens,
                  turnReasoningTokens,
                ),
              },
            } satisfies ProviderRuntimeEvent);
          }

          emit({
            ...makeEventBase(ctx),
            type: "turn.completed",
            payload: { state, stopReason: response.stopReason },
          } satisfies ProviderRuntimeEvent);
          ctx.turnState = undefined;
          ctx.promptReject = undefined;
        })
        .catch((error: unknown) => {
          const detail = toMessage(error, "Turn failed.");
          console.error(`[gemini-acp] prompt() rejected:`, error);
          if (!ctx.stopped) {
            emit({
              ...makeEventBase(ctx),
              type: "turn.completed",
              payload: {
                state: "failed",
                errorMessage: detail,
              },
            } satisfies ProviderRuntimeEvent);
          }
          ctx.turnState = undefined;
          ctx.promptReject = undefined;
        });

      void promptPromise;

      return {
        threadId: input.threadId,
        turnId: TurnIdBrand.makeUnsafe(turnId),
        resumeCursor: {
          threadId: input.threadId,
          acpSessionId: ctx.acpSessionId,
        },
      };
    });

  // ------------------------------------------------------------------
  // Adapter method: interruptTurn
  // ------------------------------------------------------------------

  const interruptTurn: GeminiAcpAdapterShape["interruptTurn"] = (
    threadId: ThreadId,
    _turnId?: TurnIdBrand,
  ) =>
    Effect.gen(function* () {
      const ctx = sessions.get(threadId);
      if (!ctx) {
        return yield* new ProviderAdapterSessionNotFoundError({
          provider: PROVIDER,
          threadId,
        });
      }

      yield* Effect.tryPromise({
        try: () => ctx.connection.cancel({ sessionId: ctx.acpSessionId }),
        catch: (cause) => toRequestError(threadId, "cancel", cause),
      });
    });

  // ------------------------------------------------------------------
  // Adapter method: respondToRequest
  // ------------------------------------------------------------------

  const respondToRequest: GeminiAcpAdapterShape["respondToRequest"] = (
    threadId: ThreadId,
    requestId: ApprovalRequestId,
    decision: ProviderApprovalDecision,
  ) =>
    Effect.gen(function* () {
      const ctx = sessions.get(threadId);
      if (!ctx) {
        return yield* new ProviderAdapterSessionNotFoundError({
          provider: PROVIDER,
          threadId,
        });
      }

      const pending = ctx.pendingApprovals.get(requestId);
      if (!pending) {
        return yield* new ProviderAdapterRequestError({
          provider: PROVIDER,
          method: "respondToRequest",
          detail: `No pending approval found for requestId "${requestId}".`,
        });
      }

      ctx.pendingApprovals.delete(requestId);

      const base = makeEventBase(ctx);
      emit({
        ...base,
        requestId: RuntimeRequestId.makeUnsafe(requestId),
        type: "request.resolved",
        payload: {
          requestType: pending.requestType,
          decision,
        },
      } satisfies ProviderRuntimeEvent);

      // Map T3Code decision to ACP response
      if (decision === "accept" || decision === "acceptForSession") {
        pending.resolve({
          outcome: {
            outcome: "selected",
            optionId: decision === "acceptForSession" ? "proceed_always" : "proceed_once",
          },
        });
      } else {
        pending.resolve({
          outcome: { outcome: "cancelled" },
        });
      }
    });

  // ------------------------------------------------------------------
  // Adapter method: respondToUserInput
  // ------------------------------------------------------------------

  const respondToUserInput: GeminiAcpAdapterShape["respondToUserInput"] = (
    _threadId: ThreadId,
    _requestId: ApprovalRequestId,
    _answers: ProviderUserInputAnswers,
  ) => Effect.void;

  // ------------------------------------------------------------------
  // Adapter method: stopSession
  // ------------------------------------------------------------------

  const stopSession: GeminiAcpAdapterShape["stopSession"] = (threadId: ThreadId) =>
    Effect.gen(function* () {
      const ctx = sessions.get(threadId);
      if (!ctx) {
        return yield* new ProviderAdapterSessionNotFoundError({
          provider: PROVIDER,
          threadId,
        });
      }

      ctx.stopped = true;
      sessions.delete(threadId);

      for (const [, pending] of ctx.pendingApprovals) {
        pending.resolve({ outcome: { outcome: "cancelled" } });
      }
      ctx.pendingApprovals.clear();

      yield* Effect.sync(() => {
        try {
          ctx.child.kill();
        } catch {
          // Process may already have exited
        }
      });

      emit({
        ...makeEventBase(ctx),
        type: "session.exited",
        payload: {
          reason: "Session stopped by user.",
          recoverable: false,
          exitKind: "graceful",
        },
      } satisfies ProviderRuntimeEvent);
    });

  // ------------------------------------------------------------------
  // Adapter method: stopAll
  // ------------------------------------------------------------------

  const stopAll: GeminiAcpAdapterShape["stopAll"] = () =>
    Effect.sync(() => {
      for (const [, ctx] of sessions) {
        stopSessionInternal(ctx, true);
        emit({
          ...makeEventBase(ctx),
          type: "session.exited",
          payload: {
            reason: "All sessions stopped.",
            recoverable: false,
            exitKind: "graceful",
          },
        } satisfies ProviderRuntimeEvent);
      }
      sessions.clear();
    });

  // ------------------------------------------------------------------
  // Simple query methods
  // ------------------------------------------------------------------

  const listSessions: GeminiAcpAdapterShape["listSessions"] = () =>
    Effect.succeed([...sessions.values()].map((ctx) => ctx.session));

  const hasSession: GeminiAcpAdapterShape["hasSession"] = (threadId: ThreadId) =>
    Effect.succeed(sessions.has(threadId) && !sessions.get(threadId)!.stopped);

  const readThread: GeminiAcpAdapterShape["readThread"] = (threadId: ThreadId) =>
    Effect.succeed({ threadId, turns: [] } satisfies ProviderThreadSnapshot);

  const rollbackThread: GeminiAcpAdapterShape["rollbackThread"] = (
    threadId: ThreadId,
    _numTurns: number,
  ) => Effect.succeed({ threadId, turns: [] } satisfies ProviderThreadSnapshot);

  // ------------------------------------------------------------------
  // Cleanup
  // ------------------------------------------------------------------

  yield* Effect.addFinalizer(() =>
    Effect.sync(() => {
      for (const ctx of sessions.values()) {
        stopSessionInternal(ctx, false);
      }
      sessions.clear();
    }).pipe(Effect.tap(() => Queue.shutdown(runtimeEventQueue))),
  );

  // ------------------------------------------------------------------
  // Return adapter shape
  // ------------------------------------------------------------------

  return GeminiAcpAdapter.of({
    provider: PROVIDER,
    capabilities: {
      sessionModelSwitch: "restart-session",
    },
    startSession,
    sendTurn,
    interruptTurn,
    respondToRequest,
    respondToUserInput,
    stopSession,
    listSessions,
    hasSession,
    readThread,
    rollbackThread,
    stopAll,
    streamEvents: Stream.fromQueue(runtimeEventQueue),
  });
});

export const GeminiAcpAdapterLive = Layer.effect(GeminiAcpAdapter, makeGeminiAcpAdapter);
