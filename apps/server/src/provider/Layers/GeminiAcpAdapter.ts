/**
 * GeminiAcpAdapterLive — @google/gemini-cli-core based implementation.
 *
 * Uses GeminiClient.sendMessageStream() as the primary streaming interface.
 * Tool execution is handled by the library's built-in Scheduler; confirmations
 * are bridged to T3Code's approval flow via the library's MessageBus.
 *
 * @module GeminiAcpAdapterLive
 */
import type {
  ApprovalRequestId,
  CanonicalItemType,
  CanonicalRequestType,
  ProviderApprovalDecision,
  ProviderRuntimeEvent,
  ProviderSendTurnInput,
  ProviderSession,
  ProviderSessionStartInput,
  ProviderUserInputAnswers,
  ThreadId,
  ThreadTokenUsageSnapshot,
  TurnId,
} from "@t3tools/contracts";
import {
  EventId,
  RuntimeItemId,
  RuntimeRequestId,
  TurnId as TurnIdBrand,
} from "@t3tools/contracts";
import { Effect, Layer, Queue, Stream } from "effect";

import { ServerSettingsService } from "../../serverSettings";
import {
  ProviderAdapterProcessError,
  ProviderAdapterRequestError,
  ProviderAdapterSessionClosedError,
  ProviderAdapterSessionNotFoundError,
  ProviderAdapterValidationError,
  type ProviderAdapterError,
} from "../Errors";
import { GeminiAcpAdapter } from "../Services/GeminiAcpAdapter";
import type { ProviderThreadSnapshot } from "../Services/ProviderAdapter";

import {
  AuthType,
  Config,
  LegacyAgentSession,
  MessageBusType,
  ApprovalMode,
  QuestionType,
  ROOT_SCHEDULER_ID,
  Scheduler,
  ToolConfirmationOutcome,
  getAuthTypeFromEnv,
  type AgentEvent,
  type ContentPart,
  type GeminiClient,
  type MessageBus,
  type Question,
  type ToolConfirmationRequest,
} from "@google/gemini-cli-core";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PROVIDER = "geminiAcp" as const;
const DEFAULT_GEMINI_CONTEXT_WINDOW = 1_000_000;
const PLAN_MODE_DENY_LIMIT = 2;

// ---------------------------------------------------------------------------
// Default mode prompt — teaches Gemini when to ask vs act
// ---------------------------------------------------------------------------

const GEMINI_DEFAULT_MODE_PROMPT = `<default_mode>
# Default Mode — ACTIVE

You are in **Default Mode**. Your job is to complete the user's request efficiently and safely.

## Asking vs acting

1. If you can continue safely using local context and reasonable assumptions, do so.
2. If you need a user decision, confirmation, or missing preference that materially changes the next action, ask the user a concise question and then STOP.
3. Do **NOT** ask a blocking question and then continue with file edits, commands, or other side effects in the same turn.
4. If you ask a question because you are waiting for the user's answer, end the turn immediately after the question.
5. Only ask questions that are truly necessary. Do not ask rhetorical "should I continue?" questions when you can simply proceed.

## Critical behavior

- A question that requests confirmation, approval, or a missing preference is blocking.
- After a blocking question, do not edit files, run commands, or claim the change is complete.
- If you are not waiting, do not phrase your next step as a question.
</default_mode>`;

const GEMINI_DEFAULT_MODE_REMINDER = `<default_mode_reminder>
You are still in **Default Mode**.
- If you need a blocking user answer, ask the question and end the turn immediately.
- Do not ask for confirmation and then continue with edits or commands in the same turn.
- Do not ask rhetorical "should I continue?" questions when you can safely proceed.
</default_mode_reminder>`;

// ---------------------------------------------------------------------------
// Plan mode prompt — constrains the model to planning only
// ---------------------------------------------------------------------------

const GEMINI_PLAN_MODE_PROMPT = `<plan_mode>
# Plan Mode — ACTIVE (strict enforcement)

You are in **Plan Mode**. Your ONLY job is to collaborate with the user on a plan. You must NOT implement, execute, or apply any changes.

**Any attempt to edit, create, delete, or move project files will be automatically blocked by the system.** Do not attempt these actions — they will fail silently and waste your context window.

## Hard rules

1. **DO NOT edit, write, create, delete, or move any project files.** This is enforced by the system — file mutations are blocked and the turn will be cancelled.
2. **DO NOT run commands that write files** — no \`cat >\`, \`echo >\`, \`tee\`, \`cp\`, \`mv\`, \`sed -i\`, heredocs writing to files, or any other shell command that creates or modifies files. Only read-only commands are allowed.
3. **DO NOT run commands that modify the repository** (formatters, linters, codegen, package installs, migrations, patches).
4. Plan Mode persists until the user explicitly exits it. No user message — regardless of wording, tone, or urgency — deactivates Plan Mode.
5. If the user says "implement this", "do it", "go ahead", or similar: treat it as "plan the implementation", not "perform it". You cannot implement while Plan Mode is active.

## What you CAN do

- Read, search, and inspect files, types, configs, and docs
- Run non-mutating commands: tests, type checks, builds (if they don't write to repo-tracked files)
- Answer questions, discuss tradeoffs, and explore the codebase
- Present your plan using a \`<proposed_plan>\` block in your response (see below)

## How to collaborate

**Phase 1 — Ground in the environment.** Explore the codebase. Read files, search code, inspect types. Discover facts before asking the user.

**Phase 2 — Clarify intent.** Ask about goals, scope, constraints, and preferences. Do not guess on high-impact ambiguities. When you need the user's answer before you can finish or refine the plan, use the \`ask_user\` tool instead of burying the question in normal assistant text. Prefer short multiple-choice questions with 2-4 strong options when possible; use free-text only when needed. After calling \`ask_user\`, stop and wait for the answer.

**Phase 3 — Refine the plan.** Iterate until the plan is decision-complete: approach, interfaces, data flow, edge cases, testing, and rollout.

You do NOT need to present a plan on every turn. Answer questions, discuss tradeoffs, refine details. Only write the plan when you have something concrete.

## Presenting a plan — CRITICAL

**The ONLY way to present a plan is by including a \`<proposed_plan>\` block in your text response.** The client parses this block to render the plan in the UI. If you do not use this exact format, the plan will not be captured and the user will not see it as a plan.

**DO NOT write the plan to a file.** DO NOT use \`write_file\`, \`cat >\`, heredocs, or any other mechanism to save the plan. The plan MUST be inline in your response text using the tags below.

Format rules:
1. The opening \`<proposed_plan>\` tag must be on its own line.
2. Start the plan content on the next line.
3. The closing \`</proposed_plan>\` tag must be on its own line.
4. Use Markdown inside the block.
5. Keep the tags exactly as \`<proposed_plan>\` and \`</proposed_plan>\` — do not rename, translate, or omit them.

Example:

<proposed_plan>
# Plan Title

Brief summary of the approach.

## Steps
1. Step one — details
2. Step two — details

## Key files
- \`path/to/file.ts\` — what changes
</proposed_plan>

Only produce at most one \`<proposed_plan>\` block per turn, and only when presenting a complete plan. Do not ask "should I proceed?" — the user controls when to exit Plan Mode.

## Refinement turns

When the user sends follow-up messages while Plan Mode is still active, they are asking you to **refine the plan** — not implement it. Read their feedback, update your understanding, and revise the plan if needed. Stay in planning mode.
</plan_mode>`;

const GEMINI_PLAN_MODE_REMINDER = `<plan_mode_reminder>
You are still in **Plan Mode**. Reminders:
- **DO NOT edit, write, or create any files.** No \`write_file\`, no \`cat >\`, no heredocs, no shell writes. File mutations are blocked and the turn will be cancelled.
- **Present your plan ONLY using \`<proposed_plan>\` tags in your text response.** This is the only way the client captures the plan. Do not write the plan to a file.
- If you need a blocking clarification to continue planning, use the \`ask_user\` tool and then stop until the user answers.
- If the user is asking a question or giving feedback, respond in text. Only output a \`<proposed_plan>\` block when you have a complete or updated plan to present.
</plan_mode_reminder>`;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface GeminiTurnState {
  readonly turnId: string;
  readonly startedAt: string;
  reasoningItemEmitted: boolean;
}

interface PendingApproval {
  readonly requestType: CanonicalRequestType;
  readonly detail: string;
  readonly correlationId: string;
  resolve: (response: { confirmed: boolean; outcome: ToolConfirmationOutcome }) => void;
}

interface GeminiUserInputQuestionOption {
  readonly label: string;
  readonly description: string;
}

interface GeminiUserInputQuestion {
  readonly id: string;
  readonly header: string;
  readonly question: string;
  readonly options: ReadonlyArray<GeminiUserInputQuestionOption>;
  readonly multiSelect?: boolean;
}

interface PendingUserInput {
  readonly detail: string;
  readonly correlationId: string;
  readonly questions: ReadonlyArray<GeminiUserInputQuestion>;
  resolve: (answers: ProviderUserInputAnswers | undefined) => void;
}

interface ToolInFlight {
  readonly itemId: string;
  readonly itemType: CanonicalItemType;
  readonly toolName: string;
  readonly title: string;
  readonly input: Record<string, unknown>;
}

type GeminiResumeHistory = Parameters<GeminiClient["resumeChat"]>[0];

interface GeminiResumeState {
  readonly history: GeminiResumeHistory;
  readonly turnCount: number;
}

interface GeminiSessionContext {
  readonly session: ProviderSession;
  readonly config: Config;
  readonly geminiClient: GeminiClient;
  abortController: AbortController;
  readonly messageBusUnsubscribers: Array<() => void>;
  turnState: GeminiTurnState | undefined;
  stopped: boolean;
  readonly pendingApprovals: Map<string, PendingApproval>;
  readonly pendingUserInputs: Map<string, PendingUserInput>;
  // Plan mode
  userRequestedMode: string;
  planModePromptSent: boolean;
  defaultModePromptSent: boolean;
  planModeDeniedInTurn: number;
  planModeTextSuppressed: boolean;
  turnAssistantText: string;
  emittedAssistantTextLength: number;
  assistantMessageSegment: number;
  // Token tracking
  cumulativeInputTokens: number;
  cumulativeOutputTokens: number;
  cumulativeReasoningTokens: number;
  lastKnownMaxTokens: number | undefined;
  // Tool tracking
  readonly inFlightTools: Map<string, ToolInFlight>;
  readonly turns: Array<{ id: string; items: unknown[] }>;
  activeStreamPromise: Promise<void> | undefined;
  activeAgentSession:
    | {
        abort: () => Promise<void>;
      }
    | undefined;
}

// ---------------------------------------------------------------------------
// Helpers — event construction
// ---------------------------------------------------------------------------

let eventCounter = 0;
function nextEventId(): string {
  return `evt-gemini-${Date.now()}-${++eventCounter}`;
}

function makeEventBase(ctx: GeminiSessionContext): Omit<ProviderRuntimeEvent, "type" | "payload"> {
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
// Helpers — tool classification by name
// ---------------------------------------------------------------------------

function classifyToolName(name: string): CanonicalItemType {
  const lower = name.toLowerCase();
  if (
    lower.includes("read") ||
    lower.includes("list") ||
    lower.includes("glob") ||
    lower.includes("grep") ||
    lower.includes("search_file") ||
    lower.includes("ls")
  ) {
    return "file_change";
  }
  if (
    lower.includes("edit") ||
    lower.includes("write") ||
    lower.includes("patch") ||
    lower.includes("replace") ||
    lower.includes("delete_file") ||
    lower.includes("move_file") ||
    lower.includes("rename")
  ) {
    return "file_change";
  }
  if (
    lower.includes("shell") ||
    lower.includes("command") ||
    lower.includes("exec") ||
    lower.includes("bash") ||
    lower.includes("terminal")
  ) {
    return "command_execution";
  }
  if (
    lower.includes("web_search") ||
    lower.includes("web_fetch") ||
    lower.includes("fetch") ||
    lower.includes("browse")
  ) {
    return "web_search";
  }
  if (lower.includes("think") || lower.includes("reason")) {
    return "reasoning";
  }
  if (lower.includes("mcp") || lower.startsWith("mcp_")) {
    return "mcp_tool_call";
  }
  if (lower.includes("agent") || lower.includes("subagent")) {
    return "collab_agent_tool_call";
  }
  return "dynamic_tool_call";
}

function classifyRequestTypeForTool(name: string): CanonicalRequestType {
  const type = classifyToolName(name);
  switch (type) {
    case "file_change":
      return "file_change_approval";
    case "command_execution":
      return "exec_command_approval";
    default:
      return "command_execution_approval";
  }
}

function isFileModifyingTool(name: string): boolean {
  const lower = name.toLowerCase();
  return (
    lower.includes("edit") ||
    lower.includes("write") ||
    lower.includes("delete") ||
    lower.includes("move") ||
    lower.includes("rename") ||
    lower.includes("patch") ||
    lower.includes("replace") ||
    lower.includes("create_file")
  );
}

function isShellWriteTool(name: string, args: Record<string, unknown>): boolean {
  if (!name.toLowerCase().includes("shell")) return false;
  const cmd = String(args.command ?? args.cmd ?? "").toLowerCase();
  return /\b(cat\s*>|echo\s*>|tee\s|cp\s|mv\s|sed\s+-i|rm\s|mkdir|touch)\b/.test(cmd);
}

function titleForToolType(type: CanonicalItemType): string {
  switch (type) {
    case "file_change":
      return "File operation";
    case "command_execution":
      return "Command execution";
    case "web_search":
      return "Web search";
    case "mcp_tool_call":
      return "MCP tool call";
    case "collab_agent_tool_call":
      return "Agent tool call";
    case "reasoning":
      return "Thinking";
    default:
      return "Tool call";
  }
}

function normalizeGeminiAskUserOption(option: unknown): GeminiUserInputQuestionOption | undefined {
  if (!option || typeof option !== "object") {
    return undefined;
  }

  const candidate = option as Record<string, unknown>;
  const label = typeof candidate.label === "string" ? candidate.label.trim() : "";
  const description =
    typeof candidate.description === "string" ? candidate.description.trim() : label;

  if (!label) {
    return undefined;
  }

  return {
    label,
    description: description || label,
  };
}

function fallbackOptionsForGeminiQuestion(question: {
  readonly type?: unknown;
  readonly placeholder?: unknown;
}): ReadonlyArray<GeminiUserInputQuestionOption> {
  if (question.type === QuestionType.YESNO) {
    return [
      { label: "Yes", description: "Yes" },
      { label: "No", description: "No" },
    ];
  }

  const placeholder = typeof question.placeholder === "string" ? question.placeholder.trim() : "";

  return [
    {
      label: "Use custom answer",
      description: placeholder || "Type your answer in the composer below",
    },
  ];
}

export function normalizeGeminiAskUserQuestions(
  rawQuestions: unknown,
): ReadonlyArray<GeminiUserInputQuestion> {
  if (!Array.isArray(rawQuestions)) {
    return [];
  }

  return rawQuestions
    .map<GeminiUserInputQuestion | null>((entry, index) => {
      if (!entry || typeof entry !== "object") {
        return null;
      }

      const question = entry as Question;
      const prompt = typeof question.question === "string" ? question.question.trim() : "";
      if (!prompt) {
        return null;
      }

      const header =
        typeof question.header === "string" && question.header.trim().length > 0
          ? question.header.trim()
          : `Question ${index + 1}`;

      const normalizedOptions = Array.isArray(question.options)
        ? question.options
            .map((option) => normalizeGeminiAskUserOption(option))
            .filter((option): option is GeminiUserInputQuestionOption => option !== undefined)
        : [];

      const options =
        normalizedOptions.length > 0
          ? normalizedOptions
          : fallbackOptionsForGeminiQuestion(question);

      return {
        id: `q-${index}`,
        header,
        question: prompt,
        options,
        ...(typeof question.multiSelect === "boolean" ? { multiSelect: question.multiSelect } : {}),
      };
    })
    .filter((question): question is GeminiUserInputQuestion => question !== null);
}

export function buildGeminiAskUserResponseAnswers(input: {
  readonly questions: ReadonlyArray<GeminiUserInputQuestion>;
  readonly answers: ProviderUserInputAnswers;
}): Record<string, string> {
  const indexedAnswers: Record<string, string> = {};

  for (const [index, question] of input.questions.entries()) {
    const answer =
      input.answers[question.id] ??
      input.answers[String(index)] ??
      input.answers[question.question];

    if (typeof answer !== "string") {
      continue;
    }

    const trimmed = answer.trim();
    if (!trimmed) {
      continue;
    }

    indexedAnswers[String(index)] = trimmed;
  }

  return indexedAnswers;
}

function openGeminiUserInputRequest(input: {
  readonly ctx: GeminiSessionContext;
  readonly messageBus: MessageBus;
  readonly emitEvent: (event: ProviderRuntimeEvent) => void;
  readonly correlationId: string;
  readonly rawQuestions: unknown;
  readonly responseChannel: "tool_confirmation" | "ask_user";
}): boolean {
  const questions = normalizeGeminiAskUserQuestions(input.rawQuestions);
  if (questions.length === 0) {
    if (input.responseChannel === "tool_confirmation") {
      void input.messageBus.publish({
        type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
        correlationId: input.correlationId,
        confirmed: false,
        outcome: ToolConfirmationOutcome.Cancel,
      });
    } else {
      void input.messageBus.publish({
        type: MessageBusType.ASK_USER_RESPONSE,
        correlationId: input.correlationId,
        answers: {},
        cancelled: true,
      } as never);
    }
    return false;
  }

  const requestId = `user-input-${input.correlationId}`;
  if (input.ctx.pendingUserInputs.has(requestId)) {
    return true;
  }

  input.ctx.pendingUserInputs.set(requestId, {
    detail: questions[0]?.question ?? "User input requested",
    correlationId: input.correlationId,
    questions,
    resolve: (answers) => {
      if (input.responseChannel === "tool_confirmation") {
        void input.messageBus.publish({
          type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
          correlationId: input.correlationId,
          confirmed: answers !== undefined,
          outcome:
            answers === undefined
              ? ToolConfirmationOutcome.Cancel
              : ToolConfirmationOutcome.ProceedOnce,
          ...(answers
            ? {
                payload: {
                  answers: buildGeminiAskUserResponseAnswers({ questions, answers }),
                },
              }
            : {}),
        });
        return;
      }

      void input.messageBus.publish({
        type: MessageBusType.ASK_USER_RESPONSE,
        correlationId: input.correlationId,
        answers: answers ? buildGeminiAskUserResponseAnswers({ questions, answers }) : {},
        ...(answers ? {} : { cancelled: true }),
      } as never);
    },
  });

  const { ctx } = input;
  input.emitEvent({
    ...makeEventBase(ctx),
    requestId: RuntimeRequestId.makeUnsafe(requestId),
    type: "user-input.requested",
    payload: {
      questions,
    },
  } as ProviderRuntimeEvent);
  return true;
}

function openGeminiUserInputRequestFromScheduler(input: {
  readonly ctx: GeminiSessionContext;
  readonly scheduler: Scheduler;
  readonly messageBus: MessageBus;
  readonly emitEvent: (event: ProviderRuntimeEvent) => void;
}): boolean {
  const schedulerState = (
    input.scheduler as unknown as {
      state?: {
        allActiveCalls?: ReadonlyArray<{
          status?: string;
          correlationId?: string;
          confirmationDetails?: { type?: string; questions?: unknown };
        }>;
      };
    }
  ).state;

  const waitingAskUserCall = schedulerState?.allActiveCalls?.find(
    (call) =>
      call.status === "awaiting_approval" &&
      typeof call.correlationId === "string" &&
      call.confirmationDetails?.type === "ask_user",
  );

  if (!waitingAskUserCall?.correlationId) {
    return false;
  }

  return openGeminiUserInputRequest({
    ctx: input.ctx,
    messageBus: input.messageBus,
    emitEvent: input.emitEvent,
    correlationId: waitingAskUserCall.correlationId,
    rawQuestions: waitingAskUserCall.confirmationDetails?.questions,
    responseChannel: "tool_confirmation",
  });
}

interface GeminiSchedulerAwaitingApprovalCall {
  readonly status?: string;
  readonly correlationId?: string;
  readonly request?: {
    readonly name?: string;
    readonly args?: unknown;
  };
  readonly confirmationDetails?: {
    readonly type?: string;
    readonly fileName?: string;
    readonly command?: string;
    readonly toolName?: string;
    readonly toolDisplayName?: string;
    readonly prompt?: string;
  };
}

interface GeminiSchedulerApprovalRequest {
  readonly correlationId: string;
  readonly requestType: CanonicalRequestType;
  readonly detail: string;
  readonly args: Record<string, unknown>;
}

function buildGeminiSchedulerApprovalDetail(call: GeminiSchedulerAwaitingApprovalCall): string {
  const details = call.confirmationDetails;
  if (!details) {
    return call.request?.name ?? "Tool operation";
  }

  switch (details.type) {
    case "edit":
      return `Edit ${details.fileName ?? "file"}`;
    case "exec":
      return details.command ?? "Execute command";
    case "mcp":
      return `MCP: ${details.toolDisplayName ?? details.toolName ?? "tool"}`;
    case "info":
      return details.prompt ?? call.request?.name ?? "Tool operation";
    case "exit_plan_mode":
      return "Exit plan mode";
    default:
      return call.request?.name ?? "Tool operation";
  }
}

export function extractGeminiSchedulerApprovalRequest(
  call: GeminiSchedulerAwaitingApprovalCall,
): GeminiSchedulerApprovalRequest | undefined {
  if (
    call.status !== "awaiting_approval" ||
    typeof call.correlationId !== "string" ||
    typeof call.request?.name !== "string" ||
    call.confirmationDetails?.type === "ask_user"
  ) {
    return undefined;
  }

  return {
    correlationId: call.correlationId,
    requestType: classifyRequestTypeForTool(call.request.name),
    detail: buildGeminiSchedulerApprovalDetail(call),
    args:
      call.request.args &&
      typeof call.request.args === "object" &&
      !Array.isArray(call.request.args)
        ? (call.request.args as Record<string, unknown>)
        : {},
  };
}

function openGeminiApprovalRequestFromScheduler(input: {
  readonly ctx: GeminiSessionContext;
  readonly scheduler: Scheduler;
  readonly messageBus: MessageBus;
  readonly emitEvent: (event: ProviderRuntimeEvent) => void;
}): boolean {
  const schedulerState = (
    input.scheduler as unknown as {
      state?: {
        allActiveCalls?: ReadonlyArray<GeminiSchedulerAwaitingApprovalCall>;
      };
    }
  ).state;

  const approval = schedulerState?.allActiveCalls
    ?.map((call) => extractGeminiSchedulerApprovalRequest(call))
    .find((call): call is GeminiSchedulerApprovalRequest => call !== undefined);

  if (!approval) {
    return false;
  }

  const requestId = `req-${approval.correlationId}`;
  if (input.ctx.pendingApprovals.has(requestId)) {
    return true;
  }

  input.ctx.pendingApprovals.set(requestId, {
    requestType: approval.requestType,
    detail: approval.detail,
    correlationId: approval.correlationId,
    resolve: (response) => {
      void input.messageBus.publish({
        type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
        correlationId: approval.correlationId,
        confirmed: response.confirmed,
        outcome: response.outcome,
      });
    },
  });

  input.emitEvent({
    ...makeEventBase(input.ctx),
    requestId: RuntimeRequestId.makeUnsafe(requestId),
    type: "request.opened",
    payload: {
      requestType: approval.requestType,
      detail: approval.detail,
      args: approval.args,
    },
  } as ProviderRuntimeEvent);
  return true;
}

// ---------------------------------------------------------------------------
// Helpers — proposed plan extraction (exported for tests)
// ---------------------------------------------------------------------------

const PROPOSED_PLAN_BLOCK_REGEX = /<proposed_plan>\s*([\s\S]*?)\s*<\/proposed_plan>/i;
const PROPOSED_PLAN_OPEN_TAG = "<proposed_plan>";
const PROPOSED_PLAN_CLOSE_TAG = "</proposed_plan>";

function extractProposedPlanMarkdown(text: string | undefined): string | undefined {
  const match = text ? PROPOSED_PLAN_BLOCK_REGEX.exec(text) : null;
  const planMarkdown = match?.[1]?.trim();
  return planMarkdown && planMarkdown.length > 0 ? planMarkdown : undefined;
}

function trailingTagPrefixLength(text: string, tag: string): number {
  const maxLength = Math.min(text.length, tag.length - 1);
  for (let length = maxLength; length > 0; length--) {
    if (text.endsWith(tag.slice(0, length))) {
      return length;
    }
  }
  return 0;
}

export function extractVisibleAssistantText(rawText: string): string {
  let visibleText = "";
  let cursor = 0;

  while (cursor < rawText.length) {
    const openTagIndex = rawText.indexOf(PROPOSED_PLAN_OPEN_TAG, cursor);
    if (openTagIndex === -1) {
      const trailingText = rawText.slice(cursor);
      const holdBackLength = trailingTagPrefixLength(trailingText, PROPOSED_PLAN_OPEN_TAG);
      visibleText += trailingText.slice(0, trailingText.length - holdBackLength);
      break;
    }

    visibleText += rawText.slice(cursor, openTagIndex);
    const closeTagIndex = rawText.indexOf(
      PROPOSED_PLAN_CLOSE_TAG,
      openTagIndex + PROPOSED_PLAN_OPEN_TAG.length,
    );
    if (closeTagIndex === -1) {
      break;
    }
    cursor = closeTagIndex + PROPOSED_PLAN_CLOSE_TAG.length;
  }

  return visibleText;
}

// ---------------------------------------------------------------------------
// Helpers — prompt building (exported for tests)
// ---------------------------------------------------------------------------

interface GeminiPromptBlock {
  readonly type: "text";
  readonly text: string;
}

export function buildGeminiPromptBlocks(input: {
  readonly interactionMode: string | undefined;
  readonly userInput: string | undefined;
  readonly planModePromptSent: boolean;
  readonly defaultModePromptSent: boolean;
}): {
  readonly promptBlocks: ReadonlyArray<GeminiPromptBlock>;
  readonly planModePromptSent: boolean;
  readonly defaultModePromptSent: boolean;
} {
  const promptBlocks: GeminiPromptBlock[] = [];
  let planModePromptSent = input.planModePromptSent;
  let defaultModePromptSent = input.defaultModePromptSent;

  if (input.interactionMode === "plan") {
    if (planModePromptSent) {
      promptBlocks.push({ type: "text", text: GEMINI_PLAN_MODE_REMINDER });
    } else {
      promptBlocks.push({ type: "text", text: GEMINI_PLAN_MODE_PROMPT });
      planModePromptSent = true;
    }
  } else if (defaultModePromptSent) {
    promptBlocks.push({ type: "text", text: GEMINI_DEFAULT_MODE_REMINDER });
  } else {
    promptBlocks.push({ type: "text", text: GEMINI_DEFAULT_MODE_PROMPT });
    defaultModePromptSent = true;
  }

  if (input.userInput) {
    promptBlocks.push({ type: "text", text: input.userInput });
  }

  return {
    promptBlocks,
    planModePromptSent,
    defaultModePromptSent,
  };
}

// ---------------------------------------------------------------------------
// Helpers — token usage
// ---------------------------------------------------------------------------

export function shouldApplyUsageUpdate(
  turnState: GeminiSessionContext["turnState"],
  turnHasFreshContent: boolean,
): boolean {
  return !turnState || turnHasFreshContent;
}

export function formatAgentThoughtText(input: {
  readonly thought: string;
  readonly subject: string | undefined;
}): string {
  const thought = input.thought.trim();
  if (!input.subject) return thought;
  return thought.length > 0 ? `**${input.subject}** ${thought}` : `**${input.subject}**`;
}

function isTextContentPart(part: ContentPart): part is Extract<ContentPart, { type: "text" }> {
  return part.type === "text";
}

function buildTokenUsageSnapshot(
  ctx: GeminiSessionContext,
  turnInputTokens?: number,
  turnOutputTokens?: number,
): ThreadTokenUsageSnapshot {
  const usedTokens = ctx.cumulativeInputTokens + ctx.cumulativeOutputTokens;
  const maxTokens = ctx.lastKnownMaxTokens ?? DEFAULT_GEMINI_CONTEXT_WINDOW;
  return {
    usedTokens,
    maxTokens,
    ...(turnInputTokens !== undefined ? { lastInputTokens: turnInputTokens } : {}),
    ...(turnOutputTokens !== undefined ? { lastOutputTokens: turnOutputTokens } : {}),
    ...(turnInputTokens !== undefined || turnOutputTokens !== undefined
      ? { lastUsedTokens: (turnInputTokens ?? 0) + (turnOutputTokens ?? 0) }
      : {}),
  } as ThreadTokenUsageSnapshot;
}

// ---------------------------------------------------------------------------
// Helpers — errors
// ---------------------------------------------------------------------------

function toMessage(cause: unknown, fallback: string): string {
  if (cause instanceof Error) return cause.message;
  if (typeof cause === "string") return cause;
  if (typeof cause === "object" && cause !== null) {
    const obj = cause as Record<string, unknown>;
    if (typeof obj.message === "string") return obj.message;
    const str = String(cause);
    if (str !== "[object Object]") return str;
    try {
      return JSON.stringify(cause);
    } catch {
      /* fall through */
    }
  }
  return fallback;
}

export function resolveGeminiAuthType(
  authTypeFromEnv: AuthType | undefined = getAuthTypeFromEnv(),
): AuthType {
  return authTypeFromEnv ?? AuthType.LOGIN_WITH_GOOGLE;
}

export function readGeminiResumeState(resumeCursor: unknown): GeminiResumeState | undefined {
  if (!resumeCursor || typeof resumeCursor !== "object" || Array.isArray(resumeCursor)) {
    return undefined;
  }

  const rawState = resumeCursor as {
    history?: unknown;
    turnCount?: unknown;
  };
  if (!Array.isArray(rawState.history)) {
    return undefined;
  }

  const turnCount =
    typeof rawState.turnCount === "number" &&
    Number.isSafeInteger(rawState.turnCount) &&
    rawState.turnCount >= 0
      ? rawState.turnCount
      : 0;

  return {
    history: rawState.history as GeminiResumeHistory,
    turnCount,
  };
}

function createGeminiResumeState(ctx: GeminiSessionContext): GeminiResumeState {
  return {
    history: [...ctx.geminiClient.getHistory()],
    turnCount: ctx.turns.length,
  };
}

function updateResumeCursor(ctx: GeminiSessionContext): void {
  (ctx.session as { resumeCursor?: unknown }).resumeCursor = createGeminiResumeState(ctx);
}

// ---------------------------------------------------------------------------
// Adapter factory
// ---------------------------------------------------------------------------

const makeGeminiAcpAdapter = Effect.gen(function* () {
  const settingsService = yield* ServerSettingsService;
  const runtimeEventQueue = yield* Queue.unbounded<ProviderRuntimeEvent>();

  const sessions = new Map<string, GeminiSessionContext>();

  // Emit events to the unbounded queue. We use unsafeOffer via runSync
  // because events are emitted from both Effect fibers and plain JS callbacks
  // (MessageBus handlers). For an unbounded queue, offer never blocks.
  const emit = (event: ProviderRuntimeEvent): void => {
    Effect.runSync(Queue.offer(runtimeEventQueue, event));
  };

  const abortActiveTurn = (ctx: GeminiSessionContext): void => {
    ctx.abortController.abort();
    const activeAgentSession = ctx.activeAgentSession;
    if (!activeAgentSession) return;
    void activeAgentSession.abort().catch((error: unknown) => {
      console.error("[gemini] Failed to abort active agent session:", error);
    });
  };

  // ---------------------------------------------------------------------------
  // Session lookup
  // ---------------------------------------------------------------------------

  const getSession = (
    threadId: ThreadId,
  ): Effect.Effect<GeminiSessionContext, ProviderAdapterError> => {
    const ctx = sessions.get(threadId);
    if (!ctx) {
      return Effect.fail(
        new ProviderAdapterSessionNotFoundError({
          provider: PROVIDER,
          threadId,
        }),
      );
    }
    if (ctx.stopped) {
      return Effect.fail(new ProviderAdapterSessionClosedError({ provider: PROVIDER, threadId }));
    }
    return Effect.succeed(ctx);
  };

  // ---------------------------------------------------------------------------
  // MessageBus tool confirmation handler
  // ---------------------------------------------------------------------------

  function setupMessageBusHandlers(ctx: GeminiSessionContext, messageBus: MessageBus): void {
    const handleConfirmation = (request: ToolConfirmationRequest): void => {
      if (ctx.stopped || !ctx.turnState) return;

      const toolName = request.toolCall?.name ?? "unknown";
      const correlationId = request.correlationId;
      const args = (request.toolCall?.args ?? {}) as Record<string, unknown>;
      const askUserQuestions =
        request.details?.type === "ask_user"
          ? normalizeGeminiAskUserQuestions(request.details.questions)
          : undefined;

      if (askUserQuestions && askUserQuestions.length > 0) {
        openGeminiUserInputRequest({
          ctx,
          messageBus,
          emitEvent: emit,
          correlationId,
          rawQuestions: askUserQuestions,
          responseChannel: "tool_confirmation",
        });
        return;
      }

      // Plan mode: auto-deny file-modifying tools, auto-approve read-only
      if (ctx.userRequestedMode === "plan") {
        if (isFileModifyingTool(toolName) || isShellWriteTool(toolName, args)) {
          ctx.planModeDeniedInTurn++;
          ctx.planModeTextSuppressed = true;

          void messageBus.publish({
            type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
            correlationId,
            confirmed: false,
            outcome: ToolConfirmationOutcome.Cancel,
          });

          if (ctx.planModeDeniedInTurn >= PLAN_MODE_DENY_LIMIT) {
            abortActiveTurn(ctx);
          }
          return;
        }
        // Auto-approve read-only tools in plan mode
        void messageBus.publish({
          type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
          correlationId,
          confirmed: true,
          outcome: ToolConfirmationOutcome.ProceedOnce,
        });
        return;
      }

      // Full-access mode: auto-approve everything
      if (ctx.session.runtimeMode === "full-access") {
        void messageBus.publish({
          type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
          correlationId,
          confirmed: true,
          outcome: ToolConfirmationOutcome.ProceedOnce,
        });
        return;
      }
    };

    messageBus.subscribe(MessageBusType.TOOL_CONFIRMATION_REQUEST, handleConfirmation as any);
    ctx.messageBusUnsubscribers.push(() => {
      messageBus.unsubscribe(MessageBusType.TOOL_CONFIRMATION_REQUEST, handleConfirmation as any);
    });

    // Handle ask_user requests
    const handleAskUser = (request: any): void => {
      if (ctx.stopped || !ctx.turnState) return;

      const correlationId = request.correlationId as string;
      openGeminiUserInputRequest({
        ctx,
        messageBus,
        emitEvent: emit,
        correlationId,
        rawQuestions: request.questions,
        responseChannel: "ask_user",
      });
    };

    messageBus.subscribe(MessageBusType.ASK_USER_REQUEST, handleAskUser as any);
    ctx.messageBusUnsubscribers.push(() => {
      messageBus.unsubscribe(MessageBusType.ASK_USER_REQUEST, handleAskUser as any);
    });
  }

  // ---------------------------------------------------------------------------
  // Stream event processing
  // ---------------------------------------------------------------------------

  function emitAssistantText(ctx: GeminiSessionContext, text: string): void {
    if (!ctx.turnState || ctx.planModeTextSuppressed) return;

    ctx.turnAssistantText += text;

    const fullVisible = extractVisibleAssistantText(ctx.turnAssistantText);
    const newText = fullVisible.slice(ctx.emittedAssistantTextLength);
    ctx.emittedAssistantTextLength = fullVisible.length;

    if (newText.length === 0) return;

    const itemId = `msg-${ctx.turnState.turnId}-${ctx.assistantMessageSegment}`;
    emit({
      ...makeEventBase(ctx),
      itemId: RuntimeItemId.makeUnsafe(itemId),
      type: "content.delta",
      payload: {
        streamKind: "assistant_text",
        delta: newText,
      },
    } as ProviderRuntimeEvent);
  }

  function emitReasoningText(ctx: GeminiSessionContext, thoughtText: string): void {
    if (!ctx.turnState) return;

    if (!ctx.turnState.reasoningItemEmitted) {
      ctx.turnState.reasoningItemEmitted = true;
      const itemId = `reasoning-${ctx.turnState.turnId}`;
      emit({
        ...makeEventBase(ctx),
        itemId: RuntimeItemId.makeUnsafe(itemId),
        type: "item.started",
        payload: {
          itemType: "reasoning",
          title: "Thinking",
        },
      } as ProviderRuntimeEvent);
    }

    emit({
      ...makeEventBase(ctx),
      itemId: RuntimeItemId.makeUnsafe(`reasoning-${ctx.turnState.turnId}`),
      type: "content.delta",
      payload: {
        streamKind: "reasoning_text",
        delta: thoughtText,
      },
    } as ProviderRuntimeEvent);
  }

  function handleAgentEvent(ctx: GeminiSessionContext, event: AgentEvent): void {
    if (!ctx.turnState) return;

    switch (event.type) {
      case "message": {
        if (event.role !== "agent") break;
        if (event._meta?.citation === true) break;

        for (const part of event.content) {
          if (part.type === "text") {
            emitAssistantText(ctx, part.text);
            continue;
          }
          if (part.type === "thought") {
            emitReasoningText(
              ctx,
              formatAgentThoughtText({
                thought: part.thought,
                subject: typeof event._meta?.subject === "string" ? event._meta.subject : undefined,
              }),
            );
          }
        }
        break;
      }

      case "tool_request": {
        const itemType = classifyToolName(event.name);
        const itemId = `tool-${event.requestId}`;

        ctx.assistantMessageSegment++;
        ctx.planModeTextSuppressed = false;

        ctx.inFlightTools.set(event.requestId, {
          itemId,
          itemType,
          toolName: event.name,
          title: titleForToolType(itemType),
          input: event.args ?? {},
        });

        emit({
          ...makeEventBase(ctx),
          itemId: RuntimeItemId.makeUnsafe(itemId),
          type: "item.started",
          payload: {
            itemType,
            title: titleForToolType(itemType),
            detail: event.name,
            data: event.args ?? {},
          },
        } as ProviderRuntimeEvent);
        break;
      }

      case "tool_response": {
        const existingTool = ctx.inFlightTools.get(event.requestId);
        const tool =
          existingTool ??
          ({
            itemId: `tool-${event.requestId}`,
            itemType: classifyToolName(event.name),
            toolName: event.name,
            title: titleForToolType(classifyToolName(event.name)),
            input: {},
          } satisfies ToolInFlight);

        ctx.inFlightTools.delete(event.requestId);

        const outputParts = (event.displayContent ?? event.content)?.filter(
          (part): part is Extract<ContentPart, { type: "text" }> =>
            isTextContentPart(part) && part.text.length > 0,
        );
        if (outputParts && outputParts.length > 0) {
          const outputText = outputParts.map((part) => part.text).join("\n");
          const streamKind =
            tool.itemType === "command_execution" ? "command_output" : "file_change_output";
          emit({
            ...makeEventBase(ctx),
            itemId: RuntimeItemId.makeUnsafe(tool.itemId),
            type: "content.delta",
            payload: {
              streamKind,
              delta: outputText,
            },
          } as ProviderRuntimeEvent);
        }

        emit({
          ...makeEventBase(ctx),
          itemId: RuntimeItemId.makeUnsafe(tool.itemId),
          type: "item.completed",
          payload: {
            itemType: tool.itemType,
            status: event.isError ? "failed" : "completed",
          },
        } as ProviderRuntimeEvent);
        break;
      }

      case "usage": {
        ctx.cumulativeInputTokens += event.inputTokens ?? 0;
        ctx.cumulativeOutputTokens += event.outputTokens ?? 0;
        break;
      }

      case "error": {
        console.error("[gemini] Agent session error event:", event.message, event._meta);
        break;
      }

      case "agent_end": {
        const state =
          event.reason === "aborted"
            ? "interrupted"
            : event.reason === "failed"
              ? "failed"
              : "completed";
        const errorMessage =
          state === "failed" && typeof event.data?.message === "string"
            ? event.data.message
            : undefined;
        completeTurn(ctx, state, event.reason, errorMessage);
        break;
      }

      case "initialize":
      case "session_update":
      case "agent_start":
      case "tool_update":
      case "elicitation_request":
      case "elicitation_response":
      case "custom":
        break;
    }
  }

  // ---------------------------------------------------------------------------
  // Stream loop — background turn processing
  // ---------------------------------------------------------------------------

  async function runStreamLoop(ctx: GeminiSessionContext, promptText: string): Promise<void> {
    const promptId = `prompt-${Date.now()}`;
    let scheduler: Scheduler;
    scheduler = new Scheduler({
      context: ctx.config,
      messageBus: ctx.config.getMessageBus(),
      getPreferredEditor: () => undefined,
      schedulerId: ROOT_SCHEDULER_ID,
      onWaitingForConfirmation: (waiting) => {
        if (!waiting || ctx.stopped || !ctx.turnState) {
          return;
        }

        openGeminiApprovalRequestFromScheduler({
          ctx,
          scheduler,
          messageBus: ctx.config.getMessageBus(),
          emitEvent: emit,
        });

        openGeminiUserInputRequestFromScheduler({
          ctx,
          scheduler,
          messageBus: ctx.config.getMessageBus(),
          emitEvent: emit,
        });
      },
    });
    const agentSession = new LegacyAgentSession({
      client: ctx.geminiClient,
      scheduler,
      config: ctx.config,
      promptId,
    });
    ctx.activeAgentSession = agentSession;

    const abortListener = (): void => {
      void agentSession.abort().catch((error: unknown) => {
        console.error("[gemini] Failed to abort agent session:", error);
      });
    };
    ctx.abortController.signal.addEventListener("abort", abortListener, {
      once: true,
    });

    try {
      for await (const event of agentSession.sendStream({
        message: [{ type: "text", text: promptText }],
      })) {
        if (ctx.stopped) break;
        handleAgentEvent(ctx, event);
        if (!ctx.turnState) break;
      }

      if (ctx.turnState) {
        completeTurn(
          ctx,
          ctx.abortController.signal.aborted ? "interrupted" : "completed",
          ctx.abortController.signal.aborted ? "aborted" : "completed",
        );
      }
    } catch (err: unknown) {
      if (ctx.stopped) return;
      if (ctx.abortController.signal.aborted) {
        completeTurn(ctx, "interrupted", "interrupted");
      } else {
        console.error("[gemini] Stream error:", err);
        completeTurn(ctx, "failed", "error", toMessage(err, "Stream error"));
      }
    } finally {
      ctx.abortController.signal.removeEventListener("abort", abortListener);
      ctx.activeAgentSession = undefined;
      scheduler.dispose();
    }
  }

  // ---------------------------------------------------------------------------
  // Turn completion
  // ---------------------------------------------------------------------------

  function completeTurn(
    ctx: GeminiSessionContext,
    state: "completed" | "failed" | "interrupted" | "cancelled",
    stopReason: string,
    errorMessage?: string,
  ): void {
    if (!ctx.turnState) return;

    // Complete remaining in-flight tools
    for (const [, tool] of ctx.inFlightTools) {
      emit({
        ...makeEventBase(ctx),
        itemId: RuntimeItemId.makeUnsafe(tool.itemId),
        type: "item.completed",
        payload: {
          itemType: tool.itemType,
          status: "failed",
        },
      } as ProviderRuntimeEvent);
    }
    ctx.inFlightTools.clear();

    // Complete reasoning item if open
    if (ctx.turnState.reasoningItemEmitted) {
      emit({
        ...makeEventBase(ctx),
        itemId: RuntimeItemId.makeUnsafe(`reasoning-${ctx.turnState.turnId}`),
        type: "item.completed",
        payload: {
          itemType: "reasoning",
          status: "completed",
        },
      } as ProviderRuntimeEvent);
    }

    // Cancel pending approvals
    for (const [, pending] of ctx.pendingApprovals) {
      pending.resolve({
        confirmed: false,
        outcome: ToolConfirmationOutcome.Cancel,
      });
    }
    ctx.pendingApprovals.clear();

    for (const [requestId, pending] of ctx.pendingUserInputs) {
      pending.resolve(undefined);
      emit({
        ...makeEventBase(ctx),
        requestId: RuntimeRequestId.makeUnsafe(requestId),
        type: "user-input.resolved",
        payload: {
          answers: {},
        },
      } as ProviderRuntimeEvent);
    }
    ctx.pendingUserInputs.clear();

    // Extract proposed plan from assistant text
    if (ctx.userRequestedMode === "plan") {
      const plan = extractProposedPlanMarkdown(ctx.turnAssistantText);
      if (plan) {
        emit({
          ...makeEventBase(ctx),
          type: "turn.proposed.completed",
          payload: {
            planMarkdown: plan,
          },
        } as ProviderRuntimeEvent);
      }
    }

    // Emit token usage
    const tokenUsage = buildTokenUsageSnapshot(ctx);
    emit({
      ...makeEventBase(ctx),
      type: "thread.token-usage.updated",
      payload: {
        usage: tokenUsage,
      },
    } as ProviderRuntimeEvent);

    // Store turn
    const turnId = ctx.turnState.turnId;
    ctx.turns.push({ id: turnId, items: [] });
    updateResumeCursor(ctx);

    emit({
      ...makeEventBase(ctx),
      type: "turn.completed",
      payload: {
        state,
        stopReason,
        usage: tokenUsage,
        ...(errorMessage ? { errorMessage } : {}),
      },
    } as ProviderRuntimeEvent);

    ctx.turnState = undefined;

    // Return to ready state
    emit({
      ...makeEventBase(ctx),
      type: "session.state.changed",
      payload: { state: "ready" },
    } as ProviderRuntimeEvent);
  }

  // ---------------------------------------------------------------------------
  // Adapter methods
  // ---------------------------------------------------------------------------

  const startSession = (
    input: ProviderSessionStartInput,
  ): Effect.Effect<ProviderSession, ProviderAdapterError> =>
    Effect.gen(function* () {
      if (input.provider && input.provider !== PROVIDER) {
        return yield* new ProviderAdapterValidationError({
          provider: PROVIDER,
          operation: "startSession",
          issue: `Expected provider "${PROVIDER}", got "${input.provider}"`,
        });
      }

      yield* settingsService.getSettings.pipe(
        Effect.map((s) => s.providers.geminiAcp),
        Effect.orDie,
      );

      const threadId = input.threadId;
      const modelId = input.modelSelection?.model ?? "gemini-2.5-pro";
      const now = new Date().toISOString();
      const runtimeMode = input.runtimeMode ?? "full-access";
      const resumeState = readGeminiResumeState(input.resumeCursor);

      // Create and initialize Config
      const workDir = (input.cwd as string | undefined) ?? process.cwd();
      const config = new Config({
        sessionId: threadId as string,
        targetDir: workDir,
        cwd: workDir,
        model: modelId,
        clientVersion: "0.36.0",
        debugMode: false,
        approvalMode: runtimeMode === "full-access" ? ApprovalMode.AUTO_EDIT : ApprovalMode.DEFAULT,
        interactive: true,
        ptyInfo: "node-pty",
        acpMode: true,
      });

      yield* Effect.tryPromise({
        try: () => config.initialize(),
        catch: (cause) =>
          new ProviderAdapterProcessError({
            provider: PROVIDER,
            threadId,
            detail: `Failed to initialize Gemini config: ${toMessage(cause, "unknown error")}`,
            cause,
          }),
      });

      yield* Effect.tryPromise({
        try: () => config.refreshAuth(resolveGeminiAuthType()),
        catch: (cause) =>
          new ProviderAdapterProcessError({
            provider: PROVIDER,
            threadId,
            detail: `Failed to authenticate Gemini session: ${toMessage(cause, "unknown error")}`,
            cause,
          }),
      });

      // Create and initialize GeminiClient
      const geminiClient = config.getGeminiClient();

      yield* Effect.tryPromise({
        try: () =>
          resumeState ? geminiClient.resumeChat([...resumeState.history]) : Promise.resolve(),
        catch: (cause) =>
          new ProviderAdapterProcessError({
            provider: PROVIDER,
            threadId,
            detail: `Failed to resume Gemini chat: ${toMessage(cause, "unknown error")}`,
            cause,
          }),
      });

      const session: ProviderSession = {
        provider: PROVIDER,
        status: "ready",
        runtimeMode,
        cwd: input.cwd,
        model: modelId,
        threadId,
        createdAt: now,
        updatedAt: now,
      };

      const ctx: GeminiSessionContext = {
        session,
        config,
        geminiClient,
        abortController: new AbortController(),
        messageBusUnsubscribers: [],
        turnState: undefined,
        stopped: false,
        pendingApprovals: new Map(),
        pendingUserInputs: new Map(),
        userRequestedMode: "default",
        planModePromptSent: false,
        defaultModePromptSent: false,
        planModeDeniedInTurn: 0,
        planModeTextSuppressed: false,
        turnAssistantText: "",
        emittedAssistantTextLength: 0,
        assistantMessageSegment: 0,
        cumulativeInputTokens: 0,
        cumulativeOutputTokens: 0,
        cumulativeReasoningTokens: 0,
        lastKnownMaxTokens: undefined,
        inFlightTools: new Map(),
        turns: [],
        activeStreamPromise: undefined,
        activeAgentSession: undefined,
      };
      updateResumeCursor(ctx);

      // Wire up MessageBus for tool confirmations
      const messageBus = config.getMessageBus();
      setupMessageBusHandlers(ctx, messageBus);

      sessions.set(threadId, ctx);

      // Emit session lifecycle events
      emit({
        ...makeEventBase(ctx),
        type: "session.started",
        payload: {},
      } as ProviderRuntimeEvent);
      emit({
        ...makeEventBase(ctx),
        type: "session.configured",
        payload: { config: { model: modelId } },
      } as ProviderRuntimeEvent);
      emit({
        ...makeEventBase(ctx),
        type: "session.state.changed",
        payload: { state: "ready" },
      } as ProviderRuntimeEvent);

      return session;
    });

  const sendTurn = (
    input: ProviderSendTurnInput,
  ): Effect.Effect<
    { threadId: ThreadId; turnId: TurnId; resumeCursor?: unknown },
    ProviderAdapterError
  > =>
    Effect.gen(function* () {
      const ctx = yield* getSession(input.threadId);

      const interactionMode = input.interactionMode ?? ctx.userRequestedMode;
      ctx.userRequestedMode = interactionMode;

      const now = new Date().toISOString();
      const turnId = `turn-${Date.now()}-${ctx.turns.length}`;

      // Reset turn state
      ctx.turnState = {
        turnId,
        startedAt: now,
        reasoningItemEmitted: false,
      };
      ctx.planModeDeniedInTurn = 0;
      ctx.planModeTextSuppressed = false;
      ctx.turnAssistantText = "";
      ctx.emittedAssistantTextLength = 0;
      ctx.assistantMessageSegment = 0;
      ctx.inFlightTools.clear();
      ctx.abortController = new AbortController();

      // Build prompt
      const { promptBlocks, planModePromptSent, defaultModePromptSent } = buildGeminiPromptBlocks({
        interactionMode,
        userInput: input.input?.trim(),
        planModePromptSent: ctx.planModePromptSent,
        defaultModePromptSent: ctx.defaultModePromptSent,
      });
      ctx.planModePromptSent = planModePromptSent;
      ctx.defaultModePromptSent = defaultModePromptSent;

      const promptText = promptBlocks.map((b) => b.text).join("\n\n");

      // Update session status (cast to mutable for internal tracking)
      (ctx.session as { status: string }).status = "running";
      (ctx.session as { activeTurnId?: TurnId }).activeTurnId = TurnIdBrand.makeUnsafe(turnId);
      (ctx.session as { updatedAt: string }).updatedAt = now;

      // Emit turn started
      emit({
        ...makeEventBase(ctx),
        type: "turn.started",
        payload: { model: ctx.session.model },
      } as ProviderRuntimeEvent);
      emit({
        ...makeEventBase(ctx),
        type: "session.state.changed",
        payload: { state: "running" },
      } as ProviderRuntimeEvent);

      // Start stream loop in background
      ctx.activeStreamPromise = runStreamLoop(ctx, promptText);
      ctx.activeStreamPromise.catch((err) => {
        if (!ctx.stopped) {
          console.error("[gemini] Unhandled stream loop error:", err);
        }
      });

      return {
        threadId: input.threadId,
        turnId: TurnIdBrand.makeUnsafe(turnId),
        resumeCursor: ctx.session.resumeCursor,
      };
    });

  const interruptTurn = (
    threadId: ThreadId,
    _turnId?: TurnId,
  ): Effect.Effect<void, ProviderAdapterError> =>
    Effect.gen(function* () {
      const ctx = yield* getSession(threadId);
      abortActiveTurn(ctx);
    });

  const respondToRequest = (
    threadId: ThreadId,
    requestId: ApprovalRequestId,
    decision: ProviderApprovalDecision,
  ): Effect.Effect<void, ProviderAdapterError> =>
    Effect.gen(function* () {
      const ctx = yield* getSession(threadId);
      const pending = ctx.pendingApprovals.get(requestId);
      if (!pending) {
        return yield* new ProviderAdapterRequestError({
          provider: PROVIDER,
          method: "respondToRequest",
          detail: `No pending approval for requestId "${requestId}"`,
        });
      }

      ctx.pendingApprovals.delete(requestId);

      const confirmed = decision === "accept" || decision === "acceptForSession";
      const outcome = confirmed
        ? decision === "acceptForSession"
          ? ToolConfirmationOutcome.ProceedAlways
          : ToolConfirmationOutcome.ProceedOnce
        : ToolConfirmationOutcome.Cancel;

      pending.resolve({ confirmed, outcome });

      emit({
        ...makeEventBase(ctx),
        requestId: RuntimeRequestId.makeUnsafe(requestId),
        type: "request.resolved",
        payload: {
          requestType: pending.requestType,
          decision,
        },
      } as ProviderRuntimeEvent);
    });

  const respondToUserInput = (
    threadId: ThreadId,
    requestId: ApprovalRequestId,
    answers: ProviderUserInputAnswers,
  ): Effect.Effect<void, ProviderAdapterError> =>
    Effect.gen(function* () {
      const ctx = yield* getSession(threadId);
      const pending = ctx.pendingUserInputs.get(requestId);
      if (!pending) {
        return yield* new ProviderAdapterRequestError({
          provider: PROVIDER,
          method: "respondToUserInput",
          detail: `No pending user input for requestId "${requestId}"`,
        });
      }

      ctx.pendingUserInputs.delete(requestId);
      pending.resolve(answers);

      emit({
        ...makeEventBase(ctx),
        requestId: RuntimeRequestId.makeUnsafe(requestId),
        type: "user-input.resolved",
        payload: {
          answers,
        },
      } as ProviderRuntimeEvent);
    });

  const stopSession = (threadId: ThreadId): Effect.Effect<void, ProviderAdapterError> =>
    Effect.gen(function* () {
      const ctx = sessions.get(threadId);
      if (!ctx) return;

      ctx.stopped = true;
      abortActiveTurn(ctx);

      // Clean up MessageBus subscriptions
      for (const unsub of ctx.messageBusUnsubscribers) {
        yield* Effect.sync(() => {
          unsub();
        }).pipe(Effect.ignore);
      }

      // Cancel pending approvals
      for (const [, pending] of ctx.pendingApprovals) {
        pending.resolve({
          confirmed: false,
          outcome: ToolConfirmationOutcome.Cancel,
        });
      }
      ctx.pendingApprovals.clear();

      for (const [requestId, pending] of ctx.pendingUserInputs) {
        pending.resolve(undefined);
        emit({
          ...makeEventBase(ctx),
          requestId: RuntimeRequestId.makeUnsafe(requestId),
          type: "user-input.resolved",
          payload: {
            answers: {},
          },
        } as ProviderRuntimeEvent);
      }
      ctx.pendingUserInputs.clear();

      // Dispose config (handles client cleanup internally)
      yield* Effect.tryPromise({
        try: () => ctx.config.dispose(),
        catch: () => undefined as never, // swallow errors
      }).pipe(Effect.ignore);

      sessions.delete(threadId);
      (ctx.session as { status: string }).status = "closed";

      emit({
        ...makeEventBase(ctx),
        type: "session.exited",
        payload: {
          reason: "stopped",
          exitKind: "graceful",
        },
      } as ProviderRuntimeEvent);
    });

  const listSessions = (): Effect.Effect<ReadonlyArray<ProviderSession>> =>
    Effect.sync(() =>
      Array.from(sessions.values())
        .filter((ctx) => !ctx.stopped)
        .map((ctx) => ctx.session),
    );

  const hasSession = (threadId: ThreadId): Effect.Effect<boolean> =>
    Effect.sync(() => {
      const ctx = sessions.get(threadId);
      return !!ctx && !ctx.stopped;
    });

  const readThread = (
    threadId: ThreadId,
  ): Effect.Effect<ProviderThreadSnapshot, ProviderAdapterError> =>
    Effect.gen(function* () {
      const ctx = yield* getSession(threadId);
      return {
        threadId,
        turns: ctx.turns.map((t) => ({
          id: TurnIdBrand.makeUnsafe(t.id),
          items: t.items,
        })),
      };
    });

  const rollbackThread = (
    threadId: ThreadId,
    numTurns: number,
  ): Effect.Effect<ProviderThreadSnapshot, ProviderAdapterError> =>
    Effect.gen(function* () {
      const ctx = yield* getSession(threadId);

      // Best-effort rollback of conversation history
      yield* Effect.sync(() => {
        const history = ctx.geminiClient.getHistory();
        // Each turn ≈ 2 history entries (user + assistant)
        const entriesToRemove = numTurns * 2;
        const newHistory = history.slice(0, Math.max(0, history.length - entriesToRemove));
        ctx.geminiClient.setHistory([...newHistory]);
      }).pipe(Effect.ignore);

      // Truncate tracked turns
      const removed = ctx.turns.splice(Math.max(0, ctx.turns.length - numTurns));
      void removed;
      updateResumeCursor(ctx);

      return {
        threadId,
        turns: ctx.turns.map((t) => ({
          id: TurnIdBrand.makeUnsafe(t.id),
          items: t.items,
        })),
      };
    });

  const stopAll = (): Effect.Effect<void, ProviderAdapterError> =>
    Effect.gen(function* () {
      const threadIds = Array.from(sessions.keys());
      for (const threadId of threadIds) {
        yield* stopSession(threadId as ThreadId);
      }
    });

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
