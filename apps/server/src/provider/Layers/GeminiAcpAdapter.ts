/**
 * GeminiAcpAdapterLive — @google/gemini-cli-core based implementation.
 *
 * Uses Gemini CLI core chat/tool primitives directly and projects them into
 * T3 Code's provider runtime events and approval flow.
 *
 * @module GeminiAcpAdapterLive
 */
import { readFile } from "node:fs/promises";

import type {
  CanonicalItemType,
  CanonicalRequestType,
  ProviderRuntimeEvent,
  ProviderSession,
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
import type { Content, FunctionCall, Part } from "@google/genai";

import { ServerSettingsService } from "../../serverSettings";
import {
  ProviderAdapterProcessError,
  ProviderAdapterRequestError,
  ProviderAdapterSessionClosedError,
  ProviderAdapterSessionNotFoundError,
  ProviderAdapterValidationError,
  type ProviderAdapterError,
} from "../Errors";
import { GeminiAcpAdapter, type GeminiAcpAdapterShape } from "../Services/GeminiAcpAdapter";
import { GeminiAuthRuntimeState } from "../Services/GeminiAuthRuntimeState";
import {
  type ProviderRuntimeBinding,
  ProviderSessionDirectory,
} from "../Services/ProviderSessionDirectory";

import {
  CoreEvent,
  DiscoveredMCPTool,
  InvalidStreamError,
  LlmRole,
  MessageBusType,
  QuestionType,
  ToolConfirmationOutcome,
  type Config,
  type GeminiClient,
  type GeminiChat,
  type MessageBus,
  type Question,
  type RetryAttemptPayload,
  type ToolConfirmationRequest,
  type ToolCallConfirmationDetails,
  type ToolConfirmationPayload,
  type ToolLiveOutput,
  type ToolResult,
  type RoutingContext,
  type SubagentActivityMessage,
  StreamEventType,
  convertToFunctionResponse,
  coreEvents,
  getCoreSystemPrompt,
  logToolCall,
  ToolCallEvent,
} from "@google/gemini-cli-core";
import {
  buildGeminiManualAuthRequiredMessage,
  resolveGeminiAuthProbeResult,
} from "./GeminiAcpProvider";
import {
  createGeminiCoreConfig,
  installGeminiCliCustomHeaders,
  resolveGeminiApprovalMode,
  resolveGeminiAuthType,
} from "./GeminiCoreConfig";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PROVIDER = "geminiAcp" as const;
const DEFAULT_GEMINI_CONTEXT_WINDOW = 1_000_000;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface GeminiTurnState {
  readonly turnId: string;
  readonly startedAt: string;
  reasoningItemEmitted: boolean;
  readonly capturedProposedPlanKeys: Set<string>;
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
  readonly turnHistoryLengths?: ReadonlyArray<number>;
}

interface GeminiTrackedTurn {
  readonly id: string;
  readonly items: unknown[];
  readonly historyLength?: number;
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
  planModeTextSuppressed: boolean;
  assistantMessageSegment: number;
  assistantMessageText: string;
  // Token tracking
  cumulativeInputTokens: number;
  cumulativeOutputTokens: number;
  // Tool tracking
  readonly inFlightTools: Map<string, ToolInFlight>;
  readonly turns: Array<GeminiTrackedTurn>;
  activeStreamPromise: Promise<void> | undefined;
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

function isGeminiReadOnlyToolName(name: string): boolean {
  return (
    name.includes("read") ||
    name.includes("list") ||
    name.includes("glob") ||
    name.includes("grep") ||
    name.includes("search_file") ||
    name.includes("ls")
  );
}

function isGeminiFileMutationToolName(name: string): boolean {
  return (
    name.includes("edit") ||
    name.includes("write") ||
    name.includes("patch") ||
    name.includes("replace") ||
    name.includes("delete_file") ||
    name.includes("move_file") ||
    name.includes("rename")
  );
}

function isGeminiCommandToolName(name: string): boolean {
  return (
    name.includes("shell") ||
    name.includes("command") ||
    name.includes("exec") ||
    name.includes("bash") ||
    name.includes("terminal")
  );
}

function classifyToolName(name: string): CanonicalItemType {
  const lower = name.toLowerCase();
  if (isGeminiReadOnlyToolName(lower) || isGeminiFileMutationToolName(lower)) {
    return "file_change";
  }
  if (isGeminiCommandToolName(lower)) {
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

export function classifyRequestTypeForTool(name: string): CanonicalRequestType {
  const lower = name.toLowerCase();
  if (isGeminiReadOnlyToolName(lower)) {
    return "file_read_approval";
  }
  if (isGeminiFileMutationToolName(lower)) {
    return "file_change_approval";
  }
  if (isGeminiCommandToolName(lower)) {
    return "exec_command_approval";
  }
  return "command_execution_approval";
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

function buildGeminiPendingId(prefix: "req" | "user-input"): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function forcedGeminiToolDecision(
  ctx: GeminiSessionContext,
  _toolName: string,
  _args: Record<string, unknown>,
): "allow" | "deny" | "ask_user" | undefined {
  if (ctx.userRequestedMode === "plan") {
    return undefined;
  }

  return ctx.session.runtimeMode === "full-access" ? "allow" : undefined;
}

function buildGeminiApprovalDetailFromConfirmation(
  toolName: string,
  confirmation: ToolCallConfirmationDetails,
): string {
  switch (confirmation.type) {
    case "edit":
      return `Edit ${confirmation.filePath || confirmation.fileName || "file"}`;
    case "exec":
      return confirmation.command || "Execute command";
    case "sandbox_expansion":
      return confirmation.command || confirmation.rootCommand || "Expand sandbox";
    case "mcp":
      return `MCP: ${confirmation.toolDisplayName || confirmation.toolName || toolName}`;
    case "exit_plan_mode":
      return "Exit plan mode";
    case "info":
      return confirmation.prompt || toolName;
    case "ask_user":
      return confirmation.questions[0]?.question?.trim() || "User input requested";
    default:
      return toolName;
  }
}

function isGracefulGeminiTurnError(error: unknown): boolean {
  if (error instanceof InvalidStreamError) {
    return true;
  }

  if (!error || typeof error !== "object" || !("type" in error)) {
    return false;
  }

  const type = (error as { type?: unknown }).type;
  return (
    type === "NO_RESPONSE_TEXT" ||
    type === "NO_FINISH_REASON" ||
    type === "MALFORMED_FUNCTION_CALL" ||
    type === "UNEXPECTED_TOOL_CALL"
  );
}

function isGeminiTextPart(part: Part): part is Part & { text: string } {
  return typeof part.text === "string" && part.text.length > 0 && part.thought !== true;
}

function consolidateGeminiModelParts(parts: ReadonlyArray<Part>): Part[] {
  const consolidated: Part[] = [];

  for (const part of parts) {
    const lastPart = consolidated[consolidated.length - 1];
    if (lastPart && isGeminiTextPart(lastPart) && isGeminiTextPart(part)) {
      lastPart.text += part.text;
      continue;
    }
    consolidated.push(part);
  }

  return consolidated;
}

export function buildGeminiAssistantHistoryEntry(parts: ReadonlyArray<Part>): Content | undefined {
  const consolidated = consolidateGeminiModelParts(parts);
  if (consolidated.length === 0) {
    return undefined;
  }

  return {
    role: "model",
    parts: consolidated,
  };
}

function appendGeminiAssistantHistory(chat: GeminiChat, parts: ReadonlyArray<Part>): void {
  const historyEntry = buildGeminiAssistantHistoryEntry(parts);
  if (!historyEntry) {
    return;
  }

  chat.addHistory(historyEntry);
}

function longestGeminiAssistantTextOverlap(existingText: string, incomingText: string): number {
  const maxOverlap = Math.min(existingText.length, incomingText.length);
  for (let overlap = maxOverlap; overlap > 0; overlap--) {
    if (existingText.endsWith(incomingText.slice(0, overlap))) {
      return overlap;
    }
  }
  return 0;
}

export function applyGeminiAssistantTextChunk(
  existingText: string,
  incomingText: string,
): { readonly nextText: string; readonly delta: string } {
  if (incomingText.length === 0) {
    return { nextText: existingText, delta: "" };
  }

  if (existingText.length === 0) {
    return { nextText: incomingText, delta: incomingText };
  }

  // Gemini may stream the full visible message-so-far instead of a true delta.
  if (incomingText.startsWith(existingText)) {
    return {
      nextText: incomingText,
      delta: incomingText.slice(existingText.length),
    };
  }

  if (existingText.startsWith(incomingText) || existingText.includes(incomingText)) {
    return { nextText: existingText, delta: "" };
  }

  const overlap = longestGeminiAssistantTextOverlap(existingText, incomingText);
  if (overlap > 0) {
    const delta = incomingText.slice(overlap);
    return {
      nextText: existingText + delta,
      delta,
    };
  }

  return {
    nextText: existingText + incomingText,
    delta: incomingText,
  };
}

function buildGeminiToolErrorResponse(
  toolName: string,
  callId: string,
  error: Error,
): ReadonlyArray<Part> {
  return [
    {
      functionResponse: {
        id: callId,
        name: toolName,
        response: { error: error.message },
      },
    },
  ];
}

function summarizeGeminiToolResult(result: ToolResult): string | undefined {
  if (typeof result.returnDisplay === "string") {
    return result.returnDisplay;
  }

  if (typeof result.llmContent === "string") {
    return result.llmContent;
  }

  if (Array.isArray(result.llmContent)) {
    const text = result.llmContent
      .map((part) =>
        part && typeof part === "object" && "text" in part && typeof part.text === "string"
          ? part.text
          : "",
      )
      .join("");
    return text.length > 0 ? text : undefined;
  }

  return result.error?.message;
}

function restoreGeminiHistoryFromResumeCursor(ctx: GeminiSessionContext): void {
  const resumeState = readGeminiResumeState(ctx.session.resumeCursor);
  if (!resumeState) {
    return;
  }

  ctx.geminiClient.setHistory([...resumeState.history]);
}

function requestGeminiApproval(input: {
  readonly ctx: GeminiSessionContext;
  readonly toolName: string;
  readonly args: Record<string, unknown>;
  readonly confirmation: Exclude<ToolCallConfirmationDetails, { type: "ask_user" }>;
  readonly emitEvent: (event: ProviderRuntimeEvent) => void;
}): Promise<{ confirmed: boolean; outcome: ToolConfirmationOutcome }> {
  if (input.ctx.abortController.signal.aborted) {
    return Promise.resolve({
      confirmed: false,
      outcome: ToolConfirmationOutcome.Cancel,
    });
  }

  const requestId = buildGeminiPendingId("req");
  return new Promise((resolve) => {
    input.ctx.pendingApprovals.set(requestId, {
      requestType:
        input.confirmation.type === "edit"
          ? "file_change_approval"
          : input.confirmation.type === "exec" || input.confirmation.type === "sandbox_expansion"
            ? "exec_command_approval"
            : classifyRequestTypeForTool(input.toolName),
      detail: buildGeminiApprovalDetailFromConfirmation(input.toolName, input.confirmation),
      correlationId: requestId,
      resolve,
    });

    input.emitEvent({
      ...makeEventBase(input.ctx),
      requestId: RuntimeRequestId.makeUnsafe(requestId),
      type: "request.opened",
      payload: {
        requestType:
          input.confirmation.type === "edit"
            ? "file_change_approval"
            : input.confirmation.type === "exec" || input.confirmation.type === "sandbox_expansion"
              ? "exec_command_approval"
              : classifyRequestTypeForTool(input.toolName),
        detail: buildGeminiApprovalDetailFromConfirmation(input.toolName, input.confirmation),
        args: input.args,
      },
    } as ProviderRuntimeEvent);
  });
}

function requestGeminiUserInput(input: {
  readonly ctx: GeminiSessionContext;
  readonly rawQuestions: unknown;
  readonly emitEvent: (event: ProviderRuntimeEvent) => void;
}): Promise<ProviderUserInputAnswers | undefined> {
  const questions = normalizeGeminiAskUserQuestions(input.rawQuestions);
  if (questions.length === 0 || input.ctx.abortController.signal.aborted) {
    return Promise.resolve(undefined);
  }

  const requestId = buildGeminiPendingId("user-input");
  return new Promise((resolve) => {
    input.ctx.pendingUserInputs.set(requestId, {
      detail: questions[0]?.question ?? "User input requested",
      correlationId: requestId,
      questions,
      resolve,
    });

    input.emitEvent({
      ...makeEventBase(input.ctx),
      requestId: RuntimeRequestId.makeUnsafe(requestId),
      type: "user-input.requested",
      payload: {
        questions,
      },
    } as ProviderRuntimeEvent);
  });
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
                  answers: buildGeminiAskUserResponseAnswers({
                    questions,
                    answers,
                  }),
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

export async function readGeminiPlanMarkdownFromFile(
  planPath: string,
): Promise<string | undefined> {
  const planMarkdown = (await readFile(planPath, "utf8")).trim();
  return planMarkdown.length > 0 ? planMarkdown : undefined;
}

function formatRetryDelayMs(delayMs: number): string {
  if (!Number.isFinite(delayMs) || delayMs <= 0) {
    return "a moment";
  }
  if (delayMs < 1_000) {
    return `${Math.max(1, Math.round(delayMs))}ms`;
  }
  if (delayMs < 60_000) {
    return `${Math.max(1, Math.round(delayMs / 1_000))}s`;
  }
  const minutes = Math.floor(delayMs / 60_000);
  const seconds = Math.round((delayMs % 60_000) / 1_000);
  if (seconds === 0) {
    return `${minutes}m`;
  }
  return `${minutes}m ${seconds}s`;
}

function isGeminiCapacityRetry(payload: Pick<RetryAttemptPayload, "error">): boolean {
  const normalizedError = payload.error?.toLowerCase() ?? "";
  return (
    normalizedError.includes("capacity") ||
    normalizedError.includes("resource_exhausted") ||
    normalizedError.includes("rate_limit") ||
    normalizedError.includes("429") ||
    normalizedError.includes("overloaded")
  );
}

export function formatGeminiRetryWarningMessage(
  payload: Pick<RetryAttemptPayload, "attempt" | "maxAttempts" | "delayMs" | "error">,
): string {
  const retryWindow = formatRetryDelayMs(payload.delayMs);
  const attemptLabel = `attempt ${payload.attempt}/${payload.maxAttempts}`;
  if (isGeminiCapacityRetry(payload)) {
    return `Capacity exhausted, retrying in ${retryWindow} (${attemptLabel})`;
  }
  return `Request retrying in ${retryWindow} (${attemptLabel})`;
}

// ---------------------------------------------------------------------------
// Helpers — token usage
// ---------------------------------------------------------------------------

export function formatGeminiSubagentActivityDetail(message: {
  readonly subagentName: string;
  readonly activity: {
    readonly type: "thought" | "tool_call";
    readonly content: string;
    readonly displayName?: string | undefined;
    readonly description?: string | undefined;
    readonly args?: string | undefined;
    readonly status: "running" | "completed" | "error" | "cancelled";
  };
}): string {
  if (message.activity.type === "thought") {
    return `${message.subagentName}: ${message.activity.content}`;
  }

  const toolLabel =
    message.activity.displayName?.trim() || message.activity.content.trim() || "tool";
  const detailParts = [`${message.subagentName}: ${toolLabel}`];
  if (message.activity.description?.trim()) {
    detailParts.push(message.activity.description.trim());
  }
  if (message.activity.args?.trim()) {
    detailParts.push(message.activity.args.trim());
  }
  if (message.activity.status !== "running") {
    detailParts.push(`status=${message.activity.status}`);
  }
  return detailParts.join(" - ");
}

function buildTokenUsageSnapshot(
  ctx: GeminiSessionContext,
  turnInputTokens?: number,
  turnOutputTokens?: number,
): ThreadTokenUsageSnapshot {
  const usedTokens = ctx.cumulativeInputTokens + ctx.cumulativeOutputTokens;
  return {
    usedTokens,
    maxTokens: DEFAULT_GEMINI_CONTEXT_WINDOW,
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

export function readGeminiResumeState(resumeCursor: unknown): GeminiResumeState | undefined {
  if (!resumeCursor || typeof resumeCursor !== "object" || Array.isArray(resumeCursor)) {
    return undefined;
  }

  const rawState = resumeCursor as {
    history?: unknown;
    turnCount?: unknown;
    turnHistoryLengths?: unknown;
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

  const history = rawState.history as GeminiResumeHistory;
  const explicitTurnHistoryLengths = normalizeGeminiTurnHistoryLengths(
    rawState.turnHistoryLengths,
    turnCount,
    history.length,
  );
  const inferredTurnHistoryLengths =
    explicitTurnHistoryLengths === undefined && turnCount > 0
      ? inferGeminiTurnHistoryLengths(history).slice(0, turnCount)
      : undefined;
  const turnHistoryLengths = explicitTurnHistoryLengths ?? inferredTurnHistoryLengths;

  return {
    history,
    turnCount,
    ...(turnHistoryLengths !== undefined ? { turnHistoryLengths } : {}),
  };
}

function normalizeGeminiTurnHistoryLengths(
  rawLengths: unknown,
  turnCount: number,
  historyLength: number,
): ReadonlyArray<number> | undefined {
  if (!Array.isArray(rawLengths)) {
    return undefined;
  }

  const normalized: number[] = [];
  let previousLength = 0;

  for (const rawLength of rawLengths.slice(0, turnCount)) {
    if (
      typeof rawLength !== "number" ||
      !Number.isSafeInteger(rawLength) ||
      rawLength < 0 ||
      rawLength <= previousLength ||
      rawLength > historyLength
    ) {
      return undefined;
    }

    normalized.push(rawLength);
    previousLength = rawLength;
  }

  if (normalized.length === 0 && turnCount > 0) {
    return undefined;
  }

  return normalized;
}

function isGeminiFunctionResponsePart(
  part: unknown,
): part is { readonly functionResponse: unknown } {
  return (
    typeof part === "object" &&
    part !== null &&
    "functionResponse" in part &&
    part.functionResponse !== undefined
  );
}

function isGeminiToolResponseContent(entry: Content): boolean {
  return (
    entry.role === "user" &&
    Array.isArray(entry.parts) &&
    entry.parts.length > 0 &&
    entry.parts.every(isGeminiFunctionResponsePart)
  );
}

export function inferGeminiTurnHistoryLengths(history: GeminiResumeHistory): ReadonlyArray<number> {
  const turnHistoryLengths: number[] = [];
  let activeTurn = false;

  for (const [index, entry] of history.entries()) {
    if (entry.role !== "user" || isGeminiToolResponseContent(entry as Content)) {
      continue;
    }

    if (activeTurn) {
      turnHistoryLengths.push(index);
    }
    activeTurn = true;
  }

  if (activeTurn) {
    turnHistoryLengths.push(history.length);
  }

  return turnHistoryLengths;
}

function createGeminiResumeState(ctx: GeminiSessionContext): GeminiResumeState {
  const turnHistoryLengths = ctx.turns
    .map((turn) => turn.historyLength)
    .filter((historyLength): historyLength is number => typeof historyLength === "number");

  return {
    history: [...ctx.geminiClient.getHistory()],
    turnCount: ctx.turns.length,
    ...(turnHistoryLengths.length === ctx.turns.length ? { turnHistoryLengths } : {}),
  };
}

function updateResumeCursor(ctx: GeminiSessionContext): void {
  (ctx.session as { resumeCursor?: unknown }).resumeCursor = createGeminiResumeState(ctx);
}

function abortActiveTurn(ctx: GeminiSessionContext): void {
  ctx.abortController.abort();
}

export function buildGeminiPersistedBinding(input: {
  readonly session: ProviderSession;
  readonly status: ProviderRuntimeBinding["status"];
  readonly lastRuntimeEvent: string;
  readonly lastRuntimeEventAt: string;
}): ProviderRuntimeBinding {
  const { session } = input;
  return {
    threadId: session.threadId,
    provider: session.provider,
    runtimeMode: session.runtimeMode,
    ...(input.status !== undefined ? { status: input.status } : {}),
    ...(session.resumeCursor !== undefined ? { resumeCursor: session.resumeCursor } : {}),
    runtimePayload: {
      cwd: session.cwd ?? null,
      model: session.model ?? null,
      activeTurnId: session.activeTurnId ?? null,
      lastError: session.lastError ?? null,
      lastRuntimeEvent: input.lastRuntimeEvent,
      lastRuntimeEventAt: input.lastRuntimeEventAt,
    },
  };
}

// ---------------------------------------------------------------------------
// Adapter factory
// ---------------------------------------------------------------------------

const makeGeminiAcpAdapter = Effect.fn("makeGeminiAcpAdapter")(function* () {
  const settingsService = yield* ServerSettingsService;
  const authRuntimeState = yield* GeminiAuthRuntimeState;
  const sessionDirectory = yield* ProviderSessionDirectory;
  const services = yield* Effect.context<never>();
  const runFork = Effect.runForkWith(services);
  const runtimeEventQueue = yield* Queue.unbounded<ProviderRuntimeEvent>();

  const sessions = new Map<string, GeminiSessionContext>();

  // Emit events to the unbounded queue. We use unsafeOffer via runSync
  // because events are emitted from both Effect fibers and plain JS callbacks
  // (MessageBus handlers). For an unbounded queue, offer never blocks.
  const emit = (event: ProviderRuntimeEvent): void => {
    Effect.runSyncWith(services)(Queue.offer(runtimeEventQueue, event));
  };

  const setupRetryAttemptHandler = (ctx: GeminiSessionContext): void => {
    const onRetryAttempt = (payload: RetryAttemptPayload): void => {
      if (!ctx.turnState || !ctx.activeStreamPromise || ctx.stopped) {
        return;
      }
      if (payload.model !== ctx.session.model) {
        return;
      }

      const activeMatchingSessions = [...sessions.values()].filter(
        (candidate) =>
          !candidate.stopped &&
          candidate.turnState !== undefined &&
          candidate.activeStreamPromise !== undefined &&
          candidate.session.model === payload.model,
      );
      if (activeMatchingSessions.length !== 1 || activeMatchingSessions[0] !== ctx) {
        return;
      }

      const message = formatGeminiRetryWarningMessage(payload);
      emit({
        ...makeEventBase(ctx),
        type: "runtime.warning",
        payload: {
          message,
          detail: {
            attempt: payload.attempt,
            maxAttempts: payload.maxAttempts,
            delayMs: payload.delayMs,
            error: payload.error,
            model: payload.model,
          },
        },
      } as ProviderRuntimeEvent);
    };

    coreEvents.on(CoreEvent.RetryAttempt, onRetryAttempt);
    ctx.messageBusUnsubscribers.push(() => {
      coreEvents.off(CoreEvent.RetryAttempt, onRetryAttempt);
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

      const correlationId = request.correlationId;
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

      if (ctx.userRequestedMode !== "plan" && ctx.session.runtimeMode === "full-access") {
        void messageBus.publish({
          type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
          correlationId,
          confirmed: true,
          outcome: ToolConfirmationOutcome.ProceedOnce,
        });
        return;
      }

      // Tell Gemini CLI core to surface the tool's own confirmation details
      // immediately instead of waiting for the MessageBus request timeout.
      void messageBus.publish({
        type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
        correlationId,
        confirmed: false,
        requiresUserConfirmation: true,
      });
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

    const handleSubagentActivity = (message: SubagentActivityMessage): void => {
      if (ctx.stopped || !ctx.turnState) return;

      const activeTool = [...ctx.inFlightTools.values()]
        .toReversed()
        .find(
          (tool) =>
            tool.itemType === "collab_agent_tool_call" && tool.toolName === message.subagentName,
        );
      if (!activeTool) {
        return;
      }

      emit({
        ...makeEventBase(ctx),
        itemId: RuntimeItemId.makeUnsafe(activeTool.itemId),
        type: "item.updated",
        payload: {
          itemType: activeTool.itemType,
          title: activeTool.toolName,
          detail: formatGeminiSubagentActivityDetail({
            subagentName: message.subagentName,
            activity: message.activity,
          }),
          data: {
            subagentName: message.subagentName,
            activityType: message.activity.type,
            status: message.activity.status,
          },
        },
      } as ProviderRuntimeEvent);
    };

    messageBus.subscribe(MessageBusType.SUBAGENT_ACTIVITY, handleSubagentActivity as any);
    ctx.messageBusUnsubscribers.push(() => {
      messageBus.unsubscribe(MessageBusType.SUBAGENT_ACTIVITY, handleSubagentActivity as any);
    });
  }

  function emitGeminiProposedPlanCompleted(ctx: GeminiSessionContext, planMarkdown: string): void {
    if (!ctx.turnState) {
      return;
    }

    const trimmedPlan = planMarkdown.trim();
    if (trimmedPlan.length === 0) {
      return;
    }

    const captureKey = `plan:${trimmedPlan}`;
    if (ctx.turnState.capturedProposedPlanKeys.has(captureKey)) {
      return;
    }
    ctx.turnState.capturedProposedPlanKeys.add(captureKey);

    emit({
      ...makeEventBase(ctx),
      type: "turn.proposed.completed",
      payload: {
        planMarkdown: trimmedPlan,
      },
    } as ProviderRuntimeEvent);
  }

  // ---------------------------------------------------------------------------
  // Stream event processing
  // ---------------------------------------------------------------------------

  function emitAssistantText(ctx: GeminiSessionContext, text: string): void {
    if (!ctx.turnState || ctx.planModeTextSuppressed) return;

    const { nextText, delta } = applyGeminiAssistantTextChunk(ctx.assistantMessageText, text);
    ctx.assistantMessageText = nextText;

    if (delta.length === 0) return;

    const itemId = `msg-${ctx.turnState.turnId}-${ctx.assistantMessageSegment}`;
    emit({
      ...makeEventBase(ctx),
      itemId: RuntimeItemId.makeUnsafe(itemId),
      type: "content.delta",
      payload: {
        streamKind: "assistant_text",
        delta,
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

  function beginGeminiToolCall(
    ctx: GeminiSessionContext,
    requestId: string,
    toolName: string,
    args: Record<string, unknown>,
  ): ToolInFlight {
    const itemType = classifyToolName(toolName);
    const tool = {
      itemId: `tool-${requestId}`,
      itemType,
      toolName,
      title: titleForToolType(itemType),
      input: args,
    } satisfies ToolInFlight;

    ctx.assistantMessageSegment++;
    ctx.assistantMessageText = "";
    if (ctx.turnState?.capturedProposedPlanKeys.size === 0) {
      ctx.planModeTextSuppressed = false;
    }
    ctx.inFlightTools.set(requestId, tool);

    emit({
      ...makeEventBase(ctx),
      itemId: RuntimeItemId.makeUnsafe(tool.itemId),
      type: "item.started",
      payload: {
        itemType,
        title: tool.title,
        detail: toolName,
        data: args,
      },
    } as ProviderRuntimeEvent);

    return tool;
  }

  function emitGeminiToolOutput(
    ctx: GeminiSessionContext,
    tool: ToolInFlight,
    output: string,
  ): void {
    const trimmed = output.trim();
    if (trimmed.length === 0) {
      return;
    }

    emit({
      ...makeEventBase(ctx),
      itemId: RuntimeItemId.makeUnsafe(tool.itemId),
      type: "content.delta",
      payload: {
        streamKind: tool.itemType === "command_execution" ? "command_output" : "file_change_output",
        delta: trimmed,
      },
    } as ProviderRuntimeEvent);
  }

  function completeGeminiToolCall(
    ctx: GeminiSessionContext,
    tool: ToolInFlight,
    status: "completed" | "failed",
  ): void {
    ctx.inFlightTools.delete(tool.itemId.replace(/^tool-/, ""));
    emit({
      ...makeEventBase(ctx),
      itemId: RuntimeItemId.makeUnsafe(tool.itemId),
      type: "item.completed",
      payload: {
        itemType: tool.itemType,
        status,
      },
    } as ProviderRuntimeEvent);
  }

  async function runGeminiTool(
    ctx: GeminiSessionContext,
    promptId: string,
    functionCall: FunctionCall,
  ): Promise<ReadonlyArray<Part>> {
    const callId =
      functionCall.id ?? `tool-${Date.now()}-${ctx.inFlightTools.size + ctx.pendingApprovals.size}`;
    const toolName = functionCall.name ?? "";
    const args = (functionCall.args ?? {}) as Record<string, unknown>;
    const startedAt = Date.now();

    if (!toolName) {
      return buildGeminiToolErrorResponse("unknown", callId, new Error("Missing function name"));
    }

    const tool = ctx.config.getToolRegistry().getTool(toolName);
    if (!tool) {
      return buildGeminiToolErrorResponse(
        toolName,
        callId,
        new Error(`Tool "${toolName}" not found in registry.`),
      );
    }

    const inFlightTool = beginGeminiToolCall(ctx, callId, toolName, args);

    try {
      const invocation = tool.build(args);
      const explanation =
        typeof invocation.getExplanation === "function" ? invocation.getExplanation() : "";
      if (explanation) {
        emitReasoningText(ctx, explanation);
      }

      const forcedDecision = forcedGeminiToolDecision(ctx, toolName, args);
      const confirmation = await invocation.shouldConfirmExecute(
        ctx.abortController.signal,
        forcedDecision,
      );

      if (confirmation && confirmation.type === "exit_plan_mode") {
        const planMarkdown = await readGeminiPlanMarkdownFromFile(confirmation.planPath);
        if (!planMarkdown) {
          throw new Error(`Plan file "${confirmation.planPath}" is empty.`);
        }

        emitGeminiProposedPlanCompleted(ctx, planMarkdown);
        ctx.planModeTextSuppressed = true;

        const toolResult = {
          llmContent: [
            "The client captured your proposed plan.",
            `Plan file: ${confirmation.planPath}`,
            "Stop here and wait for the user's feedback or implementation request in a later turn.",
          ].join("\n\n"),
          returnDisplay: `Captured plan: ${confirmation.planPath}`,
        } satisfies ToolResult;

        const resultText = summarizeGeminiToolResult(toolResult);
        if (resultText) {
          emitGeminiToolOutput(ctx, inFlightTool, resultText);
        }

        completeGeminiToolCall(ctx, inFlightTool, "completed");

        logToolCall(
          ctx.config,
          new ToolCallEvent(
            undefined,
            toolName,
            args,
            Date.now() - startedAt,
            true,
            promptId,
            tool instanceof DiscoveredMCPTool ? "mcp" : "native",
          ),
        );

        return convertToFunctionResponse(
          toolName,
          callId,
          toolResult.llmContent,
          ctx.config.getActiveModel(),
          ctx.config,
        );
      }

      if (confirmation) {
        if (confirmation.type === "ask_user") {
          const answers = await requestGeminiUserInput({
            ctx,
            rawQuestions: confirmation.questions,
            emitEvent: emit,
          });
          const payload =
            answers === undefined
              ? undefined
              : ({
                  answers: buildGeminiAskUserResponseAnswers({
                    questions: normalizeGeminiAskUserQuestions(confirmation.questions),
                    answers,
                  }),
                } satisfies ToolConfirmationPayload);

          await confirmation.onConfirm(
            answers === undefined
              ? ToolConfirmationOutcome.Cancel
              : ToolConfirmationOutcome.ProceedOnce,
            payload,
          );

          if (answers === undefined) {
            throw new Error(`Tool "${toolName}" was canceled by the user.`);
          }
        } else {
          const approval = await requestGeminiApproval({
            ctx,
            toolName,
            args,
            confirmation,
            emitEvent: emit,
          });

          await confirmation.onConfirm(approval.outcome);

          if (!approval.confirmed) {
            throw new Error(`Tool "${toolName}" was canceled by the user.`);
          }
        }
      }

      let liveOutputBuffer = "";
      const toolResult = await invocation.execute(
        ctx.abortController.signal,
        (output: ToolLiveOutput) => {
          if (typeof output !== "string") {
            return;
          }
          liveOutputBuffer += output;
          emitGeminiToolOutput(ctx, inFlightTool, output);
        },
      );

      const resultText = summarizeGeminiToolResult(toolResult);
      if (resultText && liveOutputBuffer.trim().length === 0) {
        emitGeminiToolOutput(ctx, inFlightTool, resultText);
      }

      completeGeminiToolCall(ctx, inFlightTool, "completed");

      logToolCall(
        ctx.config,
        new ToolCallEvent(
          undefined,
          toolName,
          args,
          Date.now() - startedAt,
          true,
          promptId,
          tool instanceof DiscoveredMCPTool ? "mcp" : "native",
        ),
      );

      return convertToFunctionResponse(
        toolName,
        callId,
        toolResult.llmContent,
        ctx.config.getActiveModel(),
        ctx.config,
      );
    } catch (error) {
      const toolError = error instanceof Error ? error : new Error(String(error));

      emitGeminiToolOutput(ctx, inFlightTool, toolError.message);
      completeGeminiToolCall(ctx, inFlightTool, "failed");

      logToolCall(
        ctx.config,
        new ToolCallEvent(
          undefined,
          toolName,
          args,
          Date.now() - startedAt,
          false,
          promptId,
          tool instanceof DiscoveredMCPTool ? "mcp" : "native",
          toolError.message,
        ),
      );

      return buildGeminiToolErrorResponse(toolName, callId, toolError);
    }
  }

  // ---------------------------------------------------------------------------
  // Stream loop — background turn processing
  // ---------------------------------------------------------------------------

  async function runStreamLoop(ctx: GeminiSessionContext, promptText: string): Promise<void> {
    const promptId = `prompt-${Date.now()}`;
    const chat = ctx.geminiClient.getChat();
    const baseSystemInstruction = getCoreSystemPrompt(
      ctx.config,
      ctx.config.getSystemInstructionMemory(),
    );
    chat.setSystemInstruction(baseSystemInstruction);
    let nextMessage: Content | null = {
      role: "user",
      parts: [{ text: promptText }],
    };

    try {
      while (nextMessage !== null && ctx.turnState && !ctx.stopped) {
        if (ctx.abortController.signal.aborted) {
          completeTurn(ctx, "interrupted", "aborted");
          return;
        }

        const functionCalls: FunctionCall[] = [];
        const modelParts: Part[] = [];
        const routingContext: RoutingContext = {
          history: chat.getHistory(true),
          request: nextMessage.parts ?? [],
          signal: ctx.abortController.signal,
          requestedModel: ctx.config.getModel(),
        };
        const { model } = await ctx.config.getModelRouterService().route(routingContext);

        let responseStream: AsyncGenerator<any>;
        try {
          responseStream = await chat.sendMessageStream(
            { model },
            nextMessage.parts ?? [],
            promptId,
            ctx.abortController.signal,
            LlmRole.MAIN,
          );
        } catch (error) {
          if (ctx.abortController.signal.aborted) {
            completeTurn(ctx, "interrupted", "aborted");
            return;
          }
          throw error;
        }

        nextMessage = null;

        try {
          for await (const response of responseStream) {
            if (ctx.abortController.signal.aborted) {
              completeTurn(ctx, "interrupted", "aborted");
              return;
            }

            if (response.type !== StreamEventType.CHUNK) {
              continue;
            }

            const usage = response.value?.usageMetadata;
            if (usage) {
              ctx.cumulativeInputTokens += usage.promptTokenCount ?? 0;
              ctx.cumulativeOutputTokens += usage.candidatesTokenCount ?? 0;
            }

            if (Array.isArray(response.value?.functionCalls)) {
              functionCalls.push(...response.value.functionCalls);
            }

            const candidate = response.value?.candidates?.[0];
            for (const part of candidate?.content?.parts ?? []) {
              if (part.thought && typeof part.text === "string") {
                emitReasoningText(ctx, part.text);
                continue;
              }

              if (typeof part.text === "string") {
                modelParts.push(part);
                emitAssistantText(ctx, part.text);
              }
            }
          }
        } catch (error) {
          if (ctx.abortController.signal.aborted) {
            completeTurn(ctx, "interrupted", "aborted");
            return;
          }

          if (functionCalls.length === 0 && isGracefulGeminiTurnError(error)) {
            appendGeminiAssistantHistory(chat, modelParts);
            break;
          }

          throw error;
        }

        if (functionCalls.length === 0) {
          break;
        }

        const toolResponseParts: Part[] = [];
        for (const functionCall of functionCalls) {
          const responses = await runGeminiTool(ctx, promptId, functionCall);
          toolResponseParts.push(...responses);
        }

        if (ctx.abortController.signal.aborted) {
          completeTurn(ctx, "interrupted", "aborted");
          return;
        }

        nextMessage = {
          role: "user",
          parts: toolResponseParts,
        };
      }

      if (ctx.turnState) {
        completeTurn(
          ctx,
          ctx.abortController.signal.aborted ? "interrupted" : "completed",
          ctx.abortController.signal.aborted ? "aborted" : "completed",
        );
      }
    } catch (error) {
      if (ctx.stopped) {
        return;
      }

      if (ctx.abortController.signal.aborted) {
        completeTurn(ctx, "interrupted", "interrupted");
        return;
      }

      console.error("[gemini] Stream error:", error);
      completeTurn(ctx, "failed", "error", toMessage(error, "Stream error"));
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

    // Emit token usage
    const tokenUsage = buildTokenUsageSnapshot(ctx);
    emit({
      ...makeEventBase(ctx),
      type: "thread.token-usage.updated",
      payload: {
        usage: tokenUsage,
      },
    } as ProviderRuntimeEvent);

    // Only completed turns advance the durable Gemini history snapshot.
    const turnId = ctx.turnState.turnId;
    if (state === "completed") {
      ctx.turns.push({
        id: turnId,
        items: [],
        historyLength: ctx.geminiClient.getHistory().length,
      });
      updateResumeCursor(ctx);
    }
    const completedAt = new Date().toISOString();
    const mutableSession = ctx.session as {
      status: ProviderSession["status"];
      activeTurnId: TurnId | undefined;
      updatedAt: string;
      lastError?: string;
    };
    mutableSession.status = state === "failed" ? "error" : "ready";
    mutableSession.activeTurnId = undefined;
    mutableSession.updatedAt = completedAt;
    if (errorMessage) {
      mutableSession.lastError = errorMessage;
    }
    runFork(
      sessionDirectory
        .upsert(
          buildGeminiPersistedBinding({
            session: ctx.session,
            status: state === "failed" ? "error" : "running",
            lastRuntimeEvent: "turn.completed",
            lastRuntimeEventAt: completedAt,
          }),
        )
        .pipe(
          Effect.catchCause((cause) =>
            Effect.logWarning("failed to persist Gemini session after turn completion", {
              threadId: ctx.session.threadId,
              status: state,
              cause,
            }),
          ),
        ),
    );

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

  const startSession: GeminiAcpAdapterShape["startSession"] = Effect.fn("startSession")(
    function* (input) {
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
      const geminiAuthHeaders = installGeminiCliCustomHeaders();
      const authType = resolveGeminiAuthType();
      const manualAuthFailure = yield* authRuntimeState.getFailure;

      if (manualAuthFailure) {
        const currentAuthProbe = resolveGeminiAuthProbeResult(authType);
        if (currentAuthProbe.auth.status === "authenticated") {
          yield* authRuntimeState.clearFailure;
        } else {
          return yield* new ProviderAdapterProcessError({
            provider: PROVIDER,
            threadId,
            detail: manualAuthFailure.message,
          });
        }
      }

      // Create config, authenticate first, then initialize.
      // This matches Gemini CLI ACP and ensures the content generator exists
      // before GeminiClient.initialize() runs during config.initialize().
      const workDir = (input.cwd as string | undefined) ?? process.cwd();
      const config = yield* Effect.tryPromise({
        try: () =>
          createGeminiCoreConfig({
            sessionId: threadId as string,
            cwd: workDir,
            model: modelId,
            runtimeMode,
            interactive: true,
          }),
        catch: (cause) =>
          new ProviderAdapterProcessError({
            provider: PROVIDER,
            threadId,
            detail: `Failed to create Gemini config: ${toMessage(cause, "unknown error")}`,
            cause,
          }),
      });

      yield* Effect.tryPromise({
        try: () => config.refreshAuth(authType, undefined, undefined, geminiAuthHeaders),
        catch: (cause) =>
          new ProviderAdapterProcessError({
            provider: PROVIDER,
            threadId,
            detail: `Failed to authenticate Gemini session: ${toMessage(cause, "unknown error")}`,
            cause,
          }),
      }).pipe(
        Effect.tap(() => authRuntimeState.clearFailure),
        Effect.tapError((error) =>
          authRuntimeState.requireManualLogin({
            authType,
            failedAt: now,
            message: buildGeminiManualAuthRequiredMessage({
              detail: error.detail,
            }),
          }),
        ),
      );

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
        planModeTextSuppressed: false,
        assistantMessageSegment: 0,
        assistantMessageText: "",
        cumulativeInputTokens: 0,
        cumulativeOutputTokens: 0,
        inFlightTools: new Map(),
        turns: Array.from({ length: resumeState?.turnCount ?? 0 }, (_, index) => {
          const historyLength = resumeState?.turnHistoryLengths?.[index];
          return {
            id: `restored-turn-${index}`,
            items: [],
            ...(historyLength !== undefined ? { historyLength } : {}),
          };
        }),
        activeStreamPromise: undefined,
      };
      updateResumeCursor(ctx);
      setupRetryAttemptHandler(ctx);

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
    },
  );

  const sendTurn: GeminiAcpAdapterShape["sendTurn"] = Effect.fn("sendTurn")(function* (input) {
    const ctx = yield* getSession(input.threadId);
    yield* Effect.sync(() => {
      restoreGeminiHistoryFromResumeCursor(ctx);
    }).pipe(
      Effect.mapError(
        (cause) =>
          new ProviderAdapterProcessError({
            provider: PROVIDER,
            threadId: input.threadId,
            detail: `Failed to restore Gemini history before turn start: ${toMessage(cause, "unknown error")}`,
            cause,
          }),
      ),
    );

    const interactionMode = input.interactionMode ?? ctx.userRequestedMode;
    ctx.userRequestedMode = interactionMode;
    ctx.config.setApprovalMode(
      resolveGeminiApprovalMode({
        interactionMode,
        runtimeMode: ctx.session.runtimeMode,
      }),
    );

    const now = new Date().toISOString();
    const turnId = `turn-${Date.now()}-${ctx.turns.length}`;

    // Reset turn state
    ctx.turnState = {
      turnId,
      startedAt: now,
      reasoningItemEmitted: false,
      capturedProposedPlanKeys: new Set(),
    };
    ctx.planModeTextSuppressed = false;
    ctx.assistantMessageSegment = 0;
    ctx.assistantMessageText = "";
    ctx.inFlightTools.clear();
    ctx.abortController = new AbortController();

    const promptText = input.input?.trim() ?? "";

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

  const interruptTurn: GeminiAcpAdapterShape["interruptTurn"] = Effect.fn("interruptTurn")(
    function* (threadId, _turnId) {
      const ctx = yield* getSession(threadId);
      abortActiveTurn(ctx);
    },
  );

  const respondToRequest: GeminiAcpAdapterShape["respondToRequest"] = Effect.fn("respondToRequest")(
    function* (threadId, requestId, decision) {
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
    },
  );

  const respondToUserInput: GeminiAcpAdapterShape["respondToUserInput"] = Effect.fn(
    "respondToUserInput",
  )(function* (threadId, requestId, answers) {
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

  const stopSession: GeminiAcpAdapterShape["stopSession"] = Effect.fn("stopSession")(
    function* (threadId) {
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
    },
  );

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

  const readThread: GeminiAcpAdapterShape["readThread"] = Effect.fn("readThread")(
    function* (threadId) {
      const ctx = yield* getSession(threadId);
      return {
        threadId,
        turns: ctx.turns.map((t) => ({
          id: TurnIdBrand.makeUnsafe(t.id),
          items: t.items,
        })),
      };
    },
  );

  const rollbackThread: GeminiAcpAdapterShape["rollbackThread"] = Effect.fn("rollbackThread")(
    function* (threadId, numTurns) {
      const ctx = yield* getSession(threadId);
      const nextTurnCount = Math.max(0, ctx.turns.length - numTurns);
      const retainedTurns = ctx.turns.slice(0, nextTurnCount);

      // Roll back Gemini history to the retained turn boundary when available.
      yield* Effect.sync(() => {
        const history = ctx.geminiClient.getHistory();
        const targetHistoryLength =
          retainedTurns.length === 0 ? 0 : retainedTurns.at(-1)?.historyLength;
        const newHistory =
          targetHistoryLength !== undefined
            ? history.slice(0, Math.min(history.length, targetHistoryLength))
            : history.slice(0, Math.max(0, history.length - numTurns * 2));
        ctx.geminiClient.setHistory([...newHistory]);
      }).pipe(Effect.ignore);

      // Truncate tracked turns
      const removed = ctx.turns.splice(nextTurnCount);
      void removed;
      updateResumeCursor(ctx);

      return {
        threadId,
        turns: ctx.turns.map((t) => ({
          id: TurnIdBrand.makeUnsafe(t.id),
          items: t.items,
        })),
      };
    },
  );

  const stopAll: GeminiAcpAdapterShape["stopAll"] = Effect.fn("stopAll")(function* () {
    const threadIds = Array.from(sessions.keys());
    for (const threadId of threadIds) {
      yield* stopSession(threadId as ThreadId);
    }
  });

  return {
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
    get streamEvents() {
      return Stream.fromQueue(runtimeEventQueue);
    },
  } satisfies GeminiAcpAdapterShape;
});

export const GeminiAcpAdapterLive = Layer.effect(GeminiAcpAdapter, makeGeminiAcpAdapter());
