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
  ProviderApprovalDecision,
  CanonicalItemType,
  CanonicalRequestType,
  ProviderRuntimeEvent,
  ProviderSendTurnInput,
  ProviderSession,
  ProviderTurnStartResult,
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
  readonly requestId: string;
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

function toGeminiAdapterError(
  threadId: ThreadId,
  operation: string,
  cause: unknown,
): ProviderAdapterError {
  const errorTag =
    typeof cause === "object" && cause !== null ? (cause as { _tag?: unknown })._tag : undefined;
  if (
    errorTag === "ProviderAdapterProcessError" ||
    errorTag === "ProviderAdapterRequestError" ||
    errorTag === "ProviderAdapterSessionClosedError" ||
    errorTag === "ProviderAdapterSessionNotFoundError" ||
    errorTag === "ProviderAdapterValidationError"
  ) {
    return cause as ProviderAdapterError;
  }

  return new ProviderAdapterProcessError({
    provider: PROVIDER,
    threadId,
    detail: `${operation}: ${toMessage(cause, "unknown error")}`,
    cause,
  });
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

  // Emit events to the unbounded queue. We use unsafeOffer via runSync
  // because events are emitted from both Effect fibers and plain JS callbacks
  // (MessageBus handlers). For an unbounded queue, offer never blocks.
  const emit = (event: ProviderRuntimeEvent): void => {
    Effect.runSyncWith(services)(Queue.offer(runtimeEventQueue, event));
  };
  interface QueuedGeminiTurn {
    readonly promptText: string;
  }

  class GeminiRuntimeSession implements GeminiSessionContext {
    readonly messageBusUnsubscribers: Array<() => void> = [];
    turnState: GeminiTurnState | undefined = undefined;
    stopped = false;
    readonly pendingApprovals = new Map<string, PendingApproval>();
    readonly pendingUserInputs = new Map<string, PendingUserInput>();
    userRequestedMode = "default";
    planModeTextSuppressed = false;
    assistantMessageSegment = 0;
    assistantMessageText = "";
    cumulativeInputTokens = 0;
    cumulativeOutputTokens = 0;
    readonly inFlightTools = new Map<string, ToolInFlight>();
    readonly turns: Array<GeminiTrackedTurn>;
    activeStreamPromise: Promise<void> | undefined = undefined;

    private readonly messageBus: MessageBus;
    private queuedTurn: QueuedGeminiTurn | undefined = undefined;
    private turnQueueWaiter: (() => void) | undefined = undefined;
    private readonly workerPromise: Promise<void>;

    constructor(
      readonly session: ProviderSession,
      readonly config: Config,
      readonly geminiClient: GeminiClient,
      resumeState: GeminiResumeState | undefined,
    ) {
      this.abortController = new AbortController();
      this.turns = Array.from({ length: resumeState?.turnCount ?? 0 }, (_, index) => {
        const historyLength = resumeState?.turnHistoryLengths?.[index];
        return {
          id: `restored-turn-${index}`,
          items: [],
          ...(historyLength !== undefined ? { historyLength } : {}),
        };
      });
      updateResumeCursor(this);
      this.messageBus = config.getMessageBus();
      this.setupRetryAttemptHandler();
      this.setupMessageBusHandlers();
      this.workerPromise = this.runWorker().catch((error) => {
        if (this.stopped) {
          return;
        }

        console.error("[gemini] Session worker crashed:", error);
        if (this.turnState) {
          this.completeTurn("failed", "error", toMessage(error, "Session worker error"));
        }
      });
    }

    abortController: AbortController;

    startTurn(input: ProviderSendTurnInput): ProviderTurnStartResult {
      if (this.turnState || this.queuedTurn || this.activeStreamPromise) {
        throw new ProviderAdapterRequestError({
          provider: PROVIDER,
          method: "sendTurn",
          detail: `Session "${this.session.threadId}" already has an active turn.`,
        });
      }

      try {
        restoreGeminiHistoryFromResumeCursor(this);
      } catch (cause) {
        throw new ProviderAdapterProcessError({
          provider: PROVIDER,
          threadId: input.threadId,
          detail: `Failed to restore Gemini history before turn start: ${toMessage(cause, "unknown error")}`,
          cause,
        });
      }

      const interactionMode = input.interactionMode ?? this.userRequestedMode;
      this.userRequestedMode = interactionMode;
      this.config.setApprovalMode(
        resolveGeminiApprovalMode({
          interactionMode,
          runtimeMode: this.session.runtimeMode,
        }),
      );

      const now = new Date().toISOString();
      const turnId = `turn-${Date.now()}-${this.turns.length}`;

      this.turnState = {
        turnId,
        startedAt: now,
        reasoningItemEmitted: false,
        capturedProposedPlanKeys: new Set(),
      };
      this.planModeTextSuppressed = false;
      this.assistantMessageSegment = 0;
      this.assistantMessageText = "";
      this.inFlightTools.clear();
      this.abortController = new AbortController();

      const promptText = input.input?.trim() ?? "";

      const mutableSession = this.session as {
        status: ProviderSession["status"];
        activeTurnId?: TurnId;
        updatedAt: string;
      };
      mutableSession.status = "running";
      mutableSession.activeTurnId = TurnIdBrand.makeUnsafe(turnId);
      mutableSession.updatedAt = now;

      emit({
        ...makeEventBase(this),
        type: "turn.started",
        payload: { model: this.session.model },
      } as ProviderRuntimeEvent);
      emit({
        ...makeEventBase(this),
        type: "session.state.changed",
        payload: { state: "running" },
      } as ProviderRuntimeEvent);

      this.queuedTurn = { promptText };
      this.notifyTurnWorker();

      return {
        threadId: input.threadId,
        turnId: TurnIdBrand.makeUnsafe(turnId),
        ...(this.session.resumeCursor !== undefined
          ? { resumeCursor: this.session.resumeCursor }
          : {}),
      };
    }

    interrupt(): void {
      abortActiveTurn(this);
    }

    resolveApproval(requestId: string, decision: ProviderApprovalDecision): void {
      const pending = this.pendingApprovals.get(requestId);
      if (!pending) {
        throw new ProviderAdapterRequestError({
          provider: PROVIDER,
          method: "respondToRequest",
          detail: `No pending approval for requestId "${requestId}"`,
        });
      }

      this.pendingApprovals.delete(requestId);

      const confirmed = decision === "accept" || decision === "acceptForSession";
      const outcome = confirmed
        ? decision === "acceptForSession"
          ? ToolConfirmationOutcome.ProceedAlways
          : ToolConfirmationOutcome.ProceedOnce
        : ToolConfirmationOutcome.Cancel;

      pending.resolve({ confirmed, outcome });

      emit({
        ...makeEventBase(this),
        requestId: RuntimeRequestId.makeUnsafe(requestId),
        type: "request.resolved",
        payload: {
          requestType: pending.requestType,
          decision,
        },
      } as ProviderRuntimeEvent);
    }

    resolveUserInput(requestId: string, answers: ProviderUserInputAnswers): void {
      const pending = this.pendingUserInputs.get(requestId);
      if (!pending) {
        throw new ProviderAdapterRequestError({
          provider: PROVIDER,
          method: "respondToUserInput",
          detail: `No pending user input for requestId "${requestId}"`,
        });
      }

      this.pendingUserInputs.delete(requestId);
      pending.resolve(answers);

      emit({
        ...makeEventBase(this),
        requestId: RuntimeRequestId.makeUnsafe(requestId),
        type: "user-input.resolved",
        payload: {
          answers,
        },
      } as ProviderRuntimeEvent);
    }

    async stop(): Promise<void> {
      if (this.stopped) {
        return;
      }

      this.stopped = true;
      this.queuedTurn = undefined;
      abortActiveTurn(this);
      if (this.turnState && !this.activeStreamPromise) {
        this.completeTurn("interrupted", "aborted");
      }

      this.notifyTurnWorker();
      for (const unsubscribe of this.messageBusUnsubscribers.splice(0)) {
        try {
          unsubscribe();
        } catch {
          // ignore cleanup failures while stopping
        }
      }

      for (const [, pending] of this.pendingApprovals) {
        pending.resolve({
          confirmed: false,
          outcome: ToolConfirmationOutcome.Cancel,
        });
      }
      this.pendingApprovals.clear();

      for (const [requestId, pending] of this.pendingUserInputs) {
        pending.resolve(undefined);
        emit({
          ...makeEventBase(this),
          requestId: RuntimeRequestId.makeUnsafe(requestId),
          type: "user-input.resolved",
          payload: {
            answers: {},
          },
        } as ProviderRuntimeEvent);
      }
      this.pendingUserInputs.clear();

      await this.activeStreamPromise?.catch(() => undefined);
      await this.workerPromise;

      try {
        await this.config.dispose();
      } catch {
        // swallow shutdown errors
      }

      (this.session as { status: ProviderSession["status"] }).status = "closed";

      emit({
        ...makeEventBase(this),
        type: "session.exited",
        payload: {
          reason: "stopped",
          exitKind: "graceful",
        },
      } as ProviderRuntimeEvent);
    }

    readThreadSnapshot() {
      return {
        threadId: this.session.threadId,
        turns: this.turns.map((turn) => ({
          id: TurnIdBrand.makeUnsafe(turn.id),
          items: turn.items,
        })),
      };
    }

    rollbackThread(numTurns: number) {
      const nextTurnCount = Math.max(0, this.turns.length - numTurns);
      const retainedTurns = this.turns.slice(0, nextTurnCount);

      const history = this.geminiClient.getHistory();
      const targetHistoryLength =
        retainedTurns.length === 0 ? 0 : retainedTurns.at(-1)?.historyLength;
      const newHistory =
        targetHistoryLength !== undefined
          ? history.slice(0, Math.min(history.length, targetHistoryLength))
          : history.slice(0, Math.max(0, history.length - numTurns * 2));
      this.geminiClient.setHistory([...newHistory]);

      void this.turns.splice(nextTurnCount);
      updateResumeCursor(this);

      return this.readThreadSnapshot();
    }

    private notifyTurnWorker(): void {
      const waiter = this.turnQueueWaiter;
      this.turnQueueWaiter = undefined;
      waiter?.();
    }

    private async waitForQueuedTurn(): Promise<QueuedGeminiTurn | undefined> {
      while (!this.stopped) {
        if (this.queuedTurn) {
          const queuedTurn = this.queuedTurn;
          this.queuedTurn = undefined;
          return queuedTurn;
        }

        await new Promise<void>((resolve) => {
          this.turnQueueWaiter = resolve;
        });
      }

      return undefined;
    }

    private async runWorker(): Promise<void> {
      while (!this.stopped) {
        const queuedTurn = await this.waitForQueuedTurn();
        if (!queuedTurn) {
          return;
        }

        this.activeStreamPromise = this.runStreamLoop(queuedTurn.promptText);
        try {
          await this.activeStreamPromise;
        } finally {
          this.activeStreamPromise = undefined;
        }
      }
    }

    private setupRetryAttemptHandler(): void {
      const onRetryAttempt = (payload: RetryAttemptPayload): void => {
        if (!this.turnState || !this.activeStreamPromise || this.stopped) {
          return;
        }
        if (payload.model !== this.session.model) {
          return;
        }

        const activeMatchingSessions = [...sessions.values()].filter(
          (candidate) =>
            !candidate.stopped &&
            candidate.turnState !== undefined &&
            candidate.activeStreamPromise !== undefined &&
            candidate.session.model === payload.model,
        );
        if (activeMatchingSessions.length !== 1 || activeMatchingSessions[0] !== this) {
          return;
        }

        emit({
          ...makeEventBase(this),
          type: "runtime.warning",
          payload: {
            message: formatGeminiRetryWarningMessage(payload),
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
      this.messageBusUnsubscribers.push(() => {
        coreEvents.off(CoreEvent.RetryAttempt, onRetryAttempt);
      });
    }

    private setupMessageBusHandlers(): void {
      const handleConfirmation = (request: ToolConfirmationRequest): void => {
        if (this.stopped || !this.turnState) {
          return;
        }

        const correlationId = request.correlationId;
        const askUserQuestions =
          request.details?.type === "ask_user"
            ? normalizeGeminiAskUserQuestions(request.details.questions)
            : undefined;

        if (askUserQuestions && askUserQuestions.length > 0) {
          this.openUserInputRequest({
            correlationId,
            rawQuestions: askUserQuestions,
            responseChannel: "tool_confirmation",
          });
          return;
        }

        if (this.userRequestedMode !== "plan" && this.session.runtimeMode === "full-access") {
          void this.messageBus.publish({
            type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
            correlationId,
            confirmed: true,
            outcome: ToolConfirmationOutcome.ProceedOnce,
          });
          return;
        }

        // Surface the tool's native confirmation details immediately rather
        // than waiting for the MessageBus request timeout.
        void this.messageBus.publish({
          type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
          correlationId,
          confirmed: false,
          requiresUserConfirmation: true,
        });
      };

      this.messageBus.subscribe(
        MessageBusType.TOOL_CONFIRMATION_REQUEST,
        handleConfirmation as any,
      );
      this.messageBusUnsubscribers.push(() => {
        this.messageBus.unsubscribe(
          MessageBusType.TOOL_CONFIRMATION_REQUEST,
          handleConfirmation as any,
        );
      });

      const handleAskUser = (request: { correlationId: string; questions: unknown }): void => {
        if (this.stopped || !this.turnState) {
          return;
        }

        this.openUserInputRequest({
          correlationId: request.correlationId,
          rawQuestions: request.questions,
          responseChannel: "ask_user",
        });
      };

      this.messageBus.subscribe(MessageBusType.ASK_USER_REQUEST, handleAskUser as any);
      this.messageBusUnsubscribers.push(() => {
        this.messageBus.unsubscribe(MessageBusType.ASK_USER_REQUEST, handleAskUser as any);
      });

      const handleSubagentActivity = (message: SubagentActivityMessage): void => {
        if (this.stopped || !this.turnState) {
          return;
        }

        const activeTool = [...this.inFlightTools.values()]
          .toReversed()
          .find(
            (tool) =>
              tool.itemType === "collab_agent_tool_call" && tool.toolName === message.subagentName,
          );
        if (!activeTool) {
          return;
        }

        emit({
          ...makeEventBase(this),
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

      this.messageBus.subscribe(MessageBusType.SUBAGENT_ACTIVITY, handleSubagentActivity as any);
      this.messageBusUnsubscribers.push(() => {
        this.messageBus.unsubscribe(
          MessageBusType.SUBAGENT_ACTIVITY,
          handleSubagentActivity as any,
        );
      });
    }

    private requestApproval(input: {
      readonly toolName: string;
      readonly args: Record<string, unknown>;
      readonly confirmation: Exclude<ToolCallConfirmationDetails, { type: "ask_user" }>;
    }): Promise<{ confirmed: boolean; outcome: ToolConfirmationOutcome }> {
      return requestGeminiApproval({
        ctx: this,
        toolName: input.toolName,
        args: input.args,
        confirmation: input.confirmation,
        emitEvent: emit,
      });
    }

    private requestUserInput(rawQuestions: unknown): Promise<ProviderUserInputAnswers | undefined> {
      return requestGeminiUserInput({
        ctx: this,
        rawQuestions,
        emitEvent: emit,
      });
    }

    private openUserInputRequest(input: {
      readonly correlationId: string;
      readonly rawQuestions: unknown;
      readonly responseChannel: "tool_confirmation" | "ask_user";
    }): boolean {
      return openGeminiUserInputRequest({
        ctx: this,
        messageBus: this.messageBus,
        emitEvent: emit,
        correlationId: input.correlationId,
        rawQuestions: input.rawQuestions,
        responseChannel: input.responseChannel,
      });
    }

    private emitProposedPlanCompleted(planMarkdown: string): void {
      if (!this.turnState) {
        return;
      }

      const trimmedPlan = planMarkdown.trim();
      if (trimmedPlan.length === 0) {
        return;
      }

      const captureKey = `plan:${trimmedPlan}`;
      if (this.turnState.capturedProposedPlanKeys.has(captureKey)) {
        return;
      }
      this.turnState.capturedProposedPlanKeys.add(captureKey);

      emit({
        ...makeEventBase(this),
        type: "turn.proposed.completed",
        payload: {
          planMarkdown: trimmedPlan,
        },
      } as ProviderRuntimeEvent);
    }

    private emitAssistantText(text: string): void {
      if (!this.turnState || this.planModeTextSuppressed) {
        return;
      }

      const { nextText, delta } = applyGeminiAssistantTextChunk(this.assistantMessageText, text);
      this.assistantMessageText = nextText;

      if (delta.length === 0) {
        return;
      }

      emit({
        ...makeEventBase(this),
        itemId: RuntimeItemId.makeUnsafe(
          `msg-${this.turnState.turnId}-${this.assistantMessageSegment}`,
        ),
        type: "content.delta",
        payload: {
          streamKind: "assistant_text",
          delta,
        },
      } as ProviderRuntimeEvent);
    }

    private emitReasoningText(thoughtText: string): void {
      if (!this.turnState) {
        return;
      }

      if (!this.turnState.reasoningItemEmitted) {
        this.turnState.reasoningItemEmitted = true;
        emit({
          ...makeEventBase(this),
          itemId: RuntimeItemId.makeUnsafe(`reasoning-${this.turnState.turnId}`),
          type: "item.started",
          payload: {
            itemType: "reasoning",
            title: "Thinking",
          },
        } as ProviderRuntimeEvent);
      }

      emit({
        ...makeEventBase(this),
        itemId: RuntimeItemId.makeUnsafe(`reasoning-${this.turnState.turnId}`),
        type: "content.delta",
        payload: {
          streamKind: "reasoning_text",
          delta: thoughtText,
        },
      } as ProviderRuntimeEvent);
    }

    private beginToolCall(
      requestId: string,
      toolName: string,
      args: Record<string, unknown>,
    ): ToolInFlight {
      const itemType = classifyToolName(toolName);
      const tool = {
        requestId,
        itemId: `tool-${requestId}`,
        itemType,
        toolName,
        title: titleForToolType(itemType),
        input: args,
      } satisfies ToolInFlight;

      this.assistantMessageSegment++;
      this.assistantMessageText = "";
      if (this.turnState?.capturedProposedPlanKeys.size === 0) {
        this.planModeTextSuppressed = false;
      }
      this.inFlightTools.set(requestId, tool);

      emit({
        ...makeEventBase(this),
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

    private emitToolOutput(tool: ToolInFlight, output: string): void {
      const trimmed = output.trim();
      if (trimmed.length === 0) {
        return;
      }

      emit({
        ...makeEventBase(this),
        itemId: RuntimeItemId.makeUnsafe(tool.itemId),
        type: "content.delta",
        payload: {
          streamKind:
            tool.itemType === "command_execution" ? "command_output" : "file_change_output",
          delta: trimmed,
        },
      } as ProviderRuntimeEvent);
    }

    private completeToolCall(tool: ToolInFlight, status: "completed" | "failed"): void {
      this.inFlightTools.delete(tool.requestId);
      emit({
        ...makeEventBase(this),
        itemId: RuntimeItemId.makeUnsafe(tool.itemId),
        type: "item.completed",
        payload: {
          itemType: tool.itemType,
          status,
        },
      } as ProviderRuntimeEvent);
    }

    private async runTool(
      promptId: string,
      functionCall: FunctionCall,
    ): Promise<ReadonlyArray<Part>> {
      const callId =
        functionCall.id ??
        `tool-${Date.now()}-${this.inFlightTools.size + this.pendingApprovals.size}`;
      const toolName = functionCall.name ?? "";
      const args = (functionCall.args ?? {}) as Record<string, unknown>;
      const startedAt = Date.now();

      if (!toolName) {
        return buildGeminiToolErrorResponse("unknown", callId, new Error("Missing function name"));
      }

      const tool = this.config.getToolRegistry().getTool(toolName);
      if (!tool) {
        return buildGeminiToolErrorResponse(
          toolName,
          callId,
          new Error(`Tool "${toolName}" not found in registry.`),
        );
      }

      const inFlightTool = this.beginToolCall(callId, toolName, args);

      try {
        const invocation = tool.build(args);
        const explanation =
          typeof invocation.getExplanation === "function" ? invocation.getExplanation() : "";
        if (explanation) {
          this.emitReasoningText(explanation);
        }

        const forcedDecision = forcedGeminiToolDecision(this, toolName, args);
        const confirmation = await invocation.shouldConfirmExecute(
          this.abortController.signal,
          forcedDecision,
        );

        if (confirmation && confirmation.type === "exit_plan_mode") {
          const planMarkdown = await readGeminiPlanMarkdownFromFile(confirmation.planPath);
          if (!planMarkdown) {
            throw new Error(`Plan file "${confirmation.planPath}" is empty.`);
          }

          this.emitProposedPlanCompleted(planMarkdown);
          this.planModeTextSuppressed = true;

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
            this.emitToolOutput(inFlightTool, resultText);
          }

          this.completeToolCall(inFlightTool, "completed");

          logToolCall(
            this.config,
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
            this.config.getActiveModel(),
            this.config,
          );
        }

        if (confirmation) {
          if (confirmation.type === "ask_user") {
            const answers = await this.requestUserInput(confirmation.questions);
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
            const approval = await this.requestApproval({
              toolName,
              args,
              confirmation,
            });

            await confirmation.onConfirm(approval.outcome);

            if (!approval.confirmed) {
              throw new Error(`Tool "${toolName}" was canceled by the user.`);
            }
          }
        }

        let liveOutputBuffer = "";
        const toolResult = await invocation.execute(
          this.abortController.signal,
          (output: ToolLiveOutput) => {
            if (typeof output !== "string") {
              return;
            }
            liveOutputBuffer += output;
            this.emitToolOutput(inFlightTool, output);
          },
        );

        const resultText = summarizeGeminiToolResult(toolResult);
        if (resultText && liveOutputBuffer.trim().length === 0) {
          this.emitToolOutput(inFlightTool, resultText);
        }

        this.completeToolCall(inFlightTool, "completed");

        logToolCall(
          this.config,
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
          this.config.getActiveModel(),
          this.config,
        );
      } catch (error) {
        const toolError = error instanceof Error ? error : new Error(String(error));

        this.emitToolOutput(inFlightTool, toolError.message);
        this.completeToolCall(inFlightTool, "failed");

        logToolCall(
          this.config,
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

    private async runStreamLoop(promptText: string): Promise<void> {
      const promptId = `prompt-${Date.now()}`;
      const chat = this.geminiClient.getChat();
      const baseSystemInstruction = getCoreSystemPrompt(
        this.config,
        this.config.getSystemInstructionMemory(),
      );
      chat.setSystemInstruction(baseSystemInstruction);
      let nextMessage: Content | null = {
        role: "user",
        parts: [{ text: promptText }],
      };

      try {
        while (nextMessage !== null && this.turnState && !this.stopped) {
          if (this.abortController.signal.aborted) {
            this.completeTurn("interrupted", "aborted");
            return;
          }

          const functionCalls: FunctionCall[] = [];
          const modelParts: Part[] = [];
          const routingContext: RoutingContext = {
            history: chat.getHistory(true),
            request: nextMessage.parts ?? [],
            signal: this.abortController.signal,
            requestedModel: this.config.getModel(),
          };
          const { model } = await this.config.getModelRouterService().route(routingContext);

          let responseStream: AsyncGenerator<any>;
          try {
            responseStream = await chat.sendMessageStream(
              { model },
              nextMessage.parts ?? [],
              promptId,
              this.abortController.signal,
              LlmRole.MAIN,
            );
          } catch (error) {
            if (this.abortController.signal.aborted) {
              this.completeTurn("interrupted", "aborted");
              return;
            }
            throw error;
          }

          nextMessage = null;

          try {
            for await (const response of responseStream) {
              if (this.abortController.signal.aborted) {
                this.completeTurn("interrupted", "aborted");
                return;
              }

              if (response.type !== StreamEventType.CHUNK) {
                continue;
              }

              const usage = response.value?.usageMetadata;
              if (usage) {
                this.cumulativeInputTokens += usage.promptTokenCount ?? 0;
                this.cumulativeOutputTokens += usage.candidatesTokenCount ?? 0;
              }

              if (Array.isArray(response.value?.functionCalls)) {
                functionCalls.push(...response.value.functionCalls);
              }

              const candidate = response.value?.candidates?.[0];
              for (const part of candidate?.content?.parts ?? []) {
                if (part.thought && typeof part.text === "string") {
                  this.emitReasoningText(part.text);
                  continue;
                }

                if (typeof part.text === "string") {
                  modelParts.push(part);
                  this.emitAssistantText(part.text);
                }
              }
            }
          } catch (error) {
            if (this.abortController.signal.aborted) {
              this.completeTurn("interrupted", "aborted");
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
            const responses = await this.runTool(promptId, functionCall);
            toolResponseParts.push(...responses);
          }

          if (this.abortController.signal.aborted) {
            this.completeTurn("interrupted", "aborted");
            return;
          }

          nextMessage = {
            role: "user",
            parts: toolResponseParts,
          };
        }

        if (this.turnState) {
          this.completeTurn(
            this.abortController.signal.aborted ? "interrupted" : "completed",
            this.abortController.signal.aborted ? "aborted" : "completed",
          );
        }
      } catch (error) {
        if (this.abortController.signal.aborted) {
          this.completeTurn("interrupted", "interrupted");
          return;
        }

        console.error("[gemini] Stream error:", error);
        this.completeTurn("failed", "error", toMessage(error, "Stream error"));
      }
    }

    private completeTurn(
      state: "completed" | "failed" | "interrupted" | "cancelled",
      stopReason: string,
      errorMessage?: string,
    ): void {
      if (!this.turnState) {
        return;
      }

      for (const [, tool] of this.inFlightTools) {
        emit({
          ...makeEventBase(this),
          itemId: RuntimeItemId.makeUnsafe(tool.itemId),
          type: "item.completed",
          payload: {
            itemType: tool.itemType,
            status: "failed",
          },
        } as ProviderRuntimeEvent);
      }
      this.inFlightTools.clear();

      if (this.turnState.reasoningItemEmitted) {
        emit({
          ...makeEventBase(this),
          itemId: RuntimeItemId.makeUnsafe(`reasoning-${this.turnState.turnId}`),
          type: "item.completed",
          payload: {
            itemType: "reasoning",
            status: "completed",
          },
        } as ProviderRuntimeEvent);
      }

      for (const [, pending] of this.pendingApprovals) {
        pending.resolve({
          confirmed: false,
          outcome: ToolConfirmationOutcome.Cancel,
        });
      }
      this.pendingApprovals.clear();

      for (const [requestId, pending] of this.pendingUserInputs) {
        pending.resolve(undefined);
        emit({
          ...makeEventBase(this),
          requestId: RuntimeRequestId.makeUnsafe(requestId),
          type: "user-input.resolved",
          payload: {
            answers: {},
          },
        } as ProviderRuntimeEvent);
      }
      this.pendingUserInputs.clear();

      const tokenUsage = buildTokenUsageSnapshot(this);
      emit({
        ...makeEventBase(this),
        type: "thread.token-usage.updated",
        payload: {
          usage: tokenUsage,
        },
      } as ProviderRuntimeEvent);

      const turnId = this.turnState.turnId;
      if (state === "completed") {
        this.turns.push({
          id: turnId,
          items: [],
          historyLength: this.geminiClient.getHistory().length,
        });
        updateResumeCursor(this);
      }

      const completedAt = new Date().toISOString();
      const mutableSession = this.session as {
        status: ProviderSession["status"];
        activeTurnId: TurnId | undefined;
        updatedAt: string;
        lastError?: string;
      };
      mutableSession.status = this.stopped ? "closed" : state === "failed" ? "error" : "ready";
      mutableSession.activeTurnId = undefined;
      mutableSession.updatedAt = completedAt;
      if (errorMessage) {
        mutableSession.lastError = errorMessage;
      }

      if (!this.stopped) {
        runFork(
          sessionDirectory
            .upsert(
              buildGeminiPersistedBinding({
                session: this.session,
                status: state === "failed" ? "error" : "running",
                lastRuntimeEvent: "turn.completed",
                lastRuntimeEventAt: completedAt,
              }),
            )
            .pipe(
              Effect.catchCause((cause) =>
                Effect.logWarning("failed to persist Gemini session after turn completion", {
                  threadId: this.session.threadId,
                  status: state,
                  cause,
                }),
              ),
            ),
        );
      }

      emit({
        ...makeEventBase(this),
        type: "turn.completed",
        payload: {
          state,
          stopReason,
          usage: tokenUsage,
          ...(errorMessage ? { errorMessage } : {}),
        },
      } as ProviderRuntimeEvent);

      this.turnState = undefined;

      if (!this.stopped) {
        emit({
          ...makeEventBase(this),
          type: "session.state.changed",
          payload: { state: "ready" },
        } as ProviderRuntimeEvent);
      }
    }
  }

  const sessions = new Map<string, GeminiRuntimeSession>();

  const getSession = (
    threadId: ThreadId,
  ): Effect.Effect<GeminiRuntimeSession, ProviderAdapterError> => {
    const session = sessions.get(threadId);
    if (!session) {
      return Effect.fail(
        new ProviderAdapterSessionNotFoundError({
          provider: PROVIDER,
          threadId,
        }),
      );
    }
    if (session.stopped) {
      return Effect.fail(new ProviderAdapterSessionClosedError({ provider: PROVIDER, threadId }));
    }
    return Effect.succeed(session);
  };

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

      const runtimeSession = new GeminiRuntimeSession(session, config, geminiClient, resumeState);
      sessions.set(threadId, runtimeSession);

      // Emit session lifecycle events
      emit({
        ...makeEventBase(runtimeSession),
        type: "session.started",
        payload: {},
      } as ProviderRuntimeEvent);
      emit({
        ...makeEventBase(runtimeSession),
        type: "session.configured",
        payload: { config: { model: modelId } },
      } as ProviderRuntimeEvent);
      emit({
        ...makeEventBase(runtimeSession),
        type: "session.state.changed",
        payload: { state: "ready" },
      } as ProviderRuntimeEvent);

      return session;
    },
  );

  const sendTurn: GeminiAcpAdapterShape["sendTurn"] = Effect.fn("sendTurn")(function* (input) {
    const session = yield* getSession(input.threadId);
    return yield* Effect.try({
      try: () => session.startTurn(input),
      catch: (cause) => toGeminiAdapterError(input.threadId, "Failed to start Gemini turn", cause),
    });
  });

  const interruptTurn: GeminiAcpAdapterShape["interruptTurn"] = Effect.fn("interruptTurn")(
    function* (threadId, _turnId) {
      const session = yield* getSession(threadId);
      yield* Effect.try({
        try: () => session.interrupt(),
        catch: (cause) => toGeminiAdapterError(threadId, "Failed to interrupt Gemini turn", cause),
      });
    },
  );

  const respondToRequest: GeminiAcpAdapterShape["respondToRequest"] = Effect.fn("respondToRequest")(
    function* (threadId, requestId, decision) {
      const session = yield* getSession(threadId);
      yield* Effect.try({
        try: () => session.resolveApproval(requestId, decision),
        catch: (cause) =>
          toGeminiAdapterError(threadId, "Failed to resolve Gemini approval", cause),
      });
    },
  );

  const respondToUserInput: GeminiAcpAdapterShape["respondToUserInput"] = Effect.fn(
    "respondToUserInput",
  )(function* (threadId, requestId, answers) {
    const session = yield* getSession(threadId);
    yield* Effect.try({
      try: () => session.resolveUserInput(requestId, answers),
      catch: (cause) =>
        toGeminiAdapterError(threadId, "Failed to resolve Gemini user input", cause),
    });
  });

  const stopSession: GeminiAcpAdapterShape["stopSession"] = Effect.fn("stopSession")(
    function* (threadId) {
      const session = sessions.get(threadId);
      if (!session) {
        return;
      }

      yield* Effect.tryPromise({
        try: () => session.stop(),
        catch: (cause) => toGeminiAdapterError(threadId, "Failed to stop Gemini session", cause),
      });
      sessions.delete(threadId);
    },
  );

  const listSessions = (): Effect.Effect<ReadonlyArray<ProviderSession>> =>
    Effect.sync(() =>
      Array.from(sessions.values())
        .filter((session) => !session.stopped)
        .map((session) => session.session),
    );

  const hasSession = (threadId: ThreadId): Effect.Effect<boolean> =>
    Effect.sync(() => {
      const session = sessions.get(threadId);
      return !!session && !session.stopped;
    });

  const readThread: GeminiAcpAdapterShape["readThread"] = Effect.fn("readThread")(
    function* (threadId) {
      const session = yield* getSession(threadId);
      return session.readThreadSnapshot();
    },
  );

  const rollbackThread: GeminiAcpAdapterShape["rollbackThread"] = Effect.fn("rollbackThread")(
    function* (threadId, numTurns) {
      const session = yield* getSession(threadId);
      return yield* Effect.try({
        try: () => session.rollbackThread(numTurns),
        catch: (cause) =>
          toGeminiAdapterError(threadId, "Failed to roll back Gemini thread", cause),
      });
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
