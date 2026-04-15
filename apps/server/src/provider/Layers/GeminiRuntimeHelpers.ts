import { readFile } from "node:fs/promises";

import type {
  CanonicalItemType,
  CanonicalRequestType,
  ProviderSession,
  ProviderUserInputAnswers,
  ThreadTokenUsageSnapshot,
} from "@t3tools/contracts";
import { FinishReason, type Content, type Part } from "@google/genai";
import {
  MessageBusType,
  QuestionType,
  type GeminiClient,
  type MessageBus,
  type Question,
  type RetryAttemptPayload,
  type ServerGeminiStreamEvent,
  type ThoughtSummary,
  tokenLimit,
} from "@google/gemini-cli-core";
import type { ProviderRuntimeBinding } from "../Services/ProviderSessionDirectory";

export const GEMINI_PROVIDER = "geminiAcp" as const;
export const DEFAULT_GEMINI_CONTEXT_WINDOW = 1_000_000;

const GEMINI_SCHEDULER_PATCHED = Symbol.for("t3code.gemini.scheduler-patched");

export interface GeminiTurnState {
  readonly turnId: string;
  readonly startedAt: string;
  activeModel: string | undefined;
  reasoningItemEmitted: boolean;
  readonly capturedProposedPlanKeys: Set<string>;
}

export interface GeminiPendingApproval {
  readonly requestType: CanonicalRequestType;
  readonly detail: string;
  readonly correlationId: string;
}

export interface GeminiUserInputQuestionOption {
  readonly label: string;
  readonly description: string;
}

export interface GeminiUserInputQuestion {
  readonly id: string;
  readonly header: string;
  readonly question: string;
  readonly options: ReadonlyArray<GeminiUserInputQuestionOption>;
  readonly multiSelect?: boolean;
}

export interface GeminiPendingUserInput {
  readonly detail: string;
  readonly correlationId: string;
  readonly questions: ReadonlyArray<GeminiUserInputQuestion>;
  readonly responseChannel: "tool_confirmation" | "ask_user";
}

export interface GeminiToolInFlight {
  readonly requestId: string;
  readonly itemId: string;
  readonly itemType: CanonicalItemType;
  readonly toolName: string;
  readonly title: string;
  readonly detail?: string;
  readonly input: Record<string, unknown>;
  readonly schedulerId?: string | undefined;
}

export type GeminiResumeHistory = Parameters<GeminiClient["resumeChat"]>[0];

export interface GeminiResumeState {
  readonly history: GeminiResumeHistory;
  readonly turnCount: number;
  readonly turnHistoryLengths?: ReadonlyArray<number>;
}

export interface GeminiUsageCounts {
  readonly promptTokens: number;
  readonly cachedInputTokens: number;
  readonly toolUsePromptTokens: number;
  readonly outputTokens: number;
  readonly reasoningOutputTokens: number;
  readonly totalTokens: number;
}

export interface GeminiTrackedTurn {
  readonly id: string;
  readonly items: unknown[];
  readonly historyLength?: number;
}

interface GeminiPatchedMessageBus extends MessageBus {
  [GEMINI_SCHEDULER_PATCHED]?: true;
}

export function patchGeminiMessageBusForScheduler(messageBus: MessageBus): MessageBus {
  const patchedBus = messageBus as GeminiPatchedMessageBus;
  if (patchedBus[GEMINI_SCHEDULER_PATCHED]) {
    return messageBus;
  }

  const originalSubscribe = messageBus.subscribe.bind(messageBus);
  const originalDerive = messageBus.derive?.bind(messageBus);
  let ignoredFirstToolConfirmationListener = false;

  messageBus.subscribe = ((type, listener) => {
    if (
      type === MessageBusType.TOOL_CONFIRMATION_REQUEST &&
      ignoredFirstToolConfirmationListener === false
    ) {
      ignoredFirstToolConfirmationListener = true;
      return;
    }

    originalSubscribe(type as never, listener as never);
  }) as typeof messageBus.subscribe;

  if (originalDerive) {
    messageBus.derive = ((subagentName: string) => {
      const child = originalDerive(subagentName);
      return patchGeminiMessageBusForScheduler(child);
    }) as typeof messageBus.derive;
  }

  patchedBus[GEMINI_SCHEDULER_PATCHED] = true;
  return messageBus;
}

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

export function classifyGeminiToolItemType(name: string): CanonicalItemType {
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

export function titleForGeminiToolType(type: CanonicalItemType): string {
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

export function summarizeGeminiToolRequest(
  toolName: string,
  args: Record<string, unknown>,
): string | undefined {
  const commandValue = args.command ?? args.cmd;
  if (typeof commandValue === "string" && commandValue.trim().length > 0) {
    return commandValue.trim();
  }

  const fileValue =
    args.filePath ?? args.filepath ?? args.path ?? args.file ?? args.targetPath ?? args.target;
  if (typeof fileValue === "string" && fileValue.trim().length > 0) {
    return `${toolName}: ${fileValue.trim()}`;
  }

  return undefined;
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

      return {
        id: `q-${index}`,
        header,
        question: prompt,
        options:
          normalizedOptions.length > 0
            ? normalizedOptions
            : fallbackOptionsForGeminiQuestion(question),
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
  return seconds === 0 ? `${minutes}m` : `${minutes}m ${seconds}s`;
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
  return isGeminiCapacityRetry(payload)
    ? `Capacity exhausted, retrying in ${retryWindow} (${attemptLabel})`
    : `Request retrying in ${retryWindow} (${attemptLabel})`;
}

export function formatGeminiThoughtSummary(
  summary: Pick<ThoughtSummary, "subject" | "description">,
): string {
  const subject = summary.subject.trim();
  const description = summary.description.trim();
  if (subject && description) {
    return `${subject}: ${description}`;
  }
  return subject || description;
}

export function formatGeminiFinishReasonMessage(
  reason: FinishReason | undefined,
): string | undefined {
  switch (reason) {
    case undefined:
    case FinishReason.FINISH_REASON_UNSPECIFIED:
    case FinishReason.STOP:
      return undefined;
    case FinishReason.MAX_TOKENS:
      return "Response truncated due to token limits.";
    case FinishReason.SAFETY:
      return "Response stopped due to safety reasons.";
    case FinishReason.RECITATION:
      return "Response stopped due to recitation policy.";
    case FinishReason.LANGUAGE:
      return "Response stopped due to unsupported language.";
    case FinishReason.BLOCKLIST:
      return "Response stopped due to forbidden terms.";
    case FinishReason.PROHIBITED_CONTENT:
      return "Response stopped due to prohibited content.";
    case FinishReason.SPII:
      return "Response stopped due to sensitive personally identifiable information.";
    case FinishReason.OTHER:
      return "Response stopped for other reasons.";
    case FinishReason.MALFORMED_FUNCTION_CALL:
      return "Response stopped due to malformed function call.";
    case FinishReason.IMAGE_SAFETY:
      return "Response stopped due to image safety violations.";
    case FinishReason.UNEXPECTED_TOOL_CALL:
      return "Response stopped due to unexpected tool call.";
    case FinishReason.IMAGE_PROHIBITED_CONTENT:
      return "Response stopped due to prohibited image content.";
    case FinishReason.NO_IMAGE:
      return "Response stopped because no image was generated.";
  }
}

export function formatGeminiChatCompressionMessage(input: {
  readonly originalTokenCount: number;
  readonly newTokenCount: number;
  readonly limit: number;
}): string {
  const safeLimit = input.limit > 0 ? input.limit : DEFAULT_GEMINI_CONTEXT_WINDOW;
  const originalPercentage = Math.round((input.originalTokenCount / safeLimit) * 100);
  const newPercentage = Math.round((input.newTokenCount / safeLimit) * 100);
  return `Context compressed from ${originalPercentage}% to ${newPercentage}%.`;
}

export function formatGeminiContextWindowOverflowMessage(input: {
  readonly estimatedRequestTokenCount: number;
  readonly remainingTokenCount: number;
  readonly limit: number;
}): string {
  const safeLimit = input.limit > 0 ? input.limit : DEFAULT_GEMINI_CONTEXT_WINDOW;
  const moreThanQuarterUsed = input.remainingTokenCount < safeLimit * 0.75;
  let message =
    `Sending this message (${input.estimatedRequestTokenCount} tokens) might exceed the ` +
    `context window limit (${input.remainingTokenCount} tokens left).`;
  if (moreThanQuarterUsed) {
    message += " Please reduce the request size or compress the conversation before retrying.";
  }
  return message;
}

export function formatGeminiAgentExecutionMessage(
  kind: "blocked" | "stopped",
  reason: string,
  systemMessage?: string,
): string {
  const detail = systemMessage?.trim() || reason;
  return kind === "blocked"
    ? `Agent execution blocked: ${detail}`
    : `Agent execution stopped: ${detail}`;
}

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

export function buildGeminiToolStreamKind(
  itemType: CanonicalItemType,
): "command_output" | "file_change_output" {
  return itemType === "command_execution" ? "command_output" : "file_change_output";
}

export function buildTokenUsageSnapshot(input: {
  readonly usage: GeminiUsageCounts;
  readonly totalProcessedTokens?: number | undefined;
  readonly model?: string | null | undefined;
}): ThreadTokenUsageSnapshot | undefined {
  const inputTokens = input.usage.promptTokens + input.usage.toolUsePromptTokens;
  const outputTokens = input.usage.outputTokens;
  const reasoningOutputTokens = input.usage.reasoningOutputTokens;
  const usedTokens = inputTokens + outputTokens;
  if (usedTokens <= 0) {
    return undefined;
  }

  return {
    usedTokens,
    ...(input.totalProcessedTokens !== undefined && input.totalProcessedTokens > usedTokens
      ? { totalProcessedTokens: input.totalProcessedTokens }
      : {}),
    maxTokens: tokenLimit(input.model ?? "") || DEFAULT_GEMINI_CONTEXT_WINDOW,
    ...(inputTokens > 0 ? { inputTokens } : {}),
    ...(input.usage.cachedInputTokens > 0
      ? { cachedInputTokens: input.usage.cachedInputTokens }
      : {}),
    ...(outputTokens > 0 ? { outputTokens } : {}),
    ...(reasoningOutputTokens > 0 ? { reasoningOutputTokens } : {}),
    lastUsedTokens: usedTokens,
    ...(inputTokens > 0 ? { lastInputTokens: inputTokens } : {}),
    ...(input.usage.cachedInputTokens > 0
      ? { lastCachedInputTokens: input.usage.cachedInputTokens }
      : {}),
    ...(outputTokens > 0 ? { lastOutputTokens: outputTokens } : {}),
    ...(reasoningOutputTokens > 0 ? { lastReasoningOutputTokens: reasoningOutputTokens } : {}),
  };
}

export function summarizeGeminiToolResultDisplay(result: {
  readonly returnDisplay: unknown;
  readonly llmContent: unknown;
  readonly error?: { readonly message?: string } | undefined;
}): string | undefined {
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

export function createGeminiResumeState(input: {
  readonly geminiClient: GeminiClient;
  readonly turns: ReadonlyArray<GeminiTrackedTurn>;
}): GeminiResumeState {
  const turnHistoryLengths = input.turns
    .map((turn) => turn.historyLength)
    .filter((historyLength): historyLength is number => typeof historyLength === "number");

  return {
    history: [...input.geminiClient.getHistory()],
    turnCount: input.turns.length,
    ...(turnHistoryLengths.length === input.turns.length ? { turnHistoryLengths } : {}),
  };
}

export function restoreGeminiHistoryFromResumeCursor(input: {
  readonly geminiClient: GeminiClient;
  readonly resumeCursor: unknown;
}): void {
  const resumeState = readGeminiResumeState(input.resumeCursor);
  if (!resumeState) {
    return;
  }

  input.geminiClient.setHistory([...resumeState.history]);
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

export function buildGeminiRawEvent(event: ServerGeminiStreamEvent): {
  readonly source: "gemini.acp.session-update";
  readonly method: string;
  readonly payload: ServerGeminiStreamEvent;
} {
  return {
    source: "gemini.acp.session-update",
    method: `gemini/${event.type}`,
    payload: event,
  };
}
