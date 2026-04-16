import Anthropic from "@anthropic-ai/sdk";
import {
  type CanonicalItemType,
  type CanonicalRequestType,
  DEFAULT_MODEL_BY_PROVIDER,
  EventId,
  ProviderItemId,
  type ProviderRuntimeEvent,
  type ProviderRuntimeTurnStatus,
  type ProviderSendTurnInput,
  type ProviderSession,
  RuntimeItemId,
  RuntimeRequestId,
  RuntimeTaskId,
  type ThreadTokenUsageSnapshot,
  ThreadId,
  TurnId,
  type UserInputQuestion,
} from "@t3tools/contracts";
import { resolveModelSlugForProvider } from "@t3tools/shared/model";
import {
  Exit,
  Cause,
  DateTime,
  Effect,
  Fiber,
  FileSystem,
  Layer,
  Queue,
  Random,
  Scope,
  Stream,
} from "effect";

import { resolveAttachmentPath } from "../../attachmentStore.ts";
import { ServerConfig } from "../../config.ts";
import { ServerSettingsService } from "../../serverSettings.ts";
import {
  ProviderAdapterProcessError,
  ProviderAdapterRequestError,
  ProviderAdapterSessionClosedError,
  ProviderAdapterSessionNotFoundError,
  ProviderAdapterValidationError,
  type ProviderAdapterError,
} from "../Errors.ts";
import { ANTIGRAVITY_DEFAULT_BASE_URL, ANTIGRAVITY_DEFAULT_MAX_TOKENS } from "../antigravity";
import {
  AntigravityAdapter,
  type AntigravityAdapterShape,
} from "../Services/AntigravityAdapter.ts";

const PROVIDER = "antigravity" as const;
const SUPPORTED_IMAGE_MIME_TYPES = new Set(["image/gif", "image/jpeg", "image/png", "image/webp"]);

type AntigravityToolKind = "tool_use" | "server_tool_use" | "mcp_tool_use";
type AntigravityToolResultKind = "tool_result" | "mcp_tool_result";

type AntigravityMessageContentBlock =
  | {
      type: "text";
      text: string;
    }
  | {
      type: "thinking";
      thinking: string;
      signature?: string;
    }
  | {
      type: AntigravityToolKind;
      id: string;
      name: string;
      input: Record<string, unknown>;
      server_name?: string;
    }
  | {
      type: AntigravityToolResultKind;
      tool_use_id: string;
      content?: string;
      is_error?: boolean;
    }
  | {
      type: "image";
      source: {
        type: "base64";
        media_type: string;
        data: string;
      };
    };

interface AntigravityMessageInput {
  role: "user" | "assistant";
  content: string | AntigravityMessageContentBlock[];
}

interface AssistantTextBlockState {
  readonly itemId: string;
  readonly blockIndex: number;
  emittedTextDelta: boolean;
  fallbackText: string;
  streamClosed: boolean;
  completionEmitted: boolean;
}

interface ToolBlockState {
  readonly kind: AntigravityToolKind;
  readonly blockIndex: number;
  readonly itemId: string;
  readonly toolUseId: string;
  readonly toolName: string;
  readonly itemType: CanonicalItemType;
  readonly title: string;
  readonly serverName?: string;
  input: Record<string, unknown>;
  detail?: string;
  partialInputJson: string;
  lastEmittedInputFingerprint?: string;
}

interface ToolBlockAliasState {
  readonly kind: "tool_alias";
  readonly blockIndex: number;
  readonly toolUseId: string;
  readonly target: ToolBlockState;
  partialInputJson: string;
}

interface ThinkingBlockState {
  readonly kind: "thinking";
  readonly blockIndex: number;
  readonly itemId: string;
  readonly taskId: string;
  thinking: string;
  signature?: string;
  completionEmitted: boolean;
  lastProgressSummary?: string;
}

type AntigravityHistoricalBlock =
  | {
      readonly kind: "text";
      readonly text: string;
    }
  | {
      readonly kind: "thinking";
      readonly thinking: string;
      readonly signature?: string;
    }
  | {
      readonly kind: AntigravityToolKind;
      readonly toolUseId: string;
      readonly toolName: string;
      readonly input: Record<string, unknown>;
      readonly serverName?: string;
    };

interface AntigravityReplayProbe {
  readonly index: number;
  readonly kind: AntigravityHistoricalBlock["kind"];
  readonly bufferedEvents: Array<{
    readonly event: string;
    readonly data: unknown;
  }>;
  text: string;
  thinking: string;
  signature?: string;
  toolUseId?: string;
  toolName?: string;
  serverName?: string;
  input: Record<string, unknown>;
  partialInputJson: string;
}

interface AntigravityReplayState {
  readonly expectedBlocks: ReadonlyArray<AntigravityHistoricalBlock>;
  readonly pendingProbes: Map<number, AntigravityReplayProbe>;
  readonly skippedBlockIndices: Set<number>;
  matchedCount: number;
  active: boolean;
}

type ResponseBlockState =
  | {
      readonly kind: "text";
      text: string;
    }
  | ThinkingBlockState
  | ToolBlockAliasState
  | ToolBlockState;

interface PendingApproval {
  readonly requestType: CanonicalRequestType;
  readonly detail?: string;
  readonly interactionType?: string;
  readonly tool: ToolBlockState;
}

interface PendingUserInput {
  readonly interactionType: string;
  readonly tool: ToolBlockState;
  readonly questions: ReadonlyArray<UserInputQuestion>;
}

interface AntigravityCompletedTurn {
  readonly id: TurnId;
  readonly items: Array<unknown>;
  readonly messages: Array<AntigravityMessageInput>;
}

interface AntigravityTurnState {
  readonly turnId: TurnId;
  readonly startedAt: string;
  readonly items: Array<unknown>;
  readonly turnMessages: Array<AntigravityMessageInput>;
  readonly historyCheckpoint: number;
  readonly assistantTextBlocks: Map<number, AssistantTextBlockState>;
  readonly assistantTextBlockOrder: Array<AssistantTextBlockState>;
  readonly responseBlocks: Map<number, ResponseBlockState>;
  readonly inFlightTools: Map<string, ToolBlockState>;
  stopReason: string | null | undefined;
  usage:
    | {
        readonly input_tokens?: number;
        readonly output_tokens?: number;
      }
    | undefined;
  replayState: AntigravityReplayState | undefined;
}

interface AntigravityResumeCursor {
  readonly containerId?: string;
  readonly turnCount?: number;
}

interface AntigravitySessionContext {
  session: ProviderSession;
  readonly baseUrl: string;
  readonly messages: Array<AntigravityMessageInput>;
  readonly turns: Array<AntigravityCompletedTurn>;
  readonly pendingApprovals: Map<string, PendingApproval>;
  readonly pendingUserInputs: Map<string, PendingUserInput>;
  turnState: AntigravityTurnState | undefined;
  requestFiber: Fiber.Fiber<void, never> | undefined;
  requestAbortController: AbortController | undefined;
  containerId: string | undefined;
  stopped: boolean;
}

function makeQuestionOptions(values: ReadonlyArray<string>) {
  return values
    .map((value) => value.trim())
    .filter((value) => value.length > 0)
    .map((value) => ({
      label: value,
      description: value,
    }));
}

function appendTurnMessage(context: AntigravitySessionContext, message: AntigravityMessageInput) {
  context.messages.push(message);
  if (context.turnState) {
    context.turnState.turnMessages.push(message);
    context.turnState.items.push(message);
  }
}

function clearPendingInteractions(context: AntigravitySessionContext) {
  context.pendingApprovals.clear();
  context.pendingUserInputs.clear();
}

function clearRequestFiber(context: AntigravitySessionContext) {
  context.requestFiber = undefined;
  context.requestAbortController = undefined;
}

function resolveToolById(context: AntigravitySessionContext, toolUseId: string) {
  return context.turnState?.inFlightTools.get(toolUseId);
}

function snapshotThread(context: AntigravitySessionContext) {
  return {
    threadId: context.session.threadId,
    turns: context.turns.map((turn) => ({
      id: turn.id,
      items: [...turn.items],
    })),
  };
}

export interface AntigravityAdapterLiveOptions {
  readonly fetch?: typeof globalThis.fetch;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function toMessage(cause: unknown, fallback: string): string {
  if (cause instanceof Error && cause.message.trim().length > 0) {
    return cause.message;
  }
  return fallback;
}

function normalizeInteractionType(type: string): string {
  return type.replace(/[A-Z]/g, (match) => `_${match.toLowerCase()}`);
}

function classifyToolItemType(toolName: string): CanonicalItemType {
  const normalized = toolName.toLowerCase();
  if (normalized.includes("agent")) {
    return "collab_agent_tool_call";
  }
  if (
    normalized.includes("bash") ||
    normalized.includes("command") ||
    normalized.includes("shell") ||
    normalized.includes("terminal") ||
    normalized.includes("browser")
  ) {
    return "command_execution";
  }
  if (
    normalized.includes("edit") ||
    normalized.includes("write") ||
    normalized.includes("file") ||
    normalized.includes("patch") ||
    normalized.includes("replace") ||
    normalized.includes("create") ||
    normalized.includes("delete")
  ) {
    return "file_change";
  }
  if (normalized.includes("mcp")) {
    return "mcp_tool_call";
  }
  if (normalized.includes("read") || normalized.includes("search")) {
    return "web_search";
  }
  return "dynamic_tool_call";
}

function isReadOnlyToolName(toolName: string): boolean {
  const normalized = toolName.toLowerCase();
  return (
    normalized.includes("read") ||
    normalized.includes("view") ||
    normalized.includes("search") ||
    normalized.includes("url_content")
  );
}

function classifyRequestType(toolName: string): CanonicalRequestType {
  if (isReadOnlyToolName(toolName)) {
    return "file_read_approval";
  }
  const itemType = classifyToolItemType(toolName);
  return itemType === "command_execution"
    ? "command_execution_approval"
    : itemType === "file_change"
      ? "file_change_approval"
      : "dynamic_tool_call";
}

function summarizeToolRequest(toolName: string, input: Record<string, unknown>): string {
  const commandValue = input.command ?? input.cmd ?? input.url;
  const command = typeof commandValue === "string" ? commandValue : undefined;
  if (command && command.trim().length > 0) {
    return `${toolName}: ${command.trim().slice(0, 400)}`;
  }

  const serialized = JSON.stringify(input);
  if (serialized === "{}") {
    return toolName;
  }
  if (serialized.length <= 400) {
    return `${toolName}: ${serialized}`;
  }
  return `${toolName}: ${serialized.slice(0, 397)}...`;
}

function titleForTool(itemType: CanonicalItemType): string {
  switch (itemType) {
    case "command_execution":
      return "Command run";
    case "file_change":
      return "File change";
    case "mcp_tool_call":
      return "MCP tool call";
    case "collab_agent_tool_call":
      return "Subagent task";
    case "web_search":
      return "Read operation";
    default:
      return "Tool call";
  }
}

function toolInputFingerprint(input: Record<string, unknown>): string | undefined {
  try {
    return JSON.stringify(input);
  } catch {
    return undefined;
  }
}

function safeParseJsonRecord(value: string): Record<string, unknown> | undefined {
  try {
    const parsed = JSON.parse(value);
    return isRecord(parsed) ? parsed : undefined;
  } catch {
    return undefined;
  }
}

function normalizeProgressSummary(value: string): string | undefined {
  const normalized = value.replace(/\s+/g, " ").trim();
  return normalized.length > 0 ? normalized : undefined;
}

function buildToolLifecycleData(tool: ToolBlockState) {
  return {
    toolName: tool.toolName,
    input: tool.input,
    item: {
      input: tool.input,
      ...(typeof tool.input.command === "string" ? { command: tool.input.command } : {}),
      ...(typeof tool.input.url === "string" ? { url: tool.input.url } : {}),
      ...(typeof tool.input.target_file_uri === "string"
        ? { path: tool.input.target_file_uri }
        : {}),
      ...(typeof tool.input.absolute_path_uri === "string"
        ? { path: tool.input.absolute_path_uri }
        : {}),
      ...(typeof tool.input.command_id === "string" ? { commandId: tool.input.command_id } : {}),
    },
  };
}

function hasMeaningfulToolInput(input: Record<string, unknown>): boolean {
  return Object.keys(input).length > 0;
}

function normalizeTokenUsage(input: {
  readonly input_tokens?: number;
  readonly output_tokens?: number;
}): ThreadTokenUsageSnapshot | undefined {
  const inputTokens =
    typeof input.input_tokens === "number" && Number.isFinite(input.input_tokens)
      ? input.input_tokens
      : 0;
  const outputTokens =
    typeof input.output_tokens === "number" && Number.isFinite(input.output_tokens)
      ? input.output_tokens
      : 0;
  const usedTokens = inputTokens + outputTokens;
  if (usedTokens <= 0) {
    return undefined;
  }
  return {
    usedTokens,
    lastUsedTokens: usedTokens,
    ...(inputTokens > 0 ? { inputTokens } : {}),
    ...(outputTokens > 0 ? { outputTokens } : {}),
  };
}

function buildToolResultKind(tool: ToolBlockState): AntigravityToolResultKind {
  return tool.kind === "mcp_tool_use" ? "mcp_tool_result" : "tool_result";
}

function safeStringifyJson(value: unknown, fallback = "{}"): string {
  try {
    return JSON.stringify(value ?? {});
  } catch {
    return fallback;
  }
}

function historicalBlockInputFingerprint(
  block: Extract<AntigravityHistoricalBlock, { input: Record<string, unknown> }>,
): string {
  return safeStringifyJson(block.input);
}

function contentBlockToHistoricalBlock(
  block: AntigravityMessageContentBlock,
): AntigravityHistoricalBlock | undefined {
  switch (block.type) {
    case "text":
      return {
        kind: "text",
        text: block.text,
      };
    case "thinking":
      return {
        kind: "thinking",
        thinking: block.thinking,
        ...(block.signature ? { signature: block.signature } : {}),
      };
    case "tool_use":
    case "server_tool_use":
    case "mcp_tool_use":
      return {
        kind: block.type,
        toolUseId: block.id,
        toolName: block.name,
        input: block.input,
        ...(block.server_name ? { serverName: block.server_name } : {}),
      };
    default:
      return undefined;
  }
}

function responseBlockToHistoricalBlock(
  block: ResponseBlockState,
): AntigravityHistoricalBlock | undefined {
  switch (block.kind) {
    case "text":
      return block.text.length > 0
        ? {
            kind: "text",
            text: block.text,
          }
        : undefined;
    case "thinking":
      return block.thinking.length > 0
        ? {
            kind: "thinking",
            thinking: block.thinking,
            ...(block.signature ? { signature: block.signature } : {}),
          }
        : undefined;
    case "tool_use":
    case "server_tool_use":
    case "mcp_tool_use":
      return {
        kind: block.kind,
        toolUseId: block.toolUseId,
        toolName: block.toolName,
        input: block.input,
        ...(block.serverName ? { serverName: block.serverName } : {}),
      };
    case "tool_alias":
      return undefined;
  }
}

function assistantMessageToHistoricalBlocks(
  message: AntigravityMessageInput,
): ReadonlyArray<AntigravityHistoricalBlock> {
  if (message.role !== "assistant" || typeof message.content === "string") {
    return [];
  }
  return message.content.flatMap((block) => {
    const historical = contentBlockToHistoricalBlock(block);
    return historical ? [historical] : [];
  });
}

function buildReplayExpectedBlocks(
  context: AntigravitySessionContext,
): ReadonlyArray<AntigravityHistoricalBlock> {
  const completedTurnBlocks = context.turns.flatMap((turn) =>
    turn.messages.flatMap((message) => assistantMessageToHistoricalBlocks(message)),
  );
  const activeTurnBlocks = context.turnState
    ? [...context.turnState.responseBlocks.entries()]
        .toSorted(([left], [right]) => left - right)
        .flatMap(([, block]) => {
          const historical = responseBlockToHistoricalBlock(block);
          return historical ? [historical] : [];
        })
    : [];
  return [...completedTurnBlocks, ...activeTurnBlocks];
}

function buildReplayProbe(
  index: number,
  contentBlock: Record<string, unknown>,
  event: {
    readonly event: string;
    readonly data: unknown;
  },
): AntigravityReplayProbe | undefined {
  if (typeof contentBlock.type !== "string") {
    return undefined;
  }

  switch (contentBlock.type) {
    case "text":
      return {
        index,
        kind: "text",
        bufferedEvents: [event],
        text: typeof contentBlock.text === "string" ? contentBlock.text : "",
        thinking: "",
        input: {},
        partialInputJson: "",
      };
    case "thinking":
      return {
        index,
        kind: "thinking",
        bufferedEvents: [event],
        text: "",
        thinking: typeof contentBlock.thinking === "string" ? contentBlock.thinking : "",
        ...(typeof contentBlock.signature === "string"
          ? { signature: contentBlock.signature }
          : {}),
        input: {},
        partialInputJson: "",
      };
    case "tool_use":
    case "server_tool_use":
    case "mcp_tool_use":
      return {
        index,
        kind: contentBlock.type,
        bufferedEvents: [event],
        text: "",
        thinking: "",
        ...(typeof contentBlock.id === "string" ? { toolUseId: contentBlock.id } : {}),
        ...(typeof contentBlock.name === "string" ? { toolName: contentBlock.name } : {}),
        ...(typeof contentBlock.server_name === "string"
          ? { serverName: contentBlock.server_name }
          : {}),
        input: isRecord(contentBlock.input) ? contentBlock.input : {},
        partialInputJson: "",
      };
    default:
      return undefined;
  }
}

function isReplayProbeCompatible(
  expected: AntigravityHistoricalBlock,
  probe: AntigravityReplayProbe,
): boolean {
  if (expected.kind !== probe.kind) {
    return false;
  }

  if (
    expected.kind === "tool_use" ||
    expected.kind === "server_tool_use" ||
    expected.kind === "mcp_tool_use"
  ) {
    return (
      expected.toolUseId === probe.toolUseId &&
      expected.toolName === probe.toolName &&
      (expected.kind !== "mcp_tool_use" || expected.serverName === probe.serverName)
    );
  }

  return true;
}

function doesReplayProbeStillMatchExpected(
  expected: AntigravityHistoricalBlock,
  probe: AntigravityReplayProbe,
): boolean {
  if (!isReplayProbeCompatible(expected, probe)) {
    return false;
  }

  switch (expected.kind) {
    case "text":
      return expected.text.startsWith(probe.text);
    case "thinking":
      return (
        expected.thinking.startsWith(probe.thinking) &&
        (probe.signature === undefined || expected.signature === probe.signature)
      );
    case "tool_use":
    case "server_tool_use":
    case "mcp_tool_use": {
      const expectedInput = historicalBlockInputFingerprint(expected);
      if (probe.partialInputJson.length > 0) {
        return expectedInput.startsWith(probe.partialInputJson);
      }
      return safeStringifyJson(probe.input) === expectedInput;
    }
  }
}

function isReplayProbeFullyMatched(
  expected: AntigravityHistoricalBlock,
  probe: AntigravityReplayProbe,
): boolean {
  if (!doesReplayProbeStillMatchExpected(expected, probe)) {
    return false;
  }

  switch (expected.kind) {
    case "text":
      return probe.text === expected.text;
    case "thinking":
      return probe.thinking === expected.thinking && expected.signature === probe.signature;
    case "tool_use":
    case "server_tool_use":
    case "mcp_tool_use":
      return safeStringifyJson(probe.input) === historicalBlockInputFingerprint(expected);
  }
}

function readResumeCursor(value: unknown): AntigravityResumeCursor | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  const containerId =
    typeof value.containerId === "string" && value.containerId.trim().length > 0
      ? value.containerId
      : undefined;
  const turnCount =
    typeof value.turnCount === "number" && Number.isInteger(value.turnCount) && value.turnCount >= 0
      ? value.turnCount
      : undefined;
  return {
    ...(containerId ? { containerId } : {}),
    ...(turnCount !== undefined ? { turnCount } : {}),
  };
}

function buildToolQuestions(tool: ToolBlockState): ReadonlyArray<UserInputQuestion> | undefined {
  const normalizedInteraction = normalizeInteractionType(tool.toolName);
  if (normalizedInteraction !== "elicitation") {
    return undefined;
  }

  const message = typeof tool.input.message === "string" ? tool.input.message.trim() : "";
  const header = typeof tool.input.server_name === "string" ? tool.input.server_name : "Input";
  const schemaText =
    typeof tool.input.requested_schema_json === "string"
      ? tool.input.requested_schema_json
      : undefined;
  if (!schemaText) {
    return undefined;
  }

  let schema: unknown;
  try {
    schema = JSON.parse(schemaText);
  } catch {
    return undefined;
  }

  if (isRecord(schema)) {
    const properties = isRecord(schema.properties) ? schema.properties : undefined;
    if (properties) {
      const questions = Object.entries(properties)
        .map(([key, value], index): UserInputQuestion | null => {
          if (!isRecord(value)) {
            return null;
          }
          const enumValues = Array.isArray(value.enum)
            ? value.enum.filter((candidate): candidate is string => typeof candidate === "string")
            : undefined;
          if (!enumValues || enumValues.length === 0) {
            return null;
          }
          return {
            id: key,
            header: header || `Question ${index + 1}`,
            question:
              typeof value.title === "string"
                ? value.title
                : message || `Choose a value for ${key}`,
            options: makeQuestionOptions(enumValues),
          };
        })
        .filter((question): question is UserInputQuestion => question !== null);
      if (questions.length > 0) {
        return questions;
      }
    }

    if (schema.type === "boolean") {
      return [
        {
          id: "value",
          header,
          question: message || "Choose an option",
          options: [
            { label: "Yes", description: "Submit `true`." },
            { label: "No", description: "Submit `false`." },
          ],
        },
      ];
    }

    const enumValues = Array.isArray(schema.enum)
      ? schema.enum.filter((candidate): candidate is string => typeof candidate === "string")
      : undefined;
    if (enumValues && enumValues.length > 0) {
      return [
        {
          id: "value",
          header,
          question: message || "Choose an option",
          options: makeQuestionOptions(enumValues),
        },
      ];
    }
  }

  return undefined;
}

function normalizeAnthropicStreamEvent(event: Record<string, unknown>): {
  readonly event: string;
  readonly data: unknown;
} | null {
  const type = typeof event.type === "string" ? event.type : null;
  if (!type) {
    return null;
  }

  switch (type) {
    case "message_start":
      return { event: type, data: { message: event.message } };
    case "message_delta":
      return { event: type, data: { delta: event.delta, usage: event.usage } };
    case "message_stop":
      return { event: type, data: {} };
    case "content_block_start":
      return {
        event: type,
        data: {
          index: event.index,
          content_block: event.content_block,
        },
      };
    case "content_block_delta":
      return {
        event: type,
        data: {
          index: event.index,
          delta: event.delta,
        },
      };
    case "content_block_stop":
      return {
        event: type,
        data: {
          index: event.index,
        },
      };
    default:
      return null;
  }
}

const buildUserMessageEffect = Effect.fn("buildUserMessageEffect")(function* (
  input: ProviderSendTurnInput,
  dependencies: {
    readonly fileSystem: FileSystem.FileSystem;
    readonly attachmentsDir: string;
  },
) {
  const blocks: Array<AntigravityMessageContentBlock> = [];
  const text = input.input?.trim();
  if (text && text.length > 0) {
    blocks.push({
      type: "text",
      text,
    });
  }

  for (const attachment of input.attachments ?? []) {
    if (attachment.type !== "image") {
      continue;
    }
    if (!SUPPORTED_IMAGE_MIME_TYPES.has(attachment.mimeType)) {
      return yield* new ProviderAdapterRequestError({
        provider: PROVIDER,
        method: "turn/start",
        detail: `Unsupported Antigravity image attachment type '${attachment.mimeType}'.`,
      });
    }

    const attachmentPath = resolveAttachmentPath({
      attachmentsDir: dependencies.attachmentsDir,
      attachment,
    });
    if (!attachmentPath) {
      return yield* new ProviderAdapterRequestError({
        provider: PROVIDER,
        method: "turn/start",
        detail: `Invalid attachment id '${attachment.id}'.`,
      });
    }

    const bytes = yield* dependencies.fileSystem.readFile(attachmentPath).pipe(
      Effect.mapError(
        (cause) =>
          new ProviderAdapterRequestError({
            provider: PROVIDER,
            method: "turn/start",
            detail: toMessage(cause, "Failed to read attachment file."),
            cause,
          }),
      ),
    );

    blocks.push({
      type: "image",
      source: {
        type: "base64",
        media_type: attachment.mimeType,
        data: Buffer.from(bytes).toString("base64"),
      },
    });
  }

  if (blocks.length === 0) {
    return yield* new ProviderAdapterValidationError({
      provider: PROVIDER,
      operation: "turn/start",
      issue: "Antigravity turns require text or at least one image attachment.",
    });
  }

  return {
    role: "user",
    content: blocks,
  } satisfies AntigravityMessageInput;
});

const makeAntigravityAdapter = Effect.fn("makeAntigravityAdapter")(function* (
  options?: AntigravityAdapterLiveOptions,
) {
  const fetchImpl = options?.fetch ?? globalThis.fetch.bind(globalThis);
  const fileSystem = yield* FileSystem.FileSystem;
  const serverConfig = yield* ServerConfig;
  const serverSettings = yield* ServerSettingsService;
  const services = yield* Effect.context<never>();
  const runPromiseWith = Effect.runPromiseWith(services);
  const runtimeEventQueue = yield* Queue.unbounded<ProviderRuntimeEvent>();
  const adapterScope = yield* Scope.make("sequential");
  const sessions = new Map<ThreadId, AntigravitySessionContext>();

  const nowIso = Effect.map(DateTime.now, DateTime.formatIso);
  const nextEventId = Effect.map(Random.nextUUIDv4, (id) => EventId.make(id));
  const makeEventStamp = () => Effect.all({ eventId: nextEventId, createdAt: nowIso });

  const offerRuntimeEvent = (event: ProviderRuntimeEvent): Effect.Effect<void> =>
    Queue.offer(runtimeEventQueue, event).pipe(Effect.asVoid);

  const asRuntimeItemId = (value: string) => RuntimeItemId.make(value);
  const asRuntimeRequestId = (value: string) => RuntimeRequestId.make(value);
  const asRuntimeTaskId = (value: string) => RuntimeTaskId.make(value);

  const updateResumeCursor = Effect.fn("updateResumeCursor")(function* (
    context: AntigravitySessionContext,
  ) {
    context.session = {
      ...context.session,
      ...(context.containerId
        ? {
            resumeCursor: {
              containerId: context.containerId,
              turnCount: context.turns.length,
            },
          }
        : {}),
      ...(context.containerId ? {} : { resumeCursor: undefined }),
      updatedAt: yield* nowIso,
    };
  });

  const setSessionState = Effect.fn("setSessionState")(function* (
    context: AntigravitySessionContext,
    state: "ready" | "running" | "waiting",
    reason?: string,
  ) {
    const stamp = yield* makeEventStamp();
    yield* offerRuntimeEvent({
      type: "session.state.changed",
      eventId: stamp.eventId,
      provider: PROVIDER,
      createdAt: stamp.createdAt,
      threadId: context.session.threadId,
      ...(context.turnState ? { turnId: context.turnState.turnId } : {}),
      payload: {
        state,
        ...(reason ? { reason } : {}),
      },
      providerRefs: {},
    });
  });

  const ensureAssistantTextBlock = Effect.fn("ensureAssistantTextBlock")(function* (
    context: AntigravitySessionContext,
    blockIndex: number,
    fallbackText = "",
  ) {
    const turnState = context.turnState;
    if (!turnState) {
      return undefined;
    }
    const existing = turnState.assistantTextBlocks.get(blockIndex);
    if (existing && !existing.completionEmitted) {
      if (existing.fallbackText.length === 0 && fallbackText.length > 0) {
        existing.fallbackText = fallbackText;
      }
      return existing;
    }

    const block: AssistantTextBlockState = {
      itemId: yield* Random.nextUUIDv4,
      blockIndex,
      emittedTextDelta: false,
      fallbackText,
      streamClosed: false,
      completionEmitted: false,
    };
    turnState.assistantTextBlocks.set(blockIndex, block);
    turnState.assistantTextBlockOrder.push(block);
    return block;
  });

  const completeAssistantTextBlock = Effect.fn("completeAssistantTextBlock")(function* (
    context: AntigravitySessionContext,
    block: AssistantTextBlockState,
    force = false,
  ) {
    if (!context.turnState || block.completionEmitted) {
      return;
    }
    if (!force && !block.streamClosed) {
      return;
    }

    if (!block.emittedTextDelta && block.fallbackText.length > 0) {
      const deltaStamp = yield* makeEventStamp();
      yield* offerRuntimeEvent({
        type: "content.delta",
        eventId: deltaStamp.eventId,
        provider: PROVIDER,
        createdAt: deltaStamp.createdAt,
        threadId: context.session.threadId,
        turnId: context.turnState.turnId,
        itemId: asRuntimeItemId(block.itemId),
        payload: {
          streamKind: "assistant_text",
          delta: block.fallbackText,
        },
        providerRefs: {},
        raw: {
          source: "antigravity.anthropic.stream",
          method: "antigravity/content_block_delta/text_delta",
          payload: {
            index: block.blockIndex,
            delta: block.fallbackText,
          },
        },
      });
    }

    block.completionEmitted = true;
    context.turnState.assistantTextBlocks.delete(block.blockIndex);
    const stamp = yield* makeEventStamp();
    yield* offerRuntimeEvent({
      type: "item.completed",
      eventId: stamp.eventId,
      provider: PROVIDER,
      createdAt: stamp.createdAt,
      threadId: context.session.threadId,
      turnId: context.turnState.turnId,
      itemId: asRuntimeItemId(block.itemId),
      payload: {
        itemType: "assistant_message",
        status: "completed",
        title: "Assistant message",
        ...(block.fallbackText.length > 0 ? { detail: block.fallbackText } : {}),
      },
      providerRefs: {},
    });
  });

  const emitRuntimeError = Effect.fn("emitRuntimeError")(function* (
    context: AntigravitySessionContext,
    message: string,
    detail?: unknown,
  ) {
    const stamp = yield* makeEventStamp();
    yield* offerRuntimeEvent({
      type: "runtime.error",
      eventId: stamp.eventId,
      provider: PROVIDER,
      createdAt: stamp.createdAt,
      threadId: context.session.threadId,
      ...(context.turnState ? { turnId: context.turnState.turnId } : {}),
      payload: {
        message,
        class: "provider_error",
        ...(detail !== undefined ? { detail } : {}),
      },
      providerRefs: {},
    });
  });

  const emitRuntimeWarning = Effect.fn("emitRuntimeWarning")(function* (
    context: AntigravitySessionContext,
    message: string,
    detail?: unknown,
  ) {
    const stamp = yield* makeEventStamp();
    yield* offerRuntimeEvent({
      type: "runtime.warning",
      eventId: stamp.eventId,
      provider: PROVIDER,
      createdAt: stamp.createdAt,
      threadId: context.session.threadId,
      ...(context.turnState ? { turnId: context.turnState.turnId } : {}),
      payload: {
        message,
        ...(detail !== undefined ? { detail } : {}),
      },
      providerRefs: {},
    });
  });

  const emitTokenUsage = Effect.fn("emitTokenUsage")(function* (
    context: AntigravitySessionContext,
    usage: {
      readonly input_tokens?: number;
      readonly output_tokens?: number;
    },
  ) {
    const normalized = normalizeTokenUsage(usage);
    if (!normalized) {
      return;
    }
    const stamp = yield* makeEventStamp();
    yield* offerRuntimeEvent({
      type: "thread.token-usage.updated",
      eventId: stamp.eventId,
      provider: PROVIDER,
      createdAt: stamp.createdAt,
      threadId: context.session.threadId,
      ...(context.turnState ? { turnId: context.turnState.turnId } : {}),
      payload: {
        usage: normalized,
      },
      providerRefs: {},
    });
  });

  const emitToolUpdate = Effect.fn("emitToolUpdate")(function* (
    context: AntigravitySessionContext,
    turnState: AntigravityTurnState,
    tool: ToolBlockState,
    raw?: {
      readonly method: string;
      readonly payload: unknown;
    },
  ) {
    const stamp = yield* makeEventStamp();
    yield* offerRuntimeEvent({
      type: "item.updated",
      eventId: stamp.eventId,
      provider: PROVIDER,
      createdAt: stamp.createdAt,
      threadId: context.session.threadId,
      turnId: turnState.turnId,
      itemId: asRuntimeItemId(tool.itemId),
      payload: {
        itemType: tool.itemType,
        status: "inProgress",
        title: tool.title,
        ...(tool.detail ? { detail: tool.detail } : {}),
        data: buildToolLifecycleData(tool),
      },
      providerRefs: {
        providerItemId: ProviderItemId.make(tool.toolUseId),
      },
      ...(raw
        ? {
            raw: {
              source: "antigravity.anthropic.stream",
              method: raw.method,
              payload: raw.payload,
            },
          }
        : {}),
    });
  });

  const syncToolInput = Effect.fn("syncToolInput")(function* (
    context: AntigravitySessionContext,
    turnState: AntigravityTurnState,
    tool: ToolBlockState,
    input: Record<string, unknown>,
    raw?: {
      readonly method: string;
      readonly payload: unknown;
    },
  ) {
    tool.input = input;
    tool.detail = summarizeToolRequest(tool.toolName, input);
    const nextFingerprint = toolInputFingerprint(input);
    if (nextFingerprint && tool.lastEmittedInputFingerprint !== nextFingerprint) {
      tool.lastEmittedInputFingerprint = nextFingerprint;
      yield* emitToolUpdate(context, turnState, tool, raw);
    }
  });

  const emitReasoningProgress = Effect.fn("emitReasoningProgress")(function* (
    context: AntigravitySessionContext,
    turnState: AntigravityTurnState,
    block: ThinkingBlockState,
    force = false,
  ) {
    const summary = normalizeProgressSummary(block.thinking);
    if (!summary) {
      return;
    }
    if (!force && block.lastProgressSummary !== undefined) {
      return;
    }
    if (block.lastProgressSummary === summary) {
      return;
    }
    block.lastProgressSummary = summary;
    const stamp = yield* makeEventStamp();
    yield* offerRuntimeEvent({
      type: "task.progress",
      eventId: stamp.eventId,
      provider: PROVIDER,
      createdAt: stamp.createdAt,
      threadId: context.session.threadId,
      turnId: turnState.turnId,
      payload: {
        taskId: asRuntimeTaskId(block.taskId),
        description: "Reasoning",
        summary,
      },
      providerRefs: {},
    });
  });

  const completeReasoningBlock = Effect.fn("completeReasoningBlock")(function* (
    context: AntigravitySessionContext,
    turnState: AntigravityTurnState,
    block: ThinkingBlockState,
    raw?: {
      readonly method: string;
      readonly payload: unknown;
    },
  ) {
    if (block.completionEmitted) {
      return;
    }
    yield* emitReasoningProgress(context, turnState, block, true);
    block.completionEmitted = true;
    const stamp = yield* makeEventStamp();
    yield* offerRuntimeEvent({
      type: "item.completed",
      eventId: stamp.eventId,
      provider: PROVIDER,
      createdAt: stamp.createdAt,
      threadId: context.session.threadId,
      turnId: turnState.turnId,
      itemId: asRuntimeItemId(block.itemId),
      payload: {
        itemType: "reasoning",
        status: "completed",
        title: "Reasoning",
        ...(normalizeProgressSummary(block.thinking)
          ? { detail: normalizeProgressSummary(block.thinking) }
          : {}),
      },
      providerRefs: {},
      ...(raw
        ? {
            raw: {
              source: "antigravity.anthropic.stream",
              method: raw.method,
              payload: raw.payload,
            },
          }
        : {}),
    });
  });

  const finalizeAssistantMessage = Effect.fn("finalizeAssistantMessage")(function* (
    context: AntigravitySessionContext,
  ) {
    const turnState = context.turnState;
    if (!turnState || turnState.responseBlocks.size === 0) {
      return;
    }

    const contentBlocks: AntigravityMessageContentBlock[] = [];
    for (const [, block] of [...turnState.responseBlocks.entries()].toSorted(
      ([left], [right]) => left - right,
    )) {
      if (block.kind === "text" && block.text.length > 0) {
        contentBlocks.push({
          type: "text",
          text: block.text,
        });
        continue;
      }
      if (block.kind === "thinking" && block.thinking.length > 0) {
        contentBlocks.push({
          type: "thinking",
          thinking: block.thinking,
          ...(block.signature ? { signature: block.signature } : {}),
        });
        continue;
      }
      if (
        block.kind === "tool_use" ||
        block.kind === "server_tool_use" ||
        block.kind === "mcp_tool_use"
      ) {
        contentBlocks.push({
          type: block.kind,
          id: block.toolUseId,
          name: block.toolName,
          input: block.input,
          ...(block.serverName ? { server_name: block.serverName } : {}),
        });
      }
    }

    if (contentBlocks.length > 0) {
      appendTurnMessage(context, {
        role: "assistant",
        content: contentBlocks,
      });
    }

    for (const block of turnState.assistantTextBlockOrder) {
      yield* completeAssistantTextBlock(context, block, true);
    }

    for (const [, block] of turnState.responseBlocks) {
      if (block.kind === "thinking") {
        yield* completeReasoningBlock(context, turnState, block);
      }
    }

    turnState.responseBlocks.clear();
    turnState.assistantTextBlocks.clear();
    turnState.assistantTextBlockOrder.length = 0;
  });

  const updateContainerId = Effect.fn("updateContainerId")(function* (
    context: AntigravitySessionContext,
    containerId: string,
  ) {
    if (!containerId || context.containerId === containerId) {
      return;
    }
    context.containerId = containerId;
    yield* updateResumeCursor(context);
    const stamp = yield* makeEventStamp();
    yield* offerRuntimeEvent({
      type: "thread.started",
      eventId: stamp.eventId,
      provider: PROVIDER,
      createdAt: stamp.createdAt,
      threadId: context.session.threadId,
      payload: {
        providerThreadId: containerId,
      },
      providerRefs: {},
      raw: {
        source: "antigravity.anthropic.stream",
        method: "antigravity/message_delta/container",
        payload: {
          containerId,
        },
      },
    });
  });

  const completeTurn = Effect.fn("completeTurn")(function* (
    context: AntigravitySessionContext,
    status: ProviderRuntimeTurnStatus,
    options?: {
      readonly stopReason?: string | null;
      readonly errorMessage?: string;
      readonly rollbackHistory?: boolean;
    },
  ) {
    const turnState = context.turnState;
    if (!turnState) {
      return;
    }

    yield* finalizeAssistantMessage(context);

    for (const tool of turnState.inFlightTools.values()) {
      const stamp = yield* makeEventStamp();
      yield* offerRuntimeEvent({
        type: "item.completed",
        eventId: stamp.eventId,
        provider: PROVIDER,
        createdAt: stamp.createdAt,
        threadId: context.session.threadId,
        turnId: turnState.turnId,
        itemId: asRuntimeItemId(tool.itemId),
        payload: {
          itemType: tool.itemType,
          status: status === "completed" ? "completed" : "failed",
          title: tool.title,
          ...(tool.detail ? { detail: tool.detail } : {}),
          data: buildToolLifecycleData(tool),
        },
        providerRefs: {
          providerItemId: ProviderItemId.make(tool.toolUseId),
        },
      });
    }

    if (turnState.usage) {
      yield* emitTokenUsage(context, turnState.usage);
    }

    const turnCompletedStamp = yield* makeEventStamp();
    yield* offerRuntimeEvent({
      type: "turn.completed",
      eventId: turnCompletedStamp.eventId,
      provider: PROVIDER,
      createdAt: turnCompletedStamp.createdAt,
      threadId: context.session.threadId,
      turnId: turnState.turnId,
      payload: {
        state: status,
        ...(options?.stopReason !== undefined ? { stopReason: options.stopReason } : {}),
        ...(turnState.usage ? { usage: turnState.usage } : {}),
        ...(options?.errorMessage ? { errorMessage: options.errorMessage } : {}),
      },
      providerRefs: {},
    });

    if (options?.rollbackHistory) {
      context.messages.splice(turnState.historyCheckpoint);
      context.containerId = undefined;
    } else {
      context.turns.push({
        id: turnState.turnId,
        items: [...turnState.items],
        messages: [...turnState.turnMessages],
      });
    }

    clearPendingInteractions(context);
    context.turnState = undefined;
    context.session = {
      ...context.session,
      status: "ready",
      activeTurnId: undefined,
      updatedAt: yield* nowIso,
      ...(status === "failed" && options?.errorMessage ? { lastError: options.errorMessage } : {}),
    };
    yield* updateResumeCursor(context);
    yield* setSessionState(context, "ready");
  });

  const failActiveTurn = Effect.fn("failActiveTurn")(function* (
    context: AntigravitySessionContext,
    status: Extract<ProviderRuntimeTurnStatus, "failed" | "interrupted" | "cancelled">,
    message: string,
  ) {
    if (status === "failed") {
      yield* emitRuntimeError(context, message);
    } else {
      yield* emitRuntimeWarning(context, message);
    }
    yield* completeTurn(context, status, {
      errorMessage: message,
      rollbackHistory: true,
    });
  });

  const openApprovalRequest = Effect.fn("openApprovalRequest")(function* (
    context: AntigravitySessionContext,
    interactionType: string | undefined,
    tool: ToolBlockState,
  ) {
    const requestId = yield* Random.nextUUIDv4;
    const requestType = classifyRequestType(tool.toolName);
    const detail = tool.detail;
    context.pendingApprovals.set(requestId, {
      requestType,
      tool,
      ...(detail ? { detail } : {}),
      ...(interactionType ? { interactionType } : {}),
    });

    const stamp = yield* makeEventStamp();
    yield* offerRuntimeEvent({
      type: "request.opened",
      eventId: stamp.eventId,
      provider: PROVIDER,
      createdAt: stamp.createdAt,
      threadId: context.session.threadId,
      ...(context.turnState ? { turnId: context.turnState.turnId } : {}),
      requestId: asRuntimeRequestId(requestId),
      payload: {
        requestType,
        ...(detail ? { detail } : {}),
        args: {
          toolName: tool.toolName,
          input: tool.input,
          toolUseId: tool.toolUseId,
          ...(interactionType ? { interactionType } : {}),
        },
      },
      providerRefs: {
        providerItemId: ProviderItemId.make(tool.toolUseId),
      },
      raw: {
        source: "antigravity.anthropic.stream",
        method: "antigravity/request.opened",
        payload: {
          interactionType,
          tool,
        },
      },
    });
  });

  const openUserInputRequest = Effect.fn("openUserInputRequest")(function* (
    context: AntigravitySessionContext,
    interactionType: string,
    tool: ToolBlockState,
    questions: ReadonlyArray<UserInputQuestion>,
  ) {
    const requestId = yield* Random.nextUUIDv4;
    context.pendingUserInputs.set(requestId, {
      interactionType,
      tool,
      questions,
    });

    const stamp = yield* makeEventStamp();
    yield* offerRuntimeEvent({
      type: "user-input.requested",
      eventId: stamp.eventId,
      provider: PROVIDER,
      createdAt: stamp.createdAt,
      threadId: context.session.threadId,
      ...(context.turnState ? { turnId: context.turnState.turnId } : {}),
      requestId: asRuntimeRequestId(requestId),
      payload: {
        questions,
      },
      providerRefs: {
        providerItemId: ProviderItemId.make(tool.toolUseId),
      },
      raw: {
        source: "antigravity.anthropic.stream",
        method: "antigravity/user-input.requested",
        payload: {
          interactionType,
          tool,
          questions,
        },
      },
    });
  });

  const handlePauseTurn = Effect.fn("handlePauseTurn")(function* (
    context: AntigravitySessionContext,
    details: Record<string, unknown>,
  ) {
    const interactionType =
      typeof details.interaction === "string" ? details.interaction : undefined;
    const toolUseId = typeof details.tool_use_id === "string" ? details.tool_use_id : undefined;
    const tool =
      toolUseId && resolveToolById(context, toolUseId)
        ? resolveToolById(context, toolUseId)
        : context.turnState
          ? [...context.turnState.inFlightTools.values()].at(-1)
          : undefined;

    if (!interactionType || !tool) {
      yield* failActiveTurn(
        context,
        "failed",
        "Antigravity returned an interaction request without a resolvable tool context.",
      );
      return;
    }

    const questions = buildToolQuestions(tool);
    yield* setSessionState(context, "waiting", `interaction:${interactionType}`);
    if (questions && questions.length > 0) {
      yield* openUserInputRequest(context, interactionType, tool, questions);
      return;
    }
    yield* openApprovalRequest(context, interactionType, tool);
  });

  const replayBufferedEvents = Effect.fn("replayBufferedEvents")(function* (
    context: AntigravitySessionContext,
    events: ReadonlyArray<{
      readonly event: string;
      readonly data: unknown;
    }>,
  ) {
    for (const event of events) {
      yield* processResponseEventDirect(context, event);
    }
  });

  const processResponseEventDirect = Effect.fn("processResponseEventDirect")(function* (
    context: AntigravitySessionContext,
    event: {
      readonly event: string;
      readonly data: unknown;
    },
  ) {
    const turnState = context.turnState;
    if (!turnState) {
      return;
    }
    const data = isRecord(event.data) ? event.data : {};
    console.log("event", event);
    switch (event.event) {
      case "message_start": {
        const message = isRecord(data.message) ? data.message : undefined;
        if (message && isRecord(message.container) && typeof message.container.id === "string") {
          yield* updateContainerId(context, message.container.id);
        }
        const usage = message && isRecord(message.usage) ? message.usage : undefined;
        if (usage) {
          const nextUsage = {
            ...(typeof usage.input_tokens === "number" ? { input_tokens: usage.input_tokens } : {}),
            ...(typeof usage.output_tokens === "number"
              ? { output_tokens: usage.output_tokens }
              : {}),
          };
          if (Object.keys(nextUsage).length > 0) {
            turnState.usage = {
              ...turnState.usage,
              ...nextUsage,
            };
            yield* emitTokenUsage(context, turnState.usage);
          }
        }
        return;
      }

      case "content_block_start": {
        const index = typeof data.index === "number" ? data.index : undefined;
        const contentBlock = isRecord(data.content_block) ? data.content_block : undefined;
        if (index === undefined || !contentBlock || typeof contentBlock.type !== "string") {
          return;
        }

        if (contentBlock.type === "text") {
          turnState.responseBlocks.set(index, {
            kind: "text",
            text: typeof contentBlock.text === "string" ? contentBlock.text : "",
          });
          yield* ensureAssistantTextBlock(
            context,
            index,
            typeof contentBlock.text === "string" ? contentBlock.text : "",
          );
          return;
        }

        if (contentBlock.type === "thinking") {
          const thinkingBlock: ThinkingBlockState = {
            kind: "thinking",
            blockIndex: index,
            itemId: yield* Random.nextUUIDv4,
            taskId: yield* Random.nextUUIDv4,
            thinking: typeof contentBlock.thinking === "string" ? contentBlock.thinking : "",
            ...(typeof contentBlock.signature === "string"
              ? { signature: contentBlock.signature }
              : {}),
            completionEmitted: false,
          };
          turnState.responseBlocks.set(index, thinkingBlock);
          const stamp = yield* makeEventStamp();
          yield* offerRuntimeEvent({
            type: "item.started",
            eventId: stamp.eventId,
            provider: PROVIDER,
            createdAt: stamp.createdAt,
            threadId: context.session.threadId,
            turnId: turnState.turnId,
            itemId: asRuntimeItemId(thinkingBlock.itemId),
            payload: {
              itemType: "reasoning",
              status: "inProgress",
              title: "Reasoning",
            },
            providerRefs: {},
            raw: {
              source: "antigravity.anthropic.stream",
              method: `antigravity/${event.event}`,
              payload: event.data,
            },
          });
          yield* emitReasoningProgress(context, turnState, thinkingBlock);
          return;
        }

        if (
          contentBlock.type !== "tool_use" &&
          contentBlock.type !== "server_tool_use" &&
          contentBlock.type !== "mcp_tool_use"
        ) {
          return;
        }

        const toolName = typeof contentBlock.name === "string" ? contentBlock.name : "tool";
        const input = isRecord(contentBlock.input) ? contentBlock.input : {};
        const toolUseId =
          typeof contentBlock.id === "string" && contentBlock.id.length > 0
            ? contentBlock.id
            : `antigravity-tool-${index}`;
        const existingTool = turnState.inFlightTools.get(toolUseId);
        if (existingTool) {
          turnState.responseBlocks.set(index, {
            kind: "tool_alias",
            blockIndex: index,
            toolUseId,
            target: existingTool,
            partialInputJson: "",
          });
          if (hasMeaningfulToolInput(input)) {
            yield* syncToolInput(context, turnState, existingTool, input, {
              method: `antigravity/${event.event}`,
              payload: event.data,
            });
          }
          return;
        }
        const itemType = classifyToolItemType(toolName);
        const detail = summarizeToolRequest(toolName, input);
        const lastEmittedInputFingerprint = toolInputFingerprint(input);
        const tool: ToolBlockState = {
          kind: contentBlock.type,
          blockIndex: index,
          itemId: yield* Random.nextUUIDv4,
          toolUseId,
          toolName,
          itemType,
          title: titleForTool(itemType),
          ...(typeof contentBlock.server_name === "string"
            ? { serverName: contentBlock.server_name }
            : {}),
          input,
          partialInputJson: "",
          ...(detail ? { detail } : {}),
          ...(lastEmittedInputFingerprint ? { lastEmittedInputFingerprint } : {}),
        };
        turnState.responseBlocks.set(index, tool);
        turnState.inFlightTools.set(tool.toolUseId, tool);

        const stamp = yield* makeEventStamp();
        yield* offerRuntimeEvent({
          type: "item.started",
          eventId: stamp.eventId,
          provider: PROVIDER,
          createdAt: stamp.createdAt,
          threadId: context.session.threadId,
          turnId: turnState.turnId,
          itemId: asRuntimeItemId(tool.itemId),
          payload: {
            itemType: tool.itemType,
            status: "inProgress",
            title: tool.title,
            ...(tool.detail ? { detail: tool.detail } : {}),
            data: buildToolLifecycleData(tool),
          },
          providerRefs: {
            providerItemId: ProviderItemId.make(tool.toolUseId),
          },
          raw: {
            source: "antigravity.anthropic.stream",
            method: `antigravity/${event.event}`,
            payload: event.data,
          },
        });
        yield* emitToolUpdate(context, turnState, tool, {
          method: `antigravity/${event.event}`,
          payload: event.data,
        });
        return;
      }

      case "content_block_delta": {
        const index = typeof data.index === "number" ? data.index : undefined;
        const delta = isRecord(data.delta) ? data.delta : undefined;
        if (index === undefined || !delta || typeof delta.type !== "string") {
          return;
        }
        if (delta.type === "text_delta") {
          const block = turnState.responseBlocks.get(index);
          const text = typeof delta.text === "string" ? delta.text : "";
          if (block?.kind === "text") {
            block.text += text;
          }
          const assistantBlock = yield* ensureAssistantTextBlock(context, index);
          if (assistantBlock) {
            assistantBlock.emittedTextDelta = true;
            assistantBlock.fallbackText += text;
          }
          if (text.length === 0) {
            return;
          }
          const stamp = yield* makeEventStamp();
          yield* offerRuntimeEvent({
            type: "content.delta",
            eventId: stamp.eventId,
            provider: PROVIDER,
            createdAt: stamp.createdAt,
            threadId: context.session.threadId,
            turnId: turnState.turnId,
            ...(assistantBlock ? { itemId: asRuntimeItemId(assistantBlock.itemId) } : {}),
            payload: {
              streamKind: "assistant_text",
              delta: text,
            },
            providerRefs: {},
            raw: {
              source: "antigravity.anthropic.stream",
              method: "antigravity/content_block_delta/text_delta",
              payload: event.data,
            },
          });
          return;
        }

        if (delta.type === "thinking_delta") {
          const block = turnState.responseBlocks.get(index);
          const thinking =
            typeof delta.thinking === "string"
              ? delta.thinking
              : typeof delta.text === "string"
                ? delta.text
                : "";
          if (block?.kind === "thinking") {
            block.thinking += thinking;
            yield* emitReasoningProgress(context, turnState, block);
          }
          if (thinking.length === 0) {
            return;
          }
          const stamp = yield* makeEventStamp();
          yield* offerRuntimeEvent({
            type: "content.delta",
            eventId: stamp.eventId,
            provider: PROVIDER,
            createdAt: stamp.createdAt,
            threadId: context.session.threadId,
            turnId: turnState.turnId,
            payload: {
              streamKind: "reasoning_text",
              delta: thinking,
            },
            providerRefs: {},
            raw: {
              source: "antigravity.anthropic.stream",
              method: "antigravity/content_block_delta/thinking_delta",
              payload: event.data,
            },
          });
          return;
        }

        if (delta.type === "signature_delta") {
          const block = turnState.responseBlocks.get(index);
          if (block?.kind === "thinking" && typeof delta.signature === "string") {
            block.signature = delta.signature;
          }
          return;
        }
        if (delta.type === "input_json_delta") {
          const block = turnState.responseBlocks.get(index);
          if (block?.kind === "tool_alias") {
            const partialJson = typeof delta.partial_json === "string" ? delta.partial_json : "";
            block.partialInputJson += partialJson;
            const parsed = safeParseJsonRecord(block.partialInputJson);
            if (parsed) {
              yield* syncToolInput(context, turnState, block.target, parsed, {
                method: "antigravity/content_block_delta/input_json_delta",
                payload: event.data,
              });
            }
            return;
          }
          if (
            !block ||
            (block.kind !== "tool_use" &&
              block.kind !== "server_tool_use" &&
              block.kind !== "mcp_tool_use")
          ) {
            return;
          }
          const partialJson = typeof delta.partial_json === "string" ? delta.partial_json : "";
          block.partialInputJson += partialJson;
          const parsed = safeParseJsonRecord(block.partialInputJson);
          if (parsed) {
            yield* syncToolInput(context, turnState, block, parsed, {
              method: "antigravity/content_block_delta/input_json_delta",
              payload: event.data,
            });
          }
          return;
        }
        return;
      }

      case "content_block_stop": {
        const index = typeof data.index === "number" ? data.index : undefined;
        if (index === undefined) {
          return;
        }
        const assistantBlock = turnState.assistantTextBlocks.get(index);
        if (assistantBlock) {
          assistantBlock.streamClosed = true;
          yield* completeAssistantTextBlock(context, assistantBlock);
          return;
        }
        const block = turnState.responseBlocks.get(index);
        if (block?.kind === "thinking") {
          yield* completeReasoningBlock(context, turnState, block, {
            method: "antigravity/content_block_stop",
            payload: event.data,
          });
        }
        return;
      }

      case "message_delta": {
        const delta = isRecord(data.delta) ? data.delta : undefined;
        if (delta && typeof delta.stop_reason === "string") {
          turnState.stopReason = delta.stop_reason;
        } else if (delta && delta.stop_reason === null) {
          turnState.stopReason = null;
        }
        if (delta && isRecord(delta.container) && typeof delta.container.id === "string") {
          yield* updateContainerId(context, delta.container.id);
        }
        const usage = isRecord(data.usage) ? data.usage : undefined;
        if (usage) {
          const nextUsage = {
            ...(typeof usage.input_tokens === "number" ? { input_tokens: usage.input_tokens } : {}),
            ...(typeof usage.output_tokens === "number"
              ? { output_tokens: usage.output_tokens }
              : {}),
          };
          turnState.usage = Object.keys(nextUsage).length > 0 ? nextUsage : undefined;
          if (turnState.usage) {
            yield* emitTokenUsage(context, turnState.usage);
          }
        }
        return;
      }

      case "message_stop": {
        yield* finalizeAssistantMessage(context);
        const stopDetails =
          isRecord(turnState.stopReason) || !isRecord(data) ? undefined : undefined;
        void stopDetails;
        return;
      }

      case "error": {
        const error = isRecord(data.error) ? data.error : undefined;
        const message =
          typeof error?.message === "string" ? error.message : "Antigravity request failed.";
        yield* failActiveTurn(context, "failed", message);
        return;
      }

      default:
        return;
    }
  });

  const handleResponseEvent = Effect.fn("handleResponseEvent")(function* (
    context: AntigravitySessionContext,
    event: {
      readonly event: string;
      readonly data: unknown;
    },
  ) {
    const replayState = context.turnState?.replayState;
    if (replayState?.active && event.event !== "message_start" && event.event !== "message_delta") {
      const data = isRecord(event.data) ? event.data : {};

      if (event.event === "content_block_start") {
        const index = typeof data.index === "number" ? data.index : undefined;
        const contentBlock = isRecord(data.content_block) ? data.content_block : undefined;
        const expected = replayState.expectedBlocks.at(replayState.matchedCount);
        if (index === undefined || !contentBlock || !expected) {
          replayState.active = false;
        } else {
          const probe = buildReplayProbe(index, contentBlock, event);
          if (!probe || !isReplayProbeCompatible(expected, probe)) {
            replayState.active = false;
          } else {
            replayState.pendingProbes.set(index, probe);
            if (isReplayProbeFullyMatched(expected, probe)) {
              replayState.pendingProbes.delete(index);
              replayState.skippedBlockIndices.add(index);
              replayState.matchedCount += 1;
            }
            return;
          }
        }
      }

      if (event.event === "content_block_delta") {
        const index = typeof data.index === "number" ? data.index : undefined;
        const delta = isRecord(data.delta) ? data.delta : undefined;
        const probe = index === undefined ? undefined : replayState.pendingProbes.get(index);
        if (probe && delta && typeof delta.type === "string") {
          const blockIndex = index!;
          probe.bufferedEvents.push(event);
          if (delta.type === "text_delta" && typeof delta.text === "string") {
            probe.text += delta.text;
          } else if (delta.type === "thinking_delta") {
            const thinking =
              typeof delta.thinking === "string"
                ? delta.thinking
                : typeof delta.text === "string"
                  ? delta.text
                  : "";
            probe.thinking += thinking;
          } else if (delta.type === "signature_delta" && typeof delta.signature === "string") {
            probe.signature = delta.signature;
          } else if (delta.type === "input_json_delta") {
            const partialJson = typeof delta.partial_json === "string" ? delta.partial_json : "";
            probe.partialInputJson += partialJson;
            const parsed = safeParseJsonRecord(probe.partialInputJson);
            if (parsed) {
              probe.input = parsed;
            }
          }

          const expected = replayState.expectedBlocks.at(replayState.matchedCount);
          if (!expected || !doesReplayProbeStillMatchExpected(expected, probe)) {
            replayState.pendingProbes.delete(blockIndex);
            replayState.active = false;
            yield* replayBufferedEvents(context, probe.bufferedEvents);
            return;
          }
          if (isReplayProbeFullyMatched(expected, probe)) {
            replayState.pendingProbes.delete(blockIndex);
            replayState.skippedBlockIndices.add(blockIndex);
            replayState.matchedCount += 1;
          }
          return;
        }
        if (index !== undefined && replayState.skippedBlockIndices.has(index)) {
          return;
        }
      }

      if (event.event === "content_block_stop") {
        const index = typeof data.index === "number" ? data.index : undefined;
        if (index !== undefined && replayState.skippedBlockIndices.delete(index)) {
          return;
        }
        const probe = index === undefined ? undefined : replayState.pendingProbes.get(index);
        if (probe) {
          const blockIndex = index!;
          replayState.pendingProbes.delete(blockIndex);
          probe.bufferedEvents.push(event);
          const expected = replayState.expectedBlocks.at(replayState.matchedCount);
          if (expected && isReplayProbeFullyMatched(expected, probe)) {
            replayState.matchedCount += 1;
            return;
          }
          replayState.active = false;
          yield* replayBufferedEvents(context, probe.bufferedEvents);
          return;
        }
      }

      if (event.event === "message_stop" && replayState.pendingProbes.size > 0) {
        const bufferedEvents = [...replayState.pendingProbes.entries()]
          .toSorted(([left], [right]) => left - right)
          .flatMap(([, probe]) => probe.bufferedEvents);
        replayState.pendingProbes.clear();
        replayState.active = false;
        yield* replayBufferedEvents(context, bufferedEvents);
      }
    }

    yield* processResponseEventDirect(context, event);
  });

  const runRequest = Effect.fn("runRequest")(function* (
    context: AntigravitySessionContext,
    requestMessages: ReadonlyArray<AntigravityMessageInput>,
  ) {
    const currentTurn = context.turnState;
    if (!currentTurn) {
      return;
    }

    const model = context.session.model ?? DEFAULT_MODEL_BY_PROVIDER.antigravity;
    const useContainer = context.containerId !== undefined;
    const replayExpectedBlocks = useContainer ? buildReplayExpectedBlocks(context) : [];
    currentTurn.replayState =
      replayExpectedBlocks.length > 0
        ? {
            expectedBlocks: replayExpectedBlocks,
            pendingProbes: new Map(),
            skippedBlockIndices: new Set(),
            matchedCount: 0,
            active: true,
          }
        : undefined;
    const body = {
      model,
      max_tokens: ANTIGRAVITY_DEFAULT_MAX_TOKENS,
      messages: useContainer ? requestMessages : context.messages,
      ...(useContainer && context.containerId ? { container: context.containerId } : {}),
    };

    const controller = new AbortController();
    context.requestAbortController = controller;
    const client = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY ?? "local-test-key",
      baseURL: context.baseUrl,
      defaultHeaders: context.session.cwd ? { "x-cwd": context.session.cwd } : undefined,
      fetch: fetchImpl,
      maxRetries: 0,
    });

    const pendingPauseDetails = yield* Effect.tryPromise({
      try: async () => {
        const stream = client.messages.stream({ ...body, stream: true } as any, {
          signal: controller.signal,
        });
        let nextPendingPauseDetails: Record<string, unknown> | undefined;
        for await (const rawEvent of stream as AsyncIterable<Record<string, unknown>>) {
          const event = normalizeAnthropicStreamEvent(rawEvent);
          if (!event) {
            continue;
          }
          await runPromiseWith(handleResponseEvent(context, event));
          if (event.event === "message_delta" && isRecord(event.data)) {
            const delta = isRecord(event.data.delta) ? event.data.delta : undefined;
            if (delta && delta.stop_reason === "pause_turn") {
              nextPendingPauseDetails = isRecord(delta.stop_details)
                ? delta.stop_details
                : undefined;
            }
          }
        }
        return nextPendingPauseDetails;
      },
      catch: (cause) =>
        new ProviderAdapterProcessError({
          provider: PROVIDER,
          threadId: context.session.threadId,
          detail: toMessage(cause, "Failed to reach Antigravity proxy."),
          cause,
        }),
    });

    clearRequestFiber(context);

    if (!context.turnState) {
      return;
    }

    if (context.turnState.stopReason === "pause_turn") {
      if (pendingPauseDetails) {
        yield* handlePauseTurn(context, pendingPauseDetails);
        return;
      }
      return yield* failActiveTurn(
        context,
        "failed",
        "Antigravity paused the turn without interaction details.",
      );
    }

    yield* completeTurn(context, "completed", {
      stopReason: context.turnState.stopReason ?? null,
    });
  });

  const launchRequest = Effect.fn("launchRequest")(function* (
    context: AntigravitySessionContext,
    requestMessages: ReadonlyArray<AntigravityMessageInput>,
  ) {
    const fiber = yield* runRequest(context, requestMessages).pipe(
      Effect.catchCause((cause) => {
        clearRequestFiber(context);
        const message = Cause.pretty(cause);
        return failActiveTurn(
          context,
          message.toLowerCase().includes("abort") ? "interrupted" : "failed",
          message,
        );
      }),
      Effect.asVoid,
      Effect.forkIn(adapterScope),
    );
    context.requestFiber = fiber;
  });

  const requireSession = (
    threadId: ThreadId,
  ): Effect.Effect<AntigravitySessionContext, ProviderAdapterError> => {
    const context = sessions.get(threadId);
    if (!context) {
      return Effect.fail(
        new ProviderAdapterSessionNotFoundError({
          provider: PROVIDER,
          threadId,
        }),
      );
    }
    if (context.stopped || context.session.status === "closed") {
      return Effect.fail(
        new ProviderAdapterSessionClosedError({
          provider: PROVIDER,
          threadId,
        }),
      );
    }
    return Effect.succeed(context);
  };

  const startSession: AntigravityAdapterShape["startSession"] = Effect.fn("startSession")(
    function* (input) {
      console.log("startSessssion", input);

      if (input.provider !== undefined && input.provider !== PROVIDER) {
        return yield* new ProviderAdapterValidationError({
          provider: PROVIDER,
          operation: "startSession",
          issue: `Expected provider '${PROVIDER}' but received '${input.provider}'.`,
        });
      }
      console.log("startSessssion", input);
      const antigravitySettings = yield* serverSettings.getSettings.pipe(
        Effect.map((settings) => settings.providers.antigravity),
        Effect.mapError(
          (error) =>
            new ProviderAdapterProcessError({
              provider: PROVIDER,
              threadId: input.threadId,
              detail: error.message,
              cause: error,
            }),
        ),
      );
      const startedAt = yield* nowIso;
      const modelSelection =
        input.modelSelection?.provider === PROVIDER ? input.modelSelection : undefined;
      const resumeCursor = readResumeCursor(input.resumeCursor);

      const session: ProviderSession = {
        provider: PROVIDER,
        status: "ready",
        runtimeMode: input.runtimeMode,
        ...(input.cwd ? { cwd: input.cwd } : {}),
        model: resolveModelSlugForProvider(
          PROVIDER,
          modelSelection?.model ?? DEFAULT_MODEL_BY_PROVIDER.antigravity,
        ),
        threadId: input.threadId,
        ...(resumeCursor?.containerId
          ? {
              resumeCursor: {
                containerId: resumeCursor.containerId,
                turnCount: resumeCursor.turnCount ?? 0,
              },
            }
          : {}),
        createdAt: startedAt,
        updatedAt: startedAt,
      };

      const context: AntigravitySessionContext = {
        session,
        baseUrl: antigravitySettings.baseUrl.trim() || ANTIGRAVITY_DEFAULT_BASE_URL,
        messages: [],
        turns: [],
        pendingApprovals: new Map(),
        pendingUserInputs: new Map(),
        turnState: undefined,
        requestFiber: undefined,
        requestAbortController: undefined,
        containerId: resumeCursor?.containerId,
        stopped: false,
      };
      sessions.set(input.threadId, context);

      const startedStamp = yield* makeEventStamp();
      yield* offerRuntimeEvent({
        type: "session.started",
        eventId: startedStamp.eventId,
        provider: PROVIDER,
        createdAt: startedStamp.createdAt,
        threadId: input.threadId,
        payload: input.resumeCursor !== undefined ? { resume: input.resumeCursor } : {},
        providerRefs: {},
      });

      const configuredStamp = yield* makeEventStamp();
      yield* offerRuntimeEvent({
        type: "session.configured",
        eventId: configuredStamp.eventId,
        provider: PROVIDER,
        createdAt: configuredStamp.createdAt,
        threadId: input.threadId,
        payload: {
          config: {
            baseUrl: antigravitySettings.baseUrl,
            model: session.model,
            ...(input.cwd ? { cwd: input.cwd } : {}),
          },
        },
        providerRefs: {},
      });

      yield* setSessionState(context, "ready");
      return { ...session };
    },
  );

  const sendTurn: AntigravityAdapterShape["sendTurn"] = Effect.fn("sendTurn")(function* (input) {
    console.log("sendTurn", input);
    const context = yield* requireSession(input.threadId);
    if (context.turnState) {
      return yield* new ProviderAdapterRequestError({
        provider: PROVIDER,
        method: "turn/start",
        detail: "Antigravity already has an active turn for this thread.",
      });
    }

    const modelSelection =
      input.modelSelection?.provider === PROVIDER ? input.modelSelection : undefined;
    const model = resolveModelSlugForProvider(
      PROVIDER,
      modelSelection?.model ?? context.session.model ?? DEFAULT_MODEL_BY_PROVIDER.antigravity,
    );

    const userMessage = yield* buildUserMessageEffect(input, {
      fileSystem,
      attachmentsDir: serverConfig.attachmentsDir,
    });
    const turnId = TurnId.make(yield* Random.nextUUIDv4);
    const startedAt = yield* nowIso;
    context.turnState = {
      turnId,
      startedAt,
      items: [],
      turnMessages: [],
      historyCheckpoint: context.messages.length,
      assistantTextBlocks: new Map(),
      assistantTextBlockOrder: [],
      responseBlocks: new Map(),
      inFlightTools: new Map(),
      stopReason: null,
      usage: undefined,
      replayState: undefined,
    };
    context.session = {
      ...context.session,
      status: "running",
      model,
      activeTurnId: turnId,
      updatedAt: startedAt,
    };

    appendTurnMessage(context, userMessage);

    const turnStartedStamp = yield* makeEventStamp();
    yield* offerRuntimeEvent({
      type: "turn.started",
      eventId: turnStartedStamp.eventId,
      provider: PROVIDER,
      createdAt: turnStartedStamp.createdAt,
      threadId: context.session.threadId,
      turnId,
      payload: {
        model,
      },
      providerRefs: {},
    });
    yield* setSessionState(context, "running");
    yield* launchRequest(context, [userMessage]);

    return {
      threadId: context.session.threadId,
      turnId,
      ...(context.session.resumeCursor !== undefined
        ? { resumeCursor: context.session.resumeCursor }
        : {}),
    };
  });

  const interruptTurn: AntigravityAdapterShape["interruptTurn"] = Effect.fn("interruptTurn")(
    function* (threadId) {
      const context = yield* requireSession(threadId);
      context.requestAbortController?.abort();
      clearPendingInteractions(context);
      yield* failActiveTurn(context, "interrupted", "Antigravity turn interrupted.");
    },
  );

  const buildToolResultMessage = (
    pending: PendingApproval | PendingUserInput,
    options: {
      readonly content?: string;
      readonly isError: boolean;
    },
  ): AntigravityMessageInput => ({
    role: "user",
    content: [
      {
        type: buildToolResultKind(pending.tool),
        tool_use_id: pending.tool.toolUseId,
        ...(options.content !== undefined ? { content: options.content } : {}),
        ...(options.isError ? { is_error: true } : {}),
      },
    ],
  });

  const respondToRequest: AntigravityAdapterShape["respondToRequest"] = Effect.fn(
    "respondToRequest",
  )(function* (threadId, requestId, decision) {
    const context = yield* requireSession(threadId);
    const pending = context.pendingApprovals.get(requestId);
    if (!pending) {
      return yield* new ProviderAdapterRequestError({
        provider: PROVIDER,
        method: "request/respond",
        detail: `Unknown pending approval request: ${requestId}`,
      });
    }
    context.pendingApprovals.delete(requestId);

    const resolvedStamp = yield* makeEventStamp();
    yield* offerRuntimeEvent({
      type: "request.resolved",
      eventId: resolvedStamp.eventId,
      provider: PROVIDER,
      createdAt: resolvedStamp.createdAt,
      threadId: context.session.threadId,
      ...(context.turnState ? { turnId: context.turnState.turnId } : {}),
      requestId: asRuntimeRequestId(requestId),
      payload: {
        requestType: pending.requestType,
        decision,
      },
      providerRefs: {
        providerItemId: ProviderItemId.make(pending.tool.toolUseId),
      },
    });

    const itemStamp = yield* makeEventStamp();
    yield* offerRuntimeEvent({
      type: "item.completed",
      eventId: itemStamp.eventId,
      provider: PROVIDER,
      createdAt: itemStamp.createdAt,
      threadId: context.session.threadId,
      ...(context.turnState ? { turnId: context.turnState.turnId } : {}),
      itemId: asRuntimeItemId(pending.tool.itemId),
      payload: {
        itemType: pending.tool.itemType,
        status: decision === "accept" || decision === "acceptForSession" ? "completed" : "declined",
        title: pending.tool.title,
        ...(pending.tool.detail ? { detail: pending.tool.detail } : {}),
        data: buildToolLifecycleData(pending.tool),
      },
      providerRefs: {
        providerItemId: ProviderItemId.make(pending.tool.toolUseId),
      },
    });

    context.turnState?.inFlightTools.delete(pending.tool.toolUseId);

    const toolResultMessage = buildToolResultMessage(pending, {
      content: "{}",
      isError: decision !== "accept" && decision !== "acceptForSession",
    });
    appendTurnMessage(context, toolResultMessage);
    yield* setSessionState(context, "running", "interaction-resolved");
    yield* launchRequest(context, [toolResultMessage]);
  });

  const respondToUserInput: AntigravityAdapterShape["respondToUserInput"] = Effect.fn(
    "respondToUserInput",
  )(function* (threadId, requestId, answers) {
    const context = yield* requireSession(threadId);
    const pending = context.pendingUserInputs.get(requestId);
    if (!pending) {
      return yield* new ProviderAdapterRequestError({
        provider: PROVIDER,
        method: "user-input/respond",
        detail: `Unknown pending user-input request: ${requestId}`,
      });
    }
    context.pendingUserInputs.delete(requestId);

    const resolvedStamp = yield* makeEventStamp();
    yield* offerRuntimeEvent({
      type: "user-input.resolved",
      eventId: resolvedStamp.eventId,
      provider: PROVIDER,
      createdAt: resolvedStamp.createdAt,
      threadId: context.session.threadId,
      ...(context.turnState ? { turnId: context.turnState.turnId } : {}),
      requestId: asRuntimeRequestId(requestId),
      payload: {
        answers,
      },
      providerRefs: {
        providerItemId: ProviderItemId.make(pending.tool.toolUseId),
      },
    });

    const itemStamp = yield* makeEventStamp();
    yield* offerRuntimeEvent({
      type: "item.completed",
      eventId: itemStamp.eventId,
      provider: PROVIDER,
      createdAt: itemStamp.createdAt,
      threadId: context.session.threadId,
      ...(context.turnState ? { turnId: context.turnState.turnId } : {}),
      itemId: asRuntimeItemId(pending.tool.itemId),
      payload: {
        itemType: pending.tool.itemType,
        status: "completed",
        title: pending.tool.title,
        ...(pending.tool.detail ? { detail: pending.tool.detail } : {}),
        data: {
          ...buildToolLifecycleData(pending.tool),
          answers,
        },
      },
      providerRefs: {
        providerItemId: ProviderItemId.make(pending.tool.toolUseId),
      },
    });

    context.turnState?.inFlightTools.delete(pending.tool.toolUseId);

    const payload = JSON.stringify(answers);
    const toolResultMessage = buildToolResultMessage(pending, {
      content: payload,
      isError: false,
    });
    appendTurnMessage(context, toolResultMessage);
    yield* setSessionState(context, "running", "user-input-resolved");
    yield* launchRequest(context, [toolResultMessage]);
  });

  const stopSessionInternal = Effect.fn("stopSessionInternal")(function* (
    context: AntigravitySessionContext,
    emitExitEvent: boolean,
  ) {
    if (context.stopped) {
      return;
    }
    context.stopped = true;
    context.requestAbortController?.abort();
    clearPendingInteractions(context);
    if (context.turnState) {
      yield* completeTurn(context, "interrupted", {
        errorMessage: "Session stopped.",
        rollbackHistory: true,
      });
    }
    context.session = {
      ...context.session,
      status: "closed",
      activeTurnId: undefined,
      updatedAt: yield* nowIso,
    };
    if (emitExitEvent) {
      const stamp = yield* makeEventStamp();
      yield* offerRuntimeEvent({
        type: "session.exited",
        eventId: stamp.eventId,
        provider: PROVIDER,
        createdAt: stamp.createdAt,
        threadId: context.session.threadId,
        payload: {
          reason: "Session stopped",
          exitKind: "graceful",
        },
        providerRefs: {},
      });
    }
    sessions.delete(context.session.threadId);
  });

  const stopSession: AntigravityAdapterShape["stopSession"] = Effect.fn("stopSession")(
    function* (threadId) {
      const context = yield* requireSession(threadId);
      yield* stopSessionInternal(context, true);
    },
  );

  const listSessions: AntigravityAdapterShape["listSessions"] = () =>
    Effect.sync(() => Array.from(sessions.values(), ({ session }) => ({ ...session })));

  const hasSession: AntigravityAdapterShape["hasSession"] = (threadId) =>
    Effect.sync(() => {
      const context = sessions.get(threadId);
      return context !== undefined && !context.stopped;
    });

  const readThread: AntigravityAdapterShape["readThread"] = Effect.fn("readThread")(
    function* (threadId) {
      const context = yield* requireSession(threadId);
      return snapshotThread(context);
    },
  );

  const rollbackThread: AntigravityAdapterShape["rollbackThread"] = Effect.fn("rollbackThread")(
    function* (threadId, numTurns) {
      const context = yield* requireSession(threadId);
      if (context.turnState) {
        return yield* new ProviderAdapterRequestError({
          provider: PROVIDER,
          method: "thread/rollback",
          detail: "Cannot roll back an Antigravity thread while a turn is active.",
        });
      }
      const nextLength = Math.max(0, context.turns.length - numTurns);
      context.turns.splice(nextLength);
      context.messages.splice(
        0,
        context.messages.length,
        ...context.turns.flatMap((turn) => turn.messages),
      );
      context.containerId = undefined;
      yield* updateResumeCursor(context);
      return snapshotThread(context);
    },
  );

  const stopAll: AntigravityAdapterShape["stopAll"] = () =>
    Effect.forEach(sessions, ([, context]) => stopSessionInternal(context, false), {
      discard: true,
    });

  yield* Effect.addFinalizer(() =>
    Effect.forEach(sessions, ([, context]) => stopSessionInternal(context, false), {
      discard: true,
    }).pipe(
      Effect.tap(() => Queue.shutdown(runtimeEventQueue)),
      Effect.andThen(Scope.close(adapterScope, Exit.void)),
    ),
  );

  return {
    provider: PROVIDER,
    capabilities: {
      sessionModelSwitch: "in-session",
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
  } satisfies AntigravityAdapterShape;
});

export const AntigravityAdapterLive = Layer.effect(AntigravityAdapter, makeAntigravityAdapter());

export function makeAntigravityAdapterLive(options?: AntigravityAdapterLiveOptions) {
  return Layer.effect(AntigravityAdapter, makeAntigravityAdapter(options));
}
