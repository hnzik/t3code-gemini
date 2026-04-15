import {
  GeminiEventType,
  tokenLimit,
  type GeminiClient,
  type ServerGeminiStreamEvent,
  type ToolCallRequestInfo,
} from "@google/gemini-cli-core";
import { Data, Effect, Stream } from "effect";

import {
  DEFAULT_GEMINI_CONTEXT_WINDOW,
  formatGeminiAgentExecutionMessage,
  formatGeminiChatCompressionMessage,
  formatGeminiContextWindowOverflowMessage,
  formatGeminiFinishReasonMessage,
  formatGeminiThoughtSummary,
  type GeminiUsageCounts,
  type GeminiTurnState,
} from "./GeminiRuntimeHelpers";

export interface GeminiStreamTerminalTurnResult {
  readonly state: "completed" | "failed" | "interrupted" | "cancelled";
  readonly stopReason: string;
  readonly errorMessage?: string;
}

export interface GeminiStreamProcessingResult {
  readonly toolCallRequests: ReadonlyArray<ToolCallRequestInfo>;
  readonly terminalTurnResult?: GeminiStreamTerminalTurnResult;
}

export interface GeminiStreamProcessorInput {
  readonly stream: AsyncIterable<ServerGeminiStreamEvent>;
  readonly geminiClient: GeminiClient;
  readonly sessionModel: string | undefined;
  readonly turnState: GeminiTurnState | undefined;
  readonly onNativeEvent?: (event: ServerGeminiStreamEvent) => void;
  readonly onThought: (text: string) => void;
  readonly onContent: (text: string) => void;
  readonly onModelInfo: (model: string, event: ServerGeminiStreamEvent) => void;
  readonly onRuntimeWarning: (
    message: string,
    detail?: unknown,
    event?: ServerGeminiStreamEvent,
  ) => void;
  readonly onUsage: (usage: GeminiUsageCounts) => void;
}

export class GeminiStreamProcessorError extends Data.TaggedError("GeminiStreamProcessorError")<{
  readonly message: string;
  readonly cause?: unknown;
}> {}

function toGeminiStreamError(cause: unknown): GeminiStreamProcessorError {
  if (cause instanceof Error && cause.message.length > 0) {
    return new GeminiStreamProcessorError({
      message: cause.message,
      cause,
    });
  }
  if (typeof cause === "string" && cause.length > 0) {
    return new GeminiStreamProcessorError({
      message: cause,
      cause,
    });
  }
  return new GeminiStreamProcessorError({
    message: "Gemini stream failed.",
    cause,
  });
}

export function processGeminiStreamEvents(
  input: GeminiStreamProcessorInput,
): Effect.Effect<GeminiStreamProcessingResult, GeminiStreamProcessorError> {
  const toolCallRequests: ToolCallRequestInfo[] = [];
  let terminalTurnResult: GeminiStreamTerminalTurnResult | undefined;

  const processEvent = (event: ServerGeminiStreamEvent) =>
    Effect.try({
      try: () => {
        input.onNativeEvent?.(event);

        switch (event.type) {
          case GeminiEventType.Thought: {
            const thoughtText = formatGeminiThoughtSummary(event.value);
            if (thoughtText.length > 0) {
              input.onThought(thoughtText);
            }
            break;
          }

          case GeminiEventType.Content:
            input.onContent(event.value);
            break;

          case GeminiEventType.ToolCallRequest:
            toolCallRequests.push(event.value);
            break;

          case GeminiEventType.ToolCallConfirmation:
          case GeminiEventType.ToolCallResponse:
          case GeminiEventType.Retry:
            break;

          case GeminiEventType.Citation:
            input.onRuntimeWarning(event.value, { geminiEventType: event.type }, event);
            break;

          case GeminiEventType.ChatCompressed: {
            if (event.value) {
              const limit =
                tokenLimit(input.turnState?.activeModel ?? input.sessionModel ?? "") ||
                DEFAULT_GEMINI_CONTEXT_WINDOW;
              input.onRuntimeWarning(
                formatGeminiChatCompressionMessage({
                  originalTokenCount: event.value.originalTokenCount,
                  newTokenCount: event.value.newTokenCount,
                  limit,
                }),
                {
                  geminiEventType: event.type,
                  originalTokenCount: event.value.originalTokenCount,
                  newTokenCount: event.value.newTokenCount,
                  limit,
                },
                event,
              );
            }
            break;
          }

          case GeminiEventType.Finished: {
            const usage = event.value.usageMetadata;
            if (usage) {
              const promptTokens = usage.promptTokenCount ?? 0;
              const cachedInputTokens = usage.cachedContentTokenCount ?? 0;
              const toolUsePromptTokens = usage.toolUsePromptTokenCount ?? 0;
              const outputTokens = usage.candidatesTokenCount ?? 0;
              const reasoningOutputTokens = usage.thoughtsTokenCount ?? 0;
              const totalTokens =
                usage.totalTokenCount ??
                promptTokens + toolUsePromptTokens + outputTokens + reasoningOutputTokens;

              if (
                promptTokens > 0 ||
                cachedInputTokens > 0 ||
                toolUsePromptTokens > 0 ||
                outputTokens > 0 ||
                reasoningOutputTokens > 0 ||
                totalTokens > 0
              ) {
                input.onUsage({
                  promptTokens,
                  cachedInputTokens,
                  toolUsePromptTokens,
                  outputTokens,
                  reasoningOutputTokens,
                  totalTokens,
                });
              }
            }

            const finishMessage = formatGeminiFinishReasonMessage(event.value.reason);
            if (finishMessage) {
              input.onRuntimeWarning(
                finishMessage,
                {
                  geminiEventType: event.type,
                  finishReason: event.value.reason,
                  usageMetadata: usage,
                },
                event,
              );
            }
            break;
          }

          case GeminiEventType.ModelInfo:
            input.onModelInfo(event.value, event);
            break;

          case GeminiEventType.ContextWindowWillOverflow: {
            const limit =
              tokenLimit(input.turnState?.activeModel ?? input.sessionModel ?? "") ||
              DEFAULT_GEMINI_CONTEXT_WINDOW;
            input.onRuntimeWarning(
              formatGeminiContextWindowOverflowMessage({
                estimatedRequestTokenCount: event.value.estimatedRequestTokenCount,
                remainingTokenCount: event.value.remainingTokenCount,
                limit,
              }),
              {
                geminiEventType: event.type,
                estimatedRequestTokenCount: event.value.estimatedRequestTokenCount,
                remainingTokenCount: event.value.remainingTokenCount,
                limit,
              },
              event,
            );
            terminalTurnResult = {
              state: "cancelled",
              stopReason: "context_window_will_overflow",
            };
            break;
          }

          case GeminiEventType.MaxSessionTurns:
            input.onRuntimeWarning(
              "The session has reached the maximum number of turns.",
              { geminiEventType: event.type },
              event,
            );
            terminalTurnResult = {
              state: "cancelled",
              stopReason: "max_session_turns",
            };
            break;

          case GeminiEventType.LoopDetected:
            input.onRuntimeWarning(
              "A potential loop was detected. The request has been halted.",
              { geminiEventType: event.type },
              event,
            );
            terminalTurnResult = {
              state: "interrupted",
              stopReason: "loop_detected",
            };
            break;

          case GeminiEventType.AgentExecutionBlocked:
            input.onRuntimeWarning(
              formatGeminiAgentExecutionMessage(
                "blocked",
                event.value.reason,
                event.value.systemMessage,
              ),
              {
                geminiEventType: event.type,
                reason: event.value.reason,
                systemMessage: event.value.systemMessage,
                contextCleared: event.value.contextCleared,
              },
              event,
            );
            if (event.value.contextCleared) {
              input.onRuntimeWarning("Conversation context has been cleared.", undefined, event);
            }
            break;

          case GeminiEventType.AgentExecutionStopped:
            input.onRuntimeWarning(
              formatGeminiAgentExecutionMessage(
                "stopped",
                event.value.reason,
                event.value.systemMessage,
              ),
              {
                geminiEventType: event.type,
                reason: event.value.reason,
                systemMessage: event.value.systemMessage,
                contextCleared: event.value.contextCleared,
              },
              event,
            );
            if (event.value.contextCleared) {
              input.onRuntimeWarning("Conversation context has been cleared.", undefined, event);
            }
            terminalTurnResult = {
              state: "completed",
              stopReason: event.value.reason || "agent_execution_stopped",
            };
            break;

          case GeminiEventType.UserCancelled:
            terminalTurnResult = {
              state: "interrupted",
              stopReason: "aborted",
            };
            break;

          case GeminiEventType.InvalidStream:
            input.onRuntimeWarning(
              "Gemini returned an invalid stream response. The request will stop here.",
              { geminiEventType: event.type },
              event,
            );
            terminalTurnResult = {
              state: "failed",
              stopReason: "invalid_stream",
              errorMessage: "Gemini returned an invalid stream response.",
            };
            break;

          case GeminiEventType.Error:
            throw event.value.error;

          default: {
            const exhaustiveCheck: never = event;
            return exhaustiveCheck;
          }
        }
      },
      catch: toGeminiStreamError,
    });

  return Stream.fromAsyncIterable(input.stream, toGeminiStreamError).pipe(
    Stream.takeUntilEffect((event) =>
      processEvent(event).pipe(Effect.as(terminalTurnResult !== undefined)),
    ),
    Stream.runDrain,
    Effect.flatMap(() =>
      Effect.sync(() => ({
        toolCallRequests,
        ...(terminalTurnResult ? { terminalTurnResult } : {}),
      })),
    ),
  );
}
