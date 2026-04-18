/**
 * GeminiAcpAdapterLive - Gemini CLI core based provider adapter.
 *
 * Keeps the external provider/runtime contract stable while structuring the
 * internals like Gemini CLI's runtime: adapter wiring, per-session runtime
 * coordinator, central stream processor, and scheduler-backed tool bridge.
 *
 * @module GeminiAcpAdapterLive
 */
import type { ProviderRuntimeEvent, ProviderSession, ThreadId } from "@t3tools/contracts";
import { EventId } from "@t3tools/contracts";
import type { Config } from "@google/gemini-cli-core";
import { Effect, Exit, Fiber, Layer, Queue, Scope, Stream } from "effect";
import {
  installGeminiCliCustomHeaders,
  resolveGeminiApprovalMode,
  resolveGeminiAuthType,
  createGeminiCoreConfig,
  DEFAULT_GEMINI_MODEL,
} from "./GeminiCoreConfig.ts";
import {
  buildGeminiManualAuthRequiredMessage,
  resolveGeminiAuthProbeResult,
} from "./GeminiAcpProvider.ts";
import {
  ProviderAdapterProcessError,
  ProviderAdapterSessionClosedError,
  ProviderAdapterSessionNotFoundError,
  ProviderAdapterValidationError,
  type ProviderAdapterError,
} from "../Errors.ts";
import { GeminiAcpAdapter, type GeminiAcpAdapterShape } from "../Services/GeminiAcpAdapter.ts";
import { GeminiAuthRuntimeState } from "../Services/GeminiAuthRuntimeState.ts";
import {
  ProviderSessionDirectory,
  type ProviderRuntimeBinding,
} from "../Services/ProviderSessionDirectory.ts";
import { ServerSettingsService } from "../../serverSettings.ts";
import { makeEventNdjsonLogger, type EventNdjsonLogger } from "./EventNdjsonLogger.ts";
import { GeminiRuntimeSession } from "./GeminiRuntimeSession.ts";
import {
  GEMINI_PROVIDER,
  buildGeminiPersistedBinding,
  readGeminiResumeState,
} from "./GeminiRuntimeHelpers.ts";

export {
  applyGeminiAssistantTextChunk,
  buildGeminiAssistantHistoryEntry,
  buildGeminiAskUserResponseAnswers,
  buildGeminiPersistedBinding,
  classifyRequestTypeForTool,
  formatGeminiChatCompressionMessage,
  formatGeminiContextWindowOverflowMessage,
  formatGeminiFinishReasonMessage,
  formatGeminiRetryWarningMessage,
  formatGeminiSubagentActivityDetail,
  formatGeminiThoughtSummary,
  inferGeminiTurnHistoryLengths,
  normalizeGeminiAskUserQuestions,
  readGeminiPlanMarkdownFromFile,
  readGeminiResumeState,
} from "./GeminiRuntimeHelpers.ts";

export interface GeminiAcpAdapterLiveOptions {
  readonly createConfig?: (input: {
    readonly sessionId: string;
    readonly cwd: string;
    readonly model?: string | null | undefined;
    readonly interactionMode?: string | null | undefined;
    readonly runtimeMode?: string | null | undefined;
    readonly interactive?: boolean | undefined;
  }) => Promise<Config>;
  readonly nativeEventLogPath?: string;
  readonly nativeEventLogger?: EventNdjsonLogger;
}

function toMessage(cause: unknown, fallback: string): string {
  if (cause instanceof Error && cause.message.length > 0) {
    return cause.message;
  }
  if (typeof cause === "string" && cause.length > 0) {
    return cause;
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
    provider: GEMINI_PROVIDER,
    threadId,
    detail: `${operation}: ${toMessage(cause, "unknown error")}`,
    cause,
  });
}

const makeGeminiAcpAdapter = Effect.fn("makeGeminiAcpAdapter")(function* (
  options?: GeminiAcpAdapterLiveOptions,
) {
  const settingsService = yield* ServerSettingsService;
  const authRuntimeState = yield* GeminiAuthRuntimeState;
  const sessionDirectory = yield* ProviderSessionDirectory;
  const services = yield* Effect.context<never>();
  const runFork = Effect.runForkWith(services);
  const runPromise = Effect.runPromiseWith(services);
  const runSync = Effect.runSyncWith(services);
  const runtimeEventQueue = yield* Queue.unbounded<ProviderRuntimeEvent>();
  const nativeEventLogger =
    options?.nativeEventLogger ??
    (options?.nativeEventLogPath !== undefined
      ? yield* makeEventNdjsonLogger(options.nativeEventLogPath, {
          stream: "native",
        })
      : undefined);

  const emit = (event: ProviderRuntimeEvent): void => {
    runSync(Queue.offer(runtimeEventQueue, event));
  };

  const persistBinding = (binding: ProviderRuntimeBinding): void => {
    runFork(
      sessionDirectory.upsert(binding).pipe(
        Effect.catchCause((cause) =>
          Effect.logWarning("failed to persist Gemini session binding", {
            threadId: binding.threadId,
            cause,
          }),
        ),
      ),
    );
  };

  const writeNativeRecord = (record: unknown, threadId: ThreadId): void => {
    if (!nativeEventLogger) {
      return;
    }

    runFork(
      nativeEventLogger.write(record, threadId).pipe(
        Effect.catchCause((cause) =>
          Effect.logWarning("failed to write Gemini native event log", {
            threadId,
            cause,
          }),
        ),
      ),
    );
  };

  const createConfig =
    options?.createConfig ??
    ((input: {
      readonly sessionId: string;
      readonly cwd: string;
      readonly model?: string | null | undefined;
      readonly interactionMode?: string | null | undefined;
      readonly runtimeMode?: string | null | undefined;
      readonly interactive?: boolean | undefined;
    }) =>
      createGeminiCoreConfig({
        sessionId: input.sessionId,
        cwd: input.cwd,
        model: input.model,
        interactionMode: input.interactionMode,
        runtimeMode: input.runtimeMode,
        interactive: input.interactive,
      }));

  const sessions = new Map<string, GeminiRuntimeSession>();

  const getSession = (
    threadId: ThreadId,
  ): Effect.Effect<GeminiRuntimeSession, ProviderAdapterError> => {
    const session = sessions.get(threadId);
    if (!session) {
      return Effect.fail(
        new ProviderAdapterSessionNotFoundError({
          provider: GEMINI_PROVIDER,
          threadId,
        }),
      );
    }
    if (session.stopped) {
      return Effect.fail(
        new ProviderAdapterSessionClosedError({
          provider: GEMINI_PROVIDER,
          threadId,
        }),
      );
    }
    return Effect.succeed(session);
  };

  const startSession: GeminiAcpAdapterShape["startSession"] = Effect.fn("startSession")(
    function* (input) {
      if (input.provider && input.provider !== GEMINI_PROVIDER) {
        return yield* new ProviderAdapterValidationError({
          provider: GEMINI_PROVIDER,
          operation: "startSession",
          issue: `Expected provider "${GEMINI_PROVIDER}", got "${input.provider}"`,
        });
      }

      yield* settingsService.getSettings.pipe(
        Effect.map((settings) => settings.providers.geminiAcp),
        Effect.orDie,
      );

      const threadId = input.threadId;
      const modelId = input.modelSelection?.model ?? DEFAULT_GEMINI_MODEL;
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
            provider: GEMINI_PROVIDER,
            threadId,
            detail: manualAuthFailure.message,
          });
        }
      }

      const workDir = (input.cwd as string | undefined) ?? process.cwd();
      const config = yield* Effect.tryPromise({
        try: () =>
          createConfig({
            sessionId: threadId as string,
            cwd: workDir,
            model: modelId,
            runtimeMode,
            interactive: true,
          }),
        catch: (cause) =>
          new ProviderAdapterProcessError({
            provider: GEMINI_PROVIDER,
            threadId,
            detail: `Failed to create Gemini config: ${toMessage(cause, "unknown error")}`,
            cause,
          }),
      });

      yield* Effect.tryPromise({
        try: () => config.refreshAuth(authType, undefined, undefined, geminiAuthHeaders),
        catch: (cause) =>
          new ProviderAdapterProcessError({
            provider: GEMINI_PROVIDER,
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
            provider: GEMINI_PROVIDER,
            threadId,
            detail: `Failed to initialize Gemini config: ${toMessage(cause, "unknown error")}`,
            cause,
          }),
      });

      const geminiClient = config.getGeminiClient();
      yield* Effect.tryPromise({
        try: () =>
          resumeState ? geminiClient.resumeChat([...resumeState.history]) : Promise.resolve(),
        catch: (cause) =>
          new ProviderAdapterProcessError({
            provider: GEMINI_PROVIDER,
            threadId,
            detail: `Failed to resume Gemini chat: ${toMessage(cause, "unknown error")}`,
            cause,
          }),
      });

      const session: ProviderSession = {
        provider: GEMINI_PROVIDER,
        status: "ready",
        runtimeMode,
        cwd: input.cwd,
        model: modelId,
        threadId,
        createdAt: now,
        updatedAt: now,
      };

      const turnScope = yield* Scope.make("sequential");
      const runtimeSession = new GeminiRuntimeSession({
        session,
        config,
        geminiClient,
        resumeState,
        getPreferredEditor: () => undefined,
        emitEvent: (event) => emit(event),
        persistBinding,
        writeNativeRecord: (record, nativeThreadId) => writeNativeRecord(record, nativeThreadId),
        getActiveSessions: () => Array.from(sessions.values()),
        turnRuntime: {
          fork: (effect) =>
            runSync(
              effect.pipe(
                Effect.forkScoped({ startImmediately: true }),
                Effect.provideService(Scope.Scope, turnScope),
              ),
            ),
          await: (fiber) => runPromise(Fiber.await(fiber).pipe(Effect.asVoid)),
          close: () => runPromise(Scope.close(turnScope, Exit.void)),
        },
      });
      sessions.set(threadId, runtimeSession);

      emit({
        eventId: EventId.makeUnsafe(`evt-gemini-${Date.now()}-session-started`),
        provider: GEMINI_PROVIDER,
        threadId,
        createdAt: now,
        providerRefs: {},
        type: "session.started",
        payload: {},
      });
      emit({
        eventId: EventId.makeUnsafe(`evt-gemini-${Date.now()}-session-configured`),
        provider: GEMINI_PROVIDER,
        threadId,
        createdAt: now,
        providerRefs: {},
        type: "session.configured",
        payload: {
          config: {
            model: modelId,
            approvalMode: resolveGeminiApprovalMode({
              runtimeMode,
            }),
          },
        },
      });
      emit({
        eventId: EventId.makeUnsafe(`evt-gemini-${Date.now()}-session-ready`),
        provider: GEMINI_PROVIDER,
        threadId,
        createdAt: now,
        providerRefs: {},
        type: "session.state.changed",
        payload: { state: "ready" },
      });

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
    provider: GEMINI_PROVIDER,
    capabilities: {
      sessionModelSwitch: "unsupported",
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

export const makeGeminiAcpAdapterLive = (options?: GeminiAcpAdapterLiveOptions) =>
  Layer.effect(GeminiAcpAdapter, makeGeminiAcpAdapter(options));

export const GeminiAcpAdapterLive = makeGeminiAcpAdapterLive();
