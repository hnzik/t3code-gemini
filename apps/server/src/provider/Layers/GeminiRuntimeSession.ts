import { randomUUID } from "node:crypto";

import type {
  ProviderApprovalDecision,
  ProviderRuntimeEvent,
  ProviderSendTurnInput,
  ProviderSession,
  ProviderTurnStartResult,
  ProviderUserInputAnswers,
  ThreadId,
  TurnId,
} from "@t3tools/contracts";
import {
  EventId,
  RuntimeItemId,
  RuntimeRequestId,
  TurnId as TurnIdBrand,
} from "@t3tools/contracts";
import { type PartListUnion } from "@google/genai";
import {
  CoreEvent,
  MessageBusType,
  type MessageBus,
  ToolConfirmationOutcome,
  coreEvents,
  getCoreSystemPrompt,
  getDisplayString,
  type Config,
  type EditorType,
  type GeminiClient,
  type RetryAttemptPayload,
  type SerializableConfirmationDetails,
  type ServerGeminiStreamEvent,
  type ToolConfirmationPayload,
  type ToolConfirmationRequest,
} from "@google/gemini-cli-core";
import { Cause, Data, Effect, Exit, type Fiber } from "effect";
import { resolveGeminiApprovalMode } from "./GeminiCoreConfig";
import {
  GeminiStreamProcessorError,
  processGeminiStreamEvents,
  type GeminiStreamTerminalTurnResult,
} from "./GeminiStreamProcessor";
import { GeminiToolSchedulerBridge } from "./GeminiToolSchedulerBridge";
import {
  GEMINI_PROVIDER,
  applyGeminiAssistantTextChunk,
  buildGeminiAskUserResponseAnswers,
  buildGeminiPersistedBinding,
  buildGeminiRawEvent,
  buildTokenUsageSnapshot,
  classifyRequestTypeForTool,
  createGeminiResumeState,
  formatGeminiRetryWarningMessage,
  normalizeGeminiAskUserQuestions,
  patchGeminiMessageBusForScheduler,
  readGeminiResumeState,
  restoreGeminiHistoryFromResumeCursor,
  type GeminiPendingApproval,
  type GeminiPendingUserInput,
  type GeminiResumeState,
  type GeminiTrackedTurn,
  type GeminiTurnState,
  type GeminiUsageCounts,
} from "./GeminiRuntimeHelpers";

interface GeminiRuntimeSessionInput {
  readonly session: ProviderSession;
  readonly config: Config;
  readonly geminiClient: GeminiClient;
  readonly resumeState: GeminiResumeState | undefined;
  readonly getPreferredEditor: () => EditorType | undefined;
  readonly emitEvent: (event: ProviderRuntimeEvent) => void;
  readonly persistBinding: (binding: ReturnType<typeof buildGeminiPersistedBinding>) => void;
  readonly writeNativeRecord?: (record: unknown, threadId: ThreadId) => void;
  readonly getActiveSessions: () => ReadonlyArray<GeminiRuntimeSession>;
  readonly turnRuntime: {
    readonly fork: (effect: Effect.Effect<void, never>) => Fiber.Fiber<void, never>;
    readonly await: (fiber: Fiber.Fiber<void, never>) => Promise<void>;
    readonly close: () => Promise<void>;
  };
}

function toMessage(cause: unknown, fallback: string): string {
  if (cause instanceof Error) {
    return cause.message;
  }
  if (typeof cause === "string") {
    return cause;
  }
  if (typeof cause === "object" && cause !== null && "message" in cause) {
    const message = (cause as { message?: unknown }).message;
    if (typeof message === "string") {
      return message;
    }
  }
  return fallback;
}

function buildPendingId(prefix: "req" | "user-input"): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

class GeminiToolSchedulingError extends Data.TaggedError("GeminiToolSchedulingError")<{
  readonly message: string;
  readonly cause?: unknown;
}> {}

function requestTypeFromConfirmation(
  toolName: string,
  confirmation: Exclude<SerializableConfirmationDetails, { type: "ask_user" }>,
) {
  if (confirmation.type === "edit") {
    return "file_change_approval" as const;
  }
  if (confirmation.type === "exec" || confirmation.type === "sandbox_expansion") {
    return "exec_command_approval" as const;
  }
  return classifyRequestTypeForTool(toolName);
}

function detailFromConfirmation(
  toolName: string,
  confirmation: Exclude<SerializableConfirmationDetails, { type: "ask_user" }>,
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
    default:
      return toolName;
  }
}

export class GeminiRuntimeSession {
  private readonly emitEvent: (event: ProviderRuntimeEvent) => void;
  private readonly pendingApprovals = new Map<string, GeminiPendingApproval>();
  private readonly pendingUserInputs = new Map<string, GeminiPendingUserInput>();
  private readonly turns: Array<GeminiTrackedTurn>;
  private readonly messageBusUnsubscribers: Array<() => void> = [];
  private readonly toolBridge: GeminiToolSchedulerBridge;
  private readonly messageBus: MessageBus;
  private activeTurnFiber: Fiber.Fiber<void, never> | undefined;
  private eventCounter = 0;
  private turnState: GeminiTurnState | undefined = undefined;
  private abortController = new AbortController();
  private userRequestedMode = "default";
  private planModeTextSuppressed = false;
  private assistantMessageSegment = 0;
  private assistantMessageText = "";
  private assistantItemStarted = false;
  private lastKnownUsage: GeminiUsageCounts | undefined;
  private totalProcessedTokens = 0;
  stopped = false;

  constructor(private readonly input: GeminiRuntimeSessionInput) {
    this.emitEvent = input.emitEvent;
    this.messageBus = patchGeminiMessageBusForScheduler(input.config.getMessageBus());
    this.turns = Array.from({ length: input.resumeState?.turnCount ?? 0 }, (_, index) => {
      const historyLength = input.resumeState?.turnHistoryLengths?.[index];
      return {
        id: `restored-turn-${index}`,
        items: [],
        ...(historyLength !== undefined ? { historyLength } : {}),
      };
    });
    this.updateResumeCursor();

    this.toolBridge = new GeminiToolSchedulerBridge({
      config: input.config,
      geminiClient: input.geminiClient,
      getPreferredEditor: input.getPreferredEditor,
      context: {
        config: input.config,
        promptId: input.session.threadId as string,
        toolRegistry: input.config.getToolRegistry(),
        promptRegistry: input.config.getPromptRegistry(),
        resourceRegistry: input.config.getResourceRegistry(),
        messageBus: this.messageBus,
        geminiClient: input.geminiClient,
        sandboxManager: input.config.sandboxManager,
      },
      callbacks: {
        emitEvent: (event) => this.emitEvent(event),
        makeEventBase: () => this.makeEventBase(),
        finalizeAssistantSegment: () => this.finalizeAssistantSegment(),
        onPlanCaptured: (planMarkdown) => this.emitProposedPlanCompleted(planMarkdown),
        onApprovalRequested: (approval) => this.openApprovalRequest(approval),
      },
    });

    this.setupRetryAttemptHandler();
    this.setupMessageBusHandlers();
  }

  get session(): ProviderSession {
    return this.input.session;
  }

  startTurn(input: ProviderSendTurnInput): ProviderTurnStartResult {
    if (this.hasActiveTurnExecution()) {
      throw new Error(`Session "${this.session.threadId}" already has an active turn.`);
    }

    restoreGeminiHistoryFromResumeCursor({
      geminiClient: this.input.geminiClient,
      resumeCursor: this.session.resumeCursor,
    });

    const interactionMode = input.interactionMode ?? this.userRequestedMode;
    this.userRequestedMode = interactionMode;
    this.input.config.setApprovalMode(
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
      activeModel: this.session.model,
      reasoningItemEmitted: false,
      capturedProposedPlanKeys: new Set(),
    };
    this.abortController = new AbortController();
    this.planModeTextSuppressed = false;
    this.assistantMessageSegment = 0;
    this.assistantMessageText = "";
    this.assistantItemStarted = false;

    const mutableSession = this.session as {
      status: ProviderSession["status"];
      activeTurnId?: TurnId;
      updatedAt: string;
    };
    mutableSession.status = "running";
    mutableSession.activeTurnId = TurnIdBrand.makeUnsafe(turnId);
    mutableSession.updatedAt = now;

    this.emitEvent({
      ...this.makeEventBase(),
      type: "turn.started",
      payload: { model: this.session.model },
    } as ProviderRuntimeEvent);
    this.emitEvent({
      ...this.makeEventBase(),
      type: "session.state.changed",
      payload: { state: "running" },
    } as ProviderRuntimeEvent);

    const promptText = input.input?.trim() ?? "";
    const turnFiber = this.input.turnRuntime.fork(
      this.runStreamLoop(promptText).pipe(
        Effect.exit,
        Effect.flatMap((exit) => this.handleTurnExit(exit)),
      ),
    );
    this.activeTurnFiber = turnFiber;
    turnFiber.addObserver(() => {
      if (this.activeTurnFiber === turnFiber) {
        this.activeTurnFiber = undefined;
      }
    });

    return {
      threadId: input.threadId,
      turnId: TurnIdBrand.makeUnsafe(turnId),
      ...(this.session.resumeCursor !== undefined
        ? { resumeCursor: this.session.resumeCursor }
        : {}),
    };
  }

  interrupt(): void {
    this.abortController.abort();
    this.toolBridge.cancelAll();
  }

  resolveApproval(requestId: string, decision: ProviderApprovalDecision): void {
    const pending = this.pendingApprovals.get(requestId);
    if (!pending) {
      throw new Error(`No pending approval for requestId "${requestId}"`);
    }

    this.pendingApprovals.delete(requestId);
    const confirmed = decision === "accept" || decision === "acceptForSession";
    const outcome = confirmed
      ? decision === "acceptForSession"
        ? ToolConfirmationOutcome.ProceedAlways
        : ToolConfirmationOutcome.ProceedOnce
      : ToolConfirmationOutcome.Cancel;

    void this.messageBus.publish({
      type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
      correlationId: pending.correlationId,
      confirmed,
      outcome,
    });

    this.emitEvent({
      ...this.makeEventBase(),
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
      throw new Error(`No pending user input for requestId "${requestId}"`);
    }

    this.pendingUserInputs.delete(requestId);

    if (pending.responseChannel === "tool_confirmation") {
      const payload: ToolConfirmationPayload = {
        answers: buildGeminiAskUserResponseAnswers({
          questions: pending.questions,
          answers,
        }),
      };
      void this.messageBus.publish({
        type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
        correlationId: pending.correlationId,
        confirmed: true,
        outcome: ToolConfirmationOutcome.ProceedOnce,
        payload,
      });
    } else {
      void this.messageBus.publish({
        type: MessageBusType.ASK_USER_RESPONSE,
        correlationId: pending.correlationId,
        answers: buildGeminiAskUserResponseAnswers({
          questions: pending.questions,
          answers,
        }),
      } as never);
    }

    this.emitEvent({
      ...this.makeEventBase(),
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
    this.interrupt();

    for (const unsubscribe of this.messageBusUnsubscribers.splice(0)) {
      try {
        unsubscribe();
      } catch {
        // ignore cleanup failures while stopping
      }
    }

    this.cancelPendingInteractions();
    if (this.activeTurnFiber) {
      try {
        await this.input.turnRuntime.await(this.activeTurnFiber);
      } catch {
        // Best-effort shutdown only.
      }
    }
    try {
      await this.input.turnRuntime.close();
    } catch {
      // swallow scoped runtime cleanup failures
    }
    this.toolBridge.dispose();

    try {
      await this.input.config.dispose();
    } catch {
      // swallow shutdown errors
    }

    (this.session as { status: ProviderSession["status"] }).status = "closed";

    this.emitEvent({
      ...this.makeEventBase(),
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

    const history = this.input.geminiClient.getHistory();
    const targetHistoryLength =
      retainedTurns.length === 0 ? 0 : retainedTurns.at(-1)?.historyLength;
    const newHistory =
      targetHistoryLength !== undefined
        ? history.slice(0, Math.min(history.length, targetHistoryLength))
        : history.slice(0, Math.max(0, history.length - numTurns * 2));
    this.input.geminiClient.setHistory([...newHistory]);

    void this.turns.splice(nextTurnCount);
    this.updateResumeCursor();

    return this.readThreadSnapshot();
  }

  private makeEventBase(): Omit<ProviderRuntimeEvent, "type" | "payload"> {
    return {
      eventId: EventId.makeUnsafe(`evt-gemini-${Date.now()}-${++this.eventCounter}`),
      provider: GEMINI_PROVIDER,
      threadId: this.session.threadId,
      createdAt: new Date().toISOString(),
      ...(this.turnState ? { turnId: TurnIdBrand.makeUnsafe(this.turnState.turnId) } : {}),
      providerRefs: {},
    };
  }

  private updateResumeCursor(): void {
    (this.session as { resumeCursor?: unknown }).resumeCursor = createGeminiResumeState({
      geminiClient: this.input.geminiClient,
      turns: this.turns,
    });
  }

  private setupRetryAttemptHandler(): void {
    const onRetryAttempt = (payload: RetryAttemptPayload): void => {
      if (!this.hasActiveTurnExecution() || this.stopped) {
        return;
      }

      const turnState = this.turnState;
      if (!turnState) {
        return;
      }

      const activeModel =
        turnState.activeModel ?? this.input.geminiClient.getCurrentSequenceModel();
      const matchingLabels = new Set<string>();
      if (this.session.model) {
        matchingLabels.add(this.session.model);
        matchingLabels.add(getDisplayString(this.session.model));
      }
      if (activeModel) {
        matchingLabels.add(activeModel);
        matchingLabels.add(getDisplayString(activeModel));
      }
      if (matchingLabels.size > 0 && !matchingLabels.has(payload.model)) {
        return;
      }

      const activeMatchingSessions = this.input.getActiveSessions().filter((candidate) => {
        if (!candidate.hasActiveTurnExecution() || candidate.stopped) {
          return false;
        }

        const candidateTurnState = candidate.turnState;
        if (!candidateTurnState) {
          return false;
        }

        const candidateActiveModel =
          candidateTurnState.activeModel ?? candidate.input.geminiClient.getCurrentSequenceModel();
        const candidateLabels = new Set<string>();
        if (candidate.session.model) {
          candidateLabels.add(candidate.session.model);
          candidateLabels.add(getDisplayString(candidate.session.model));
        }
        if (candidateActiveModel) {
          candidateLabels.add(candidateActiveModel);
          candidateLabels.add(getDisplayString(candidateActiveModel));
        }
        return candidateLabels.has(payload.model);
      });

      if (activeMatchingSessions.length !== 1 || activeMatchingSessions[0] !== this) {
        return;
      }

      this.emitRuntimeWarning(formatGeminiRetryWarningMessage(payload), {
        attempt: payload.attempt,
        maxAttempts: payload.maxAttempts,
        delayMs: payload.delayMs,
        error: payload.error,
        model: payload.model,
      });
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

      if (!request.details) {
        void this.messageBus.publish({
          type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
          correlationId: request.correlationId,
          confirmed: false,
          requiresUserConfirmation: true,
        });
        return;
      }

      this.openApprovalRequest({
        toolName: request.toolCall.name ?? "tool",
        correlationId: request.correlationId,
        details: request.details,
        args: request.toolCall.args ?? {},
      });
    };

    this.messageBus.subscribe(
      MessageBusType.TOOL_CONFIRMATION_REQUEST,
      handleConfirmation as never,
    );
    this.messageBusUnsubscribers.push(() => {
      this.messageBus.unsubscribe(
        MessageBusType.TOOL_CONFIRMATION_REQUEST,
        handleConfirmation as never,
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

    this.messageBus.subscribe(MessageBusType.ASK_USER_REQUEST, handleAskUser as never);
    this.messageBusUnsubscribers.push(() => {
      this.messageBus.unsubscribe(MessageBusType.ASK_USER_REQUEST, handleAskUser as never);
    });
  }

  private openApprovalRequest(input: {
    readonly toolName: string;
    readonly correlationId: string;
    readonly details: SerializableConfirmationDetails;
    readonly args: Record<string, unknown>;
  }): void {
    if (input.details.type === "ask_user") {
      this.openUserInputRequest({
        correlationId: input.correlationId,
        rawQuestions: input.details.questions,
        responseChannel: "tool_confirmation",
      });
      return;
    }

    if (input.details.type === "exit_plan_mode") {
      this.publishDeferredToolConfirmationResponse({
        correlationId: input.correlationId,
        confirmed: true,
        outcome: ToolConfirmationOutcome.ProceedOnce,
      });
      return;
    }

    if (this.userRequestedMode !== "plan" && this.session.runtimeMode === "full-access") {
      this.publishDeferredToolConfirmationResponse({
        correlationId: input.correlationId,
        confirmed: true,
        outcome: ToolConfirmationOutcome.ProceedOnce,
      });
      return;
    }

    const duplicate = [...this.pendingApprovals.values()].some(
      (pending) => pending.correlationId === input.correlationId,
    );
    if (duplicate) {
      return;
    }

    const requestId = buildPendingId("req");
    const requestType = requestTypeFromConfirmation(input.toolName, input.details);
    const detail = detailFromConfirmation(input.toolName, input.details);
    this.pendingApprovals.set(requestId, {
      requestType,
      detail,
      correlationId: input.correlationId,
    });

    this.emitEvent({
      ...this.makeEventBase(),
      requestId: RuntimeRequestId.makeUnsafe(requestId),
      type: "request.opened",
      payload: {
        requestType,
        detail,
        args: input.args,
      },
    } as ProviderRuntimeEvent);
  }

  private publishDeferredToolConfirmationResponse(input: {
    readonly correlationId: string;
    readonly confirmed: boolean;
    readonly outcome: ToolConfirmationOutcome;
  }): void {
    queueMicrotask(() => {
      void this.messageBus.publish({
        type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
        correlationId: input.correlationId,
        confirmed: input.confirmed,
        outcome: input.outcome,
      });
    });
  }

  private openUserInputRequest(input: {
    readonly correlationId: string;
    readonly rawQuestions: unknown;
    readonly responseChannel: "tool_confirmation" | "ask_user";
  }): boolean {
    const questions = normalizeGeminiAskUserQuestions(input.rawQuestions);
    if (questions.length === 0) {
      if (input.responseChannel === "tool_confirmation") {
        void this.messageBus.publish({
          type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
          correlationId: input.correlationId,
          confirmed: false,
          outcome: ToolConfirmationOutcome.Cancel,
        });
      } else {
        void this.messageBus.publish({
          type: MessageBusType.ASK_USER_RESPONSE,
          correlationId: input.correlationId,
          answers: {},
          cancelled: true,
        } as never);
      }
      return false;
    }

    const requestId = `user-input-${input.correlationId}`;
    if (this.pendingUserInputs.has(requestId)) {
      return true;
    }

    this.pendingUserInputs.set(requestId, {
      detail: questions[0]?.question ?? "User input requested",
      correlationId: input.correlationId,
      questions,
      responseChannel: input.responseChannel,
    });

    this.emitEvent({
      ...this.makeEventBase(),
      requestId: RuntimeRequestId.makeUnsafe(requestId),
      type: "user-input.requested",
      payload: {
        questions,
      },
    } as ProviderRuntimeEvent);
    return true;
  }

  private ensureAssistantSegmentStarted(): void {
    if (!this.turnState || this.assistantItemStarted || this.planModeTextSuppressed) {
      return;
    }

    this.assistantItemStarted = true;
    this.emitEvent({
      ...this.makeEventBase(),
      itemId: RuntimeItemId.makeUnsafe(this.currentAssistantItemId()),
      type: "item.started",
      payload: {
        itemType: "assistant_message",
        title: "Assistant message",
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

    this.ensureAssistantSegmentStarted();
    this.emitEvent({
      ...this.makeEventBase(),
      itemId: RuntimeItemId.makeUnsafe(this.currentAssistantItemId()),
      type: "content.delta",
      payload: {
        streamKind: "assistant_text",
        delta,
      },
    } as ProviderRuntimeEvent);
  }

  private finalizeAssistantSegment(): void {
    if (!this.turnState || !this.assistantItemStarted) {
      return;
    }

    this.emitEvent({
      ...this.makeEventBase(),
      itemId: RuntimeItemId.makeUnsafe(this.currentAssistantItemId()),
      type: "item.completed",
      payload: {
        itemType: "assistant_message",
        status: "completed",
        title: "Assistant message",
        ...(this.assistantMessageText.length > 0 ? { detail: this.assistantMessageText } : {}),
      },
    } as ProviderRuntimeEvent);

    this.assistantItemStarted = false;
    this.assistantMessageText = "";
    this.assistantMessageSegment += 1;
  }

  private emitReasoningText(thoughtText: string): void {
    if (!this.turnState) {
      return;
    }

    if (!this.turnState.reasoningItemEmitted) {
      this.turnState.reasoningItemEmitted = true;
      this.emitEvent({
        ...this.makeEventBase(),
        itemId: RuntimeItemId.makeUnsafe(`reasoning-${this.turnState.turnId}`),
        type: "item.started",
        payload: {
          itemType: "reasoning",
          title: "Thinking",
        },
      } as ProviderRuntimeEvent);
    }

    this.emitEvent({
      ...this.makeEventBase(),
      itemId: RuntimeItemId.makeUnsafe(`reasoning-${this.turnState.turnId}`),
      type: "content.delta",
      payload: {
        streamKind: "reasoning_text",
        delta: thoughtText,
      },
    } as ProviderRuntimeEvent);
  }

  private emitRuntimeWarning(
    message: string,
    detail?: unknown,
    event?: ServerGeminiStreamEvent,
  ): void {
    this.emitEvent({
      ...this.makeEventBase(),
      type: "runtime.warning",
      payload: {
        message,
        ...(detail !== undefined ? { detail } : {}),
      },
      ...(event ? { raw: buildGeminiRawEvent(event) } : {}),
    } as ProviderRuntimeEvent);
  }

  private emitModelRerouted(
    fromModel: string,
    toModel: string,
    event: ServerGeminiStreamEvent,
  ): void {
    this.emitEvent({
      ...this.makeEventBase(),
      type: "model.rerouted",
      payload: {
        fromModel,
        toModel,
        reason: "gemini_client_model_selection",
      },
      raw: buildGeminiRawEvent(event),
    } as ProviderRuntimeEvent);
  }

  private handleGeminiModelInfo(model: string, event: ServerGeminiStreamEvent): void {
    if (!this.turnState) {
      return;
    }

    const previousModel = this.turnState.activeModel ?? this.session.model;
    if (previousModel && previousModel !== model) {
      this.emitModelRerouted(previousModel, model, event);
    }
    this.turnState.activeModel = model;
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
    this.planModeTextSuppressed = true;

    this.emitEvent({
      ...this.makeEventBase(),
      type: "turn.proposed.completed",
      payload: {
        planMarkdown: trimmedPlan,
      },
    } as ProviderRuntimeEvent);
  }

  private currentAssistantItemId(): string {
    return `msg-${this.turnState?.turnId ?? "unknown"}-${this.assistantMessageSegment}`;
  }

  private emitNativeEvent(event: ServerGeminiStreamEvent): void {
    this.input.writeNativeRecord?.(
      {
        observedAt: new Date().toISOString(),
        event: {
          id: randomUUID(),
          kind: "notification",
          provider: GEMINI_PROVIDER,
          createdAt: new Date().toISOString(),
          method: `gemini/${event.type}`,
          providerThreadId: String(this.session.threadId),
          ...(this.turnState ? { turnId: TurnIdBrand.makeUnsafe(this.turnState.turnId) } : {}),
          payload: event,
        },
      },
      this.session.threadId,
    );
  }

  private hasActiveTurnExecution(): boolean {
    return this.turnState !== undefined && this.activeTurnFiber !== undefined;
  }

  private interruptedTurnResult(): GeminiStreamTerminalTurnResult {
    return {
      state: "interrupted",
      stopReason: "aborted",
    };
  }

  private defaultTurnCompletion(): GeminiStreamTerminalTurnResult {
    return this.abortController.signal.aborted
      ? this.interruptedTurnResult()
      : {
          state: "completed",
          stopReason: "completed",
        };
  }

  private resolveTurnCompletion(
    exit: Exit.Exit<
      GeminiStreamTerminalTurnResult | undefined,
      GeminiStreamProcessorError | GeminiToolSchedulingError
    >,
  ): GeminiStreamTerminalTurnResult {
    if (Exit.isSuccess(exit)) {
      return exit.value ?? this.defaultTurnCompletion();
    }

    if (this.abortController.signal.aborted || Cause.hasInterruptsOnly(exit.cause)) {
      return this.interruptedTurnResult();
    }

    return {
      state: "failed",
      stopReason: "error",
      errorMessage: toMessage(Cause.squash(exit.cause), "Stream error"),
    };
  }

  private completeTurnEffect(result: GeminiStreamTerminalTurnResult): Effect.Effect<void> {
    return Effect.sync(() => {
      this.completeTurn(result.state, result.stopReason, result.errorMessage);
    });
  }

  private handleTurnExit(
    exit: Exit.Exit<
      GeminiStreamTerminalTurnResult | undefined,
      GeminiStreamProcessorError | GeminiToolSchedulingError
    >,
  ): Effect.Effect<void> {
    return this.completeTurnEffect(this.resolveTurnCompletion(exit));
  }

  private runStreamLoop(
    promptText: string,
  ): Effect.Effect<
    GeminiStreamTerminalTurnResult | undefined,
    GeminiStreamProcessorError | GeminiToolSchedulingError
  > {
    const promptId = `prompt-${Date.now()}`;
    const baseSystemInstruction = getCoreSystemPrompt(
      this.input.config,
      this.input.config.getSystemInstructionMemory(),
    );
    this.input.geminiClient.getChat().setSystemInstruction(baseSystemInstruction);

    return this.runStreamRound({
      promptId,
      nextRequest: [{ text: promptText }],
      displayContent: promptText,
    });
  }

  private runStreamRound(input: {
    readonly promptId: string;
    readonly nextRequest: PartListUnion | null;
    readonly displayContent?: PartListUnion;
  }): Effect.Effect<
    GeminiStreamTerminalTurnResult | undefined,
    GeminiStreamProcessorError | GeminiToolSchedulingError
  > {
    if (input.nextRequest === null || !this.turnState || this.stopped) {
      return Effect.void.pipe(Effect.as(undefined));
    }

    if (this.abortController.signal.aborted) {
      return Effect.succeed(this.interruptedTurnResult());
    }

    const responseStream = this.input.geminiClient.sendMessageStream(
      input.nextRequest,
      this.abortController.signal,
      input.promptId,
      undefined,
      false,
      input.displayContent,
    );

    return processGeminiStreamEvents({
      stream: responseStream,
      geminiClient: this.input.geminiClient,
      sessionModel: this.session.model,
      turnState: this.turnState,
      onNativeEvent: (event) => this.emitNativeEvent(event),
      onThought: (text) => this.emitReasoningText(text),
      onContent: (text) => this.emitAssistantText(text),
      onModelInfo: (model, event) => this.handleGeminiModelInfo(model, event),
      onRuntimeWarning: (message, detail, event) => this.emitRuntimeWarning(message, detail, event),
      onUsage: (usage) => {
        this.lastKnownUsage = usage;
        this.totalProcessedTokens += usage.totalTokens;
      },
    }).pipe(
      Effect.flatMap((processed) => {
        if (this.abortController.signal.aborted) {
          return Effect.succeed(this.interruptedTurnResult());
        }

        if (processed.terminalTurnResult) {
          return Effect.succeed(processed.terminalTurnResult);
        }

        if (processed.toolCallRequests.length === 0) {
          return Effect.void.pipe(Effect.as(undefined));
        }

        return Effect.tryPromise({
          try: () =>
            this.toolBridge.scheduleToolCalls(
              processed.toolCallRequests,
              this.abortController.signal,
            ),
          catch: (cause) =>
            new GeminiToolSchedulingError({
              message: toMessage(cause, "Failed to schedule Gemini tool calls."),
              cause,
            }),
        }).pipe(
          Effect.flatMap((scheduled) => {
            if (this.abortController.signal.aborted) {
              return Effect.succeed(this.interruptedTurnResult());
            }

            if (scheduled.stopExecution) {
              return Effect.succeed({
                state: "completed" as const,
                stopReason: "tool_requested_stop",
              });
            }

            if (scheduled.fatalError) {
              return Effect.succeed({
                state: "failed" as const,
                stopReason: "fatal_tool_error",
                errorMessage: scheduled.fatalError.message,
              });
            }

            if (scheduled.allCancelled) {
              return Effect.succeed({
                state: "cancelled" as const,
                stopReason: "tool_execution_cancelled",
              });
            }

            return this.runStreamRound({
              promptId: input.promptId,
              nextRequest: [...scheduled.responseParts],
            });
          }),
        );
      }),
    );
  }

  private cancelPendingInteractions(): void {
    for (const pending of this.pendingApprovals.values()) {
      void this.messageBus.publish({
        type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
        correlationId: pending.correlationId,
        confirmed: false,
        outcome: ToolConfirmationOutcome.Cancel,
      });
    }
    this.pendingApprovals.clear();

    for (const [requestId, pending] of this.pendingUserInputs.entries()) {
      if (pending.responseChannel === "tool_confirmation") {
        void this.messageBus.publish({
          type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
          correlationId: pending.correlationId,
          confirmed: false,
          outcome: ToolConfirmationOutcome.Cancel,
        });
      } else {
        void this.messageBus.publish({
          type: MessageBusType.ASK_USER_RESPONSE,
          correlationId: pending.correlationId,
          answers: {},
          cancelled: true,
        } as never);
      }

      this.emitEvent({
        ...this.makeEventBase(),
        requestId: RuntimeRequestId.makeUnsafe(requestId),
        type: "user-input.resolved",
        payload: {
          answers: {},
        },
      } as ProviderRuntimeEvent);
    }
    this.pendingUserInputs.clear();
  }

  private completeTurn(
    state: "completed" | "failed" | "interrupted" | "cancelled",
    stopReason: string,
    errorMessage?: string,
  ): void {
    if (!this.turnState) {
      return;
    }

    this.finalizeAssistantSegment();

    if (this.turnState.reasoningItemEmitted) {
      this.emitEvent({
        ...this.makeEventBase(),
        itemId: RuntimeItemId.makeUnsafe(`reasoning-${this.turnState.turnId}`),
        type: "item.completed",
        payload: {
          itemType: "reasoning",
          status: "completed",
          title: "Thinking",
        },
      } as ProviderRuntimeEvent);
    }

    this.cancelPendingInteractions();

    const tokenUsage = this.lastKnownUsage
      ? buildTokenUsageSnapshot({
          usage: this.lastKnownUsage,
          totalProcessedTokens: this.totalProcessedTokens,
          model: this.turnState?.activeModel ?? this.session.model,
        })
      : undefined;
    if (tokenUsage) {
      this.emitEvent({
        ...this.makeEventBase(),
        type: "thread.token-usage.updated",
        payload: {
          usage: tokenUsage,
        },
      } as ProviderRuntimeEvent);
    }

    const turnId = this.turnState.turnId;
    if (state === "completed") {
      this.turns.push({
        id: turnId,
        items: [],
        historyLength: this.input.geminiClient.getHistory().length,
      });
      this.updateResumeCursor();
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
      this.input.persistBinding(
        buildGeminiPersistedBinding({
          session: this.session,
          status: state === "failed" ? "error" : "running",
          lastRuntimeEvent: "turn.completed",
          lastRuntimeEventAt: completedAt,
        }),
      );
    }

    this.emitEvent({
      ...this.makeEventBase(),
      type: "turn.completed",
      payload: {
        state,
        stopReason,
        ...(tokenUsage ? { usage: tokenUsage } : {}),
        ...(errorMessage ? { errorMessage } : {}),
      },
    } as ProviderRuntimeEvent);

    this.turnState = undefined;

    if (!this.stopped) {
      this.emitEvent({
        ...this.makeEventBase(),
        type: "session.state.changed",
        payload: { state: "ready" },
      } as ProviderRuntimeEvent);
    }
  }
}

export function restoreGeminiResumeHistory(resumeCursor: unknown): GeminiResumeState | undefined {
  return readGeminiResumeState(resumeCursor);
}
