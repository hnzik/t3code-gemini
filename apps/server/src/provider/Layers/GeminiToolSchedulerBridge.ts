import path from "node:path";

import type { Part } from "@google/genai";
import type { ProviderRuntimeEvent } from "@t3tools/contracts";
import { RuntimeItemId } from "@t3tools/contracts";
import {
  CoreToolCallStatus,
  MessageBusType,
  ROOT_SCHEDULER_ID,
  Scheduler,
  ToolErrorType,
  isFatalToolError,
  recordToolCallInteractions,
  refreshServerHierarchicalMemory,
  type AgentLoopContext,
  type CompletedToolCall,
  type Config,
  type EditorType,
  type GeminiClient,
  type MessageBus,
  type SerializableConfirmationDetails,
  type SubagentActivityMessage,
  type ToolCall,
  type ToolCallRequestInfo,
} from "@google/gemini-cli-core";

import {
  buildGeminiToolStreamKind,
  classifyGeminiToolItemType,
  formatGeminiSubagentActivityDetail,
  readGeminiPlanMarkdownFromFile,
  summarizeGeminiToolRequest,
  summarizeGeminiToolResultDisplay,
  titleForGeminiToolType,
  type GeminiToolInFlight,
} from "./GeminiRuntimeHelpers";

interface ToolBridgeCallbacks {
  readonly emitEvent: (event: ProviderRuntimeEvent) => void;
  readonly makeEventBase: () => Omit<ProviderRuntimeEvent, "type" | "payload">;
  readonly finalizeAssistantSegment: () => void;
  readonly onPlanCaptured: (planMarkdown: string) => void;
  readonly onApprovalRequested: (input: {
    readonly toolName: string;
    readonly correlationId: string;
    readonly details: SerializableConfirmationDetails;
    readonly args: Record<string, unknown>;
  }) => void;
}

interface ToolBridgeState {
  readonly item: GeminiToolInFlight;
  lastStatus?: ToolCall["status"];
  lastLiveOutput?: string;
  lastProgressDetail?: string;
  completionEmitted: boolean;
  approvalRequestEmitted: boolean;
  planCapturePromise: Promise<boolean> | undefined;
  planCaptured: boolean;
  planPath?: string;
}

export interface GeminiToolScheduleResult {
  readonly completedToolCalls: ReadonlyArray<CompletedToolCall>;
  readonly responseParts: ReadonlyArray<Part>;
  readonly stopExecution: boolean;
  readonly fatalError: Error | undefined;
  readonly allCancelled: boolean;
}

function stringifyToolOutputChunk(output: unknown): string | undefined {
  if (typeof output === "string") {
    return output;
  }
  if (!output || typeof output !== "object") {
    return undefined;
  }

  if ("text" in output && typeof output.text === "string") {
    return output.text;
  }

  return undefined;
}

function completeStatusForTool(call: ToolCall): "completed" | "failed" | "declined" {
  switch (call.status) {
    case CoreToolCallStatus.Success:
      return "completed";
    case CoreToolCallStatus.Cancelled:
      return "declined";
    default:
      return "failed";
  }
}

function normalizeSubagentAliases(
  state: ToolBridgeState,
  requestName: string,
  displayName?: string,
): ReadonlySet<string> {
  const aliases = new Set<string>();
  aliases.add(requestName.toLowerCase());
  aliases.add(state.item.toolName.toLowerCase());

  if (displayName) {
    aliases.add(displayName.toLowerCase());
  }

  for (const key of ["agent_name", "agentName", "subagent_name", "subagentName", "name"]) {
    const value = state.item.input[key];
    if (typeof value === "string" && value.trim().length > 0) {
      aliases.add(value.trim().toLowerCase());
    }
  }

  return aliases;
}

function resolveExitPlanPath(
  config: Config,
  state: Pick<ToolBridgeState, "planPath"> | undefined,
  toolCall: Pick<ToolCall, "request">,
): string | undefined {
  if (typeof state?.planPath === "string" && state.planPath.length > 0) {
    return state.planPath;
  }

  if (toolCall.request.name !== "exit_plan_mode") {
    return undefined;
  }

  const planFilename = toolCall.request.args?.["plan_filename"];
  if (typeof planFilename !== "string" || planFilename.trim().length === 0) {
    return undefined;
  }

  const resolvedPlanPath = path.join(config.storage.getPlansDir(), path.basename(planFilename));
  if (state) {
    state.planPath = resolvedPlanPath;
  }
  return resolvedPlanPath;
}

export class GeminiToolSchedulerBridge {
  private readonly messageBus: MessageBus;
  private readonly scheduler: Scheduler;
  private readonly toolStates = new Map<string, ToolBridgeState>();
  private readonly unsubscribers: Array<() => void> = [];

  constructor(
    private readonly input: {
      readonly config: Config;
      readonly geminiClient: GeminiClient;
      readonly getPreferredEditor: () => EditorType | undefined;
      readonly context: AgentLoopContext;
      readonly callbacks: ToolBridgeCallbacks;
    },
  ) {
    this.messageBus = input.context.messageBus;
    this.scheduler = new Scheduler({
      context: input.context,
      messageBus: this.messageBus,
      getPreferredEditor: input.getPreferredEditor,
      schedulerId: ROOT_SCHEDULER_ID,
    });

    this.subscribeToToolUpdates();
    this.subscribeToSubagentActivity();
  }

  dispose(): void {
    for (const unsubscribe of this.unsubscribers.splice(0)) {
      try {
        unsubscribe();
      } catch {
        // Best-effort cleanup only.
      }
    }
    this.scheduler.dispose();
  }

  cancelAll(): void {
    this.scheduler.cancelAll();
  }

  async scheduleToolCalls(
    toolCallRequests: ReadonlyArray<ToolCallRequestInfo>,
    signal: AbortSignal,
  ): Promise<GeminiToolScheduleResult> {
    const completedToolCalls = await this.scheduler.schedule([...toolCallRequests], signal);
    let capturedExitPlan = false;

    try {
      const currentModel =
        this.input.geminiClient.getCurrentSequenceModel() ?? this.input.config.getModel();
      this.input.geminiClient.getChat().recordCompletedToolCalls(currentModel, completedToolCalls);
      await recordToolCallInteractions(this.input.config, completedToolCalls);
    } catch {
      // Recording is best-effort and should not break the session.
    }

    const successfulMemorySave = completedToolCalls.some(
      (toolCall) =>
        toolCall.request.name === "save_memory" && toolCall.status === CoreToolCallStatus.Success,
    );
    if (successfulMemorySave) {
      try {
        await refreshServerHierarchicalMemory(this.input.config);
      } catch {
        // Memory refresh is best-effort.
      }
    }

    for (const toolCall of completedToolCalls) {
      const state = this.toolStates.get(toolCall.request.callId);
      const resolvedPlanPath = resolveExitPlanPath(this.input.config, state, toolCall);
      if (!resolvedPlanPath) {
        continue;
      }

      if (state) {
        capturedExitPlan =
          (await this.capturePlanIfAvailable(state, resolvedPlanPath)) || capturedExitPlan;
        continue;
      }

      try {
        const planMarkdown = await readGeminiPlanMarkdownFromFile(resolvedPlanPath);
        if (planMarkdown) {
          this.input.callbacks.onPlanCaptured(planMarkdown);
          capturedExitPlan = true;
        }
      } catch {
        // Plan capture is best-effort.
      }
    }

    const stopExecution =
      capturedExitPlan ||
      completedToolCalls.some(
        (toolCall) =>
          toolCall.response.errorType === ToolErrorType.STOP_EXECUTION &&
          toolCall.response.error !== undefined,
      );
    const fatalTool = completedToolCalls.find((toolCall) =>
      isFatalToolError(toolCall.response.errorType),
    );

    return {
      completedToolCalls,
      responseParts: completedToolCalls.flatMap((toolCall) => toolCall.response.responseParts),
      stopExecution,
      fatalError: fatalTool?.response.error,
      allCancelled:
        completedToolCalls.length > 0 &&
        completedToolCalls.every((toolCall) => toolCall.status === CoreToolCallStatus.Cancelled),
    };
  }

  private subscribeToToolUpdates(): void {
    const handleToolCallsUpdate = (message: {
      toolCalls: ToolCall[];
      schedulerId: string;
    }): void => {
      if (message.schedulerId !== ROOT_SCHEDULER_ID) {
        return;
      }

      for (const toolCall of message.toolCalls) {
        this.handleToolCallUpdate(toolCall);
      }
    };

    this.messageBus.subscribe(MessageBusType.TOOL_CALLS_UPDATE, handleToolCallsUpdate as never);
    this.unsubscribers.push(() => {
      this.messageBus.unsubscribe(MessageBusType.TOOL_CALLS_UPDATE, handleToolCallsUpdate as never);
    });
  }

  private subscribeToSubagentActivity(): void {
    const handleSubagentActivity = (message: SubagentActivityMessage): void => {
      const normalizedName = message.subagentName.toLowerCase();
      const activeAgentTool = [...this.toolStates.entries()]
        .map(([, state]) => ({ state }))
        .toReversed()
        .find(({ state }) => {
          if (
            state.item.itemType !== "collab_agent_tool_call" ||
            state.completionEmitted === true
          ) {
            return false;
          }

          const aliases = normalizeSubagentAliases(state, state.item.toolName, state.item.detail);
          return aliases.has(normalizedName);
        });

      if (!activeAgentTool) {
        const fallback = [...this.toolStates.values()]
          .toReversed()
          .find(
            (state) =>
              state.item.itemType === "collab_agent_tool_call" && state.completionEmitted === false,
          );
        if (!fallback) {
          return;
        }

        this.emitSubagentUpdate(fallback, message);
        return;
      }

      this.emitSubagentUpdate(activeAgentTool.state, message);
    };

    this.messageBus.subscribe(MessageBusType.SUBAGENT_ACTIVITY, handleSubagentActivity as never);
    this.unsubscribers.push(() => {
      this.messageBus.unsubscribe(
        MessageBusType.SUBAGENT_ACTIVITY,
        handleSubagentActivity as never,
      );
    });
  }

  private emitSubagentUpdate(state: ToolBridgeState, message: SubagentActivityMessage): void {
    this.input.callbacks.emitEvent({
      ...this.input.callbacks.makeEventBase(),
      itemId: RuntimeItemId.makeUnsafe(state.item.itemId),
      type: "item.updated",
      payload: {
        itemType: state.item.itemType,
        title: state.item.title,
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
  }

  private handleToolCallUpdate(toolCall: ToolCall): void {
    const state = this.ensureToolState(toolCall);
    const confirmationDetails =
      toolCall.status === CoreToolCallStatus.AwaitingApproval &&
      "confirmationDetails" in toolCall &&
      toolCall.confirmationDetails &&
      typeof toolCall.confirmationDetails === "object"
        ? (toolCall.confirmationDetails as SerializableConfirmationDetails)
        : undefined;

    if (confirmationDetails?.type === "exit_plan_mode") {
      state.planPath = confirmationDetails.planPath;
    }

    const resolvedPlanPath = resolveExitPlanPath(this.input.config, state, toolCall);
    if (toolCall.status === CoreToolCallStatus.AwaitingApproval) {
      void this.capturePlanIfAvailable(state, resolvedPlanPath);
    }

    if (
      toolCall.status === CoreToolCallStatus.AwaitingApproval &&
      state.approvalRequestEmitted === false &&
      "correlationId" in toolCall &&
      typeof toolCall.correlationId === "string" &&
      confirmationDetails
    ) {
      state.approvalRequestEmitted = true;
      this.input.callbacks.onApprovalRequested({
        toolName: toolCall.request.name,
        correlationId: toolCall.correlationId,
        details: confirmationDetails,
        args: toolCall.request.args,
      });
    }

    if (toolCall.status === CoreToolCallStatus.Executing) {
      const liveOutput = stringifyToolOutputChunk(toolCall.liveOutput);
      if (liveOutput && liveOutput !== state.lastLiveOutput) {
        state.lastLiveOutput = liveOutput;
        this.input.callbacks.emitEvent({
          ...this.input.callbacks.makeEventBase(),
          itemId: RuntimeItemId.makeUnsafe(state.item.itemId),
          type: "content.delta",
          payload: {
            streamKind: buildGeminiToolStreamKind(state.item.itemType),
            delta: liveOutput,
          },
        } as ProviderRuntimeEvent);
      }

      const progressDetail =
        typeof toolCall.progressMessage === "string" && toolCall.progressMessage.trim().length > 0
          ? toolCall.progressMessage.trim()
          : undefined;
      if (progressDetail && progressDetail !== state.lastProgressDetail) {
        state.lastProgressDetail = progressDetail;
        this.input.callbacks.emitEvent({
          ...this.input.callbacks.makeEventBase(),
          itemId: RuntimeItemId.makeUnsafe(state.item.itemId),
          type: "item.updated",
          payload: {
            itemType: state.item.itemType,
            title: state.item.title,
            detail: progressDetail,
            data: {
              toolName: state.item.toolName,
              progressPercent: toolCall.progressPercent,
              progress: toolCall.progress,
              progressTotal: toolCall.progressTotal,
              ...(toolCall.pid !== undefined ? { pid: toolCall.pid } : {}),
            },
          },
        } as ProviderRuntimeEvent);
      }
    }

    if (
      (toolCall.status === CoreToolCallStatus.Success ||
        toolCall.status === CoreToolCallStatus.Error ||
        toolCall.status === CoreToolCallStatus.Cancelled) &&
      state.completionEmitted === false
    ) {
      this.completeToolState(state, toolCall);
    }

    state.lastStatus = toolCall.status;
  }

  private ensureToolState(toolCall: ToolCall): ToolBridgeState {
    const existing = this.toolStates.get(toolCall.request.callId);
    if (existing) {
      return existing;
    }

    this.input.callbacks.finalizeAssistantSegment();

    const itemType = classifyGeminiToolItemType(toolCall.request.name);
    const detail =
      summarizeGeminiToolRequest(toolCall.request.name, toolCall.request.args) ??
      (toolCall.tool?.displayName || toolCall.request.name);
    const item: GeminiToolInFlight = {
      requestId: toolCall.request.callId,
      itemId: `tool-${toolCall.request.callId}`,
      itemType,
      toolName: toolCall.request.name,
      title: titleForGeminiToolType(itemType),
      detail,
      input: toolCall.request.args,
      ...(toolCall.schedulerId ? { schedulerId: toolCall.schedulerId } : {}),
    };
    const state: ToolBridgeState = {
      item,
      completionEmitted: false,
      approvalRequestEmitted: false,
      planCapturePromise: undefined,
      planCaptured: false,
    };

    this.toolStates.set(toolCall.request.callId, state);
    this.input.callbacks.emitEvent({
      ...this.input.callbacks.makeEventBase(),
      itemId: RuntimeItemId.makeUnsafe(item.itemId),
      type: "item.started",
      payload: {
        itemType: item.itemType,
        title: item.title,
        ...(item.detail ? { detail: item.detail } : {}),
        data: item.input,
      },
    } as ProviderRuntimeEvent);

    return state;
  }

  private completeToolState(state: ToolBridgeState, toolCall: ToolCall): void {
    const summary = summarizeGeminiToolResultDisplay({
      returnDisplay: "response" in toolCall ? toolCall.response.resultDisplay : undefined,
      llmContent: "response" in toolCall ? toolCall.response.responseParts : undefined,
      error: "response" in toolCall ? toolCall.response.error : undefined,
    });
    if (summary && summary !== state.lastLiveOutput && summary !== state.lastProgressDetail) {
      this.input.callbacks.emitEvent({
        ...this.input.callbacks.makeEventBase(),
        itemId: RuntimeItemId.makeUnsafe(state.item.itemId),
        type: "content.delta",
        payload: {
          streamKind: buildGeminiToolStreamKind(state.item.itemType),
          delta: summary,
        },
      } as ProviderRuntimeEvent);
    }

    this.input.callbacks.emitEvent({
      ...this.input.callbacks.makeEventBase(),
      itemId: RuntimeItemId.makeUnsafe(state.item.itemId),
      type: "item.completed",
      payload: {
        itemType: state.item.itemType,
        status: completeStatusForTool(toolCall),
        title: state.item.title,
        ...(state.item.detail ? { detail: state.item.detail } : {}),
        data: {
          toolName: state.item.toolName,
          input: state.item.input,
          ...(toolCall.schedulerId ? { schedulerId: toolCall.schedulerId } : {}),
        },
      },
    } as ProviderRuntimeEvent);
    state.completionEmitted = true;
  }

  private async capturePlanIfAvailable(
    state: ToolBridgeState,
    planPath: string | undefined,
  ): Promise<boolean> {
    if (!planPath) {
      return state.planCaptured;
    }

    if (state.planCaptured) {
      return true;
    }

    if (!state.planCapturePromise) {
      state.planCapturePromise = (async () => {
        try {
          const planMarkdown = await readGeminiPlanMarkdownFromFile(planPath);
          if (planMarkdown) {
            state.planCaptured = true;
            this.input.callbacks.onPlanCaptured(planMarkdown);
          }
        } catch {
          // Plan capture is best-effort.
        } finally {
          state.planCapturePromise = undefined;
        }
        return state.planCaptured;
      })();
    }

    await state.planCapturePromise;
    return state.planCaptured;
  }
}
