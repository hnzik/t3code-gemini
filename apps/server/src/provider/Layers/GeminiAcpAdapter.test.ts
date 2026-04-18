import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import * as NodeServices from "@effect/platform-node/NodeServices";
import { assert, describe, it } from "@effect/vitest";
import { Effect, Layer, Stream } from "effect";
import {
  AuthType,
  CoreEvent,
  CoreToolCallStatus,
  GeminiEventType,
  MessageBusType,
  QuestionType,
  ROOT_SCHEDULER_ID,
  Scheduler,
  ToolConfirmationOutcome,
  coreEvents,
  makeFakeConfig,
  type Config,
  type GeminiClient,
  type MessageBus,
  type ServerGeminiStreamEvent,
  type ToolCall,
  type ToolCallRequestInfo,
  type CompletedToolCall,
} from "@google/gemini-cli-core";
import { FinishReason, type PartListUnion } from "@google/genai";
import { ApprovalRequestId, ThreadId, type ProviderRuntimeEvent } from "@t3tools/contracts";
import { afterEach, vi } from "vitest";

import { ServerConfig } from "../../config.ts";
import { ProviderSessionRuntimeRepositoryLive } from "../../persistence/Layers/ProviderSessionRuntime.ts";
import { SqlitePersistenceMemory } from "../../persistence/Layers/Sqlite.ts";
import { ServerSettingsService } from "../../serverSettings.ts";
import { GeminiAcpAdapter, type GeminiAcpAdapterShape } from "../Services/GeminiAcpAdapter.ts";
import {
  applyGeminiAssistantTextChunk,
  buildGeminiPersistedBinding,
  buildGeminiAssistantHistoryEntry,
  buildGeminiAskUserResponseAnswers,
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
  makeGeminiAcpAdapterLive,
} from "./GeminiAcpAdapter.ts";
import { GeminiAuthRuntimeStateLive } from "./GeminiAuthRuntimeState.ts";
import { resolveGeminiAuthType } from "./GeminiCoreConfig.ts";
import { ProviderSessionDirectoryLive } from "./ProviderSessionDirectory.ts";
import { GeminiToolSchedulerBridge } from "./GeminiToolSchedulerBridge.ts";

vi.mock("@google/gemini-cli-core", async () => {
  const actual =
    await vi.importActual<typeof import("@google/gemini-cli-core")>("@google/gemini-cli-core");
  return {
    ...actual,
    getCoreSystemPrompt: vi.fn(() => "Test system prompt"),
    recordToolCallInteractions: vi.fn(async () => undefined),
    refreshServerHierarchicalMemory: vi.fn(async () => ({
      memoryContent: {
        global: "",
        project: "",
        extension: "",
      },
      fileCount: 0,
      filePaths: [],
    })),
  };
});

afterEach(() => {
  vi.restoreAllMocks();
});

const THREAD_ID = ThreadId.make("thread-gemini-adapter");

interface Deferred<T> {
  readonly promise: Promise<T>;
  resolve: (value: T | PromiseLike<T>) => void;
  reject: (reason?: unknown) => void;
}

interface FakeGeminiSendMessageCall {
  readonly query: PartListUnion;
  readonly promptId: string;
  readonly displayContent: PartListUnion | undefined;
}

type FakeGeminiStreamFactory = (input: {
  readonly query: PartListUnion;
  readonly signal: AbortSignal;
  readonly promptId: string;
  readonly displayContent: PartListUnion | undefined;
  readonly callIndex: number;
}) => AsyncIterable<ServerGeminiStreamEvent>;

interface GeminiHarness {
  readonly layer: Layer.Layer<GeminiAcpAdapter, never, never>;
  readonly config: Config;
  readonly geminiClient: FakeGeminiClient;
  readonly messageBus: MessageBus;
}

function makeDeferred<T>(): Deferred<T> {
  let resolve!: Deferred<T>["resolve"];
  let reject!: Deferred<T>["reject"];
  const promise = new Promise<T>((innerResolve, innerReject) => {
    resolve = innerResolve;
    reject = innerReject;
  });
  return {
    promise,
    resolve,
    reject,
  };
}

async function* streamFromEvents(
  events: ReadonlyArray<ServerGeminiStreamEvent>,
): AsyncIterable<ServerGeminiStreamEvent> {
  for (const event of events) {
    yield event;
  }
}

class FakeGeminiClient {
  private readonly streamFactories: FakeGeminiStreamFactory[] = [];
  private history: Array<unknown> = [];
  readonly sendCalls: Array<FakeGeminiSendMessageCall> = [];
  readonly resumeChatCalls: Array<ReadonlyArray<unknown>> = [];
  readonly addHistoryCalls: Array<unknown> = [];
  readonly chat = {
    systemInstructions: [] as Array<string>,
    completedToolCalls: [] as Array<{
      readonly model: string | undefined;
      readonly calls: ReadonlyArray<CompletedToolCall>;
    }>,
    setSystemInstruction: (instruction: string) => {
      this.chat.systemInstructions.push(instruction);
    },
    recordCompletedToolCalls: (
      model: string | undefined,
      calls: ReadonlyArray<CompletedToolCall>,
    ) => {
      this.chat.completedToolCalls.push({
        model,
        calls,
      });
    },
  };
  currentSequenceModel: string | undefined = undefined;

  enqueueEvents(events: ReadonlyArray<ServerGeminiStreamEvent>): void {
    this.streamFactories.push(() => streamFromEvents(events));
  }

  enqueueFactory(factory: FakeGeminiStreamFactory): void {
    this.streamFactories.push(factory);
  }

  sendMessageStream = (
    query: PartListUnion,
    signal: AbortSignal,
    promptId: string,
    _options?: unknown,
    _isToolResponse?: boolean,
    displayContent?: PartListUnion,
  ): AsyncIterable<ServerGeminiStreamEvent> => {
    const callIndex = this.sendCalls.length;
    this.sendCalls.push({
      query,
      promptId,
      ...(displayContent !== undefined ? { displayContent } : { displayContent: undefined }),
    });

    const factory = this.streamFactories.shift();
    if (!factory) {
      return streamFromEvents([]);
    }
    return factory({
      query,
      signal,
      promptId,
      ...(displayContent !== undefined ? { displayContent } : { displayContent: undefined }),
      callIndex,
    });
  };

  getChat() {
    return this.chat;
  }

  getHistory(): Array<unknown> {
    return [...this.history];
  }

  setHistory(history: Array<unknown>): void {
    this.history = [...history];
  }

  async resumeChat(history: Array<unknown>): Promise<void> {
    this.resumeChatCalls.push([...history]);
    this.history = [...history];
  }

  async addHistory(entry: unknown): Promise<void> {
    this.addHistoryCalls.push(entry);
    this.history.push(entry);
  }

  getCurrentSequenceModel(): string | undefined {
    return this.currentSequenceModel;
  }

  async dispose(): Promise<void> {
    return undefined;
  }
}

function makeGeminiHarness(options?: {
  readonly cwd?: string;
  readonly model?: string;
  readonly nativeEventLogger?: {
    readonly filePath: string;
    write: (event: unknown, threadId: ThreadId | null) => Effect.Effect<void>;
    close: () => Effect.Effect<void>;
  };
}): GeminiHarness {
  const cwd = options?.cwd ?? process.cwd();
  const model = options?.model ?? "gemini-2.5-pro";
  const geminiClient = new FakeGeminiClient();
  const config = makeFakeConfig({
    cwd,
    targetDir: cwd,
    model,
    sessionId: `gemini-test-${Date.now()}`,
  });
  vi.spyOn(config, "refreshAuth").mockResolvedValue(undefined);
  vi.spyOn(config, "initialize").mockResolvedValue(undefined);
  vi.spyOn(config, "dispose").mockResolvedValue(undefined);
  vi.spyOn(config, "getGeminiClient").mockReturnValue(geminiClient as unknown as GeminiClient);

  const runtimeRepositoryLayer = ProviderSessionRuntimeRepositoryLive.pipe(
    Layer.provide(SqlitePersistenceMemory),
  );
  const directoryLayer = ProviderSessionDirectoryLive.pipe(Layer.provide(runtimeRepositoryLayer));
  const layer = makeGeminiAcpAdapterLive({
    createConfig: async () => config,
    ...(options?.nativeEventLogger
      ? {
          nativeEventLogger: options.nativeEventLogger,
        }
      : {}),
  }).pipe(
    Layer.provideMerge(ServerConfig.layerTest(cwd, cwd)),
    Layer.provideMerge(ServerSettingsService.layerTest()),
    Layer.provideMerge(GeminiAuthRuntimeStateLive),
    Layer.provideMerge(directoryLayer),
    Layer.provideMerge(NodeServices.layer),
  );

  return {
    layer: layer as Layer.Layer<GeminiAcpAdapter, never, never>,
    config,
    geminiClient,
    messageBus: config.getMessageBus(),
  };
}

function makeToolCallRequest(input: {
  readonly callId: string;
  readonly name: string;
  readonly args?: Record<string, unknown> | undefined;
  readonly promptId?: string;
}): ToolCallRequestInfo {
  return {
    callId: input.callId,
    name: input.name,
    args: input.args ?? {},
    isClientInitiated: false,
    prompt_id: input.promptId ?? "prompt-test",
  };
}

function makeToolCallUpdate(input: {
  readonly callId: string;
  readonly name: string;
  readonly args?: Record<string, unknown> | undefined;
  readonly status: CoreToolCallStatus;
  readonly correlationId?: string;
  readonly resultDisplay?: string;
  readonly liveOutput?: string;
  readonly progressMessage?: string;
  readonly confirmationDetails?: unknown;
  readonly displayName?: string;
}): ToolCall {
  return {
    request: {
      callId: input.callId,
      name: input.name,
      args: input.args ?? {},
    },
    schedulerId: ROOT_SCHEDULER_ID,
    status: input.status,
    ...(input.correlationId ? { correlationId: input.correlationId } : {}),
    ...(input.liveOutput ? { liveOutput: input.liveOutput } : {}),
    ...(input.progressMessage ? { progressMessage: input.progressMessage } : {}),
    ...(input.confirmationDetails ? { confirmationDetails: input.confirmationDetails } : {}),
    ...(input.displayName ? { tool: { displayName: input.displayName } } : {}),
    ...(input.resultDisplay ? { resultDisplay: input.resultDisplay } : {}),
  } as unknown as ToolCall;
}

function makeCompletedToolCall(input: {
  readonly callId: string;
  readonly name: string;
  readonly args?: Record<string, unknown> | undefined;
  readonly status?: CoreToolCallStatus;
  readonly resultDisplay?: string;
  readonly responseParts?: ReadonlyArray<unknown>;
  readonly error?: Error;
  readonly errorType?: unknown;
}): CompletedToolCall {
  return {
    ...makeToolCallUpdate({
      callId: input.callId,
      name: input.name,
      status: input.status ?? CoreToolCallStatus.Success,
      ...(input.args !== undefined ? { args: input.args } : {}),
      ...(input.resultDisplay !== undefined ? { resultDisplay: input.resultDisplay } : {}),
    }),
    status: input.status ?? CoreToolCallStatus.Success,
    response: {
      responseParts: [...(input.responseParts ?? [])],
      ...(input.resultDisplay ? { resultDisplay: input.resultDisplay } : {}),
      ...(input.error ? { error: input.error } : {}),
      ...(input.errorType ? { errorType: input.errorType } : {}),
    },
  } as unknown as CompletedToolCall;
}

function waitForRuntimeEvent(
  adapter: { readonly streamEvents: Stream.Stream<ProviderRuntimeEvent> },
  predicate: (event: ProviderRuntimeEvent) => boolean,
) {
  return Stream.runCollect(Stream.takeUntil(adapter.streamEvents, predicate)).pipe(
    Effect.map((chunk) => {
      const events = Array.from(chunk);
      const event = events.at(-1);
      if (!event) {
        throw new Error("Expected a runtime event");
      }
      return event;
    }),
  );
}

function collectRuntimeEventsUntil(
  adapter: { readonly streamEvents: Stream.Stream<ProviderRuntimeEvent> },
  predicate: (event: ProviderRuntimeEvent) => boolean,
) {
  return Stream.runCollect(Stream.takeUntil(adapter.streamEvents, predicate)).pipe(
    Effect.map((chunk) => Array.from(chunk)),
  );
}

function startGeminiSession(
  adapter: GeminiAcpAdapterShape,
  options?: {
    readonly runtimeMode?: "full-access" | "approval-required";
    readonly model?: string;
    readonly resumeCursor?: unknown;
  },
) {
  return Effect.gen(function* () {
    const session = yield* adapter.startSession({
      threadId: THREAD_ID,
      provider: "geminiAcp",
      runtimeMode: options?.runtimeMode ?? "full-access",
      ...(options?.model
        ? {
            modelSelection: {
              provider: "geminiAcp",
              model: options.model,
            },
          }
        : {}),
      ...(options?.resumeCursor !== undefined ? { resumeCursor: options.resumeCursor } : {}),
    });

    yield* Stream.take(adapter.streamEvents, 3).pipe(Stream.runDrain);
    return session;
  });
}

function runWithGeminiHarness<T, E>(
  harness: GeminiHarness,
  effect: (adapter: GeminiAcpAdapterShape) => Effect.Effect<T, E, never>,
) {
  return Effect.gen(function* () {
    const adapter = yield* GeminiAcpAdapter;
    yield* Effect.addFinalizer(() => Effect.orDie(adapter.stopAll()));
    return yield* Effect.orDie(effect(adapter));
  }).pipe(
    Effect.scoped,
    Effect.provide(harness.layer as Layer.Layer<GeminiAcpAdapter, never, never>),
    Effect.orDie,
  ) as Effect.Effect<T, never, never>;
}

function runWithGeminiHarnessPromise<T, E>(
  harness: GeminiHarness,
  effect: (adapter: GeminiAcpAdapterShape) => Effect.Effect<T, E, never>,
): Promise<T> {
  return Effect.runPromise(runWithGeminiHarness(harness, effect));
}

describe("GeminiAcpAdapter agent-event helpers", () => {
  it("formats subagent thought updates for the work log", () => {
    assert.equal(
      formatGeminiSubagentActivityDetail({
        subagentName: "generalist",
        activity: {
          type: "thought",
          content: "Scanning the repository layout first.",
          status: "running",
        },
      }),
      "generalist: Scanning the repository layout first.",
    );
  });

  it("formats subagent tool call updates for the work log", () => {
    assert.equal(
      formatGeminiSubagentActivityDetail({
        subagentName: "codebase_investigator",
        activity: {
          type: "tool_call",
          content: "rg",
          displayName: "Search code",
          description: "Looking for the WebSocket handlers.",
          args: '{"pattern":"websocket"}',
          status: "completed",
        },
      }),
      'codebase_investigator: Search code - Looking for the WebSocket handlers. - {"pattern":"websocket"} - status=completed',
    );
  });

  it("classifies Gemini read-only tools as file reads for approval routing", () => {
    assert.equal(classifyRequestTypeForTool("glob"), "file_read_approval");
    assert.equal(classifyRequestTypeForTool("read_file"), "file_read_approval");
    assert.equal(classifyRequestTypeForTool("list_directory"), "file_read_approval");
    assert.equal(classifyRequestTypeForTool("replace"), "file_change_approval");
    assert.equal(classifyRequestTypeForTool("run_shell_command"), "exec_command_approval");
  });
});

describe("GeminiAcpAdapter auth and resume helpers", () => {
  it("defaults to Google OAuth when no Gemini auth env is set", () => {
    assert.equal(resolveGeminiAuthType(undefined), AuthType.LOGIN_WITH_GOOGLE);
  });

  it("preserves env-selected Gemini auth modes", () => {
    assert.equal(resolveGeminiAuthType(AuthType.USE_GEMINI), AuthType.USE_GEMINI);
    assert.equal(resolveGeminiAuthType(AuthType.USE_VERTEX_AI), AuthType.USE_VERTEX_AI);
  });

  it("reads persisted Gemini resume state", () => {
    const resumeState = readGeminiResumeState({
      history: [
        { role: "user", parts: [{ text: "Hello" }] },
        { role: "model", parts: [{ text: "Hi" }] },
        { role: "user", parts: [{ text: "Again" }] },
        { role: "model", parts: [{ text: "Sure" }] },
        { role: "user", parts: [{ text: "Third" }] },
        { role: "model", parts: [{ text: "Done" }] },
      ],
      turnCount: 3,
      turnHistoryLengths: [2, 4, 6],
    });

    assert.deepStrictEqual(resumeState, {
      history: [
        { role: "user", parts: [{ text: "Hello" }] },
        { role: "model", parts: [{ text: "Hi" }] },
        { role: "user", parts: [{ text: "Again" }] },
        { role: "model", parts: [{ text: "Sure" }] },
        { role: "user", parts: [{ text: "Third" }] },
        { role: "model", parts: [{ text: "Done" }] },
      ],
      turnCount: 3,
      turnHistoryLengths: [2, 4, 6],
    });
  });

  it("infers Gemini turn history boundaries for legacy resume cursors", () => {
    const history = [
      { role: "user", parts: [{ text: "First prompt" }] },
      { role: "model", parts: [{ text: "Need to inspect files" }] },
      {
        role: "user",
        parts: [{ functionResponse: { id: "tool-1", name: "rg", response: { ok: true } } }],
      },
      { role: "model", parts: [{ text: "Inspection complete" }] },
      { role: "user", parts: [{ text: "Second prompt" }] },
      { role: "model", parts: [{ text: "Second answer" }] },
    ];

    assert.deepStrictEqual(inferGeminiTurnHistoryLengths(history), [4, 6]);
    assert.deepStrictEqual(
      readGeminiResumeState({
        history,
        turnCount: 2,
      }),
      {
        history,
        turnCount: 2,
        turnHistoryLengths: [4, 6],
      },
    );
  });

  it("rejects invalid Gemini resume cursors", () => {
    assert.equal(readGeminiResumeState(undefined), undefined);
    assert.equal(readGeminiResumeState({ turnCount: 2 }), undefined);
    assert.deepStrictEqual(readGeminiResumeState({ history: [], turnCount: -1 }), {
      history: [],
      turnCount: 0,
    });
  });

  it("reads proposed plans from Gemini CLI plan files", async () => {
    const dir = await mkdtemp(join(tmpdir(), "gemini-plan-"));
    try {
      const planPath = join(dir, "ship-it.md");
      await writeFile(planPath, "\n# Ship it\n\n- one\n- two\n", "utf8");

      assert.equal(await readGeminiPlanMarkdownFromFile(planPath), "# Ship it\n\n- one\n- two");
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it("builds the persisted Gemini runtime binding from the completed session state", () => {
    const binding = buildGeminiPersistedBinding({
      session: {
        provider: "geminiAcp",
        threadId: ThreadId.makeUnsafe("thread-1"),
        status: "ready",
        runtimeMode: "full-access",
        cwd: "/tmp/project",
        model: "gemini-3.1-pro-preview",
        resumeCursor: {
          history: [{ role: "user", parts: [{ text: "Hello" }] }],
          turnCount: 1,
          turnHistoryLengths: [1],
        },
        lastError: "previous failure",
        createdAt: "2026-04-09T17:00:00.000Z",
        updatedAt: "2026-04-09T17:05:00.000Z",
      },
      status: "running",
      lastRuntimeEvent: "turn.completed",
      lastRuntimeEventAt: "2026-04-09T17:05:30.000Z",
    });

    assert.deepStrictEqual(binding, {
      threadId: ThreadId.makeUnsafe("thread-1"),
      provider: "geminiAcp",
      runtimeMode: "full-access",
      status: "running",
      resumeCursor: {
        history: [{ role: "user", parts: [{ text: "Hello" }] }],
        turnCount: 1,
        turnHistoryLengths: [1],
      },
      runtimePayload: {
        cwd: "/tmp/project",
        model: "gemini-3.1-pro-preview",
        activeTurnId: null,
        lastError: "previous failure",
        lastRuntimeEvent: "turn.completed",
        lastRuntimeEventAt: "2026-04-09T17:05:30.000Z",
      },
    });
  });
});

describe("GeminiAcpAdapter retry warnings", () => {
  it("formats capacity retries for the UI", () => {
    assert.equal(
      formatGeminiRetryWarningMessage({
        attempt: 1,
        maxAttempts: 10,
        delayMs: 30_000,
        error: "MODEL_CAPACITY_EXHAUSTED",
      }),
      "Capacity exhausted, retrying in 30s (attempt 1/10)",
    );
  });

  it("formats generic retries for the UI", () => {
    assert.equal(
      formatGeminiRetryWarningMessage({
        attempt: 2,
        maxAttempts: 10,
        delayMs: 5_000,
        error: "ECONNRESET",
      }),
      "Request retrying in 5s (attempt 2/10)",
    );
  });

  it("formats structured Gemini thought summaries for reasoning output", () => {
    assert.equal(
      formatGeminiThoughtSummary({
        subject: "Plan",
        description: "Inspect the WebSocket session lifecycle first.",
      }),
      "Plan: Inspect the WebSocket session lifecycle first.",
    );
    assert.equal(
      formatGeminiThoughtSummary({
        subject: "",
        description: "Inspect the WebSocket session lifecycle first.",
      }),
      "Inspect the WebSocket session lifecycle first.",
    );
  });

  it("formats Gemini finish reasons into user-facing warnings", () => {
    assert.equal(
      formatGeminiFinishReasonMessage(FinishReason.MAX_TOKENS),
      "Response truncated due to token limits.",
    );
    assert.equal(formatGeminiFinishReasonMessage(FinishReason.STOP), undefined);
  });

  it("formats Gemini context compression and overflow warnings", () => {
    assert.equal(
      formatGeminiChatCompressionMessage({
        originalTokenCount: 820_000,
        newTokenCount: 510_000,
        limit: 1_000_000,
      }),
      "Context compressed from 82% to 51%.",
    );
    assert.equal(
      formatGeminiContextWindowOverflowMessage({
        estimatedRequestTokenCount: 120_000,
        remainingTokenCount: 75_000,
        limit: 1_000_000,
      }),
      "Sending this message (120000 tokens) might exceed the context window limit (75000 tokens left). Please reduce the request size or compress the conversation before retrying.",
    );
  });
});

describe("GeminiAcpAdapter ask-user helpers", () => {
  it("normalizes Gemini ask_user questions for the UI", () => {
    const questions = normalizeGeminiAskUserQuestions([
      {
        question: "Pick an environment",
        header: "Environment",
        type: QuestionType.CHOICE,
        options: [
          { label: "Staging", description: "Deploy to staging" },
          { label: "Production", description: "Deploy to production" },
        ],
      },
      {
        question: "Any extra notes?",
        type: QuestionType.TEXT,
        placeholder: "Type details",
      },
      {
        question: "Continue now?",
        type: QuestionType.YESNO,
      },
    ]);

    assert.deepStrictEqual(questions, [
      {
        id: "q-0",
        header: "Environment",
        question: "Pick an environment",
        options: [
          { label: "Staging", description: "Deploy to staging" },
          { label: "Production", description: "Deploy to production" },
        ],
      },
      {
        id: "q-1",
        header: "Question 2",
        question: "Any extra notes?",
        options: [{ label: "Use custom answer", description: "Type details" }],
      },
      {
        id: "q-2",
        header: "Question 3",
        question: "Continue now?",
        options: [
          { label: "Yes", description: "Yes" },
          { label: "No", description: "No" },
        ],
      },
    ]);
  });

  it("maps UI answers back to Gemini's index-based ask_user payload", () => {
    const questions = normalizeGeminiAskUserQuestions([
      {
        question: "Pick an environment",
        header: "Environment",
        type: QuestionType.CHOICE,
        options: [
          { label: "Staging", description: "Deploy to staging" },
          { label: "Production", description: "Deploy to production" },
        ],
      },
      {
        question: "Any extra notes?",
        type: QuestionType.TEXT,
      },
    ]);

    const answers = buildGeminiAskUserResponseAnswers({
      questions,
      answers: {
        "q-0": "Production",
        "q-1": "Ship after 5pm",
      },
    });

    assert.deepStrictEqual(answers, {
      "0": "Production",
      "1": "Ship after 5pm",
    });
  });
});

describe("GeminiAcpAdapter assistant history repair", () => {
  it("extracts only the appended suffix from cumulative Gemini text snapshots", () => {
    assert.deepStrictEqual(applyGeminiAssistantTextChunk("", "Ahoj"), {
      nextText: "Ahoj",
      delta: "Ahoj",
    });
    assert.deepStrictEqual(applyGeminiAssistantTextChunk("Ahoj", "Ahoj světe"), {
      nextText: "Ahoj světe",
      delta: " světe",
    });
    assert.deepStrictEqual(applyGeminiAssistantTextChunk("Ahoj světe", "Ahoj světe"), {
      nextText: "Ahoj světe",
      delta: "",
    });
  });

  it("still appends true Gemini delta chunks without losing text", () => {
    assert.deepStrictEqual(applyGeminiAssistantTextChunk("", "Ahoj"), {
      nextText: "Ahoj",
      delta: "Ahoj",
    });
    assert.deepStrictEqual(applyGeminiAssistantTextChunk("Ahoj", " světe"), {
      nextText: "Ahoj světe",
      delta: " světe",
    });
    assert.deepStrictEqual(applyGeminiAssistantTextChunk("Ahoj světe", "!"), {
      nextText: "Ahoj světe!",
      delta: "!",
    });
  });

  it("drops repeated overlapping Gemini chunks instead of stitching them into the UI", () => {
    assert.deepStrictEqual(applyGeminiAssistantTextChunk("Ahoj světe", "světe"), {
      nextText: "Ahoj světe",
      delta: "",
    });
    assert.deepStrictEqual(applyGeminiAssistantTextChunk("Ahoj svě", "ěte"), {
      nextText: "Ahoj světe",
      delta: "te",
    });
  });

  it("builds a model history entry from streamed no-tool text chunks", () => {
    const historyEntry = buildGeminiAssistantHistoryEntry([
      { text: "Ano, takhle " },
      { text: "je to lepší." },
    ]);

    assert.deepStrictEqual(historyEntry, {
      role: "model",
      parts: [{ text: "Ano, takhle je to lepší." }],
    });
  });

  it("returns undefined when there is no visible assistant content to persist", () => {
    const historyEntry = buildGeminiAssistantHistoryEntry([]);
    assert.equal(historyEntry, undefined);
  });
});

describe("GeminiAcpAdapter runtime session flow", () => {
  it("streams assistant lifecycle events across tool continuations without duplicating cumulative content", async () => {
    const harness = makeGeminiHarness();
    const scheduleSpy = vi
      .spyOn(GeminiToolSchedulerBridge.prototype, "scheduleToolCalls")
      .mockImplementation(function (this: { messageBus: MessageBus }) {
        const bus = this.messageBus;
        bus.emit(MessageBusType.TOOL_CALLS_UPDATE, {
          type: MessageBusType.TOOL_CALLS_UPDATE,
          schedulerId: ROOT_SCHEDULER_ID,
          toolCalls: [
            makeToolCallUpdate({
              callId: "tool-read-1",
              name: "read_file",
              args: { path: "README.md" },
              status: CoreToolCallStatus.Executing,
              liveOutput: "opening README",
              progressMessage: "Reading README.md",
            }),
          ],
        } as never);
        bus.emit(MessageBusType.TOOL_CALLS_UPDATE, {
          type: MessageBusType.TOOL_CALLS_UPDATE,
          schedulerId: ROOT_SCHEDULER_ID,
          toolCalls: [
            makeCompletedToolCall({
              callId: "tool-read-1",
              name: "read_file",
              args: { path: "README.md" },
              resultDisplay: "README loaded",
              responseParts: [
                {
                  functionResponse: {
                    id: "tool-read-1",
                    name: "read_file",
                    response: { ok: true },
                  },
                },
              ],
            }),
          ],
        } as never);

        const responseParts = [
          {
            functionResponse: {
              id: "tool-read-1",
              name: "read_file",
              response: { ok: true },
            },
          },
        ];
        return Promise.resolve({
          completedToolCalls: [
            makeCompletedToolCall({
              callId: "tool-read-1",
              name: "read_file",
              args: { path: "README.md" },
              resultDisplay: "README loaded",
              responseParts,
            }),
          ],
          responseParts,
          stopExecution: false,
          fatalError: undefined,
          allCancelled: false,
        });
      });

    harness.geminiClient.enqueueEvents([
      {
        type: GeminiEventType.Thought,
        value: {
          subject: "Plan",
          description: "Inspect the repo first.",
        },
      },
      {
        type: GeminiEventType.Content,
        value: "Hello",
      },
      {
        type: GeminiEventType.Content,
        value: "Hello world",
      },
      {
        type: GeminiEventType.ToolCallRequest,
        value: makeToolCallRequest({
          callId: "tool-read-1",
          name: "read_file",
          args: { path: "README.md" },
        }),
      },
    ] as Array<ServerGeminiStreamEvent>);
    harness.geminiClient.enqueueEvents([
      {
        type: GeminiEventType.Content,
        value: "After tool",
      },
      {
        type: GeminiEventType.Finished,
        value: {
          reason: FinishReason.STOP,
          usageMetadata: {
            promptTokenCount: 11,
            candidatesTokenCount: 7,
          },
        },
      },
    ] as Array<ServerGeminiStreamEvent>);

    await runWithGeminiHarnessPromise(harness, (adapter) =>
      Effect.gen(function* () {
        yield* startGeminiSession(adapter);
        yield* adapter.sendTurn({
          threadId: THREAD_ID,
          input: "Inspect and continue",
          attachments: [],
        });

        const events = yield* collectRuntimeEventsUntil(
          adapter,
          (event) => event.type === "turn.completed",
        );

        const assistantDeltas = events
          .filter(
            (event): event is Extract<ProviderRuntimeEvent, { type: "content.delta" }> =>
              event.type === "content.delta" && event.payload.streamKind === "assistant_text",
          )
          .map((event) => event.payload.delta);
        assert.deepStrictEqual(assistantDeltas, ["Hello", " world", "After tool"]);

        const reasoningDeltas = events
          .filter(
            (event): event is Extract<ProviderRuntimeEvent, { type: "content.delta" }> =>
              event.type === "content.delta" && event.payload.streamKind === "reasoning_text",
          )
          .map((event) => event.payload.delta);
        assert.deepStrictEqual(reasoningDeltas, ["Plan: Inspect the repo first."]);

        const assistantCompletions = events.filter(
          (event): event is Extract<ProviderRuntimeEvent, { type: "item.completed" }> =>
            event.type === "item.completed" && event.payload.itemType === "assistant_message",
        );
        assert.equal(assistantCompletions.length, 2);
        assert.equal(assistantCompletions[0]?.payload.detail, "Hello world");
        assert.equal(assistantCompletions[1]?.payload.detail, "After tool");

        const toolStarted = events.find(
          (event): event is Extract<ProviderRuntimeEvent, { type: "item.started" }> =>
            event.type === "item.started" && event.payload.itemType === "file_change",
        );
        assert.equal(toolStarted?.payload.detail, "read_file: README.md");

        const turnCompleted = events.find(
          (event): event is Extract<ProviderRuntimeEvent, { type: "turn.completed" }> =>
            event.type === "turn.completed",
        );
        assert.equal(turnCompleted?.payload.state, "completed");
        assert.equal(scheduleSpy.mock.calls.length, 1);
      }),
    );
  });

  it("reports Gemini context usage from the latest round instead of accumulating prior rounds", async () => {
    const harness = makeGeminiHarness();
    vi.spyOn(GeminiToolSchedulerBridge.prototype, "scheduleToolCalls").mockResolvedValue({
      completedToolCalls: [
        makeCompletedToolCall({
          callId: "tool-read-usage-1",
          name: "read_file",
          args: { path: "README.md" },
          resultDisplay: "README loaded",
          responseParts: [
            {
              functionResponse: {
                id: "tool-read-usage-1",
                name: "read_file",
                response: { ok: true },
              },
            },
          ],
        }),
      ],
      responseParts: [
        {
          functionResponse: {
            id: "tool-read-usage-1",
            name: "read_file",
            response: { ok: true },
          },
        },
      ],
      stopExecution: false,
      fatalError: undefined,
      allCancelled: false,
    });

    harness.geminiClient.enqueueEvents([
      {
        type: GeminiEventType.ToolCallRequest,
        value: makeToolCallRequest({
          callId: "tool-read-usage-1",
          name: "read_file",
          args: { path: "README.md" },
        }),
      },
      {
        type: GeminiEventType.Finished,
        value: {
          reason: FinishReason.STOP,
          usageMetadata: {
            promptTokenCount: 200,
            cachedContentTokenCount: 30,
            candidatesTokenCount: 40,
            toolUsePromptTokenCount: 20,
            thoughtsTokenCount: 50,
            totalTokenCount: 310,
          },
        },
      },
    ] as Array<ServerGeminiStreamEvent>);
    harness.geminiClient.enqueueEvents([
      {
        type: GeminiEventType.Content,
        value: "Done",
      },
      {
        type: GeminiEventType.Finished,
        value: {
          reason: FinishReason.STOP,
          usageMetadata: {
            promptTokenCount: 230,
            cachedContentTokenCount: 35,
            candidatesTokenCount: 8,
            toolUsePromptTokenCount: 18,
            thoughtsTokenCount: 60,
            totalTokenCount: 316,
          },
        },
      },
    ] as Array<ServerGeminiStreamEvent>);

    await runWithGeminiHarnessPromise(harness, (adapter) =>
      Effect.gen(function* () {
        yield* startGeminiSession(adapter);
        yield* adapter.sendTurn({
          threadId: THREAD_ID,
          input: "Inspect and continue",
          attachments: [],
        });

        const events = yield* collectRuntimeEventsUntil(
          adapter,
          (event) => event.type === "turn.completed",
        );

        const usageEvent = events.find(
          (event): event is Extract<ProviderRuntimeEvent, { type: "thread.token-usage.updated" }> =>
            event.type === "thread.token-usage.updated",
        );
        assert.equal(usageEvent?.type, "thread.token-usage.updated");
        if (usageEvent?.type !== "thread.token-usage.updated") {
          return;
        }

        assert.equal(
          (usageEvent.payload.usage.maxTokens ?? 0) > usageEvent.payload.usage.usedTokens,
          true,
        );
        const { maxTokens: _maxTokens, ...usageWithoutMaxTokens } = usageEvent.payload.usage;
        assert.deepStrictEqual(usageWithoutMaxTokens, {
          usedTokens: 256,
          totalProcessedTokens: 626,
          inputTokens: 248,
          cachedInputTokens: 35,
          outputTokens: 8,
          reasoningOutputTokens: 60,
          lastUsedTokens: 256,
          lastInputTokens: 248,
          lastCachedInputTokens: 35,
          lastOutputTokens: 8,
          lastReasoningOutputTokens: 60,
        });
      }),
    );
  });

  it("routes tool confirmations through request.opened and request.resolved", async () => {
    const harness = makeGeminiHarness();
    let confirmationResponse: unknown | undefined;
    let preflightResponse: unknown | undefined;
    vi.spyOn(GeminiToolSchedulerBridge.prototype, "scheduleToolCalls").mockImplementation(
      function (this: { messageBus: MessageBus }) {
        const bus = this.messageBus;
        const correlationId = "tool-confirm-1";
        const policyCheckCorrelationId = "tool-policy-check-1";
        const listener = (message: unknown) => {
          const candidate = message as { correlationId?: string };
          if (candidate.correlationId === policyCheckCorrelationId) {
            preflightResponse = message;
            return;
          }
          if (candidate.correlationId !== correlationId) {
            return;
          }
          bus.unsubscribe(MessageBusType.TOOL_CONFIRMATION_RESPONSE, listener as never);
          confirmationResponse = message;
          bus.emit(MessageBusType.TOOL_CALLS_UPDATE, {
            type: MessageBusType.TOOL_CALLS_UPDATE,
            schedulerId: ROOT_SCHEDULER_ID,
            toolCalls: [
              makeCompletedToolCall({
                callId: "tool-confirmed-1",
                name: "run_shell_command",
                args: { command: "pwd" },
                responseParts: [
                  {
                    functionResponse: {
                      id: "tool-confirmed-1",
                      name: "run_shell_command",
                      response: { ok: true },
                    },
                  },
                ],
              }),
            ],
          } as never);
        };
        bus.subscribe(MessageBusType.TOOL_CONFIRMATION_RESPONSE, listener as never);
        bus.emit(MessageBusType.TOOL_CONFIRMATION_REQUEST, {
          type: MessageBusType.TOOL_CONFIRMATION_REQUEST,
          correlationId: policyCheckCorrelationId,
          toolCall: {
            name: "run_shell_command",
            args: {
              command: "pwd",
            },
          },
        } as never);
        bus.emit(MessageBusType.TOOL_CALLS_UPDATE, {
          type: MessageBusType.TOOL_CALLS_UPDATE,
          schedulerId: ROOT_SCHEDULER_ID,
          toolCalls: [
            makeToolCallUpdate({
              callId: "tool-confirmed-1",
              name: "run_shell_command",
              args: { command: "pwd" },
              status: CoreToolCallStatus.AwaitingApproval,
              correlationId,
              confirmationDetails: {
                type: "exec",
                command: "pwd",
              },
            }),
          ],
        } as never);

        return new Promise((resolve) => {
          const waitForApproval = () => {
            if (confirmationResponse === undefined) {
              setTimeout(waitForApproval, 10);
              return;
            }
            resolve({
              completedToolCalls: [
                makeCompletedToolCall({
                  callId: "tool-confirmed-1",
                  name: "run_shell_command",
                  args: { command: "pwd" },
                  responseParts: [
                    {
                      functionResponse: {
                        id: "tool-confirmed-1",
                        name: "run_shell_command",
                        response: { ok: true },
                      },
                    },
                  ],
                }),
              ],
              responseParts: [
                {
                  functionResponse: {
                    id: "tool-confirmed-1",
                    name: "run_shell_command",
                    response: { ok: true },
                  },
                },
              ],
              stopExecution: false,
              fatalError: undefined,
              allCancelled: false,
            });
          };
          waitForApproval();
        });
      },
    );

    harness.geminiClient.enqueueEvents([
      {
        type: GeminiEventType.ToolCallRequest,
        value: makeToolCallRequest({
          callId: "tool-confirmed-1",
          name: "run_shell_command",
          args: { command: "pwd" },
        }),
      },
    ] as Array<ServerGeminiStreamEvent>);

    await runWithGeminiHarnessPromise(harness, (adapter) =>
      Effect.gen(function* () {
        const session = yield* startGeminiSession(adapter, {
          runtimeMode: "approval-required",
        });
        yield* adapter.sendTurn({
          threadId: session.threadId,
          input: "Run pwd",
          attachments: [],
        });

        const requested = yield* waitForRuntimeEvent(
          adapter,
          (event) => event.type === "request.opened",
        );
        assert.equal(requested.type, "request.opened");
        if (requested.type !== "request.opened") {
          return;
        }
        assert.equal(requested.payload.requestType, "exec_command_approval");
        assert.equal(requested.payload.detail, "pwd");

        yield* adapter.respondToRequest(
          session.threadId,
          ApprovalRequestId.make(String(requested.requestId)),
          "accept",
        );

        const resolved = yield* waitForRuntimeEvent(
          adapter,
          (event) => event.type === "request.resolved",
        );
        assert.equal(resolved.type, "request.resolved");
        if (resolved.type !== "request.resolved") {
          return;
        }
        assert.equal(resolved.payload.decision, "accept");

        const response = confirmationResponse as
          | {
              confirmed?: boolean;
              outcome?: unknown;
            }
          | undefined;
        assert.equal(typeof response, "object");
        assert.equal(response?.confirmed, true);
        assert.equal(response?.outcome, ToolConfirmationOutcome.ProceedOnce);

        const policyCheckResponse = preflightResponse as
          | {
              confirmed?: boolean;
              requiresUserConfirmation?: boolean;
            }
          | undefined;
        assert.equal(typeof policyCheckResponse, "object");
        assert.equal(policyCheckResponse?.confirmed, false);
        assert.equal(policyCheckResponse?.requiresUserConfirmation, true);

        const completed = yield* waitForRuntimeEvent(
          adapter,
          (event) => event.type === "turn.completed",
        );
        assert.equal(completed.type, "turn.completed");
        if (completed.type !== "turn.completed") {
          return;
        }
        assert.equal(completed.payload.state, "completed");
      }),
    );
  });

  it("routes Gemini ask-user confirmations through user-input events and indexed answers", async () => {
    const harness = makeGeminiHarness();
    let askUserResponse: unknown | undefined;
    vi.spyOn(GeminiToolSchedulerBridge.prototype, "scheduleToolCalls").mockImplementation(
      function (this: { messageBus: MessageBus }) {
        const bus = this.messageBus;
        const correlationId = "tool-ask-user-1";
        const listener = (message: unknown) => {
          const candidate = message as { correlationId?: string };
          if (candidate.correlationId !== correlationId) {
            return;
          }
          bus.unsubscribe(MessageBusType.TOOL_CONFIRMATION_RESPONSE, listener as never);
          askUserResponse = message;
        };
        bus.subscribe(MessageBusType.TOOL_CONFIRMATION_RESPONSE, listener as never);
        bus.emit(MessageBusType.TOOL_CONFIRMATION_REQUEST, {
          type: MessageBusType.TOOL_CONFIRMATION_REQUEST,
          correlationId,
          toolCall: {
            name: "deploy_app",
            args: {},
          },
          details: {
            type: "ask_user",
            questions: [
              {
                question: "Pick an environment",
                header: "Environment",
                type: QuestionType.CHOICE,
                options: [
                  {
                    label: "Staging",
                    description: "Deploy to staging",
                  },
                  {
                    label: "Production",
                    description: "Deploy to production",
                  },
                ],
              },
              {
                question: "Any extra notes?",
                type: QuestionType.TEXT,
                placeholder: "Type details",
              },
            ],
          },
        } as never);

        return new Promise((resolve) => {
          setTimeout(resolve, 200);
        }).then(() => ({
          completedToolCalls: [
            makeCompletedToolCall({
              callId: "tool-ask-user-done",
              name: "deploy_app",
              responseParts: [
                {
                  functionResponse: {
                    id: "tool-ask-user-done",
                    name: "deploy_app",
                    response: { ok: true },
                  },
                },
              ],
            }),
          ],
          responseParts: [
            {
              functionResponse: {
                id: "tool-ask-user-done",
                name: "deploy_app",
                response: { ok: true },
              },
            },
          ],
          stopExecution: false,
          fatalError: undefined,
          allCancelled: false,
        }));
      },
    );

    harness.geminiClient.enqueueEvents([
      {
        type: GeminiEventType.ToolCallRequest,
        value: makeToolCallRequest({
          callId: "tool-ask-user-done",
          name: "deploy_app",
        }),
      },
    ] as Array<ServerGeminiStreamEvent>);

    await runWithGeminiHarnessPromise(harness, (adapter) =>
      Effect.gen(function* () {
        const session = yield* startGeminiSession(adapter, {
          runtimeMode: "approval-required",
        });
        yield* adapter.sendTurn({
          threadId: session.threadId,
          input: "Deploy it",
          attachments: [],
        });

        const requested = yield* waitForRuntimeEvent(
          adapter,
          (event) => event.type === "user-input.requested",
        );
        assert.equal(requested.type, "user-input.requested");
        if (requested.type !== "user-input.requested") {
          return;
        }
        assert.equal(requested.payload.questions[0]?.question, "Pick an environment");
        assert.equal(requested.payload.questions[0]?.id, "q-0");
        assert.equal(requested.payload.questions[1]?.id, "q-1");

        yield* adapter.respondToUserInput(
          session.threadId,
          ApprovalRequestId.make(String(requested.requestId)),
          {
            "q-0": "Production",
            "q-1": "Ship after 5pm",
          },
        );

        const resolved = yield* waitForRuntimeEvent(
          adapter,
          (event) => event.type === "user-input.resolved",
        );
        assert.equal(resolved.type, "user-input.resolved");
        if (resolved.type !== "user-input.resolved") {
          return;
        }
        assert.deepStrictEqual(resolved.payload.answers, {
          "q-0": "Production",
          "q-1": "Ship after 5pm",
        });

        const response = askUserResponse as
          | {
              confirmed?: boolean;
              payload?: {
                answers?: Record<string, string>;
              };
            }
          | undefined;
        assert.equal(typeof response, "object");
        assert.equal(response?.confirmed, true);
        assert.deepStrictEqual(response?.payload?.answers, {
          "0": "Production",
          "1": "Ship after 5pm",
        });

        const completed = yield* waitForRuntimeEvent(
          adapter,
          (event) => event.type === "turn.completed",
        );
        assert.equal(completed.type, "turn.completed");
        if (completed.type !== "turn.completed") {
          return;
        }
        assert.equal(completed.payload.state, "completed");
      }),
    );
  });

  it("emits retry warnings, model reroutes, and runtime warnings during an active turn", async () => {
    const harness = makeGeminiHarness({
      model: "gemini-2.5-pro",
    });
    const continueStream = makeDeferred<void>();
    harness.geminiClient.enqueueFactory(async function* () {
      yield {
        type: GeminiEventType.ModelInfo,
        value: "gemini-2.5-flash",
      } as ServerGeminiStreamEvent;
      await continueStream.promise;
      yield {
        type: GeminiEventType.ChatCompressed,
        value: {
          originalTokenCount: 820_000,
          newTokenCount: 510_000,
        },
      } as ServerGeminiStreamEvent;
      yield {
        type: GeminiEventType.Finished,
        value: {
          reason: FinishReason.STOP,
          usageMetadata: {
            promptTokenCount: 9,
            candidatesTokenCount: 4,
          },
        },
      } as ServerGeminiStreamEvent;
    });

    await runWithGeminiHarnessPromise(harness, (adapter) =>
      Effect.gen(function* () {
        yield* startGeminiSession(adapter, {
          model: "gemini-2.5-pro",
        });
        yield* adapter.sendTurn({
          threadId: THREAD_ID,
          input: "Hello",
          attachments: [],
        });

        const rerouted = yield* waitForRuntimeEvent(
          adapter,
          (event) => event.type === "model.rerouted",
        );
        assert.equal(rerouted.type, "model.rerouted");
        if (rerouted.type !== "model.rerouted") {
          return;
        }
        assert.equal(rerouted.payload.fromModel, "gemini-2.5-pro");
        assert.equal(rerouted.payload.toModel, "gemini-2.5-flash");

        coreEvents.emit(CoreEvent.RetryAttempt, {
          attempt: 1,
          maxAttempts: 5,
          delayMs: 30_000,
          error: "MODEL_CAPACITY_EXHAUSTED",
          model: "gemini-2.5-flash",
        });

        continueStream.resolve();

        const events = yield* collectRuntimeEventsUntil(
          adapter,
          (event) => event.type === "turn.completed",
        );
        const retryWarning = events.find(
          (event): event is Extract<ProviderRuntimeEvent, { type: "runtime.warning" }> =>
            event.type === "runtime.warning" &&
            event.payload.message.includes("Capacity exhausted"),
        );
        assert.equal(retryWarning?.type, "runtime.warning");

        const compressed = events.find(
          (event): event is Extract<ProviderRuntimeEvent, { type: "runtime.warning" }> =>
            event.type === "runtime.warning" &&
            event.payload.message.includes("Context compressed"),
        );
        assert.equal(compressed?.type, "runtime.warning");
        assert.equal(typeof compressed?.raw, "object");

        const completed = events.find(
          (event): event is Extract<ProviderRuntimeEvent, { type: "turn.completed" }> =>
            event.type === "turn.completed",
        );
        assert.equal(completed?.type, "turn.completed");
        if (!completed) {
          return;
        }
        assert.equal(completed.payload.state, "completed");
      }),
    );
  });

  it("captures exit_plan_mode files as turn.proposed.completed events and stops the plan turn immediately", async () => {
    const harness = makeGeminiHarness();
    const tempDir = await mkdtemp(join(tmpdir(), "gemini-plan-runtime-"));
    try {
      await runWithGeminiHarnessPromise(harness, (adapter) =>
        Effect.gen(function* () {
          const planPath = join(tempDir, "approved-plan.md");
          yield* Effect.promise(() =>
            writeFile(planPath, "\n# Release Plan\n\n- inspect\n- ship\n", "utf8"),
          );

          vi.spyOn(Scheduler.prototype, "schedule").mockImplementation(
            function (this: { messageBus: MessageBus }) {
              const bus = this.messageBus;
              const completedPlanToolCall = makeCompletedToolCall({
                callId: "tool-plan-1",
                name: "exit_plan_mode",
                resultDisplay: "Plan approved",
                responseParts: [
                  {
                    functionResponse: {
                      id: "tool-plan-1",
                      name: "exit_plan_mode",
                      response: { ok: true },
                    },
                  },
                ],
              });

              bus.emit(MessageBusType.TOOL_CALLS_UPDATE, {
                type: MessageBusType.TOOL_CALLS_UPDATE,
                schedulerId: ROOT_SCHEDULER_ID,
                toolCalls: [
                  makeToolCallUpdate({
                    callId: "tool-plan-1",
                    name: "exit_plan_mode",
                    status: CoreToolCallStatus.AwaitingApproval,
                    confirmationDetails: {
                      type: "exit_plan_mode",
                      planPath,
                    },
                  }),
                ],
              } as never);
              bus.emit(MessageBusType.TOOL_CALLS_UPDATE, {
                type: MessageBusType.TOOL_CALLS_UPDATE,
                schedulerId: ROOT_SCHEDULER_ID,
                toolCalls: [completedPlanToolCall],
              } as never);

              return Promise.resolve([completedPlanToolCall]);
            },
          );

          harness.geminiClient.enqueueEvents([
            {
              type: GeminiEventType.ToolCallRequest,
              value: makeToolCallRequest({
                callId: "tool-plan-1",
                name: "exit_plan_mode",
              }),
            },
          ] as Array<ServerGeminiStreamEvent>);
          harness.geminiClient.enqueueEvents([
            {
              type: GeminiEventType.Content,
              value: "This should never stream after the plan is presented.",
            },
            {
              type: GeminiEventType.Finished,
              value: {
                reason: FinishReason.STOP,
                usageMetadata: {
                  promptTokenCount: 5,
                  candidatesTokenCount: 3,
                },
              },
            },
          ] as Array<ServerGeminiStreamEvent>);

          yield* startGeminiSession(adapter);
          yield* adapter.sendTurn({
            threadId: THREAD_ID,
            input: "Plan it",
            attachments: [],
            interactionMode: "plan",
          });

          const events = yield* collectRuntimeEventsUntil(
            adapter,
            (event) => event.type === "turn.completed",
          );
          const planEvent = events.find(
            (event): event is Extract<ProviderRuntimeEvent, { type: "turn.proposed.completed" }> =>
              event.type === "turn.proposed.completed",
          );

          assert.equal(planEvent?.type, "turn.proposed.completed");
          if (planEvent?.type !== "turn.proposed.completed") {
            return;
          }
          assert.equal(planEvent.payload.planMarkdown, "# Release Plan\n\n- inspect\n- ship");

          const assistantPlanMessage = events.find(
            (event): event is Extract<ProviderRuntimeEvent, { type: "item.completed" }> =>
              event.type === "item.completed" &&
              event.payload.itemType === "assistant_message" &&
              event.payload.detail === "# Release Plan\n\n- inspect\n- ship",
          );
          assert.equal(assistantPlanMessage, undefined);

          const approvalRequested = events.find(
            (event): event is Extract<ProviderRuntimeEvent, { type: "request.opened" }> =>
              event.type === "request.opened",
          );
          assert.equal(approvalRequested, undefined);

          const leakedAssistantDelta = events.find(
            (event): event is Extract<ProviderRuntimeEvent, { type: "content.delta" }> =>
              event.type === "content.delta" &&
              event.payload.streamKind === "assistant_text" &&
              event.payload.delta.includes("This should never stream"),
          );
          assert.equal(leakedAssistantDelta, undefined);
          assert.equal(harness.geminiClient.sendCalls.length, 1);
        }),
      );
    } finally {
      await rm(tempDir, {
        recursive: true,
        force: true,
      });
    }
  });

  it("captures exit_plan_mode files from plan_filename even when no planPath is surfaced", async () => {
    const harness = makeGeminiHarness();
    const plansDir = await mkdtemp(join(tmpdir(), "gemini-plan-fallback-"));
    vi.spyOn(harness.config.storage, "getPlansDir").mockReturnValue(plansDir);

    const planPath = join(plansDir, "denied-plan.md");
    await writeFile(planPath, "\n# Denied Plan\n\n- review\n- approve\n", "utf8");

    let capturedPlan: string | undefined;
    const bridge = new GeminiToolSchedulerBridge({
      config: harness.config as unknown as Config,
      geminiClient: harness.geminiClient as unknown as GeminiClient,
      getPreferredEditor: () => undefined,
      context: {
        config: harness.config as unknown as Config,
        promptId: "prompt-test",
        toolRegistry: harness.config.getToolRegistry(),
        promptRegistry: harness.config.getPromptRegistry(),
        resourceRegistry: harness.config.getResourceRegistry(),
        messageBus: harness.messageBus,
        geminiClient: harness.geminiClient as unknown as GeminiClient,
        sandboxManager: harness.config.sandboxManager,
      },
      callbacks: {
        emitEvent: () => undefined,
        makeEventBase: () => ({
          eventId: "evt-test-plan-fallback" as never,
          provider: "geminiAcp",
          threadId: THREAD_ID,
          createdAt: new Date().toISOString(),
          providerRefs: {},
        }),
        finalizeAssistantSegment: () => undefined,
        onPlanCaptured: (planMarkdown) => {
          capturedPlan = planMarkdown;
        },
        onApprovalRequested: () => undefined,
      },
    });

    try {
      vi.spyOn(Scheduler.prototype, "schedule").mockResolvedValue([
        makeCompletedToolCall({
          callId: "tool-plan-denied-1",
          name: "exit_plan_mode",
          args: { plan_filename: "denied-plan.md" },
          status: CoreToolCallStatus.Cancelled,
          resultDisplay: 'Tool execution for "Exit Plan Mode" denied by policy.',
          responseParts: [],
        }),
      ]);

      await bridge.scheduleToolCalls(
        [
          makeToolCallRequest({
            callId: "tool-plan-denied-1",
            name: "exit_plan_mode",
            args: { plan_filename: "denied-plan.md" },
          }),
        ],
        new AbortController().signal,
      );

      assert.equal(capturedPlan, "# Denied Plan\n\n- review\n- approve");
    } finally {
      bridge.dispose();
      await rm(plansDir, {
        recursive: true,
        force: true,
      });
    }
  });

  it("defers exit_plan_mode auto-approval until Gemini is ready to receive it", async () => {
    const harness = makeGeminiHarness();
    const plansDir = await mkdtemp(join(tmpdir(), "gemini-plan-auto-approve-"));
    const planPath = join(plansDir, "approved-plan.md");
    vi.spyOn(harness.config.storage, "getPlansDir").mockReturnValue(plansDir);

    try {
      await writeFile(planPath, "\n# Approved Plan\n\n- inspect\n- refine\n", "utf8");

      await runWithGeminiHarnessPromise(harness, (adapter) =>
        Effect.gen(function* () {
          yield* startGeminiSession(adapter);
          const responseDeferred = makeDeferred<{
            correlationId?: string;
            confirmed?: boolean;
            outcome?: ToolConfirmationOutcome;
          }>();

          harness.messageBus.emit(MessageBusType.TOOL_CALLS_UPDATE, {
            type: MessageBusType.TOOL_CALLS_UPDATE,
            schedulerId: ROOT_SCHEDULER_ID,
            toolCalls: [
              makeToolCallUpdate({
                callId: "tool-plan-auto-approve",
                name: "exit_plan_mode",
                args: { plan_filename: "approved-plan.md" },
                status: CoreToolCallStatus.AwaitingApproval,
                correlationId: "corr-plan-auto-approve",
                confirmationDetails: {
                  type: "exit_plan_mode",
                  planPath,
                },
              }),
            ],
          } as never);

          const listener = (message: unknown) => {
            const candidate = message as {
              correlationId?: string;
              confirmed?: boolean;
              outcome?: ToolConfirmationOutcome;
            };
            if (candidate.correlationId !== "corr-plan-auto-approve") {
              return;
            }
            harness.messageBus.unsubscribe(
              MessageBusType.TOOL_CONFIRMATION_RESPONSE,
              listener as never,
            );
            responseDeferred.resolve(candidate);
          };
          harness.messageBus.subscribe(
            MessageBusType.TOOL_CONFIRMATION_RESPONSE,
            listener as never,
          );

          const response = yield* Effect.promise(() =>
            Promise.race([
              responseDeferred.promise,
              new Promise<never>((_, reject) => {
                setTimeout(() => {
                  harness.messageBus.unsubscribe(
                    MessageBusType.TOOL_CONFIRMATION_RESPONSE,
                    listener as never,
                  );
                  reject(new Error("Timed out waiting for deferred exit_plan_mode approval."));
                }, 100);
              }),
            ]),
          );

          assert.equal(response.correlationId, "corr-plan-auto-approve");
          assert.equal(response.confirmed, true);
          assert.equal(response.outcome, ToolConfirmationOutcome.ProceedOnce);
        }),
      );
    } finally {
      await rm(plansDir, {
        recursive: true,
        force: true,
      });
    }
  });

  it("preserves resume cursors and rolls back using persisted Gemini history boundaries", async () => {
    const harness = makeGeminiHarness();
    const history = [
      { role: "user", parts: [{ text: "First prompt" }] },
      { role: "model", parts: [{ text: "First answer" }] },
      { role: "user", parts: [{ text: "Second prompt" }] },
      { role: "model", parts: [{ text: "Second answer" }] },
    ];

    await runWithGeminiHarnessPromise(harness, (adapter) =>
      Effect.gen(function* () {
        const session = yield* startGeminiSession(adapter, {
          resumeCursor: {
            history,
            turnCount: 2,
            turnHistoryLengths: [2, 4],
          },
        });

        assert.equal(harness.geminiClient.resumeChatCalls.length, 1);
        assert.deepStrictEqual(harness.geminiClient.resumeChatCalls[0], history);

        const thread = yield* adapter.readThread(session.threadId);
        assert.equal(thread.turns.length, 2);

        const initialResumeState = readGeminiResumeState(session.resumeCursor);
        assert.equal(initialResumeState?.turnCount, 2);
        assert.deepStrictEqual(initialResumeState?.turnHistoryLengths, [2, 4]);

        const rollback = yield* adapter.rollbackThread(session.threadId, 1);
        assert.equal(rollback.turns.length, 1);
        assert.equal(harness.geminiClient.getHistory().length, 2);

        const resumeState = readGeminiResumeState(session.resumeCursor);
        assert.equal(resumeState?.turnCount, 1);
        assert.deepStrictEqual(resumeState?.turnHistoryLengths, [2]);
      }),
    );
  });

  it("writes raw Gemini native events when native logging is enabled", async () => {
    const nativeEvents: Array<{
      event?: {
        provider?: string;
        method?: string;
        providerThreadId?: string;
      };
    }> = [];
    const harness = makeGeminiHarness({
      nativeEventLogger: {
        filePath: "memory://gemini-native-events",
        write: (event) => {
          nativeEvents.push(event as (typeof nativeEvents)[number]);
          return Effect.void;
        },
        close: () => Effect.void,
      },
    });

    harness.geminiClient.enqueueEvents([
      {
        type: GeminiEventType.Thought,
        value: {
          subject: "Plan",
          description: "Warm up",
        },
      },
      {
        type: GeminiEventType.Content,
        value: "hello",
      },
      {
        type: GeminiEventType.Finished,
        value: {
          reason: FinishReason.STOP,
          usageMetadata: {
            promptTokenCount: 2,
            candidatesTokenCount: 1,
          },
        },
      },
    ] as Array<ServerGeminiStreamEvent>);

    await runWithGeminiHarnessPromise(harness, (adapter) =>
      Effect.gen(function* () {
        yield* startGeminiSession(adapter);
        yield* adapter.sendTurn({
          threadId: THREAD_ID,
          input: "hello",
          attachments: [],
        });

        yield* waitForRuntimeEvent(adapter, (event) => event.type === "turn.completed");

        const methods = nativeEvents.map((record) => record.event?.method ?? "");
        assert.equal(
          methods.some((method) => method.includes(String(GeminiEventType.Thought))),
          true,
        );
        assert.equal(
          methods.some((method) => method.includes(String(GeminiEventType.Content))),
          true,
        );
        assert.equal(
          methods.some((method) => method.includes(String(GeminiEventType.Finished))),
          true,
        );
        assert.equal(nativeEvents[0]?.event?.provider, "geminiAcp");
        assert.equal(nativeEvents[0]?.event?.providerThreadId, String(THREAD_ID));
      }),
    );
  });

  it("completes the turn as failed when the Gemini stream throws", async () => {
    const harness = makeGeminiHarness();
    harness.geminiClient.enqueueFactory(() => ({
      [Symbol.asyncIterator]() {
        return {
          next: async () => {
            throw new Error("kaboom");
          },
        };
      },
    }));

    await runWithGeminiHarnessPromise(harness, (adapter) =>
      Effect.gen(function* () {
        const session = yield* startGeminiSession(adapter);
        yield* adapter.sendTurn({
          threadId: session.threadId,
          input: "Trigger a stream failure",
          attachments: [],
        });

        const completed = yield* waitForRuntimeEvent(
          adapter,
          (event): event is Extract<ProviderRuntimeEvent, { type: "turn.completed" }> =>
            event.type === "turn.completed",
        );
        assert.equal(completed.type, "turn.completed");
        if (completed.type !== "turn.completed") {
          return;
        }

        assert.equal(completed.payload.state, "failed");
        assert.equal(completed.payload.stopReason, "error");
        assert.equal(completed.payload.errorMessage, "kaboom");
      }),
    );
  });

  it("cancels pending ask-user interactions and completes the turn as interrupted", async () => {
    const harness = makeGeminiHarness();
    let askUserResponse: unknown | undefined;
    vi.spyOn(GeminiToolSchedulerBridge.prototype, "scheduleToolCalls").mockImplementation(
      function (this: { messageBus: MessageBus }) {
        const bus = this.messageBus;
        const correlationId = "tool-ask-user-cancel";
        const listener = (message: unknown) => {
          const candidate = message as { correlationId?: string };
          if (candidate.correlationId !== correlationId) {
            return;
          }
          bus.unsubscribe(MessageBusType.ASK_USER_RESPONSE, listener as never);
          askUserResponse = message;
        };
        bus.subscribe(MessageBusType.ASK_USER_RESPONSE, listener as never);
        bus.emit(MessageBusType.ASK_USER_REQUEST, {
          type: MessageBusType.ASK_USER_REQUEST,
          correlationId,
          questions: [
            {
              question: "Continue?",
              type: QuestionType.YESNO,
            },
          ],
        } as never);

        return new Promise((resolve) => {
          setTimeout(resolve, 200);
        }).then(() => ({
          completedToolCalls: [
            makeCompletedToolCall({
              callId: "tool-cancelled-1",
              name: "ask_user_tool",
              status: CoreToolCallStatus.Cancelled,
              responseParts: [],
            }),
          ],
          responseParts: [],
          stopExecution: false,
          fatalError: undefined,
          allCancelled: true,
        }));
      },
    );

    harness.geminiClient.enqueueEvents([
      {
        type: GeminiEventType.ToolCallRequest,
        value: makeToolCallRequest({
          callId: "tool-cancelled-1",
          name: "ask_user_tool",
        }),
      },
    ] as Array<ServerGeminiStreamEvent>);

    await runWithGeminiHarnessPromise(harness, (adapter) =>
      Effect.gen(function* () {
        const session = yield* startGeminiSession(adapter);
        yield* adapter.sendTurn({
          threadId: session.threadId,
          input: "Ask the user",
          attachments: [],
        });

        const requested = yield* waitForRuntimeEvent(
          adapter,
          (event) => event.type === "user-input.requested",
        );
        assert.equal(requested.type, "user-input.requested");

        yield* adapter.interruptTurn(session.threadId);

        const postInterruptEvents = yield* collectRuntimeEventsUntil(
          adapter,
          (event) => event.type === "turn.completed",
        );
        const resolved = postInterruptEvents.find(
          (event): event is Extract<ProviderRuntimeEvent, { type: "user-input.resolved" }> =>
            event.type === "user-input.resolved",
        );
        assert.equal(resolved?.type, "user-input.resolved");
        if (!resolved || resolved.type !== "user-input.resolved") {
          return;
        }
        assert.deepStrictEqual(resolved.payload.answers, {});

        const response = askUserResponse as
          | {
              cancelled?: boolean;
              answers?: Record<string, string>;
            }
          | undefined;
        assert.equal(typeof response, "object");
        assert.equal(response?.cancelled, true);
        assert.deepStrictEqual(response?.answers, {});

        const completed = postInterruptEvents.find(
          (event): event is Extract<ProviderRuntimeEvent, { type: "turn.completed" }> =>
            event.type === "turn.completed",
        );
        assert.equal(completed?.type, "turn.completed");
        if (!completed || completed.type !== "turn.completed") {
          return;
        }
        assert.equal(completed.payload.state, "interrupted");
      }),
    );
  });

  it("emits subagent activity updates for agent-style tool calls", async () => {
    const harness = makeGeminiHarness();
    vi.spyOn(GeminiToolSchedulerBridge.prototype, "scheduleToolCalls").mockImplementation(
      function (this: { messageBus: MessageBus }) {
        const bus = this.messageBus;
        bus.emit(MessageBusType.TOOL_CALLS_UPDATE, {
          type: MessageBusType.TOOL_CALLS_UPDATE,
          schedulerId: ROOT_SCHEDULER_ID,
          toolCalls: [
            makeToolCallUpdate({
              callId: "tool-agent-1",
              name: "spawn_agent",
              args: {
                agentName: "reviewer",
              },
              status: CoreToolCallStatus.Executing,
              displayName: "Spawn agent",
            }),
          ],
        } as never);
        bus.emit(MessageBusType.SUBAGENT_ACTIVITY, {
          type: MessageBusType.SUBAGENT_ACTIVITY,
          subagentName: "reviewer",
          activity: {
            type: "thought",
            content: "Scanning the websocket code path.",
            status: "running",
          },
        } as never);
        bus.emit(MessageBusType.TOOL_CALLS_UPDATE, {
          type: MessageBusType.TOOL_CALLS_UPDATE,
          schedulerId: ROOT_SCHEDULER_ID,
          toolCalls: [
            makeCompletedToolCall({
              callId: "tool-agent-1",
              name: "spawn_agent",
              args: {
                agentName: "reviewer",
              },
              resultDisplay: "Agent finished",
              responseParts: [],
            }),
          ],
        } as never);

        return Promise.resolve({
          completedToolCalls: [
            makeCompletedToolCall({
              callId: "tool-agent-1",
              name: "spawn_agent",
              args: {
                agentName: "reviewer",
              },
              resultDisplay: "Agent finished",
              responseParts: [],
            }),
          ],
          responseParts: [],
          stopExecution: false,
          fatalError: undefined,
          allCancelled: false,
        });
      },
    );

    harness.geminiClient.enqueueEvents([
      {
        type: GeminiEventType.ToolCallRequest,
        value: makeToolCallRequest({
          callId: "tool-agent-1",
          name: "spawn_agent",
          args: {
            agentName: "reviewer",
          },
        }),
      },
    ] as Array<ServerGeminiStreamEvent>);

    await runWithGeminiHarnessPromise(harness, (adapter) =>
      Effect.gen(function* () {
        yield* startGeminiSession(adapter);
        yield* adapter.sendTurn({
          threadId: THREAD_ID,
          input: "Delegate it",
          attachments: [],
        });

        const events = yield* collectRuntimeEventsUntil(
          adapter,
          (event) => event.type === "turn.completed",
        );
        const update = events.find(
          (event): event is Extract<ProviderRuntimeEvent, { type: "item.updated" }> =>
            event.type === "item.updated" &&
            event.payload.itemType === "collab_agent_tool_call" &&
            (event.payload.detail ?? "").includes("reviewer:"),
        );

        assert.equal(
          update?.payload.detail ?? undefined,
          "reviewer: Scanning the websocket code path.",
        );
      }),
    );
  });
});
