import * as NodeServices from "@effect/platform-node/NodeServices";
import { ProviderRuntimeEvent, ThreadId } from "@t3tools/contracts";
import { assert, describe, it } from "@effect/vitest";
import { Effect, Layer, Random, Stream } from "effect";

import { ServerConfig } from "../../config.ts";
import { ServerSettingsService } from "../../serverSettings.ts";
import { AntigravityAdapter } from "../Services/AntigravityAdapter.ts";
import { makeAntigravityAdapterLive } from "./AntigravityAdapter.ts";

interface MockStreamEvent {
  readonly event: string;
  readonly data: unknown;
}

interface CapturedRequest {
  readonly url: string;
  readonly headers: Record<string, string>;
  readonly body: Record<string, unknown>;
}

function encodeSseEvent(event: MockStreamEvent): string {
  return `event: ${event.event}\ndata: ${JSON.stringify(event.data)}\n\n`;
}

function buildSseResponse(events: ReadonlyArray<MockStreamEvent>): Response {
  return new Response(events.map(encodeSseEvent).join(""), {
    status: 200,
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache",
      connection: "keep-alive",
    },
  });
}

function makeDeterministicRandomService(seed = 0x1234_5678): {
  nextIntUnsafe: () => number;
  nextDoubleUnsafe: () => number;
} {
  let state = seed >>> 0;
  const nextIntUnsafe = (): number => {
    state = (Math.imul(1_664_525, state) + 1_013_904_223) >>> 0;
    return state;
  };

  return {
    nextIntUnsafe,
    nextDoubleUnsafe: () => nextIntUnsafe() / 0x1_0000_0000,
  };
}

function makeHarness(
  responders: Array<(body: Record<string, unknown>) => ReadonlyArray<MockStreamEvent>>,
) {
  const capturedRequests: CapturedRequest[] = [];

  const fetchImpl = Object.assign(
    async (...args: Parameters<typeof globalThis.fetch>): ReturnType<typeof globalThis.fetch> => {
      const [input, init] = args;
      const request =
        input instanceof Request
          ? new Request(input)
          : input instanceof URL
            ? new Request(input.toString(), init)
            : new Request(input, init);
      const rawBody = await request.text();
      const body = JSON.parse(rawBody) as Record<string, unknown>;
      capturedRequests.push({
        url: request.url,
        headers: Object.fromEntries(request.headers.entries()),
        body,
      });
      const responder = responders.shift();
      if (!responder) {
        throw new Error("Unexpected Antigravity fetch call.");
      }
      return buildSseResponse(responder(body));
    },
    {
      preconnect: (...args: Parameters<typeof globalThis.fetch.preconnect>) =>
        globalThis.fetch.preconnect(...args),
    },
  ) satisfies typeof globalThis.fetch;

  return {
    capturedRequests,
    layer: makeAntigravityAdapterLive({
      fetch: fetchImpl,
    }).pipe(
      Layer.provideMerge(ServerConfig.layerTest("/tmp/antigravity-adapter-test", "/tmp")),
      Layer.provideMerge(
        ServerSettingsService.layerTest({
          providers: {
            antigravity: {
              enabled: true,
              baseUrl: "http://proxy.test",
            },
          },
        }),
      ),
      Layer.provideMerge(NodeServices.layer),
    ),
  };
}

const THREAD_ID = ThreadId.make("thread-antigravity-1");

const waitForRealtimeTick = (ms = 10) =>
  Effect.promise(() => new Promise<void>((resolve) => setTimeout(resolve, ms)));

function waitForCollectedEvents(
  collectedEvents: Array<ProviderRuntimeEvent>,
  predicate: (events: ReadonlyArray<ProviderRuntimeEvent>) => boolean,
) {
  return Effect.gen(function* () {
    while (!predicate(collectedEvents)) {
      yield* waitForRealtimeTick();
    }
    return [...collectedEvents];
  });
}

describe("AntigravityAdapterLive", () => {
  it.effect(
    "reuses message_start containers and filters replayed assistant history on resumed turns",
    () => {
      const harness = makeHarness([
        () => [
          {
            event: "message_start",
            data: {
              type: "message_start",
              message: {
                id: "msg_1",
                type: "message",
                role: "assistant",
                model: "gemini-3.1-pro-high",
                content: [],
                stop_reason: null,
                stop_sequence: null,
                container: {
                  id: "ctr_test",
                  expires_at: "2099-01-01T00:00:00.000Z",
                },
                usage: {
                  input_tokens: 5,
                  output_tokens: 0,
                },
              },
            },
          },
          {
            event: "content_block_start",
            data: {
              type: "content_block_start",
              index: 0,
              content_block: {
                type: "text",
                text: "",
              },
            },
          },
          {
            event: "content_block_delta",
            data: {
              type: "content_block_delta",
              index: 0,
              delta: {
                type: "text_delta",
                text: "Hello from turn one.",
              },
            },
          },
          {
            event: "content_block_stop",
            data: {
              type: "content_block_stop",
              index: 0,
            },
          },
          {
            event: "message_delta",
            data: {
              type: "message_delta",
              delta: {
                stop_reason: "end_turn",
                stop_sequence: null,
                container: null,
              },
              usage: {
                input_tokens: 5,
                output_tokens: 6,
              },
            },
          },
          {
            event: "message_stop",
            data: {
              type: "message_stop",
            },
          },
        ],
        () => [
          {
            event: "message_start",
            data: {
              type: "message_start",
              message: {
                id: "msg_2",
                type: "message",
                role: "assistant",
                model: "gemini-3.1-pro-high",
                content: [],
                stop_reason: null,
                stop_sequence: null,
                container: {
                  id: "ctr_test",
                  expires_at: "2099-01-01T00:00:00.000Z",
                },
                usage: {
                  input_tokens: 7,
                  output_tokens: 0,
                },
              },
            },
          },
          {
            event: "content_block_start",
            data: {
              type: "content_block_start",
              index: 0,
              content_block: {
                type: "text",
                text: "",
              },
            },
          },
          {
            event: "content_block_delta",
            data: {
              type: "content_block_delta",
              index: 0,
              delta: {
                type: "text_delta",
                text: "Hello from turn one.",
              },
            },
          },
          {
            event: "content_block_start",
            data: {
              type: "content_block_start",
              index: 1,
              content_block: {
                type: "text",
                text: "",
              },
            },
          },
          {
            event: "content_block_delta",
            data: {
              type: "content_block_delta",
              index: 1,
              delta: {
                type: "text_delta",
                text: "Fresh second turn answer.",
              },
            },
          },
          {
            event: "content_block_stop",
            data: {
              type: "content_block_stop",
              index: 0,
            },
          },
          {
            event: "content_block_stop",
            data: {
              type: "content_block_stop",
              index: 1,
            },
          },
          {
            event: "message_delta",
            data: {
              type: "message_delta",
              delta: {
                stop_reason: "end_turn",
                stop_sequence: null,
                container: null,
              },
              usage: {
                input_tokens: 7,
                output_tokens: 9,
              },
            },
          },
          {
            event: "message_stop",
            data: {
              type: "message_stop",
            },
          },
        ],
      ]);

      return Effect.gen(function* () {
        const adapter = yield* AntigravityAdapter;
        const scope = yield* Effect.scope;
        const collectedEvents: ProviderRuntimeEvent[] = [];
        yield* Stream.runForEach(adapter.streamEvents, (event) =>
          Effect.sync(() => {
            collectedEvents.push(event);
          }),
        ).pipe(Effect.forkIn(scope));
        yield* adapter.startSession({
          threadId: THREAD_ID,
          provider: "antigravity",
          runtimeMode: "full-access",
        });

        const firstTurnStartIndex = collectedEvents.length;
        const firstTurn = yield* adapter.sendTurn({
          threadId: THREAD_ID,
          input: "hello",
        });
        const firstTurnEvents = yield* waitForCollectedEvents(collectedEvents, (events) =>
          events
            .slice(firstTurnStartIndex)
            .some((event) => event.type === "turn.completed" && event.turnId === firstTurn.turnId),
        );
        assert.equal(
          firstTurnEvents.some(
            (event) => event.type === "turn.completed" && event.turnId === firstTurn.turnId,
          ),
          true,
        );

        const secondTurnStartIndex = collectedEvents.length;
        const secondTurn = yield* adapter.sendTurn({
          threadId: THREAD_ID,
          input: "second",
        });
        yield* waitForCollectedEvents(collectedEvents, (events) =>
          events
            .slice(secondTurnStartIndex)
            .some((event) => event.type === "turn.completed" && event.turnId === secondTurn.turnId),
        );
        const secondTurnEvents = collectedEvents.slice(secondTurnStartIndex);
        assert.equal(
          secondTurnEvents.some(
            (event) => event.type === "turn.completed" && event.turnId === secondTurn.turnId,
          ),
          true,
        );

        assert.equal(harness.capturedRequests.length, 2);
        assert.equal(harness.capturedRequests[1]?.body.container, "ctr_test");
        const secondRequestMessages = harness.capturedRequests[1]?.body.messages;
        assert.equal(Array.isArray(secondRequestMessages), true);
        if (!Array.isArray(secondRequestMessages)) {
          return;
        }
        assert.equal(secondRequestMessages.length, 1);

        const secondAssistantDeltas = secondTurnEvents.flatMap((event) =>
          event.turnId === secondTurn.turnId &&
          event.type === "content.delta" &&
          event.payload.streamKind === "assistant_text"
            ? [event.payload.delta]
            : [],
        );
        assert.deepEqual(secondAssistantDeltas, ["Fresh second turn answer."]);

        const secondAssistantCompletion = secondTurnEvents.find(
          (event) =>
            event.turnId === secondTurn.turnId &&
            event.type === "item.completed" &&
            event.payload.itemType === "assistant_message",
        );
        assert.equal(secondAssistantCompletion?.type, "item.completed");
        if (secondAssistantCompletion?.type === "item.completed") {
          assert.equal(secondAssistantCompletion.payload.detail, "Fresh second turn answer.");
        }
      }).pipe(
        Effect.provideService(Random.Random, makeDeterministicRandomService()),
        Effect.provide(harness.layer),
      );
    },
  );

  it.effect("treats proxy tool_use responses as successful tool lifecycle events", () => {
    const harness = makeHarness([
      () => [
        {
          event: "message_start",
          data: {
            type: "message_start",
            message: {
              id: "msg_tool",
              type: "message",
              role: "assistant",
              model: "gemini-3.1-pro-high",
              content: [],
              stop_reason: null,
              stop_sequence: null,
              container: {
                id: "ctr_tool",
                expires_at: "2099-01-01T00:00:00.000Z",
              },
              usage: {
                input_tokens: 4,
                output_tokens: 0,
              },
            },
          },
        },
        {
          event: "content_block_start",
          data: {
            type: "content_block_start",
            index: 0,
            content_block: {
              type: "server_tool_use",
              id: "srvu_0",
              name: "run_command",
              input: {},
            },
          },
        },
        {
          event: "content_block_delta",
          data: {
            type: "content_block_delta",
            index: 0,
            delta: {
              type: "input_json_delta",
              partial_json: '{"command":"npm test"}',
            },
          },
        },
        {
          event: "content_block_start",
          data: {
            type: "content_block_start",
            index: 1,
            content_block: {
              type: "text",
              text: "",
            },
          },
        },
        {
          event: "content_block_delta",
          data: {
            type: "content_block_delta",
            index: 1,
            delta: {
              type: "text_delta",
              text: "Done.",
            },
          },
        },
        {
          event: "content_block_stop",
          data: {
            type: "content_block_stop",
            index: 0,
          },
        },
        {
          event: "content_block_stop",
          data: {
            type: "content_block_stop",
            index: 1,
          },
        },
        {
          event: "message_delta",
          data: {
            type: "message_delta",
            delta: {
              stop_reason: "tool_use",
              stop_sequence: null,
              container: {
                id: "ctr_tool",
                expires_at: "2099-01-01T00:00:00.000Z",
              },
            },
            usage: {
              input_tokens: 4,
              output_tokens: 6,
            },
          },
        },
        {
          event: "message_stop",
          data: {
            type: "message_stop",
          },
        },
      ],
    ]);

    return Effect.gen(function* () {
      const adapter = yield* AntigravityAdapter;
      const scope = yield* Effect.scope;
      const collectedEvents: ProviderRuntimeEvent[] = [];
      yield* Stream.runForEach(adapter.streamEvents, (event) =>
        Effect.sync(() => {
          collectedEvents.push(event);
        }),
      ).pipe(Effect.forkIn(scope));
      yield* adapter.startSession({
        threadId: THREAD_ID,
        provider: "antigravity",
        runtimeMode: "full-access",
      });

      const turnStartIndex = collectedEvents.length;
      const turn = yield* adapter.sendTurn({
        threadId: THREAD_ID,
        input: "run tests",
      });
      yield* waitForCollectedEvents(collectedEvents, (entries) =>
        entries
          .slice(turnStartIndex)
          .some((event) => event.type === "turn.completed" && event.turnId === turn.turnId),
      );
      const events = collectedEvents.slice(turnStartIndex);
      assert.equal(
        events.some((event) => event.type === "turn.completed" && event.turnId === turn.turnId),
        true,
      );

      const turnCompleted = events.find(
        (event) => event.type === "turn.completed" && event.turnId === turn.turnId,
      );
      assert.equal(turnCompleted?.type, "turn.completed");
      if (turnCompleted?.type === "turn.completed") {
        assert.equal(turnCompleted.payload.state, "completed");
        assert.equal(turnCompleted.payload.stopReason, "tool_use");
      }

      const toolStarted = events.find(
        (event) =>
          event.type === "item.started" &&
          event.turnId === turn.turnId &&
          event.payload.itemType === "command_execution",
      );
      assert.equal(toolStarted?.type, "item.started");

      const toolUpdated = events.find(
        (event) =>
          event.type === "item.updated" &&
          event.turnId === turn.turnId &&
          event.payload.itemType === "command_execution" &&
          eventPayloadInput(event)?.command === "npm test",
      );
      assert.equal(toolUpdated?.type, "item.updated");
      if (toolUpdated?.type === "item.updated") {
        assert.deepEqual(eventPayloadInput(toolUpdated), {
          command: "npm test",
        });
      }

      const toolCompleted = events.find(
        (event) =>
          event.type === "item.completed" &&
          event.turnId === turn.turnId &&
          event.payload.itemType === "command_execution",
      );
      assert.equal(toolCompleted?.type, "item.completed");

      const runtimeError = events.find((event) => event.type === "runtime.error");
      assert.equal(runtimeError, undefined);
    }).pipe(
      Effect.provideService(Random.Random, makeDeterministicRandomService(0x8765_4321)),
      Effect.provide(harness.layer),
    );
  });

  it.effect("deduplicates repeated tool blocks that reuse the same tool_use id", () => {
    const harness = makeHarness([
      () => [
        {
          event: "message_start",
          data: {
            type: "message_start",
            message: {
              id: "msg_tool_dup",
              type: "message",
              role: "assistant",
              model: "gemini-3.1-pro-high",
              content: [],
              stop_reason: null,
              stop_sequence: null,
              container: {
                id: "ctr_tool_dup",
                expires_at: "2099-01-01T00:00:00.000Z",
              },
              usage: {
                input_tokens: 4,
                output_tokens: 0,
              },
            },
          },
        },
        {
          event: "content_block_start",
          data: {
            type: "content_block_start",
            index: 0,
            content_block: {
              type: "server_tool_use",
              id: "srvu_dup",
              name: "run_command",
              input: {},
            },
          },
        },
        {
          event: "content_block_delta",
          data: {
            type: "content_block_delta",
            index: 0,
            delta: {
              type: "input_json_delta",
              partial_json: '{"command":"npm test"}',
            },
          },
        },
        {
          event: "content_block_start",
          data: {
            type: "content_block_start",
            index: 1,
            content_block: {
              type: "server_tool_use",
              id: "srvu_dup",
              name: "run_command",
              input: {},
            },
          },
        },
        {
          event: "content_block_delta",
          data: {
            type: "content_block_delta",
            index: 1,
            delta: {
              type: "input_json_delta",
              partial_json: '{"command":"npm test"}',
            },
          },
        },
        {
          event: "content_block_start",
          data: {
            type: "content_block_start",
            index: 2,
            content_block: {
              type: "text",
              text: "",
            },
          },
        },
        {
          event: "content_block_delta",
          data: {
            type: "content_block_delta",
            index: 2,
            delta: {
              type: "text_delta",
              text: "Done.",
            },
          },
        },
        {
          event: "content_block_stop",
          data: {
            type: "content_block_stop",
            index: 0,
          },
        },
        {
          event: "content_block_stop",
          data: {
            type: "content_block_stop",
            index: 1,
          },
        },
        {
          event: "content_block_stop",
          data: {
            type: "content_block_stop",
            index: 2,
          },
        },
        {
          event: "message_delta",
          data: {
            type: "message_delta",
            delta: {
              stop_reason: "tool_use",
              stop_sequence: null,
              container: {
                id: "ctr_tool_dup",
                expires_at: "2099-01-01T00:00:00.000Z",
              },
            },
            usage: {
              input_tokens: 4,
              output_tokens: 6,
            },
          },
        },
        {
          event: "message_stop",
          data: {
            type: "message_stop",
          },
        },
      ],
    ]);

    return Effect.gen(function* () {
      const adapter = yield* AntigravityAdapter;
      const scope = yield* Effect.scope;
      const collectedEvents: ProviderRuntimeEvent[] = [];
      yield* Stream.runForEach(adapter.streamEvents, (event) =>
        Effect.sync(() => {
          collectedEvents.push(event);
        }),
      ).pipe(Effect.forkIn(scope));
      yield* adapter.startSession({
        threadId: THREAD_ID,
        provider: "antigravity",
        runtimeMode: "full-access",
      });

      const turnStartIndex = collectedEvents.length;
      const turn = yield* adapter.sendTurn({
        threadId: THREAD_ID,
        input: "run tests",
      });
      yield* waitForCollectedEvents(collectedEvents, (entries) =>
        entries
          .slice(turnStartIndex)
          .some((event) => event.type === "turn.completed" && event.turnId === turn.turnId),
      );
      const events = collectedEvents.slice(turnStartIndex);

      const toolStartedEvents = events.filter(
        (event) =>
          event.type === "item.started" &&
          event.turnId === turn.turnId &&
          event.payload.itemType === "command_execution",
      );
      assert.equal(toolStartedEvents.length, 1);

      const toolCompletedEvents = events.filter(
        (event) =>
          event.type === "item.completed" &&
          event.turnId === turn.turnId &&
          event.payload.itemType === "command_execution",
      );
      assert.equal(toolCompletedEvents.length, 1);

      const parsedToolUpdates = events.filter(
        (event) =>
          event.type === "item.updated" &&
          event.turnId === turn.turnId &&
          event.payload.itemType === "command_execution" &&
          eventPayloadInput(event)?.command === "npm test",
      );
      assert.equal(parsedToolUpdates.length, 1);
    }).pipe(
      Effect.provideService(Random.Random, makeDeterministicRandomService(0x1122_3344)),
      Effect.provide(harness.layer),
    );
  });

  it.effect("projects thinking blocks into visible reasoning progress events", () => {
    const harness = makeHarness([
      () => [
        {
          event: "message_start",
          data: {
            type: "message_start",
            message: {
              id: "msg_reasoning",
              type: "message",
              role: "assistant",
              model: "gemini-3.1-pro-high",
              content: [],
              stop_reason: null,
              stop_sequence: null,
              container: {
                id: "ctr_reasoning",
                expires_at: "2099-01-01T00:00:00.000Z",
              },
              usage: {
                input_tokens: 3,
                output_tokens: 0,
              },
            },
          },
        },
        {
          event: "content_block_start",
          data: {
            type: "content_block_start",
            index: 0,
            content_block: {
              type: "thinking",
              thinking: "",
            },
          },
        },
        {
          event: "content_block_delta",
          data: {
            type: "content_block_delta",
            index: 0,
            delta: {
              type: "thinking_delta",
              thinking: "Planning the next edit",
            },
          },
        },
        {
          event: "content_block_stop",
          data: {
            type: "content_block_stop",
            index: 0,
          },
        },
        {
          event: "content_block_start",
          data: {
            type: "content_block_start",
            index: 1,
            content_block: {
              type: "text",
              text: "",
            },
          },
        },
        {
          event: "content_block_delta",
          data: {
            type: "content_block_delta",
            index: 1,
            delta: {
              type: "text_delta",
              text: "Done.",
            },
          },
        },
        {
          event: "content_block_stop",
          data: {
            type: "content_block_stop",
            index: 1,
          },
        },
        {
          event: "message_delta",
          data: {
            type: "message_delta",
            delta: {
              stop_reason: "end_turn",
              stop_sequence: null,
              container: {
                id: "ctr_reasoning",
                expires_at: "2099-01-01T00:00:00.000Z",
              },
            },
            usage: {
              input_tokens: 3,
              output_tokens: 5,
            },
          },
        },
        {
          event: "message_stop",
          data: {
            type: "message_stop",
          },
        },
      ],
    ]);

    return Effect.gen(function* () {
      const adapter = yield* AntigravityAdapter;
      const scope = yield* Effect.scope;
      const collectedEvents: ProviderRuntimeEvent[] = [];
      yield* Stream.runForEach(adapter.streamEvents, (event) =>
        Effect.sync(() => {
          collectedEvents.push(event);
        }),
      ).pipe(Effect.forkIn(scope));
      yield* adapter.startSession({
        threadId: THREAD_ID,
        provider: "antigravity",
        runtimeMode: "full-access",
      });

      const turnStartIndex = collectedEvents.length;
      const turn = yield* adapter.sendTurn({
        threadId: THREAD_ID,
        input: "reason out loud",
      });
      yield* waitForCollectedEvents(collectedEvents, (entries) =>
        entries
          .slice(turnStartIndex)
          .some((event) => event.type === "turn.completed" && event.turnId === turn.turnId),
      );
      const events = collectedEvents.slice(turnStartIndex);

      const reasoningStarted = events.find(
        (event) =>
          event.type === "item.started" &&
          event.turnId === turn.turnId &&
          event.payload.itemType === "reasoning",
      );
      assert.equal(reasoningStarted?.type, "item.started");

      const reasoningProgress = events.find(
        (event) => event.type === "task.progress" && event.turnId === turn.turnId,
      );
      assert.equal(reasoningProgress?.type, "task.progress");
      if (reasoningProgress?.type === "task.progress") {
        assert.equal(reasoningProgress.payload.summary, "Planning the next edit");
      }
    }).pipe(
      Effect.provideService(Random.Random, makeDeterministicRandomService(0x2468_1357)),
      Effect.provide(harness.layer),
    );
  });

  it.effect("emits a visible tool update as soon as a tool block starts", () => {
    const harness = makeHarness([
      () => [
        {
          event: "message_start",
          data: {
            type: "message_start",
            message: {
              id: "msg_tool_start",
              type: "message",
              role: "assistant",
              model: "gemini-3.1-pro-high",
              content: [],
              stop_reason: null,
              stop_sequence: null,
              container: {
                id: "ctr_tool_start",
                expires_at: "2099-01-01T00:00:00.000Z",
              },
              usage: {
                input_tokens: 4,
                output_tokens: 0,
              },
            },
          },
        },
        {
          event: "content_block_start",
          data: {
            type: "content_block_start",
            index: 0,
            content_block: {
              type: "server_tool_use",
              id: "srvu_start",
              name: "run_command",
              input: {},
            },
          },
        },
        {
          event: "message_delta",
          data: {
            type: "message_delta",
            delta: {
              stop_reason: "tool_use",
              stop_sequence: null,
              container: {
                id: "ctr_tool_start",
                expires_at: "2099-01-01T00:00:00.000Z",
              },
            },
            usage: {
              input_tokens: 4,
              output_tokens: 2,
            },
          },
        },
        {
          event: "message_stop",
          data: {
            type: "message_stop",
          },
        },
      ],
    ]);

    return Effect.gen(function* () {
      const adapter = yield* AntigravityAdapter;
      const scope = yield* Effect.scope;
      const collectedEvents: ProviderRuntimeEvent[] = [];
      yield* Stream.runForEach(adapter.streamEvents, (event) =>
        Effect.sync(() => {
          collectedEvents.push(event);
        }),
      ).pipe(Effect.forkIn(scope));
      yield* adapter.startSession({
        threadId: THREAD_ID,
        provider: "antigravity",
        runtimeMode: "full-access",
      });

      const turnStartIndex = collectedEvents.length;
      const turn = yield* adapter.sendTurn({
        threadId: THREAD_ID,
        input: "run the command",
      });
      yield* waitForCollectedEvents(collectedEvents, (entries) =>
        entries
          .slice(turnStartIndex)
          .some((event) => event.type === "turn.completed" && event.turnId === turn.turnId),
      );
      const events = collectedEvents.slice(turnStartIndex);

      const toolUpdated = events.find(
        (event) =>
          event.type === "item.updated" &&
          event.turnId === turn.turnId &&
          event.payload.itemType === "command_execution",
      );
      assert.equal(toolUpdated?.type, "item.updated");
      if (toolUpdated?.type === "item.updated") {
        assert.equal(toolUpdated.payload.detail, "run_command");
      }
    }).pipe(
      Effect.provideService(Random.Random, makeDeterministicRandomService(0x1357_2468)),
      Effect.provide(harness.layer),
    );
  });
});

function eventPayloadInput(
  event: Extract<ProviderRuntimeEvent, { type: "item.updated" }>,
): Record<string, unknown> | undefined {
  const data = event.payload.data;
  if (!data || typeof data !== "object") {
    return undefined;
  }
  const input = (data as { input?: unknown }).input;
  return input && typeof input === "object" ? (input as Record<string, unknown>) : undefined;
}
