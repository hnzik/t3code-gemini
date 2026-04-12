import {
  CommandId,
  DEFAULT_PROVIDER_INTERACTION_MODE,
  EventId,
  ProjectId,
  ThreadId,
} from "@t3tools/contracts";
import { Effect } from "effect";
import { describe, expect, it } from "vitest";

import { decideOrchestrationCommand } from "./decider.ts";
import { createEmptyReadModel, projectEvent } from "./projector.ts";

const asEventId = (value: string): EventId => EventId.make(value);
const asProjectId = (value: string): ProjectId => ProjectId.make(value);

describe("decider project delete", () => {
  it("cascades active project threads before emitting project.deleted", async () => {
    const now = new Date().toISOString();
    const projectId = asProjectId("project-delete");
    const runningThreadId = ThreadId.make("thread-running");
    const idleThreadId = ThreadId.make("thread-idle");
    const deletedThreadId = ThreadId.make("thread-deleted");

    const initial = createEmptyReadModel(now);
    const withProject = await Effect.runPromise(
      projectEvent(initial, {
        sequence: 1,
        eventId: asEventId("evt-project-create"),
        aggregateKind: "project",
        aggregateId: projectId,
        type: "project.created",
        occurredAt: now,
        commandId: CommandId.make("cmd-project-create"),
        causationEventId: null,
        correlationId: CommandId.make("cmd-project-create"),
        metadata: {},
        payload: {
          projectId,
          title: "Delete Project",
          workspaceRoot: "/tmp/project-delete",
          defaultModelSelection: null,
          scripts: [],
          createdAt: now,
          updatedAt: now,
        },
      }),
    );
    const withRunningThread = await Effect.runPromise(
      projectEvent(withProject, {
        sequence: 2,
        eventId: asEventId("evt-thread-running-create"),
        aggregateKind: "thread",
        aggregateId: runningThreadId,
        type: "thread.created",
        occurredAt: now,
        commandId: CommandId.make("cmd-thread-running-create"),
        causationEventId: null,
        correlationId: CommandId.make("cmd-thread-running-create"),
        metadata: {},
        payload: {
          threadId: runningThreadId,
          projectId,
          title: "Running thread",
          modelSelection: {
            provider: "codex",
            model: "gpt-5-codex",
          },
          interactionMode: DEFAULT_PROVIDER_INTERACTION_MODE,
          runtimeMode: "full-access",
          branch: null,
          worktreePath: null,
          createdAt: now,
          updatedAt: now,
        },
      }),
    );
    const withRunningSession = await Effect.runPromise(
      projectEvent(withRunningThread, {
        sequence: 3,
        eventId: asEventId("evt-thread-running-session"),
        aggregateKind: "thread",
        aggregateId: runningThreadId,
        type: "thread.session-set",
        occurredAt: now,
        commandId: CommandId.make("cmd-thread-running-session"),
        causationEventId: null,
        correlationId: CommandId.make("cmd-thread-running-session"),
        metadata: {},
        payload: {
          threadId: runningThreadId,
          session: {
            threadId: runningThreadId,
            status: "running",
            providerName: "codex",
            runtimeMode: "full-access",
            activeTurnId: null,
            lastError: null,
            updatedAt: now,
          },
        },
      }),
    );
    const withIdleThread = await Effect.runPromise(
      projectEvent(withRunningSession, {
        sequence: 4,
        eventId: asEventId("evt-thread-idle-create"),
        aggregateKind: "thread",
        aggregateId: idleThreadId,
        type: "thread.created",
        occurredAt: now,
        commandId: CommandId.make("cmd-thread-idle-create"),
        causationEventId: null,
        correlationId: CommandId.make("cmd-thread-idle-create"),
        metadata: {},
        payload: {
          threadId: idleThreadId,
          projectId,
          title: "Idle thread",
          modelSelection: {
            provider: "codex",
            model: "gpt-5-codex",
          },
          interactionMode: DEFAULT_PROVIDER_INTERACTION_MODE,
          runtimeMode: "approval-required",
          branch: null,
          worktreePath: null,
          createdAt: now,
          updatedAt: now,
        },
      }),
    );
    const withDeletedThread = await Effect.runPromise(
      projectEvent(withIdleThread, {
        sequence: 5,
        eventId: asEventId("evt-thread-deleted-create"),
        aggregateKind: "thread",
        aggregateId: deletedThreadId,
        type: "thread.created",
        occurredAt: now,
        commandId: CommandId.make("cmd-thread-deleted-create"),
        causationEventId: null,
        correlationId: CommandId.make("cmd-thread-deleted-create"),
        metadata: {},
        payload: {
          threadId: deletedThreadId,
          projectId,
          title: "Already deleted thread",
          modelSelection: {
            provider: "codex",
            model: "gpt-5-codex",
          },
          interactionMode: DEFAULT_PROVIDER_INTERACTION_MODE,
          runtimeMode: "approval-required",
          branch: null,
          worktreePath: null,
          createdAt: now,
          updatedAt: now,
        },
      }),
    );
    const readModel = await Effect.runPromise(
      projectEvent(withDeletedThread, {
        sequence: 6,
        eventId: asEventId("evt-thread-deleted-delete"),
        aggregateKind: "thread",
        aggregateId: deletedThreadId,
        type: "thread.deleted",
        occurredAt: now,
        commandId: CommandId.make("cmd-thread-deleted-delete"),
        causationEventId: null,
        correlationId: CommandId.make("cmd-thread-deleted-delete"),
        metadata: {},
        payload: {
          threadId: deletedThreadId,
          deletedAt: now,
        },
      }),
    );

    const result = await Effect.runPromise(
      decideOrchestrationCommand({
        command: {
          type: "project.delete",
          commandId: CommandId.make("cmd-project-delete"),
          projectId,
        },
        readModel,
      }),
    );

    expect(Array.isArray(result)).toBe(true);
    const events = Array.isArray(result) ? result : [result];
    expect(events.map((event) => event.type)).toEqual([
      "thread.session-stop-requested",
      "thread.deleted",
      "thread.deleted",
      "project.deleted",
    ]);
    expect(events[0]).toMatchObject({
      aggregateKind: "thread",
      aggregateId: runningThreadId,
      payload: {
        threadId: runningThreadId,
      },
    });
    expect(events[1]).toMatchObject({
      aggregateKind: "thread",
      aggregateId: runningThreadId,
      payload: {
        threadId: runningThreadId,
      },
    });
    expect(events[2]).toMatchObject({
      aggregateKind: "thread",
      aggregateId: idleThreadId,
      payload: {
        threadId: idleThreadId,
      },
    });
    expect(events[3]).toMatchObject({
      aggregateKind: "project",
      aggregateId: projectId,
      payload: {
        projectId,
      },
    });
  });
});
