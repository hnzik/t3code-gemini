import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { assert, describe, it } from "@effect/vitest";
import { AuthType, QuestionType } from "@google/gemini-cli-core";
import { ThreadId } from "@t3tools/contracts";

import {
  applyGeminiAssistantTextChunk,
  buildGeminiPersistedBinding,
  buildGeminiAssistantHistoryEntry,
  buildGeminiAskUserResponseAnswers,
  formatGeminiRetryWarningMessage,
  formatGeminiSubagentActivityDetail,
  inferGeminiTurnHistoryLengths,
  normalizeGeminiAskUserQuestions,
  readGeminiPlanMarkdownFromFile,
  readGeminiResumeState,
} from "./GeminiAcpAdapter.ts";
import { resolveGeminiAuthType } from "./GeminiCoreConfig";

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
