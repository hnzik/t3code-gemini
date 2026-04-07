import { assert, describe, it } from "@effect/vitest";
import { AuthType, QuestionType } from "@google/gemini-cli-core";

import {
  buildGeminiAskUserResponseAnswers,
  buildGeminiPromptBlocks,
  extractGeminiSchedulerApprovalRequest,
  extractVisibleAssistantText,
  formatAgentThoughtText,
  normalizeGeminiAskUserQuestions,
  readGeminiResumeState,
  resolveGeminiAuthType,
  shouldApplyUsageUpdate,
} from "./GeminiAcpAdapter.ts";

describe("GeminiAcpAdapter proposed plan parsing", () => {
  it("removes complete proposed-plan blocks from visible assistant text", () => {
    const visibleText = extractVisibleAssistantText(
      [
        "Here is the plan:\n",
        "<proposed_plan>",
        "\n# Ship it\n",
        "</proposed_plan>",
        "\nAnything else?",
      ].join(""),
    );

    assert.equal(visibleText, "Here is the plan:\n\nAnything else?");
  });

  it("withholds partial proposed-plan opening tags until they are resolved", () => {
    assert.equal(extractVisibleAssistantText("Hello <proposed_"), "Hello ");
    assert.equal(extractVisibleAssistantText("Hello <proposed_x"), "Hello <proposed_x");
  });

  it("only emits text outside the proposed-plan block across streamed chunks", () => {
    const chunks = ["Intro: ", "<propose", "d_plan>\n# Plan\n", "</proposed_plan> Outro"];

    let rawText = "";
    let visibleLength = 0;
    const deltas: string[] = [];

    for (const chunk of chunks) {
      rawText += chunk;
      const visibleText = extractVisibleAssistantText(rawText);
      const delta = visibleText.slice(visibleLength);
      visibleLength = visibleText.length;
      deltas.push(delta);
    }

    assert.deepStrictEqual(deltas, ["Intro: ", "", "", " Outro"]);
  });
});

describe("GeminiAcpAdapter usage update gating", () => {
  it("ignores replay-phase usage updates until fresh content begins", () => {
    assert.equal(
      shouldApplyUsageUpdate(
        {
          turnId: "turn-1",
          startedAt: new Date().toISOString(),
          reasoningItemEmitted: false,
        },
        false,
      ),
      false,
    );
  });

  it("accepts usage updates once the turn has fresh content", () => {
    assert.equal(
      shouldApplyUsageUpdate(
        {
          turnId: "turn-1",
          startedAt: new Date().toISOString(),
          reasoningItemEmitted: false,
        },
        true,
      ),
      true,
    );
  });

  it("accepts usage updates outside an active turn", () => {
    assert.equal(shouldApplyUsageUpdate(undefined, false), true);
  });
});

describe("GeminiAcpAdapter agent-event helpers", () => {
  it("formats thought text with the upstream subject metadata", () => {
    assert.equal(
      formatAgentThoughtText({
        subject: "Check context",
        thought: "Need to verify the changed files first.",
      }),
      "**Check context** Need to verify the changed files first.",
    );
  });

  it("falls back to plain thought text when no subject is present", () => {
    assert.equal(
      formatAgentThoughtText({
        subject: undefined,
        thought: "Need to verify the changed files first.",
      }),
      "Need to verify the changed files first.",
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
      history: [{ role: "user", parts: [{ text: "Hello" }] }],
      turnCount: 3,
    });

    assert.deepStrictEqual(resumeState, {
      history: [{ role: "user", parts: [{ text: "Hello" }] }],
      turnCount: 3,
    });
  });

  it("rejects invalid Gemini resume cursors", () => {
    assert.equal(readGeminiResumeState(undefined), undefined);
    assert.equal(readGeminiResumeState({ turnCount: 2 }), undefined);
    assert.deepStrictEqual(readGeminiResumeState({ history: [], turnCount: -1 }), {
      history: [],
      turnCount: 0,
    });
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

describe("GeminiAcpAdapter scheduler approval extraction", () => {
  it("extracts edit approvals from scheduler awaiting-approval state", () => {
    const approval = extractGeminiSchedulerApprovalRequest({
      status: "awaiting_approval",
      correlationId: "corr-1",
      request: {
        name: "replace",
        args: {
          file_path: "chapters-raw/chapter-5.md",
          old_string: "old",
          new_string: "new",
        },
      },
      confirmationDetails: {
        type: "edit",
        fileName: "chapter-5.md",
      },
    });

    assert.deepStrictEqual(approval, {
      correlationId: "corr-1",
      requestType: "file_change_approval",
      detail: "Edit chapter-5.md",
      args: {
        file_path: "chapters-raw/chapter-5.md",
        old_string: "old",
        new_string: "new",
      },
    });
  });

  it("skips ask_user confirmations because they are handled separately", () => {
    const approval = extractGeminiSchedulerApprovalRequest({
      status: "awaiting_approval",
      correlationId: "corr-2",
      request: {
        name: "replace",
        args: {
          file_path: "chapters-raw/chapter-5.md",
        },
      },
      confirmationDetails: {
        type: "ask_user",
      },
    });

    assert.equal(approval, undefined);
  });
});

describe("GeminiAcpAdapter default-mode prompting", () => {
  it("prepends the default-mode prompt on the first non-plan turn", () => {
    const built = buildGeminiPromptBlocks({
      interactionMode: "default",
      userInput: "Please update the adapter.",
      planModePromptSent: false,
      defaultModePromptSent: false,
    });

    assert.equal(built.promptBlocks.length, 2);
    assert.equal(built.defaultModePromptSent, true);
    assert.equal(built.planModePromptSent, false);
    assert.equal(
      built.promptBlocks[0]?.text.includes("Do **NOT** ask a blocking question and then continue"),
      true,
    );
    assert.equal(built.promptBlocks[1]?.text, "Please update the adapter.");
  });

  it("uses the shorter default-mode reminder after the first non-plan turn", () => {
    const built = buildGeminiPromptBlocks({
      interactionMode: "default",
      userInput: "Continue.",
      planModePromptSent: false,
      defaultModePromptSent: true,
    });

    assert.equal(built.promptBlocks.length, 2);
    assert.equal(
      built.promptBlocks[0]?.text.includes("ask the question and end the turn immediately"),
      true,
    );
    assert.equal(built.promptBlocks[1]?.text, "Continue.");
  });

  it("keeps plan-mode prompting separate from default-mode prompting", () => {
    const built = buildGeminiPromptBlocks({
      interactionMode: "plan",
      userInput: "Refine the plan.",
      planModePromptSent: false,
      defaultModePromptSent: true,
    });

    assert.equal(built.promptBlocks.length, 2);
    assert.equal(built.planModePromptSent, true);
    assert.equal(built.defaultModePromptSent, true);
    assert.equal(built.promptBlocks[0]?.text.includes("# Plan Mode"), true);
    assert.equal(built.promptBlocks[1]?.text, "Refine the plan.");
  });
});
