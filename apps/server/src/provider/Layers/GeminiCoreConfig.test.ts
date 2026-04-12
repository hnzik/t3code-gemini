import { basename, dirname, join } from "node:path";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";

import { ApprovalMode, PolicyDecision, Storage } from "@google/gemini-cli-core";
import { assert, describe, it } from "@effect/vitest";
import { afterEach, beforeEach } from "vitest";

import {
  FORCED_GEMINI_USER_AGENT_MODEL,
  FORCED_GOOGLE_API_NODEJS_CLIENT_VERSION,
  GEMINI_CORE_CLIENT_VERSION,
  createGeminiCoreConfig,
  createGeminiCoreAuthHeaders,
  mergeGeminiCliCustomHeaders,
  resolveGeminiApprovalMode,
} from "./GeminiCoreConfig";

const originalGetGlobalGeminiDir = Storage.getGlobalGeminiDir;
const originalGetUserPoliciesDir = Storage.getUserPoliciesDir;
const originalGetSystemPoliciesDir = Storage.getSystemPoliciesDir;

let tempGeminiHome = "";
let tempWorkspace = "";

beforeEach(async () => {
  tempGeminiHome = await mkdtemp(join(tmpdir(), "t3-gemini-home-"));
  tempWorkspace = await mkdtemp(join(tmpdir(), "t3-gemini-workspace-"));
  Storage.getGlobalGeminiDir = () => tempGeminiHome;
  Storage.getUserPoliciesDir = () => join(tempGeminiHome, "user-policies");
  Storage.getSystemPoliciesDir = () => join(tempGeminiHome, "system-policies");
});

afterEach(async () => {
  Storage.getGlobalGeminiDir = originalGetGlobalGeminiDir;
  Storage.getUserPoliciesDir = originalGetUserPoliciesDir;
  Storage.getSystemPoliciesDir = originalGetSystemPoliciesDir;
  await rm(tempWorkspace, { recursive: true, force: true });
  await rm(tempGeminiHome, { recursive: true, force: true });
});

describe("GeminiCoreConfig user-agent overrides", () => {
  it("builds the forced Gemini CLI auth headers", () => {
    const headers = createGeminiCoreAuthHeaders();
    const googleApiNodeJsClient = `google-api-nodejs-client/${FORCED_GOOGLE_API_NODEJS_CLIENT_VERSION}`;

    assert.deepStrictEqual(headers, {
      "User-Agent": `GeminiCLI/${GEMINI_CORE_CLIENT_VERSION}/${FORCED_GEMINI_USER_AGENT_MODEL} (${process.platform}; ${process.arch}; terminal) ${googleApiNodeJsClient}`,
      "x-goog-api-client": googleApiNodeJsClient,
    });
  });

  it("merges existing Gemini CLI custom headers and replaces user-agent overrides", () => {
    const mergedHeaders = mergeGeminiCliCustomHeaders(
      [
        "X-Existing-Header:preserve-me",
        "user-agent:old-user-agent",
        "X-Goog-Api-Client:old-api-client",
      ].join(", "),
    );

    assert.equal(
      mergedHeaders,
      [
        "X-Existing-Header:preserve-me",
        `User-Agent:GeminiCLI/${GEMINI_CORE_CLIENT_VERSION}/${FORCED_GEMINI_USER_AGENT_MODEL} (${process.platform}; ${process.arch}; terminal) google-api-nodejs-client/${FORCED_GOOGLE_API_NODEJS_CLIENT_VERSION}`,
        `x-goog-api-client:google-api-nodejs-client/${FORCED_GOOGLE_API_NODEJS_CLIENT_VERSION}`,
      ].join(", "),
    );
  });
});

describe("GeminiCoreConfig approval mode mapping", () => {
  it("enters real Gemini core plan mode when the interaction mode is plan", () => {
    assert.equal(
      resolveGeminiApprovalMode({
        interactionMode: "plan",
        runtimeMode: "full-access",
      }),
      ApprovalMode.PLAN,
    );
  });

  it("maps edit-permissive runtime modes to Gemini auto-edit", () => {
    assert.equal(
      resolveGeminiApprovalMode({
        runtimeMode: "full-access",
      }),
      ApprovalMode.AUTO_EDIT,
    );
    assert.equal(
      resolveGeminiApprovalMode({
        runtimeMode: "auto-accept-edits",
      }),
      ApprovalMode.AUTO_EDIT,
    );
  });

  it("maps approval-required runtime mode to Gemini default approvals", () => {
    assert.equal(
      resolveGeminiApprovalMode({
        runtimeMode: "approval-required",
      }),
      ApprovalMode.DEFAULT,
    );
  });

  it("loads Gemini CLI policy defaults so plan mode keeps read tools approval-free", async () => {
    const config = await createGeminiCoreConfig({
      sessionId: "thread-1",
      cwd: tempWorkspace,
      runtimeMode: "approval-required",
      interactive: true,
    });
    config.geminiClient.getHistory = () =>
      [
        {
          role: "user",
          parts: [{ text: "Inspect the codebase and write a plan." }],
        },
      ] as any;

    assert.equal(
      (
        await config
          .getPolicyEngine()
          .check({ name: "read_file", args: { file_path: "README.md" } }, undefined)
      ).decision,
      PolicyDecision.ALLOW,
    );
    assert.equal(
      (
        await config
          .getPolicyEngine()
          .check(
            { name: "run_shell_command", args: { command: "touch tmp-plan-check.md" } },
            undefined,
          )
      ).decision,
      PolicyDecision.ASK_USER,
    );

    config.setApprovalMode(
      resolveGeminiApprovalMode({
        interactionMode: "plan",
        runtimeMode: "approval-required",
      }),
    );

    assert.equal(
      (
        await config
          .getPolicyEngine()
          .check({ name: "glob", args: { pattern: "src/**/*" } }, undefined)
      ).decision,
      PolicyDecision.ALLOW,
    );
    assert.equal(
      (
        await config
          .getPolicyEngine()
          .check(
            { name: "run_shell_command", args: { command: "touch tmp-plan-check.md" } },
            undefined,
          )
      ).decision,
      PolicyDecision.DENY,
    );
  });

  it("uses Gemini CLI's session-scoped plans directory layout", async () => {
    const sessionId = "thread-123";
    const config = await createGeminiCoreConfig({
      sessionId,
      cwd: tempWorkspace,
      runtimeMode: "approval-required",
      interactive: true,
    });

    await config.storage.initialize();
    const plansDir = config.storage.getPlansDir();

    assert.equal(basename(plansDir), "plans");
    assert.equal(basename(dirname(plansDir)), sessionId);
    assert.equal(dirname(dirname(plansDir)).startsWith(join(tempGeminiHome, "tmp")), true);
  });
});
