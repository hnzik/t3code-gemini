import { ApprovalMode } from "@google/gemini-cli-core";
import { assert, describe, it } from "@effect/vitest";

import {
  FORCED_GEMINI_USER_AGENT_MODEL,
  FORCED_GOOGLE_API_NODEJS_CLIENT_VERSION,
  GEMINI_CORE_CLIENT_VERSION,
  createGeminiCoreAuthHeaders,
  mergeGeminiCliCustomHeaders,
  resolveGeminiApprovalMode,
} from "./GeminiCoreConfig";

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
});
