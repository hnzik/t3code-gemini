import { describe, expect, it } from "vitest";

import { providerModelsFromSettings } from "./providerSnapshot.ts";
import {
  ANTIGRAVITY_BUILT_IN_MODELS,
  ANTIGRAVITY_GEMINI_MODEL_CAPABILITIES,
  ANTIGRAVITY_MODEL_CAPABILITIES,
  getAntigravityModelCapabilities,
} from "./antigravity.ts";

describe("Antigravity model capabilities", () => {
  it("maps built-in Gemini models to a default 1M context window", () => {
    expect(
      ANTIGRAVITY_BUILT_IN_MODELS.find((model) => model.slug === "gemini-3.1-pro-high")
        ?.capabilities,
    ).toEqual(ANTIGRAVITY_GEMINI_MODEL_CAPABILITIES);

    expect(
      ANTIGRAVITY_BUILT_IN_MODELS.find((model) => model.slug === "claude-sonnet-4-6")?.capabilities,
    ).toEqual(ANTIGRAVITY_MODEL_CAPABILITIES);
  });

  it("maps custom Gemini models to a default 1M context window without affecting other models", () => {
    const models = providerModelsFromSettings(
      ANTIGRAVITY_BUILT_IN_MODELS,
      "antigravity",
      ["gemini-3.2-pro-preview", "custom-proxy-model"],
      getAntigravityModelCapabilities,
    );

    expect(models.find((model) => model.slug === "gemini-3.2-pro-preview")?.capabilities).toEqual(
      ANTIGRAVITY_GEMINI_MODEL_CAPABILITIES,
    );
    expect(models.find((model) => model.slug === "custom-proxy-model")?.capabilities).toEqual(
      ANTIGRAVITY_MODEL_CAPABILITIES,
    );
  });
});
