import type { ModelCapabilities, ServerProviderModel } from "@t3tools/contracts";
import { createModelCapabilities } from "@t3tools/shared/model";

export const ANTIGRAVITY_DEFAULT_BASE_URL = "http://127.0.0.1:3117";
export const ANTIGRAVITY_DEFAULT_MAX_TOKENS = 8_192;

export const ANTIGRAVITY_MODEL_CAPABILITIES: ModelCapabilities = createModelCapabilities({
  optionDescriptors: [],
});

export const ANTIGRAVITY_GEMINI_MODEL_CAPABILITIES: ModelCapabilities = createModelCapabilities({
  optionDescriptors: [
    {
      id: "contextWindow",
      label: "Context Window",
      type: "select",
      options: [{ id: "1m", label: "1M", isDefault: true }],
      currentValue: "1m",
    },
  ],
});

const ANTIGRAVITY_MODEL_SLUGS = [
  "chat_20706",
  "chat_23310",
  "claude-opus-4-6-thinking",
  "claude-sonnet-4-6",
  "gemini-2.5-flash",
  "gemini-2.5-flash-lite",
  "gemini-2.5-flash-thinking",
  "gemini-2.5-pro",
  "gemini-3-flash",
  "gemini-3-flash-agent",
  "gemini-3-pro-high",
  "gemini-3-pro-low",
  "gemini-3.1-flash-image",
  "gemini-3.1-flash-lite",
  "gemini-3.1-pro-high",
  "gemini-3.1-pro-low",
  "gpt-oss-120b-medium",
  "tab_flash_lite_preview",
  "tab_jump_flash_lite_preview",
] as const;

export function getAntigravityModelCapabilities(slug: string): ModelCapabilities {
  return slug.startsWith("gemini-")
    ? ANTIGRAVITY_GEMINI_MODEL_CAPABILITIES
    : ANTIGRAVITY_MODEL_CAPABILITIES;
}

function humanizeAntigravityModelSlug(slug: string): string {
  const normalized = slug.replace(/_/g, " ").replace(/-/g, " ");
  return normalized
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => {
      const lower = part.toLowerCase();
      if (lower === "gpt") {
        return "GPT";
      }
      if (lower === "oss") {
        return "OSS";
      }
      if (lower === "tab") {
        return "Tab";
      }
      if (lower === "lite") {
        return "Lite";
      }
      if (/^\d/.test(part) || /^[A-Z0-9.]+$/.test(part)) {
        return part;
      }
      return part.charAt(0).toUpperCase() + part.slice(1);
    })
    .join(" ");
}

export const ANTIGRAVITY_BUILT_IN_MODELS: ReadonlyArray<ServerProviderModel> =
  ANTIGRAVITY_MODEL_SLUGS.map((slug) => ({
    slug,
    name: humanizeAntigravityModelSlug(slug),
    isCustom: false,
    capabilities: getAntigravityModelCapabilities(slug),
  }));
