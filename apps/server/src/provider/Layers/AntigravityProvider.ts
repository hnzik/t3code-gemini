import type { AntigravitySettings, ServerProvider } from "@t3tools/contracts";
import { ServerSettingsError } from "@t3tools/contracts";
import { Effect, Equal, Layer, Stream } from "effect";

import { ServerSettingsService } from "../../serverSettings.ts";
import { buildServerProvider, providerModelsFromSettings } from "../providerSnapshot.ts";
import { makeManagedServerProvider } from "../makeManagedServerProvider.ts";
import {
  ANTIGRAVITY_BUILT_IN_MODELS,
  ANTIGRAVITY_DEFAULT_BASE_URL,
  getAntigravityModelCapabilities,
} from "../antigravity.ts";
import { AntigravityProvider } from "../Services/AntigravityProvider.ts";

const PROVIDER = "antigravity" as const;
const ANTIGRAVITY_PRESENTATION = {
  displayName: "Antigravity",
  showInteractionModeToggle: true,
} as const;
const HEALTHCHECK_TIMEOUT_MS = 4_000;

class AntigravityHealthcheckError extends Error {
  readonly _tag = "AntigravityHealthcheckError";
  override readonly cause: unknown;

  constructor(message: string, cause: unknown) {
    super(message);
    this.cause = cause;
  }
}

function normalizeBaseUrl(value: string): string {
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : ANTIGRAVITY_DEFAULT_BASE_URL;
}

function toMessage(cause: unknown, fallback: string): string {
  return cause instanceof Error && cause.message.trim().length > 0 ? cause.message : fallback;
}

function buildHealthUrl(baseUrl: string): URL | undefined {
  try {
    return new URL("/healthz", baseUrl);
  } catch {
    return undefined;
  }
}

export const checkAntigravityProviderStatus = Effect.fn("checkAntigravityProviderStatus")(
  function* (
    fetchImpl: typeof globalThis.fetch = globalThis.fetch.bind(globalThis),
  ): Effect.fn.Return<ServerProvider, ServerSettingsError, ServerSettingsService> {
    const antigravitySettings = yield* Effect.service(ServerSettingsService).pipe(
      Effect.flatMap((service) => service.getSettings),
      Effect.map((settings) => settings.providers.antigravity),
    );
    const checkedAt = new Date().toISOString();
    const models = providerModelsFromSettings(
      ANTIGRAVITY_BUILT_IN_MODELS,
      PROVIDER,
      antigravitySettings.customModels,
      getAntigravityModelCapabilities,
    );

    if (!antigravitySettings.enabled) {
      return buildServerProvider({
        provider: PROVIDER,
        presentation: ANTIGRAVITY_PRESENTATION,
        enabled: false,
        checkedAt,
        models,
        probe: {
          installed: true,
          version: null,
          status: "warning",
          auth: { status: "unknown" },
          message: "Antigravity is disabled in T3 Code settings.",
        },
      });
    }

    const baseUrl = normalizeBaseUrl(antigravitySettings.baseUrl);
    const healthUrl = buildHealthUrl(baseUrl);
    if (!healthUrl) {
      return buildServerProvider({
        provider: PROVIDER,
        presentation: ANTIGRAVITY_PRESENTATION,
        enabled: true,
        checkedAt,
        models,
        probe: {
          installed: true,
          version: null,
          status: "error",
          auth: { status: "unknown" },
          message: `Antigravity base URL is invalid: ${baseUrl}`,
        },
      });
    }

    const healthcheck = yield* Effect.tryPromise({
      try: async () => {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), HEALTHCHECK_TIMEOUT_MS);
        try {
          return await fetchImpl(healthUrl, {
            method: "GET",
            signal: controller.signal,
          });
        } finally {
          clearTimeout(timeout);
        }
      },
      catch: (cause) =>
        new AntigravityHealthcheckError(
          toMessage(cause, "Failed to reach Antigravity proxy."),
          cause,
        ),
    }).pipe(Effect.result);

    if (healthcheck._tag === "Failure") {
      const message = toMessage(healthcheck.failure, "Failed to reach Antigravity proxy.");
      return buildServerProvider({
        provider: PROVIDER,
        presentation: ANTIGRAVITY_PRESENTATION,
        enabled: true,
        checkedAt,
        models,
        probe: {
          installed: true,
          version: null,
          status: "error",
          auth: { status: "unknown" },
          message: `Antigravity proxy is unreachable at ${baseUrl}. ${message}`,
        },
      });
    }

    const response = healthcheck.success;
    if (!response.ok) {
      return buildServerProvider({
        provider: PROVIDER,
        presentation: ANTIGRAVITY_PRESENTATION,
        enabled: true,
        checkedAt,
        models,
        probe: {
          installed: true,
          version: null,
          status: "error",
          auth: { status: "unknown" },
          message: `Antigravity proxy health check failed at ${baseUrl} with status ${response.status}.`,
        },
      });
    }

    return buildServerProvider({
      provider: PROVIDER,
      presentation: ANTIGRAVITY_PRESENTATION,
      enabled: true,
      checkedAt,
      models,
      probe: {
        installed: true,
        version: null,
        status: "ready",
        auth: {
          status: "authenticated",
          type: "proxy",
          label: "Local Proxy",
        },
        message: `Connected to Antigravity proxy at ${baseUrl}.`,
      },
    });
  },
);

const makePendingAntigravityProvider = (
  antigravitySettings: AntigravitySettings,
): ServerProvider => {
  const checkedAt = new Date().toISOString();
  const models = providerModelsFromSettings(
    ANTIGRAVITY_BUILT_IN_MODELS,
    PROVIDER,
    antigravitySettings.customModels,
    getAntigravityModelCapabilities,
  );

  if (!antigravitySettings.enabled) {
    return buildServerProvider({
      provider: PROVIDER,
      presentation: ANTIGRAVITY_PRESENTATION,
      enabled: false,
      checkedAt,
      models,
      probe: {
        installed: true,
        version: null,
        status: "warning",
        auth: { status: "unknown" },
        message: "Antigravity is disabled in T3 Code settings.",
      },
    });
  }

  return buildServerProvider({
    provider: PROVIDER,
    presentation: ANTIGRAVITY_PRESENTATION,
    enabled: true,
    checkedAt,
    models,
    probe: {
      installed: true,
      version: null,
      status: "warning",
      auth: { status: "unknown" },
      message: "Antigravity provider status has not been checked in this session yet.",
    },
  });
};

export const AntigravityProviderLive = Layer.effect(
  AntigravityProvider,
  Effect.gen(function* () {
    const serverSettings = yield* ServerSettingsService;
    const checkProvider = checkAntigravityProviderStatus().pipe(
      Effect.provideService(ServerSettingsService, serverSettings),
    );

    return yield* makeManagedServerProvider<AntigravitySettings>({
      getSettings: serverSettings.getSettings.pipe(
        Effect.map((settings) => settings.providers.antigravity),
        Effect.orDie,
      ),
      streamSettings: serverSettings.streamChanges.pipe(
        Stream.map((settings) => settings.providers.antigravity),
      ),
      haveSettingsChanged: (previous, next) => !Equal.equals(previous, next),
      initialSnapshot: makePendingAntigravityProvider,
      checkProvider,
    });
  }),
);
