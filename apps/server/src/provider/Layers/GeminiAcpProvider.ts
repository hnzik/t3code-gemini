import { existsSync } from "node:fs";

import { AuthType, Storage, UserAccountManager } from "@google/gemini-cli-core";
import type {
  GeminiSettings,
  ServerProvider,
  ServerProviderAuth,
  ServerProviderModel,
  ServerProviderState,
} from "@t3tools/contracts";
import { ServerSettingsError } from "@t3tools/contracts";
import { createModelCapabilities } from "@t3tools/shared/model";
import { Effect, Equal, Layer, Option, Result, Stream } from "effect";
import { ChildProcess, ChildProcessSpawner } from "effect/unstable/process";
import { makeManagedServerProvider } from "../makeManagedServerProvider.ts";
import { GeminiAcpProvider } from "../Services/GeminiAcpProvider.ts";
import { GeminiAuthRuntimeState } from "../Services/GeminiAuthRuntimeState.ts";
import { ServerSettingsService } from "../../serverSettings.ts";
import {
  buildServerProvider,
  DEFAULT_TIMEOUT_MS,
  detailFromResult,
  isCommandMissingCause,
  parseGenericCliVersion,
  providerModelsFromSettings,
  spawnAndCollect,
} from "../providerSnapshot.ts";
import { resolveGeminiAuthType } from "./GeminiCoreConfig.ts";

const PROVIDER = "geminiAcp" as const;
const GEMINI_ACP_PRESENTATION = {
  displayName: "Gemini",
  showInteractionModeToggle: true,
} as const;
const GEMINI_GOOGLE_AUTH_MESSAGE =
  "Gemini Google OAuth could not be verified from stored CLI state during background refresh. T3 Code will retry authentication when a chat session starts.";
const GEMINI_ADC_AUTH_MESSAGE =
  "Gemini Google ADC could not be verified during background refresh. T3 Code will retry authentication when a chat session starts.";

const GEMINI_MODEL_CAPABILITIES = createModelCapabilities({
  optionDescriptors: [],
});

const BUILT_IN_MODELS: ReadonlyArray<ServerProviderModel> = [
  {
    slug: "gemini-3.1-pro-preview",
    name: "Gemini 3.1 Pro (Preview)",
    isCustom: false,
    capabilities: GEMINI_MODEL_CAPABILITIES,
  },
  {
    slug: "gemini-3-flash-preview",
    name: "Gemini 3 Flash (Preview)",
    isCustom: false,
    capabilities: GEMINI_MODEL_CAPABILITIES,
  },
  {
    slug: "gemini-2.5-pro",
    name: "Gemini 2.5 Pro",
    isCustom: false,
    capabilities: GEMINI_MODEL_CAPABILITIES,
  },
  {
    slug: "gemini-2.5-flash",
    name: "Gemini 2.5 Flash",
    isCustom: false,
    capabilities: GEMINI_MODEL_CAPABILITIES,
  },
];

const runGeminiCommand = Effect.fn("runGeminiCommand")(function* (args: ReadonlyArray<string>) {
  const settingsService = yield* ServerSettingsService;
  const geminiSettings = yield* settingsService.getSettings.pipe(
    Effect.map((settings) => settings.providers.geminiAcp),
  );
  const command = ChildProcess.make(geminiSettings.binaryPath, [...args], {
    shell: process.platform === "win32",
  });
  return yield* spawnAndCollect(geminiSettings.binaryPath, command);
});

function geminiAuthMetadata(authType: AuthType): Pick<ServerProviderAuth, "type" | "label"> {
  switch (authType) {
    case AuthType.LOGIN_WITH_GOOGLE:
      return { type: authType, label: "Google OAuth" };
    case AuthType.USE_GEMINI:
      return { type: authType, label: "Gemini API Key" };
    case AuthType.USE_VERTEX_AI:
      return { type: authType, label: "Vertex AI" };
    case AuthType.COMPUTE_ADC:
      return { type: authType, label: "Google ADC" };
    case AuthType.LEGACY_CLOUD_SHELL:
      return { type: authType, label: "Google Cloud Shell" };
    case AuthType.GATEWAY:
      return { type: authType, label: "Gateway" };
  }
}

export interface GeminiAuthProbeResult {
  readonly status: Exclude<ServerProviderState, "disabled">;
  readonly auth: Pick<ServerProviderAuth, "status">;
  readonly message?: string;
}

export function buildGeminiManualAuthRequiredMessage(input?: { readonly detail?: string }): string {
  const detail = input?.detail?.trim();
  return detail
    ? `Gemini authentication failed. Open \`gemini\` in your terminal, complete authentication there, then retry in T3 Code. ${detail}`
    : "Gemini authentication failed. Open `gemini` in your terminal, complete authentication there, then retry in T3 Code.";
}

export function validateGeminiAuthConfiguration(
  authType: AuthType,
  env: NodeJS.ProcessEnv = process.env,
): string | undefined {
  if (authType === AuthType.USE_GEMINI && !env.GEMINI_API_KEY) {
    return "Gemini API key auth requires the GEMINI_API_KEY environment variable. Update your environment and try again.";
  }

  if (authType === AuthType.USE_VERTEX_AI) {
    const hasVertexProjectLocationConfig = Boolean(
      (env.GOOGLE_CLOUD_PROJECT || env.GOOGLE_CLOUD_PROJECT_ID) && env.GOOGLE_CLOUD_LOCATION,
    );
    const hasGoogleApiKey = Boolean(env.GOOGLE_API_KEY);

    if (!hasVertexProjectLocationConfig && !hasGoogleApiKey) {
      return "Vertex AI auth requires either GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION, or GOOGLE_API_KEY for express mode. Update your environment and try again.";
    }
  }

  return undefined;
}

export function hasGeminiGoogleOAuthSession(input?: {
  readonly env?: NodeJS.ProcessEnv;
  readonly getCachedGoogleAccount?: () => string | null;
  readonly hasOauthCredentialFile?: () => boolean;
}): boolean {
  const env = input?.env ?? process.env;
  if (env.GOOGLE_GENAI_USE_GCA === "true" && env.GOOGLE_CLOUD_ACCESS_TOKEN) {
    return true;
  }

  const cachedGoogleAccount =
    input?.getCachedGoogleAccount?.() ?? new UserAccountManager().getCachedGoogleAccount();
  if (cachedGoogleAccount) {
    return true;
  }

  return input?.hasOauthCredentialFile?.() ?? existsSync(Storage.getOAuthCredsPath());
}

export function hasGeminiAdcConfiguration(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(
    env.CLOUD_SHELL === "true" ||
    env.GEMINI_CLI_USE_COMPUTE_ADC === "true" ||
    env.GOOGLE_APPLICATION_CREDENTIALS,
  );
}

export function resolveGeminiAuthProbeResult(
  authType: AuthType,
  deps?: {
    readonly hasGoogleOAuthSession?: () => boolean;
    readonly hasAdcConfiguration?: () => boolean;
  },
): GeminiAuthProbeResult {
  const staticConfigurationError = validateGeminiAuthConfiguration(authType);
  if (staticConfigurationError) {
    return {
      status: "error",
      auth: { status: "unauthenticated" },
      message: staticConfigurationError,
    };
  }

  if (authType === AuthType.LOGIN_WITH_GOOGLE) {
    const hasGoogleOAuthSession = deps?.hasGoogleOAuthSession?.() ?? hasGeminiGoogleOAuthSession();
    return hasGoogleOAuthSession
      ? {
          status: "ready",
          auth: { status: "authenticated" },
        }
      : {
          status: "warning",
          auth: { status: "unknown" },
          message: GEMINI_GOOGLE_AUTH_MESSAGE,
        };
  }

  if (authType === AuthType.COMPUTE_ADC) {
    const hasAdcConfiguration = deps?.hasAdcConfiguration?.() ?? hasGeminiAdcConfiguration();
    return hasAdcConfiguration
      ? {
          status: "ready",
          auth: { status: "authenticated" },
        }
      : {
          status: "warning",
          auth: { status: "unknown" },
          message: GEMINI_ADC_AUTH_MESSAGE,
        };
  }

  return {
    status: "ready",
    auth: { status: "authenticated" },
  };
}

export function resolveGeminiEffectiveAuthProbeResult(input: {
  readonly authType: AuthType;
  readonly manualAuthFailure?: {
    readonly message: string;
  };
  readonly deps?: {
    readonly hasGoogleOAuthSession?: () => boolean;
    readonly hasAdcConfiguration?: () => boolean;
  };
}): GeminiAuthProbeResult {
  const authProbe = resolveGeminiAuthProbeResult(input.authType, input.deps);

  if (!input.manualAuthFailure) {
    return authProbe;
  }

  if (authProbe.auth.status === "authenticated" || authProbe.status === "error") {
    return authProbe;
  }

  return {
    status: "error",
    auth: { status: "unauthenticated" },
    message: input.manualAuthFailure.message,
  };
}

export const checkGeminiAcpProviderStatus = Effect.fn("checkGeminiAcpProviderStatus")(
  function* (deps?: {
    readonly resolveAuthType?: () => AuthType;
    readonly hasGoogleOAuthSession?: () => boolean;
    readonly hasAdcConfiguration?: () => boolean;
  }): Effect.fn.Return<
    ServerProvider,
    ServerSettingsError,
    ChildProcessSpawner.ChildProcessSpawner | GeminiAuthRuntimeState | ServerSettingsService
  > {
    const geminiSettings = yield* Effect.service(ServerSettingsService).pipe(
      Effect.flatMap((service) => service.getSettings),
      Effect.map((settings) => settings.providers.geminiAcp),
    );
    const authRuntimeState = yield* GeminiAuthRuntimeState;
    const checkedAt = new Date().toISOString();
    const models = providerModelsFromSettings(
      BUILT_IN_MODELS,
      PROVIDER,
      geminiSettings.customModels,
      GEMINI_MODEL_CAPABILITIES,
    );

    if (!geminiSettings.enabled) {
      return buildServerProvider({
        provider: PROVIDER,
        presentation: GEMINI_ACP_PRESENTATION,
        enabled: false,
        checkedAt,
        models,
        probe: {
          installed: false,
          version: null,
          status: "warning",
          auth: { status: "unknown" },
          message: "Gemini is disabled in T3 Code settings.",
        },
      });
    }

    const versionProbe = yield* runGeminiCommand(["--version"]).pipe(
      Effect.timeoutOption(DEFAULT_TIMEOUT_MS),
      Effect.result,
    );

    if (Result.isFailure(versionProbe)) {
      const error = versionProbe.failure;
      return buildServerProvider({
        provider: PROVIDER,
        presentation: GEMINI_ACP_PRESENTATION,
        enabled: geminiSettings.enabled,
        checkedAt,
        models,
        probe: {
          installed: !isCommandMissingCause(error),
          version: null,
          status: "error",
          auth: { status: "unknown" },
          message: isCommandMissingCause(error)
            ? "Gemini CLI (`gemini`) is not installed or not on PATH."
            : `Failed to execute Gemini CLI health check: ${error instanceof Error ? error.message : String(error)}.`,
        },
      });
    }

    if (Option.isNone(versionProbe.success)) {
      return buildServerProvider({
        provider: PROVIDER,
        presentation: GEMINI_ACP_PRESENTATION,
        enabled: geminiSettings.enabled,
        checkedAt,
        models,
        probe: {
          installed: true,
          version: null,
          status: "error",
          auth: { status: "unknown" },
          message: "Gemini CLI is installed but failed to run. Timed out while running command.",
        },
      });
    }

    const version = versionProbe.success.value;
    const parsedVersion = parseGenericCliVersion(`${version.stdout}\n${version.stderr}`);

    if (version.code !== 0) {
      const detail = detailFromResult(version);
      return buildServerProvider({
        provider: PROVIDER,
        presentation: GEMINI_ACP_PRESENTATION,
        enabled: geminiSettings.enabled,
        checkedAt,
        models,
        probe: {
          installed: true,
          version: parsedVersion,
          status: "error",
          auth: { status: "unknown" },
          message: detail
            ? `Gemini CLI is installed but failed to run. ${detail}`
            : "Gemini CLI is installed but failed to run.",
        },
      });
    }

    const authType = deps?.resolveAuthType?.() ?? resolveGeminiAuthType();
    const authMetadata = geminiAuthMetadata(authType);
    const manualAuthFailure = yield* authRuntimeState.getFailure;
    const authProbeDeps =
      deps &&
      ({
        ...(deps.hasGoogleOAuthSession
          ? { hasGoogleOAuthSession: deps.hasGoogleOAuthSession }
          : {}),
        ...(deps.hasAdcConfiguration ? { hasAdcConfiguration: deps.hasAdcConfiguration } : {}),
      } satisfies NonNullable<Parameters<typeof resolveGeminiAuthProbeResult>[1]>);
    const authProbe = resolveGeminiEffectiveAuthProbeResult({
      authType,
      ...(authProbeDeps ? { deps: authProbeDeps } : {}),
      ...(manualAuthFailure ? { manualAuthFailure } : {}),
    });

    if (manualAuthFailure && authProbe.auth.status === "authenticated") {
      yield* authRuntimeState.clearFailure;
    }

    return buildServerProvider({
      provider: PROVIDER,
      presentation: GEMINI_ACP_PRESENTATION,
      enabled: geminiSettings.enabled,
      checkedAt,
      models,
      probe: {
        installed: true,
        version: parsedVersion,
        status: authProbe.status,
        auth: { ...authProbe.auth, ...authMetadata },
        ...(authProbe.message ? { message: authProbe.message } : {}),
      },
    });
  },
);

const makePendingGeminiAcpProvider = (geminiSettings: GeminiSettings): ServerProvider => {
  const checkedAt = new Date().toISOString();
  const models = providerModelsFromSettings(
    BUILT_IN_MODELS,
    PROVIDER,
    geminiSettings.customModels,
    GEMINI_MODEL_CAPABILITIES,
  );

  if (!geminiSettings.enabled) {
    return buildServerProvider({
      provider: PROVIDER,
      presentation: GEMINI_ACP_PRESENTATION,
      enabled: false,
      checkedAt,
      models,
      probe: {
        installed: false,
        version: null,
        status: "warning",
        auth: { status: "unknown" },
        message: "Gemini ACP is disabled in T3 Code settings.",
      },
    });
  }

  return buildServerProvider({
    provider: PROVIDER,
    presentation: GEMINI_ACP_PRESENTATION,
    enabled: true,
    checkedAt,
    models,
    probe: {
      installed: false,
      version: null,
      status: "warning",
      auth: { status: "unknown" },
      message: "Gemini ACP provider status has not been checked in this session yet.",
    },
  });
};

export const GeminiAcpProviderLive = Layer.effect(
  GeminiAcpProvider,
  Effect.gen(function* () {
    const settingsService = yield* ServerSettingsService;
    const spawner = yield* ChildProcessSpawner.ChildProcessSpawner;
    const authRuntimeState = yield* GeminiAuthRuntimeState;

    const checkProvider = checkGeminiAcpProviderStatus().pipe(
      Effect.provideService(ServerSettingsService, settingsService),
      Effect.provideService(ChildProcessSpawner.ChildProcessSpawner, spawner),
      Effect.provideService(GeminiAuthRuntimeState, authRuntimeState),
    );

    return yield* makeManagedServerProvider<GeminiSettings>({
      getSettings: Effect.map(settingsService.getSettings, (s) => s.providers.geminiAcp).pipe(
        Effect.orDie,
      ),
      streamSettings: Stream.map(settingsService.streamChanges, (s) => s.providers.geminiAcp),
      haveSettingsChanged: (prev, next) =>
        prev.enabled !== next.enabled ||
        prev.binaryPath !== next.binaryPath ||
        !Equal.equals(prev.customModels, next.customModels),
      initialSnapshot: makePendingGeminiAcpProvider,
      checkProvider,
      refreshTriggers: authRuntimeState.streamChanges,
    });
  }),
);
