import type { GeminiSettings, ServerProvider, ServerProviderModel } from "@t3tools/contracts";
import { ServerSettingsError } from "@t3tools/contracts";
import { Effect, Equal, Layer, Option, Result, Stream } from "effect";
import { ChildProcess, ChildProcessSpawner } from "effect/unstable/process";
import { makeManagedServerProvider } from "../makeManagedServerProvider";
import { GeminiAcpProvider } from "../Services/GeminiAcpProvider";
import { ServerSettingsService } from "../../serverSettings";
import {
  buildServerProvider,
  DEFAULT_TIMEOUT_MS,
  detailFromResult,
  isCommandMissingCause,
  parseGenericCliVersion,
  providerModelsFromSettings,
  spawnAndCollect,
} from "../providerSnapshot";

const PROVIDER = "geminiAcp" as const;

const GEMINI_MODEL_CAPABILITIES = {
  reasoningEffortLevels: [],
  supportsFastMode: false,
  supportsThinkingToggle: false,
  contextWindowOptions: [],
  promptInjectedEffortLevels: [],
} as const;

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

export const checkGeminiAcpProviderStatus = Effect.fn("checkGeminiAcpProviderStatus")(
  function* (): Effect.fn.Return<
    ServerProvider,
    ServerSettingsError,
    ChildProcessSpawner.ChildProcessSpawner | ServerSettingsService
  > {
    const geminiSettings = yield* Effect.service(ServerSettingsService).pipe(
      Effect.flatMap((service) => service.getSettings),
      Effect.map((settings) => settings.providers.geminiAcp),
    );
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

    // Check if gemini CLI is installed and get version
    const versionProbe = yield* runGeminiCommand(["--version"]).pipe(
      Effect.timeoutOption(DEFAULT_TIMEOUT_MS),
      Effect.result,
    );

    if (Result.isFailure(versionProbe)) {
      const error = versionProbe.failure;
      return buildServerProvider({
        provider: PROVIDER,
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

    // Gemini CLI doesn't expose a standalone auth-check command.
    // Authentication is validated during the ACP connection flow (initialize → authenticate).
    // If --version succeeds, report the provider as ready — auth errors surface at session start.
    return buildServerProvider({
      provider: PROVIDER,
      enabled: geminiSettings.enabled,
      checkedAt,
      models,
      probe: {
        installed: true,
        version: parsedVersion,
        status: "ready",
        auth: { status: "unknown" },
      },
    });
  },
);

export const GeminiAcpProviderLive = Layer.effect(
  GeminiAcpProvider,
  Effect.gen(function* () {
    const settingsService = yield* ServerSettingsService;
    const spawner = yield* ChildProcessSpawner.ChildProcessSpawner;

    const checkProvider = checkGeminiAcpProviderStatus().pipe(
      Effect.provideService(ServerSettingsService, settingsService),
      Effect.provideService(ChildProcessSpawner.ChildProcessSpawner, spawner),
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
      checkProvider,
    });
  }),
);
