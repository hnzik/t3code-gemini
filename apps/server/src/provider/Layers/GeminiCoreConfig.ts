import {
  ApprovalMode,
  AuthType,
  Config,
  createPolicyEngineConfig,
  getAuthTypeFromEnv,
  type PolicySettings,
} from "@google/gemini-cli-core";

export const GEMINI_CORE_CLIENT_VERSION = "0.37.1";
export const DEFAULT_GEMINI_MODEL = "gemini-2.5-pro";
export const FORCED_GEMINI_USER_AGENT_MODEL = "gemini-3.1-pro-preview";
export const FORCED_GOOGLE_API_NODEJS_CLIENT_VERSION = "9.15.1";

const USER_AGENT_HEADER_NAME = "User-Agent";
const GOOGLE_API_CLIENT_HEADER_NAME = "x-goog-api-client";

export function resolveGeminiAuthType(
  authTypeFromEnv: AuthType | undefined = getAuthTypeFromEnv(),
): AuthType {
  return authTypeFromEnv ?? AuthType.LOGIN_WITH_GOOGLE;
}

export function createGeminiCoreAuthHeaders(): Record<string, string> {
  const googleApiNodeJsClient = `google-api-nodejs-client/${FORCED_GOOGLE_API_NODEJS_CLIENT_VERSION}`;

  return {
    [USER_AGENT_HEADER_NAME]: `GeminiCLI/${GEMINI_CORE_CLIENT_VERSION}/${FORCED_GEMINI_USER_AGENT_MODEL} (${process.platform}; ${process.arch}; terminal) ${googleApiNodeJsClient}`,
    [GOOGLE_API_CLIENT_HEADER_NAME]: googleApiNodeJsClient,
  };
}

function parseCustomHeaders(customHeaders: string | undefined): Array<readonly [string, string]> {
  if (!customHeaders) {
    return [];
  }

  return customHeaders
    .split(/,(?=\s*[^,:]+:)/)
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0)
    .flatMap((entry) => {
      const separatorIndex = entry.indexOf(":");
      if (separatorIndex === -1) {
        return [];
      }

      const name = entry.slice(0, separatorIndex).trim();
      const value = entry.slice(separatorIndex + 1).trim();
      if (name.length === 0) {
        return [];
      }

      return [[name, value] as const];
    });
}

export function mergeGeminiCliCustomHeaders(
  customHeaders: string | undefined,
  overrides: Record<string, string> = createGeminiCoreAuthHeaders(),
): string {
  const overriddenHeaderNames = new Set(
    Object.keys(overrides).map((headerName) => headerName.toLowerCase()),
  );

  const mergedHeaders = [
    ...parseCustomHeaders(customHeaders).filter(
      ([headerName]) => !overriddenHeaderNames.has(headerName.toLowerCase()),
    ),
    ...Object.entries(overrides),
  ];

  return mergedHeaders
    .map(([headerName, headerValue]) => `${headerName}:${headerValue}`)
    .join(", ");
}

export function installGeminiCliCustomHeaders(
  overrides: Record<string, string> = createGeminiCoreAuthHeaders(),
): Record<string, string> {
  process.env.GEMINI_CLI_CUSTOM_HEADERS = mergeGeminiCliCustomHeaders(
    process.env.GEMINI_CLI_CUSTOM_HEADERS,
    overrides,
  );

  return overrides;
}

export function resolveGeminiApprovalMode(input: {
  readonly interactionMode?: string | null | undefined;
  readonly runtimeMode?: string | null | undefined;
}): ApprovalMode {
  if (input.interactionMode === "plan") {
    return ApprovalMode.PLAN;
  }

  switch (input.runtimeMode) {
    case "full-access":
    case "auto-accept-edits":
      return ApprovalMode.AUTO_EDIT;
    default:
      return ApprovalMode.DEFAULT;
  }
}

export async function createGeminiCoreConfig(input: {
  readonly sessionId: string;
  readonly cwd: string;
  readonly model?: string | null | undefined;
  readonly interactionMode?: string | null | undefined;
  readonly runtimeMode?: string | null | undefined;
  readonly interactive?: boolean | undefined;
  readonly noBrowser?: boolean | undefined;
  readonly policySettings?: PolicySettings | undefined;
}): Promise<Config> {
  const workDir = input.cwd;
  const approvalMode = resolveGeminiApprovalMode({
    interactionMode: input.interactionMode,
    runtimeMode: input.runtimeMode,
  });
  const interactive = input.interactive ?? true;
  const policyEngineConfig = await createPolicyEngineConfig(
    input.policySettings ?? {},
    approvalMode,
    undefined,
    interactive,
  );

  return new Config({
    sessionId: input.sessionId,
    targetDir: workDir,
    cwd: workDir,
    model: input.model ?? DEFAULT_GEMINI_MODEL,
    clientVersion: GEMINI_CORE_CLIENT_VERSION,
    debugMode: false,
    approvalMode,
    policyEngineConfig,
    interactive,
    ptyInfo: "node-pty",
    acpMode: false,
    ...(input.noBrowser !== undefined ? { noBrowser: input.noBrowser } : {}),
  });
}
