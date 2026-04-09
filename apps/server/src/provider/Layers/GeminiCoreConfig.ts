import { ApprovalMode, AuthType, Config, getAuthTypeFromEnv } from "@google/gemini-cli-core";

export const GEMINI_CORE_CLIENT_VERSION = "0.37.0";
export const DEFAULT_GEMINI_MODEL = "gemini-2.5-pro";

export function resolveGeminiAuthType(
  authTypeFromEnv: AuthType | undefined = getAuthTypeFromEnv(),
): AuthType {
  return authTypeFromEnv ?? AuthType.LOGIN_WITH_GOOGLE;
}

export function createGeminiCoreConfig(input: {
  readonly sessionId: string;
  readonly cwd: string;
  readonly model?: string | null | undefined;
  readonly runtimeMode?: string | null | undefined;
  readonly interactive?: boolean | undefined;
  readonly noBrowser?: boolean | undefined;
}): Config {
  const workDir = input.cwd;

  return new Config({
    sessionId: input.sessionId,
    targetDir: workDir,
    cwd: workDir,
    model: input.model ?? DEFAULT_GEMINI_MODEL,
    clientVersion: GEMINI_CORE_CLIENT_VERSION,
    debugMode: false,
    approvalMode:
      input.runtimeMode === "full-access" ? ApprovalMode.AUTO_EDIT : ApprovalMode.DEFAULT,
    interactive: input.interactive ?? true,
    ptyInfo: "node-pty",
    acpMode: true,
    ...(input.noBrowser !== undefined ? { noBrowser: input.noBrowser } : {}),
  });
}
