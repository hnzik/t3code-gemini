import type { AuthType } from "@google/gemini-cli-core";
import { Context } from "effect";
import type { Effect, Stream } from "effect";

export interface GeminiAuthFailure {
  readonly authType: AuthType;
  readonly failedAt: string;
  readonly message: string;
}

export interface GeminiAuthRuntimeStateShape {
  readonly getFailure: Effect.Effect<GeminiAuthFailure | undefined>;
  readonly requireManualLogin: (failure: GeminiAuthFailure) => Effect.Effect<void>;
  readonly clearFailure: Effect.Effect<void>;
  readonly streamChanges: Stream.Stream<GeminiAuthFailure | undefined>;
}

export class GeminiAuthRuntimeState extends Context.Service<
  GeminiAuthRuntimeState,
  GeminiAuthRuntimeStateShape
>()("t3/provider/Services/GeminiAuthRuntimeState") {}
