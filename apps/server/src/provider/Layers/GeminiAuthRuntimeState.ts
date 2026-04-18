import { Effect, Layer, PubSub, Ref, Stream } from "effect";

import {
  GeminiAuthRuntimeState,
  type GeminiAuthFailure,
} from "../Services/GeminiAuthRuntimeState.ts";

export const GeminiAuthRuntimeStateLive = Layer.effect(
  GeminiAuthRuntimeState,
  Effect.gen(function* () {
    const failureRef = yield* Ref.make<GeminiAuthFailure | undefined>(undefined);
    const changesPubSub = yield* Effect.acquireRelease(
      PubSub.unbounded<GeminiAuthFailure | undefined>(),
      PubSub.shutdown,
    );

    const publish = (failure: GeminiAuthFailure | undefined) =>
      PubSub.publish(changesPubSub, failure).pipe(Effect.asVoid);

    return {
      getFailure: Ref.get(failureRef),
      requireManualLogin: (failure) =>
        Ref.set(failureRef, failure).pipe(Effect.flatMap(() => publish(failure))),
      clearFailure: Ref.get(failureRef).pipe(
        Effect.flatMap((failure) =>
          failure === undefined
            ? Effect.void
            : Ref.set(failureRef, undefined).pipe(Effect.flatMap(() => publish(undefined))),
        ),
      ),
      get streamChanges() {
        return Stream.fromPubSub(changesPubSub);
      },
    };
  }),
);
