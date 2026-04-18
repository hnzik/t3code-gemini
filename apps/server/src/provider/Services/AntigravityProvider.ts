import { Context } from "effect";

import type { ServerProviderShape } from "./ServerProvider.ts";

export interface AntigravityProviderShape extends ServerProviderShape {}

export class AntigravityProvider extends Context.Service<
  AntigravityProvider,
  AntigravityProviderShape
>()("t3/provider/Services/AntigravityProvider") {}
