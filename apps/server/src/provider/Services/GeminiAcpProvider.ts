import { Context } from "effect";
import type { ServerProviderShape } from "./ServerProvider.ts";

export interface GeminiAcpProviderShape extends ServerProviderShape {}

export class GeminiAcpProvider extends Context.Service<GeminiAcpProvider, GeminiAcpProviderShape>()(
  "t3/provider/Services/GeminiAcpProvider",
) {}
