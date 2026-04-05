import { ServiceMap } from "effect";
import type { ServerProviderShape } from "./ServerProvider";

export interface GeminiAcpProviderShape extends ServerProviderShape {}

export class GeminiAcpProvider extends ServiceMap.Service<
  GeminiAcpProvider,
  GeminiAcpProviderShape
>()("t3/provider/Services/GeminiAcpProvider") {}
