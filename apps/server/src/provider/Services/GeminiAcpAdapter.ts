import type { ProviderAdapterShape } from "./ProviderAdapter";
import { Context } from "effect";
import type { ProviderAdapterError } from "../Errors";

export interface GeminiAcpAdapterShape extends ProviderAdapterShape<ProviderAdapterError> {}

export class GeminiAcpAdapter extends Context.Service<GeminiAcpAdapter, GeminiAcpAdapterShape>()(
  "t3/provider/Services/GeminiAcpAdapter",
) {}
