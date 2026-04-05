import type { ProviderAdapterShape } from "./ProviderAdapter";
import { ServiceMap } from "effect";
import type { ProviderAdapterError } from "../Errors";

export interface GeminiAcpAdapterShape extends ProviderAdapterShape<ProviderAdapterError> {}

export class GeminiAcpAdapter extends ServiceMap.Service<GeminiAcpAdapter, GeminiAcpAdapterShape>()(
  "t3/provider/Services/GeminiAcpAdapter",
) {}
