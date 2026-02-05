import type { ThinkingLevel } from "@mariozechner/pi-agent-core";
import type { ReasoningLevel, ThinkLevel } from "../../auto-reply/thinking.js";
import { isGlmModel } from "../../auto-reply/thinking.js";
import type { OpenClawConfig } from "../../config/config.js";
import type { ExecToolDefaults } from "../bash-tools.js";

/**
 * Map thinking level for the session/API. The standard API uses
 * reasoning.effort with enum 'minimal'|'low'|'medium'|'high'. For GLM (Z.AI),
 * only 'on'|'off' is supported and that is set via extraBody.reasoning; we must
 * not send effort (minimal/low/...) or the provider warns. So for GLM we always
 * pass "off" here and rely on extraBody.reasoning for on/off.
 */
export function mapThinkingLevel(
  level?: ThinkLevel,
  provider?: string | null,
  modelId?: string | null,
): ThinkingLevel {
  if (!level) {
    return "off";
  }
  if (isGlmModel(provider, modelId)) {
    return "off";
  }
  return level;
}

export function resolveExecToolDefaults(config?: OpenClawConfig): ExecToolDefaults | undefined {
  const tools = config?.tools;
  if (!tools?.exec) {
    return undefined;
  }
  return tools.exec;
}

export function describeUnknownError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === "string") {
    return error;
  }
  try {
    const serialized = JSON.stringify(error);
    return serialized ?? "Unknown error";
  } catch {
    return "Unknown error";
  }
}

export type { ReasoningLevel, ThinkLevel };
