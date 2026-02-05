import type { StreamFn } from "@mariozechner/pi-agent-core";
import type { SimpleStreamOptions } from "@mariozechner/pi-ai";
import { streamSimple } from "@mariozechner/pi-ai";
import type { OpenClawConfig } from "../../config/config.js";
import { isGlmModel } from "../../auto-reply/thinking.js";
import { log } from "./logger.js";

const OPENROUTER_APP_HEADERS: Record<string, string> = {
  "HTTP-Referer": "https://openclaw.ai",
  "X-Title": "OpenClaw",
};

/**
 * Resolve provider-specific extra params from model config.
 * Used to pass through stream params like temperature/maxTokens.
 *
 * @internal Exported for testing only
 */
export function resolveExtraParams(params: {
  cfg: OpenClawConfig | undefined;
  provider: string;
  modelId: string;
}): Record<string, unknown> | undefined {
  const modelKey = `${params.provider}/${params.modelId}`;
  const modelConfig = params.cfg?.agents?.defaults?.models?.[modelKey];
  return modelConfig?.params ? { ...modelConfig.params } : undefined;
}

type CacheRetention = "none" | "short" | "long";
type CacheRetentionStreamOptions = Partial<SimpleStreamOptions> & {
  cacheRetention?: CacheRetention;
};

type ExtraStreamOptions = CacheRetentionStreamOptions & {
  extraBody?: Record<string, unknown>;
  extraHeaders?: Record<string, string>;
};

/**
 * Resolve cacheRetention from extraParams, supporting both new `cacheRetention`
 * and legacy `cacheControlTtl` values for backwards compatibility.
 *
 * Mapping: "5m" → "short", "1h" → "long"
 *
 * Only applies to Anthropic provider (OpenRouter uses openai-completions API
 * with hardcoded cache_control, not the cacheRetention stream option).
 */
function resolveCacheRetention(
  extraParams: Record<string, unknown> | undefined,
  provider: string,
): CacheRetention | undefined {
  if (provider !== "anthropic") {
    return undefined;
  }

  // Prefer new cacheRetention if present
  const newVal = extraParams?.cacheRetention;
  if (newVal === "none" || newVal === "short" || newVal === "long") {
    return newVal;
  }

  // Fall back to legacy cacheControlTtl with mapping
  const legacy = extraParams?.cacheControlTtl;
  if (legacy === "5m") {
    return "short";
  }
  if (legacy === "1h") {
    return "long";
  }
  return undefined;
}

function createStreamFnWithExtraParams(
  baseStreamFn: StreamFn | undefined,
  extraParams: Record<string, unknown> | undefined,
  provider: string,
  opts?: { stripUnsupportedForZai?: boolean },
): StreamFn | undefined {
  if (!extraParams || Object.keys(extraParams).length === 0) {
    return undefined;
  }

  const streamParams: ExtraStreamOptions = {};
  if (typeof extraParams.temperature === "number") {
    streamParams.temperature = extraParams.temperature;
  }
  if (typeof extraParams.maxTokens === "number") {
    streamParams.maxTokens = extraParams.maxTokens;
  }
  const cacheRetention = resolveCacheRetention(extraParams, provider);
  if (cacheRetention) {
    streamParams.cacheRetention = cacheRetention;
  }
  if (extraParams?.extraBody && typeof extraParams.extraBody === "object") {
    streamParams.extraBody = extraParams.extraBody as Record<string, unknown>;
  }
  if (extraParams?.extraHeaders && typeof extraParams.extraHeaders === "object") {
    streamParams.extraHeaders = extraParams.extraHeaders as Record<string, string>;
  }

  if (Object.keys(streamParams).length === 0) {
    return undefined;
  }

  log.debug(`creating streamFn wrapper with params: ${JSON.stringify(streamParams)}`);

  const stripUnsupportedForZai = opts?.stripUnsupportedForZai === true;
  const underlying = baseStreamFn ?? streamSimple;
  const wrappedStreamFn: StreamFn = (model, context, options) => {
    const merged = { ...streamParams, ...options };
    let finalContext = context;
    if (stripUnsupportedForZai) {
      // Z.AI/GLM warns on prompt_cache_key — strip so pi-ai/provider don't send it.
      // Set to undefined so any downstream merge with defaults (e.g. prompt_cache_key: sessionId) is overwritten.
      (merged as Record<string, unknown>).prompt_cache_key = undefined;
      if (merged.extraBody && typeof merged.extraBody === "object") {
        const body = merged.extraBody as Record<string, unknown>;
        const { prompt_cache_key: _dropped, ...rest } = body;
        merged.extraBody = Object.keys(rest).length > 0 ? rest : undefined;
      }
      // Z.AI/GLM warns on "developer" role — treat as "system".
      if (context && typeof context === "object" && !Array.isArray(context)) {
        const ctx = context as unknown as Record<string, unknown>;
        const rewriteRole = (item: unknown): unknown => {
          const it = item as Record<string, unknown>;
          if (it.role === "developer") {
            return { ...it, role: "system" as const };
          }
          return it;
        };
        const hasMessages = Array.isArray(ctx.messages);
        const hasInput = Array.isArray(ctx.input);
        if (hasMessages || hasInput) {
          finalContext = {
            ...ctx,
            ...(hasMessages && {
              messages: (ctx.messages as unknown[]).map(rewriteRole),
            }),
            ...(hasInput && {
              input: (ctx.input as unknown[]).map(rewriteRole),
            }),
          } as typeof context;
        }
      }
    }
    return underlying(model, finalContext, merged);
  };

  return wrappedStreamFn;
}

/** True when provider is Z.AI (not LM Studio). Z.AI accepts extraBody.reasoning "on"|"off"; LM Studio expects enum minimal|low|medium|high. */
function isZaiProvider(provider: string): boolean {
  const p = provider?.trim().toLowerCase() ?? "";
  return p === "zai" || p === "z.ai" || p === "z-ai";
}

/**
 * Apply extra params (like temperature) to an agent's streamFn.
 * For Z.AI GLM models only, pass glmReasoning "on"|"off" in extraBody.
 * LM Studio GLM expects reasoning.effort enum (minimal|low|medium|high), so we do not send "on" there to avoid HTTP 400.
 *
 * @internal Exported for testing
 */
export function applyExtraParamsToAgent(
  agent: { streamFn?: StreamFn },
  cfg: OpenClawConfig | undefined,
  provider: string,
  modelId: string,
  extraParamsOverride?: Record<string, unknown>,
  glmReasoning?: "on" | "off",
): void {
  const extraParams = resolveExtraParams({
    cfg,
    provider,
    modelId,
  });
  const override =
    extraParamsOverride && Object.keys(extraParamsOverride).length > 0
      ? Object.fromEntries(
          Object.entries(extraParamsOverride).filter(([, value]) => value !== undefined),
        )
      : undefined;
  const merged = Object.assign({}, extraParams, override);
  if (glmReasoning !== undefined && isZaiProvider(provider)) {
    const existing = (merged.extraBody as Record<string, unknown>) ?? {};
    merged.extraBody = { ...existing, reasoning: glmReasoning };
  }
  const wrappedStreamFn = createStreamFnWithExtraParams(agent.streamFn, merged, provider, {
    stripUnsupportedForZai: isGlmModel(provider, modelId),
  });

  if (wrappedStreamFn) {
    log.debug(`applying extraParams to agent streamFn for ${provider}/${modelId}`);
    agent.streamFn = wrappedStreamFn;
  }

  if (provider === "openrouter") {
    log.debug(`applying OpenRouter app attribution headers for ${provider}/${modelId}`);
    agent.streamFn = createOpenRouterHeadersWrapper(agent.streamFn);
  }
}
