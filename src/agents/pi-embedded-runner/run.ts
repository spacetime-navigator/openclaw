import fs from "node:fs/promises";
import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { SessionManager } from "@mariozechner/pi-coding-agent";
import type { ReasoningLevel, ThinkLevel } from "../../auto-reply/thinking.js";
import type { ExecElevatedDefaults } from "../bash-tools.js";
import type { OpenClawConfig } from "../../config/config.js";
import type { RunEmbeddedPiAgentParams } from "./run/params.js";
import type { EmbeddedPiAgentMeta, EmbeddedPiRunResult } from "./types.js";
import { enqueueCommandInLane } from "../../process/command-queue.js";
import { resolveUserPath } from "../../utils.js";
import { isMarkdownCapableMessageChannel } from "../../utils/message-channel.js";
import { resolveOpenClawAgentDir } from "../agent-paths.js";
import {
  isProfileInCooldown,
  markAuthProfileFailure,
  markAuthProfileGood,
  markAuthProfileUsed,
} from "../auth-profiles.js";
import {
  CONTEXT_WINDOW_HARD_MIN_TOKENS,
  CONTEXT_WINDOW_WARN_BELOW_TOKENS,
  evaluateContextWindowGuard,
  resolveContextWindowInfo,
} from "../context-window-guard.js";
import { DEFAULT_CONTEXT_TOKENS, DEFAULT_MODEL, DEFAULT_PROVIDER } from "../defaults.js";
import { FailoverError, resolveFailoverStatus } from "../failover-error.js";
import {
  ensureAuthProfileStore,
  getApiKeyForModel,
  resolveAuthProfileOrder,
  type ResolvedProviderAuth,
} from "../model-auth.js";
import { normalizeProviderId } from "../model-selection.js";
import { ensureOpenClawModelsJson } from "../models-config.js";
import {
  BILLING_ERROR_USER_MESSAGE,
  classifyFailoverReason,
  formatAssistantErrorText,
  isAuthAssistantError,
  isBillingAssistantError,
  isCompactionFailureError,
  isContextOverflowError,
  isFailoverAssistantError,
  isFailoverErrorMessage,
  parseImageSizeError,
  parseImageDimensionError,
  isRateLimitAssistantError,
  isTimeoutErrorMessage,
  pickFallbackThinkingLevel,
  type FailoverReason,
} from "../pi-embedded-helpers.js";
import { normalizeUsage, type UsageLike } from "../usage.js";
import { estimateMessagesTokens } from "../compaction.js";
import { createOpenClawCodingTools } from "../pi-tools.js";
import { compactEmbeddedPiSessionDirect } from "./compact.js";
import { resolveGlobalLane, resolveSessionLane } from "./lanes.js";
import { log } from "./logger.js";
import { resolveModel } from "./model.js";
import { runEmbeddedAttempt } from "./run/attempt.js";
import { buildEmbeddedRunPayloads } from "./run/payloads.js";
import { buildEmbeddedSystemPrompt } from "./system-prompt.js";
import { describeUnknownError } from "./utils.js";

type ApiKeyInfo = ResolvedProviderAuth;

// Avoid Anthropic's refusal test token poisoning session transcripts.
const ANTHROPIC_MAGIC_STRING_TRIGGER_REFUSAL = "ANTHROPIC_MAGIC_STRING_TRIGGER_REFUSAL";
const ANTHROPIC_MAGIC_STRING_REPLACEMENT = "ANTHROPIC MAGIC STRING TRIGGER REFUSAL (redacted)";
const DEFAULT_COMPACTION_HISTORY_LIMIT = 0.7;

function scrubAnthropicRefusalMagic(prompt: string): string {
  if (!prompt.includes(ANTHROPIC_MAGIC_STRING_TRIGGER_REFUSAL)) {
    return prompt;
  }
  return prompt.replaceAll(
    ANTHROPIC_MAGIC_STRING_TRIGGER_REFUSAL,
    ANTHROPIC_MAGIC_STRING_REPLACEMENT,
  );
}

function resolveCompactionHistoryLimit(cfg?: OpenClawConfig): number {
  const raw = cfg?.agents?.defaults?.compaction?.historyLimit;
  if (typeof raw === "number" && Number.isFinite(raw)) {
    return Math.min(0.9, Math.max(0.1, raw));
  }
  return DEFAULT_COMPACTION_HISTORY_LIMIT;
}

function estimatePromptTokens(prompt: string): number {
  if (!prompt) {
    return 0;
  }
  const message = {
    role: "user",
    content: [{ type: "text", text: prompt }],
    timestamp: Date.now(),
  } as AgentMessage;
  return estimateMessagesTokens([message]);
}

async function estimateSessionHistoryTokens(sessionFile: string): Promise<number | null> {
  try {
    await fs.stat(sessionFile);
  } catch {
    return 0;
  }
  try {
    const sessionManager = SessionManager.open(sessionFile);
    const context = sessionManager.buildSessionContext();
    const messages = Array.isArray(context?.messages) ? context.messages : [];
    return estimateMessagesTokens(messages);
  } catch {
    return null;
  }
}

async function estimateSystemPromptAndToolTokens(params: {
  workspaceDir: string;
  config?: OpenClawConfig;
  sessionKey?: string;
  sessionId: string;
  provider: string;
  modelId: string;
  agentDir: string;
  messageChannel?: string;
  messageProvider?: string;
  agentAccountId?: string;
  messageTo?: string;
  messageThreadId?: string | number;
  groupId?: string | null;
  groupChannel?: string | null;
  groupSpace?: string | null;
  spawnedBy?: string | null;
  senderId?: string;
  senderName?: string;
  senderUsername?: string;
  senderE164?: string;
  bashElevated?: ExecElevatedDefaults;
  thinkLevel?: ThinkLevel;
  reasoningLevel?: ReasoningLevel;
  extraSystemPrompt?: string;
  disableTools?: boolean;
}): Promise<number> {
  try {
    const { resolveSandboxContext } = await import("../sandbox.js");
    const { resolveBootstrapContextForRun, makeBootstrapWarn } = await import(
      "../bootstrap-files.js"
    );
    const { resolveSessionAgentIds } = await import("../agent-scope.js");
    const { buildSystemPromptParams } = await import("../system-prompt-params.js");
    const { getMachineDisplayName } = await import("../../infra/machine-name.js");
    const { normalizeMessageChannel } = await import("../../utils/message-channel.js");
    const { resolveChannelCapabilities } = await import("../../config/channel-capabilities.js");
    const { resolveModelAuthMode } = await import("../model-auth.js");
    const { resolveDefaultModelForAgent } = await import("../model-selection.js");
    const { isSubagentSessionKey } = await import("../../routing/session-key.js");
    const { resolveOpenClawDocsPath } = await import("../docs-path.js");
    const { buildTtsSystemPromptHint } = await import("../../tts/tts.js");
    const { resolveHeartbeatPrompt } = await import("../../auto-reply/heartbeat.js");
    const { loadWorkspaceSkillEntries, resolveSkillsPromptForRun } = await import("../skills.js");
    const { buildModelAliasLines } = await import("./model.js");
    const { sanitizeToolsForGoogle } = await import("./google.js");
    const { DEFAULT_BOOTSTRAP_FILENAME } = await import("../workspace.js");
    const { resolveSandboxRuntimeStatus } = await import("../sandbox.js");
    const { isReasoningTagProvider } = await import("../../utils/provider-utils.js");
    const { listChannelSupportedActions, resolveChannelMessageToolHints } = await import(
      "../channel-tools.js"
    );
    const { resolveTelegramInlineButtonsScope } = await import("../../telegram/inline-buttons.js");
    const { resolveTelegramReactionLevel } = await import("../../telegram/reaction-level.js");
    const { resolveSignalReactionLevel } = await import("../../signal/reaction-level.js");
    const { buildDbRecallContext } = await import("./run/attempt.js");

    const effectiveWorkspace = resolveUserPath(params.workspaceDir);
    const sandboxSessionKey = params.sessionKey?.trim() || params.sessionId;
    const sandbox = await resolveSandboxContext({
      config: params.config,
      sessionKey: sandboxSessionKey,
      workspaceDir: effectiveWorkspace,
    });
    const { bootstrapFiles: hookAdjustedBootstrapFiles, contextFiles } =
      await resolveBootstrapContextForRun({
        workspaceDir: effectiveWorkspace,
        config: params.config,
        sessionKey: params.sessionKey,
        sessionId: params.sessionId,
        warn: makeBootstrapWarn({
          sessionLabel: sandboxSessionKey,
          warn: (message) => log.debug(message),
        }),
      });
    const { sessionAgentId } = resolveSessionAgentIds({
      sessionKey: params.sessionKey ?? params.sessionId,
      config: params.config,
    });
    const recallContext = await buildDbRecallContext({
      config: params.config,
      agentId: sessionAgentId ?? "main",
      sessionKey: params.sessionKey ?? params.sessionId,
      query: "",
      contextFiles,
    }).catch(() => null);
    const augmentedContextFiles = recallContext
      ? [...contextFiles, recallContext]
      : contextFiles;
    const workspaceNotes = hookAdjustedBootstrapFiles.some(
      (file) => file.name === DEFAULT_BOOTSTRAP_FILENAME && !file.missing,
    )
      ? ["Reminder: commit your changes in this workspace after edits."]
      : undefined;

    const modelHasVision = false; // Conservative estimate
    const toolsRaw = params.disableTools
      ? []
      : createOpenClawCodingTools({
          exec: {
            elevated: params.bashElevated,
          },
          sandbox,
          messageProvider: params.messageChannel ?? params.messageProvider,
          agentAccountId: params.agentAccountId,
          messageTo: params.messageTo,
          messageThreadId: params.messageThreadId,
          groupId: params.groupId,
          groupChannel: params.groupChannel,
          groupSpace: params.groupSpace,
          spawnedBy: params.spawnedBy,
          senderId: params.senderId,
          senderName: params.senderName,
          senderUsername: params.senderUsername,
          senderE164: params.senderE164,
          sessionKey: params.sessionKey ?? params.sessionId,
          agentDir: params.agentDir,
          workspaceDir: effectiveWorkspace,
          config: params.config,
          abortSignal: new AbortController().signal,
          modelProvider: params.provider,
          modelId: params.modelId,
          modelAuthMode: resolveModelAuthMode(params.provider, params.config),
          modelHasVision,
        });
    const tools = sanitizeToolsForGoogle({ tools: toolsRaw, provider: params.provider });

    // Estimate tool tokens by serializing schemas
    let toolTokens = 0;
    for (const tool of tools) {
      if (tool.parameters && typeof tool.parameters === "object") {
        try {
          const schemaStr = JSON.stringify(tool.parameters);
          toolTokens += estimatePromptTokens(schemaStr);
        } catch {
          toolTokens += 100; // Fallback estimate per tool
        }
      }
      if (tool.description) {
        toolTokens += estimatePromptTokens(tool.description);
      }
      toolTokens += estimatePromptTokens(tool.name);
    }

    const machineName = await getMachineDisplayName();
    const runtimeChannel = normalizeMessageChannel(params.messageChannel ?? params.messageProvider);
    const runtimeCapabilities = runtimeChannel
      ? resolveChannelCapabilities({
          cfg: params.config,
          channel: runtimeChannel,
          accountId: params.agentAccountId,
        }) ?? []
      : undefined;
    const { sessionAgentId: sessionAgentId2 } = resolveSessionAgentIds({
      sessionKey: params.sessionKey,
      config: params.config,
    });
    const defaultModelRef = resolveDefaultModelForAgent({
      cfg: params.config ?? {},
      agentId: sessionAgentId2,
    });
    const defaultModelLabel = `${params.provider}/${params.modelId}`;
    const { runtimeInfo, userTimezone, userTime, userTimeFormat } = buildSystemPromptParams({
      config: params.config,
      agentId: sessionAgentId2,
      workspaceDir: effectiveWorkspace,
      cwd: effectiveWorkspace,
      runtime: {
        host: machineName,
        os: "unknown",
        arch: "unknown",
        node: process.version,
        model: `${params.provider}/${params.modelId}`,
        defaultModel: defaultModelLabel,
        channel: runtimeChannel,
        capabilities: runtimeCapabilities,
      },
    });
    const isDefaultAgent = sessionAgentId2 === "main";
    const promptMode = isSubagentSessionKey(params.sessionKey) ? "minimal" : "full";
    const docsPath = await resolveOpenClawDocsPath({
      workspaceDir: effectiveWorkspace,
      argv1: process.argv[1],
      cwd: effectiveWorkspace,
      moduleUrl: import.meta.url,
    });
    const ttsHint = params.config ? buildTtsSystemPromptHint(params.config) : undefined;
    const skillEntries = loadWorkspaceSkillEntries(effectiveWorkspace);
    const skillsPrompt = resolveSkillsPromptForRun({
      entries: skillEntries,
      config: params.config,
      workspaceDir: effectiveWorkspace,
    });
    const reasoningTagHint = isReasoningTagProvider(params.provider);

    const systemPrompt = buildEmbeddedSystemPrompt({
      workspaceDir: effectiveWorkspace,
      defaultThinkLevel: params.thinkLevel,
      reasoningLevel: params.reasoningLevel ?? "off",
      extraSystemPrompt: params.extraSystemPrompt,
      reasoningTagHint,
      heartbeatPrompt: isDefaultAgent
        ? resolveHeartbeatPrompt(params.config?.agents?.defaults?.heartbeat?.prompt)
        : undefined,
      skillsPrompt,
      docsPath: docsPath ?? undefined,
      ttsHint,
      workspaceNotes,
      promptMode,
      runtimeInfo,
      sandboxInfo: sandbox ? { enabled: sandbox.enabled } : undefined,
      tools,
      modelAliasLines: buildModelAliasLines(params.config),
      userTimezone,
      userTime,
      userTimeFormat,
      contextFiles: augmentedContextFiles,
    });

    const systemPromptTokens = estimatePromptTokens(systemPrompt);

    return systemPromptTokens + toolTokens;
  } catch (err) {
    log.debug(`system prompt/tool token estimation failed: ${String(err)}`);
    // Conservative fallback: assume 5000 tokens for system prompt + tools
    return 5000;
  }
}

export async function runEmbeddedPiAgent(
  params: RunEmbeddedPiAgentParams,
): Promise<EmbeddedPiRunResult> {
  const sessionLane = resolveSessionLane(params.sessionKey?.trim() || params.sessionId);
  const globalLane = resolveGlobalLane(params.lane);
  const enqueueGlobal =
    params.enqueue ?? ((task, opts) => enqueueCommandInLane(globalLane, task, opts));
  const enqueueSession =
    params.enqueue ?? ((task, opts) => enqueueCommandInLane(sessionLane, task, opts));
  const channelHint = params.messageChannel ?? params.messageProvider;
  const resolvedToolResultFormat =
    params.toolResultFormat ??
    (channelHint
      ? isMarkdownCapableMessageChannel(channelHint)
        ? "markdown"
        : "plain"
      : "markdown");
  const isProbeSession = params.sessionId?.startsWith("probe-") ?? false;

  return enqueueSession(() =>
    enqueueGlobal(async () => {
      const started = Date.now();
      const resolvedWorkspace = resolveUserPath(params.workspaceDir);
      const prevCwd = process.cwd();

      const provider = (params.provider ?? DEFAULT_PROVIDER).trim() || DEFAULT_PROVIDER;
      const modelId = (params.model ?? DEFAULT_MODEL).trim() || DEFAULT_MODEL;
      const agentDir = params.agentDir ?? resolveOpenClawAgentDir();
      const fallbackConfigured =
        (params.config?.agents?.defaults?.model?.fallbacks?.length ?? 0) > 0;
      await ensureOpenClawModelsJson(params.config, agentDir);

      const { model, error, authStorage, modelRegistry } = resolveModel(
        provider,
        modelId,
        agentDir,
        params.config,
      );
      if (!model) {
        throw new Error(error ?? `Unknown model: ${provider}/${modelId}`);
      }

      const ctxInfo = resolveContextWindowInfo({
        cfg: params.config,
        provider,
        modelId,
        modelContextWindow: model.contextWindow,
        defaultTokens: DEFAULT_CONTEXT_TOKENS,
      });
      log.debug(
        `run context window: ${ctxInfo.tokens} tokens (source=${ctxInfo.source}) for ${provider}/${modelId}`,
      );
      const compactionHistoryLimit = resolveCompactionHistoryLimit(params.config);
      const ctxGuard = evaluateContextWindowGuard({
        info: ctxInfo,
        warnBelowTokens: CONTEXT_WINDOW_WARN_BELOW_TOKENS,
        hardMinTokens: CONTEXT_WINDOW_HARD_MIN_TOKENS,
      });
      if (ctxGuard.shouldWarn) {
        log.warn(
          `low context window: ${provider}/${modelId} ctx=${ctxGuard.tokens} (warn<${CONTEXT_WINDOW_WARN_BELOW_TOKENS}) source=${ctxGuard.source}`,
        );
      }
      if (ctxGuard.shouldBlock) {
        log.error(
          `blocked model (context window too small): ${provider}/${modelId} ctx=${ctxGuard.tokens} (min=${CONTEXT_WINDOW_HARD_MIN_TOKENS}) source=${ctxGuard.source}`,
        );
        throw new FailoverError(
          `Model context window too small (${ctxGuard.tokens} tokens). Minimum is ${CONTEXT_WINDOW_HARD_MIN_TOKENS}.`,
          { reason: "unknown", provider, model: modelId },
        );
      }

      const authStore = ensureAuthProfileStore(agentDir, { allowKeychainPrompt: false });
      const preferredProfileId = params.authProfileId?.trim();
      let lockedProfileId = params.authProfileIdSource === "user" ? preferredProfileId : undefined;
      if (lockedProfileId) {
        const lockedProfile = authStore.profiles[lockedProfileId];
        if (
          !lockedProfile ||
          normalizeProviderId(lockedProfile.provider) !== normalizeProviderId(provider)
        ) {
          lockedProfileId = undefined;
        }
      }
      const profileOrder = resolveAuthProfileOrder({
        cfg: params.config,
        store: authStore,
        provider,
        preferredProfile: preferredProfileId,
      });
      if (lockedProfileId && !profileOrder.includes(lockedProfileId)) {
        throw new Error(`Auth profile "${lockedProfileId}" is not configured for ${provider}.`);
      }
      const profileCandidates = lockedProfileId
        ? [lockedProfileId]
        : profileOrder.length > 0
          ? profileOrder
          : [undefined];
      let profileIndex = 0;

      const initialThinkLevel = params.thinkLevel ?? "off";
      let thinkLevel = initialThinkLevel;
      const attemptedThinking = new Set<ThinkLevel>();
      let apiKeyInfo: ApiKeyInfo | null = null;
      let lastProfileId: string | undefined;

      const resolveAuthProfileFailoverReason = (params: {
        allInCooldown: boolean;
        message: string;
      }): FailoverReason => {
        if (params.allInCooldown) {
          return "rate_limit";
        }
        const classified = classifyFailoverReason(params.message);
        return classified ?? "auth";
      };

      const throwAuthProfileFailover = (params: {
        allInCooldown: boolean;
        message?: string;
        error?: unknown;
      }): never => {
        const fallbackMessage = `No available auth profile for ${provider} (all in cooldown or unavailable).`;
        const message =
          params.message?.trim() ||
          (params.error ? describeUnknownError(params.error).trim() : "") ||
          fallbackMessage;
        const reason = resolveAuthProfileFailoverReason({
          allInCooldown: params.allInCooldown,
          message,
        });
        if (fallbackConfigured) {
          throw new FailoverError(message, {
            reason,
            provider,
            model: modelId,
            status: resolveFailoverStatus(reason),
            cause: params.error,
          });
        }
        if (params.error instanceof Error) {
          throw params.error;
        }
        throw new Error(message);
      };

      const resolveApiKeyForCandidate = async (candidate?: string) => {
        return getApiKeyForModel({
          model,
          cfg: params.config,
          profileId: candidate,
          store: authStore,
          agentDir,
        });
      };

      const applyApiKeyInfo = async (candidate?: string): Promise<void> => {
        apiKeyInfo = await resolveApiKeyForCandidate(candidate);
        const resolvedProfileId = apiKeyInfo.profileId ?? candidate;
        if (!apiKeyInfo.apiKey) {
          if (apiKeyInfo.mode !== "aws-sdk") {
            throw new Error(
              `No API key resolved for provider "${model.provider}" (auth mode: ${apiKeyInfo.mode}).`,
            );
          }
          lastProfileId = resolvedProfileId;
          return;
        }
        if (model.provider === "github-copilot") {
          const { resolveCopilotApiToken } =
            await import("../../providers/github-copilot-token.js");
          const copilotToken = await resolveCopilotApiToken({
            githubToken: apiKeyInfo.apiKey,
          });
          authStorage.setRuntimeApiKey(model.provider, copilotToken.token);
        } else {
          authStorage.setRuntimeApiKey(model.provider, apiKeyInfo.apiKey);
        }
        lastProfileId = apiKeyInfo.profileId;
      };

      const advanceAuthProfile = async (): Promise<boolean> => {
        if (lockedProfileId) {
          return false;
        }
        let nextIndex = profileIndex + 1;
        while (nextIndex < profileCandidates.length) {
          const candidate = profileCandidates[nextIndex];
          if (candidate && isProfileInCooldown(authStore, candidate)) {
            nextIndex += 1;
            continue;
          }
          try {
            await applyApiKeyInfo(candidate);
            profileIndex = nextIndex;
            thinkLevel = initialThinkLevel;
            attemptedThinking.clear();
            return true;
          } catch (err) {
            if (candidate && candidate === lockedProfileId) {
              throw err;
            }
            nextIndex += 1;
          }
        }
        return false;
      };

      try {
        while (profileIndex < profileCandidates.length) {
          const candidate = profileCandidates[profileIndex];
          if (
            candidate &&
            candidate !== lockedProfileId &&
            isProfileInCooldown(authStore, candidate)
          ) {
            profileIndex += 1;
            continue;
          }
          await applyApiKeyInfo(profileCandidates[profileIndex]);
          break;
        }
        if (profileIndex >= profileCandidates.length) {
          throwAuthProfileFailover({ allInCooldown: true });
        }
      } catch (err) {
        if (err instanceof FailoverError) {
          throw err;
        }
        if (profileCandidates[profileIndex] === lockedProfileId) {
          throwAuthProfileFailover({ allInCooldown: false, error: err });
        }
        const advanced = await advanceAuthProfile();
        if (!advanced) {
          throwAuthProfileFailover({ allInCooldown: false, error: err });
        }
      }

      const MAX_OVERFLOW_COMPACTION_ATTEMPTS = 3;
      let overflowCompactionAttempts = 0;
      try {
        while (true) {
          attemptedThinking.add(thinkLevel);
          await fs.mkdir(resolvedWorkspace, { recursive: true });

          const prompt =
            provider === "anthropic" ? scrubAnthropicRefusalMagic(params.prompt) : params.prompt;

          const preHistoryTokens = await estimateSessionHistoryTokens(params.sessionFile);
          if (preHistoryTokens !== null) {
            const promptTokens = estimatePromptTokens(prompt);
            const systemPromptAndToolTokens = await estimateSystemPromptAndToolTokens({
              workspaceDir: params.workspaceDir,
              config: params.config,
              sessionKey: params.sessionKey,
              sessionId: params.sessionId,
              provider,
              modelId,
              agentDir,
              messageChannel: params.messageChannel,
              messageProvider: params.messageProvider,
              agentAccountId: params.agentAccountId,
              messageTo: params.messageTo,
              messageThreadId: params.messageThreadId,
              groupId: params.groupId,
              groupChannel: params.groupChannel,
              groupSpace: params.groupSpace,
              spawnedBy: params.spawnedBy,
              senderId: params.senderId ?? undefined,
              senderName: params.senderName ?? undefined,
              senderUsername: params.senderUsername ?? undefined,
              senderE164: params.senderE164 ?? undefined,
              bashElevated: params.bashElevated,
              thinkLevel,
              reasoningLevel: params.reasoningLevel,
              extraSystemPrompt: params.extraSystemPrompt,
              disableTools: params.disableTools,
            });
            const totalTokens =
              preHistoryTokens + promptTokens + systemPromptAndToolTokens;
            if (totalTokens > ctxInfo.tokens) {
              log.warn(
                `pre-run compaction: estimated ${totalTokens}/${ctxInfo.tokens} tokens ` +
                  `(history=${preHistoryTokens}, prompt=${promptTokens}, system+tools=${systemPromptAndToolTokens}) ` +
                  `for ${provider}/${modelId}`,
              );
              const compactResult = await compactEmbeddedPiSessionDirect({
                sessionId: params.sessionId,
                sessionKey: params.sessionKey,
                messageChannel: params.messageChannel,
                messageProvider: params.messageProvider,
                agentAccountId: params.agentAccountId,
                authProfileId: lastProfileId,
                sessionFile: params.sessionFile,
                workspaceDir: params.workspaceDir,
                agentDir,
                config: params.config,
                skillsSnapshot: params.skillsSnapshot,
                provider,
                model: modelId,
                thinkLevel,
                reasoningLevel: params.reasoningLevel,
                bashElevated: params.bashElevated,
                extraSystemPrompt: params.extraSystemPrompt,
                ownerNumbers: params.ownerNumbers,
              });
              if (compactResult.compacted) {
                log.info(`pre-run compaction succeeded for ${provider}/${modelId}`);
              } else {
                log.warn(
                  `pre-run compaction failed for ${provider}/${modelId}: ${
                    compactResult.reason ?? "nothing to compact"
                  }`,
                );
              }
            }
          }

          const attempt = await runEmbeddedAttempt({
            sessionId: params.sessionId,
            sessionKey: params.sessionKey,
            messageChannel: params.messageChannel,
            messageProvider: params.messageProvider,
            agentAccountId: params.agentAccountId,
            messageTo: params.messageTo,
            messageThreadId: params.messageThreadId,
            groupId: params.groupId,
            groupChannel: params.groupChannel,
            groupSpace: params.groupSpace,
            spawnedBy: params.spawnedBy,
            senderIsOwner: params.senderIsOwner,
            currentChannelId: params.currentChannelId,
            currentThreadTs: params.currentThreadTs,
            replyToMode: params.replyToMode,
            hasRepliedRef: params.hasRepliedRef,
            sessionFile: params.sessionFile,
            workspaceDir: params.workspaceDir,
            agentDir,
            config: params.config,
            skillsSnapshot: params.skillsSnapshot,
            prompt,
            images: params.images,
            disableTools: params.disableTools,
            provider,
            modelId,
            model,
            authStorage,
            modelRegistry,
            thinkLevel,
            verboseLevel: params.verboseLevel,
            reasoningLevel: params.reasoningLevel,
            toolResultFormat: resolvedToolResultFormat,
            execOverrides: params.execOverrides,
            bashElevated: params.bashElevated,
            timeoutMs: params.timeoutMs,
            runId: params.runId,
            abortSignal: params.abortSignal,
            shouldEmitToolResult: params.shouldEmitToolResult,
            shouldEmitToolOutput: params.shouldEmitToolOutput,
            onPartialReply: params.onPartialReply,
            onAssistantMessageStart: params.onAssistantMessageStart,
            onBlockReply: params.onBlockReply,
            onBlockReplyFlush: params.onBlockReplyFlush,
            blockReplyBreak: params.blockReplyBreak,
            blockReplyChunking: params.blockReplyChunking,
            onReasoningStream: params.onReasoningStream,
            onToolResult: params.onToolResult,
            onAgentEvent: params.onAgentEvent,
            extraSystemPrompt: params.extraSystemPrompt,
            streamParams: params.streamParams,
            ownerNumbers: params.ownerNumbers,
            enforceFinalTag: params.enforceFinalTag,
          });

          const { aborted, promptError, timedOut, sessionIdUsed, lastAssistant } = attempt;

          // Log assistant message errors (from streaming failures) for debugging
          if (lastAssistant?.errorMessage && !aborted) {
            const errorMsg = lastAssistant.errorMessage.trim();
            log.error(
              `embedded run assistant error: runId=${params.runId} sessionId=${sessionIdUsed} ` +
                `provider=${provider} model=${modelId} error=${errorMsg.slice(0, 500)}${errorMsg.length > 500 ? "…" : ""}`,
            );
          }

          if (promptError && !aborted) {
            const errorText = describeUnknownError(promptError);
            if (isContextOverflowError(errorText)) {
              const msgCount = attempt.messagesSnapshot?.length ?? 0;
              log.warn(
                `[context-overflow-diag] sessionKey=${params.sessionKey ?? params.sessionId} ` +
                  `provider=${provider}/${modelId} messages=${msgCount} ` +
                  `sessionFile=${params.sessionFile} compactionAttempts=${overflowCompactionAttempts} ` +
                  `error=${errorText.slice(0, 200)}`,
              );
              const isCompactionFailure = isCompactionFailureError(errorText);
              // Attempt auto-compaction on context overflow (not compaction_failure).
              // Trigger is the error message from the API or pi-ai, not our token estimate.
              if (!isCompactionFailure && !overflowCompactionAttempted) {
                log.warn(
                  `context overflow detected; attempting auto-compaction for ${provider}/${modelId} (error: ${errorText.slice(0, 200)}${errorText.length > 200 ? "…" : ""})`,
                );
                const compactResult = await compactEmbeddedPiSessionDirect({
                  sessionId: params.sessionId,
                  sessionKey: params.sessionKey,
                  messageChannel: params.messageChannel,
                  messageProvider: params.messageProvider,
                  agentAccountId: params.agentAccountId,
                  authProfileId: lastProfileId,
                  sessionFile: params.sessionFile,
                  workspaceDir: params.workspaceDir,
                  agentDir,
                  config: params.config,
                  skillsSnapshot: params.skillsSnapshot,
                  senderIsOwner: params.senderIsOwner,
                  provider,
                  model: modelId,
                  thinkLevel,
                  reasoningLevel: params.reasoningLevel,
                  bashElevated: params.bashElevated,
                  extraSystemPrompt: params.extraSystemPrompt,
                  ownerNumbers: params.ownerNumbers,
                });
                if (compactResult.compacted) {
                  log.info(`auto-compaction succeeded for ${provider}/${modelId}; retrying prompt`);
                  continue;
                }
                log.warn(
                  `auto-compaction failed for ${provider}/${modelId}: ${compactResult.reason ?? "nothing to compact"}`,
                );
              }
              const kind = isCompactionFailure ? "compaction_failure" : "context_overflow";
              return {
                payloads: [
                  {
                    text:
                      "Context overflow: prompt too large for the model. " +
                      "Try again with less input or a larger-context model.",
                    isError: true,
                  },
                ],
                meta: {
                  durationMs: Date.now() - started,
                  agentMeta: {
                    sessionId: sessionIdUsed,
                    provider,
                    model: model.id,
                  },
                  systemPromptReport: attempt.systemPromptReport,
                  error: { kind, message: errorText },
                },
              };
            }
            // Handle role ordering errors with a user-friendly message
            if (/incorrect role information|roles must alternate/i.test(errorText)) {
              return {
                payloads: [
                  {
                    text:
                      "Message ordering conflict - please try again. " +
                      "If this persists, use /new to start a fresh session.",
                    isError: true,
                  },
                ],
                meta: {
                  durationMs: Date.now() - started,
                  agentMeta: {
                    sessionId: sessionIdUsed,
                    provider,
                    model: model.id,
                  },
                  systemPromptReport: attempt.systemPromptReport,
                  error: { kind: "role_ordering", message: errorText },
                },
              };
            }
            // Handle image size errors with a user-friendly message (no retry needed)
            const imageSizeError = parseImageSizeError(errorText);
            if (imageSizeError) {
              const maxMb = imageSizeError.maxMb;
              const maxMbLabel =
                typeof maxMb === "number" && Number.isFinite(maxMb) ? `${maxMb}` : null;
              const maxBytesHint = maxMbLabel ? ` (max ${maxMbLabel}MB)` : "";
              return {
                payloads: [
                  {
                    text:
                      `Image too large for the model${maxBytesHint}. ` +
                      "Please compress or resize the image and try again.",
                    isError: true,
                  },
                ],
                meta: {
                  durationMs: Date.now() - started,
                  agentMeta: {
                    sessionId: sessionIdUsed,
                    provider,
                    model: model.id,
                  },
                  systemPromptReport: attempt.systemPromptReport,
                  error: { kind: "image_size", message: errorText },
                },
              };
            }
            const promptFailoverReason = classifyFailoverReason(errorText);
            if (promptFailoverReason && promptFailoverReason !== "timeout" && lastProfileId) {
              await markAuthProfileFailure({
                store: authStore,
                profileId: lastProfileId,
                reason: promptFailoverReason,
                cfg: params.config,
                agentDir: params.agentDir,
              });
            }
            if (
              isFailoverErrorMessage(errorText) &&
              promptFailoverReason !== "timeout" &&
              (await advanceAuthProfile())
            ) {
              continue;
            }
            const fallbackThinking = pickFallbackThinkingLevel({
              message: errorText,
              attempted: attemptedThinking,
            });
            if (fallbackThinking) {
              log.warn(
                `unsupported thinking level for ${provider}/${modelId}; retrying with ${fallbackThinking}`,
              );
              thinkLevel = fallbackThinking;
              continue;
            }
            // FIX: Throw FailoverError for prompt errors when fallbacks configured
            // This enables model fallback for quota/rate limit errors during prompt submission
            if (fallbackConfigured && isFailoverErrorMessage(errorText)) {
              throw new FailoverError(errorText, {
                reason: promptFailoverReason ?? "unknown",
                provider,
                model: modelId,
                profileId: lastProfileId,
                status: resolveFailoverStatus(promptFailoverReason ?? "unknown"),
              });
            }
            throw promptError;
          }

          const fallbackThinking = pickFallbackThinkingLevel({
            message: lastAssistant?.errorMessage,
            attempted: attemptedThinking,
          });
          if (fallbackThinking && !aborted) {
            log.warn(
              `unsupported thinking level for ${provider}/${modelId}; retrying with ${fallbackThinking}`,
            );
            thinkLevel = fallbackThinking;
            continue;
          }

          const authFailure = isAuthAssistantError(lastAssistant);
          const rateLimitFailure = isRateLimitAssistantError(lastAssistant);
          const billingFailure = isBillingAssistantError(lastAssistant);
          const failoverFailure = isFailoverAssistantError(lastAssistant);
          const assistantFailoverReason = classifyFailoverReason(lastAssistant?.errorMessage ?? "");
          const cloudCodeAssistFormatError = attempt.cloudCodeAssistFormatError;
          const imageDimensionError = parseImageDimensionError(lastAssistant?.errorMessage ?? "");

          if (imageDimensionError && lastProfileId) {
            const details = [
              imageDimensionError.messageIndex !== undefined
                ? `message=${imageDimensionError.messageIndex}`
                : null,
              imageDimensionError.contentIndex !== undefined
                ? `content=${imageDimensionError.contentIndex}`
                : null,
              imageDimensionError.maxDimensionPx !== undefined
                ? `limit=${imageDimensionError.maxDimensionPx}px`
                : null,
            ]
              .filter(Boolean)
              .join(" ");
            log.warn(
              `Profile ${lastProfileId} rejected image payload${details ? ` (${details})` : ""}.`,
            );
          }

          // Treat timeout as potential rate limit (Antigravity hangs on rate limit)
          const shouldRotate = (!aborted && failoverFailure) || timedOut;

          if (shouldRotate) {
            if (lastProfileId) {
              const reason =
                timedOut || assistantFailoverReason === "timeout"
                  ? "timeout"
                  : (assistantFailoverReason ?? "unknown");
              await markAuthProfileFailure({
                store: authStore,
                profileId: lastProfileId,
                reason,
                cfg: params.config,
                agentDir: params.agentDir,
              });
              if (timedOut && !isProbeSession) {
                log.warn(
                  `Profile ${lastProfileId} timed out (possible rate limit). Trying next account...`,
                );
              }
              if (cloudCodeAssistFormatError) {
                log.warn(
                  `Profile ${lastProfileId} hit Cloud Code Assist format error. Tool calls will be sanitized on retry.`,
                );
              }
            }

            const rotated = await advanceAuthProfile();
            if (rotated) {
              continue;
            }

            if (fallbackConfigured) {
              // Prefer formatted error message (user-friendly) over raw errorMessage
              const message =
                (lastAssistant
                  ? formatAssistantErrorText(lastAssistant, {
                      cfg: params.config,
                      sessionKey: params.sessionKey ?? params.sessionId,
                    })
                  : undefined) ||
                lastAssistant?.errorMessage?.trim() ||
                (timedOut
                  ? "LLM request timed out."
                  : rateLimitFailure
                    ? "LLM request rate limited."
                    : billingFailure
                      ? BILLING_ERROR_USER_MESSAGE
                      : authFailure
                        ? "LLM request unauthorized."
                        : "LLM request failed.");
              const status =
                resolveFailoverStatus(assistantFailoverReason ?? "unknown") ??
                (isTimeoutErrorMessage(message) ? 408 : undefined);
              throw new FailoverError(message, {
                reason: assistantFailoverReason ?? "unknown",
                provider,
                model: modelId,
                profileId: lastProfileId,
                status,
              });
            }
          }

          const usage = normalizeUsage(lastAssistant?.usage as UsageLike);
          const agentMeta: EmbeddedPiAgentMeta = {
            sessionId: sessionIdUsed,
            provider: lastAssistant?.provider ?? provider,
            model: lastAssistant?.model ?? model.id,
            usage,
          };

          const postHistoryTokens = attempt.messagesSnapshot
            ? estimateMessagesTokens(attempt.messagesSnapshot)
            : null;
          if (
            postHistoryTokens !== null &&
            postHistoryTokens > ctxInfo.tokens * compactionHistoryLimit
          ) {
            log.info(
              `proactive compaction: history ${postHistoryTokens}/${ctxInfo.tokens} ` +
                `(${Math.round(compactionHistoryLimit * 100)}% limit) for ${provider}/${modelId}`,
            );
            const compactResult = await compactEmbeddedPiSessionDirect({
              sessionId: params.sessionId,
              sessionKey: params.sessionKey,
              messageChannel: params.messageChannel,
              messageProvider: params.messageProvider,
              agentAccountId: params.agentAccountId,
              authProfileId: lastProfileId,
              sessionFile: params.sessionFile,
              workspaceDir: params.workspaceDir,
              agentDir,
              config: params.config,
              skillsSnapshot: params.skillsSnapshot,
              provider,
              model: modelId,
              thinkLevel,
              reasoningLevel: params.reasoningLevel,
              bashElevated: params.bashElevated,
              extraSystemPrompt: params.extraSystemPrompt,
              ownerNumbers: params.ownerNumbers,
            });
            if (compactResult.compacted) {
              log.info(`proactive compaction succeeded for ${provider}/${modelId}`);
            } else {
              log.warn(
                `proactive compaction failed for ${provider}/${modelId}: ${
                  compactResult.reason ?? "nothing to compact"
                }`,
              );
            }
          }

          const payloads = buildEmbeddedRunPayloads({
            assistantTexts: attempt.assistantTexts,
            toolMetas: attempt.toolMetas,
            lastAssistant: attempt.lastAssistant,
            lastToolError: attempt.lastToolError,
            config: params.config,
            sessionKey: params.sessionKey ?? params.sessionId,
            verboseLevel: params.verboseLevel,
            reasoningLevel: params.reasoningLevel,
            toolResultFormat: resolvedToolResultFormat,
            inlineToolResultsAllowed: false,
          });

          log.debug(
            `embedded run done: runId=${params.runId} sessionId=${params.sessionId} durationMs=${Date.now() - started} aborted=${aborted}`,
          );
          if (lastProfileId) {
            await markAuthProfileGood({
              store: authStore,
              provider,
              profileId: lastProfileId,
              agentDir: params.agentDir,
            });
            await markAuthProfileUsed({
              store: authStore,
              profileId: lastProfileId,
              agentDir: params.agentDir,
            });
          }
          return {
            payloads: payloads.length ? payloads : undefined,
            meta: {
              durationMs: Date.now() - started,
              agentMeta,
              aborted,
              systemPromptReport: attempt.systemPromptReport,
              // Handle client tool calls (OpenResponses hosted tools)
              stopReason: attempt.clientToolCall ? "tool_calls" : undefined,
              pendingToolCalls: attempt.clientToolCall
                ? [
                    {
                      id: `call_${Date.now()}`,
                      name: attempt.clientToolCall.name,
                      arguments: JSON.stringify(attempt.clientToolCall.params),
                    },
                  ]
                : undefined,
            },
            didSendViaMessagingTool: attempt.didSendViaMessagingTool,
            messagingToolSentTexts: attempt.messagingToolSentTexts,
            messagingToolSentTargets: attempt.messagingToolSentTargets,
          };
        }
      } finally {
        process.chdir(prevCwd);
      }
    }),
  );
}
