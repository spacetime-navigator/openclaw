import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type { ImageContent } from "@mariozechner/pi-ai";
import { streamSimple } from "@mariozechner/pi-ai";
import { createAgentSession, SessionManager, SettingsManager } from "@mariozechner/pi-coding-agent";
import fs from "node:fs/promises";
import os from "node:os";
import type { EmbeddedRunAttemptParams, EmbeddedRunAttemptResult } from "./types.js";
import { resolveHeartbeatPrompt } from "../../../auto-reply/heartbeat.js";
import { resolveChannelCapabilities } from "../../../config/channel-capabilities.js";
import { getMachineDisplayName } from "../../../infra/machine-name.js";
import { MAX_IMAGE_BYTES } from "../../../media/constants.js";
import { getGlobalHookRunner } from "../../../plugins/hook-runner-global.js";
import { isSubagentSessionKey } from "../../../routing/session-key.js";
import { resolveSignalReactionLevel } from "../../../signal/reaction-level.js";
import { resolveTelegramInlineButtonsScope } from "../../../telegram/inline-buttons.js";
import { resolveTelegramReactionLevel } from "../../../telegram/reaction-level.js";
import { buildTtsSystemPromptHint } from "../../../tts/tts.js";
import { resolveUserPath } from "../../../utils.js";
import {
  isGlmModel,
  normalizeReasoningLevelForGlm,
} from "../../../auto-reply/thinking.js";
import { normalizeMessageChannel } from "../../../utils/message-channel.js";
import { isReasoningTagProvider } from "../../../utils/provider-utils.js";
import { resolveOpenClawAgentDir } from "../../agent-paths.js";
import { resolveSessionAgentIds } from "../../agent-scope.js";
import { createAnthropicPayloadLogger } from "../../anthropic-payload-log.js";
import { makeBootstrapWarn, resolveBootstrapContextForRun } from "../../bootstrap-files.js";
import { createCacheTrace } from "../../cache-trace.js";
import {
  listChannelSupportedActions,
  resolveChannelMessageToolHints,
} from "../../channel-tools.js";
import { resolveOpenClawDocsPath } from "../../docs-path.js";
import { formatErrorMessage } from "../../../infra/errors.js";
import { isTimeoutError } from "../../failover-error.js";
import { resolveModelAuthMode } from "../../model-auth.js";
import { resolveDefaultModelForAgent } from "../../model-selection.js";
import { getMemorySearchManager } from "../../../memory/index.js";
import { createSubsystemLogger } from "../../../logging/subsystem.js";
import type { OpenClawConfig } from "../../../config/config.js";
import {
  isCloudCodeAssistFormatError,
  resolveBootstrapMaxChars,
  validateAnthropicTurns,
  validateGeminiTurns,
} from "../../pi-embedded-helpers.js";
import type { EmbeddedContextFile } from "../../pi-embedded-helpers/types.js";
import { subscribeEmbeddedPiSession } from "../../pi-embedded-subscribe.js";
import {
  ensurePiCompactionReserveTokens,
  resolveCompactionReserveTokensFloor,
} from "../../pi-settings.js";
import { toClientToolDefinitions } from "../../pi-tool-definition-adapter.js";
import { createOpenClawCodingTools } from "../../pi-tools.js";
import { resolveSandboxContext } from "../../sandbox.js";
import { resolveSandboxRuntimeStatus } from "../../sandbox/runtime-status.js";
import { repairSessionFileIfNeeded } from "../../session-file-repair.js";
import { guardSessionManager } from "../../session-tool-result-guard-wrapper.js";
import { acquireSessionWriteLock } from "../../session-write-lock.js";
import {
  applySkillEnvOverrides,
  applySkillEnvOverridesFromSnapshot,
  loadWorkspaceSkillEntries,
  resolveSkillsPromptForRun,
} from "../../skills.js";
import { buildSystemPromptParams } from "../../system-prompt-params.js";
import { buildSystemPromptReport } from "../../system-prompt-report.js";
import { resolveTranscriptPolicy } from "../../transcript-policy.js";
import { DEFAULT_BOOTSTRAP_FILENAME } from "../../workspace.js";
import { isAbortError } from "../abort.js";
import { appendCacheTtlTimestamp, isCacheTtlEligibleProvider } from "../cache-ttl.js";
import { buildEmbeddedExtensionPaths } from "../extensions.js";
import { applyExtraParamsToAgent } from "../extra-params.js";
import {
  logToolSchemasForGoogle,
  sanitizeSessionHistory,
  sanitizeToolsForGoogle,
} from "../google.js";
import { getDmHistoryLimitFromSessionKey, limitHistoryTurns } from "../history.js";
import { log } from "../logger.js";
import { buildModelAliasLines } from "../model.js";
import {
  clearActiveEmbeddedRun,
  type EmbeddedPiQueueHandle,
  setActiveEmbeddedRun,
} from "../runs.js";
import { buildEmbeddedSandboxInfo } from "../sandbox-info.js";
import { prewarmSessionFile, trackSessionManagerAccess } from "../session-manager-cache.js";
import { prepareSessionManagerForRun } from "../session-manager-init.js";
import {
  applySystemPromptOverrideToSession,
  buildEmbeddedSystemPrompt,
  createSystemPromptOverride,
} from "../system-prompt.js";
import { splitSdkTools } from "../tool-split.js";
import { describeUnknownError, mapThinkingLevel } from "../utils.js";
import { detectAndLoadPromptImages } from "./images.js";
import { resolveStorePath } from "../../../config/sessions/paths.js";
import { loadSessionStore } from "../../../config/sessions/store.js";
import { extractTextFromMessage } from "../../../tui/tui-formatters.js";

export function injectHistoryImagesIntoMessages(
  messages: AgentMessage[],
  historyImagesByIndex: Map<number, ImageContent[]>,
): boolean {
  if (historyImagesByIndex.size === 0) {
    return false;
  }
  let didMutate = false;

  for (const [msgIndex, images] of historyImagesByIndex) {
    // Bounds check: ensure index is valid before accessing
    if (msgIndex < 0 || msgIndex >= messages.length) {
      continue;
    }
    const msg = messages[msgIndex];
    if (msg && msg.role === "user") {
      // Convert string content to array format if needed
      if (typeof msg.content === "string") {
        msg.content = [{ type: "text", text: msg.content }];
        didMutate = true;
      }
      if (Array.isArray(msg.content)) {
        // Check for existing image content to avoid duplicates across turns
        const existingImageData = new Set(
          msg.content
            .filter(
              (c): c is ImageContent =>
                c != null &&
                typeof c === "object" &&
                c.type === "image" &&
                typeof c.data === "string",
            )
            .map((c) => c.data),
        );
        for (const img of images) {
          // Only add if this image isn't already in the message
          if (!existingImageData.has(img.data)) {
            msg.content.push(img);
            didMutate = true;
          }
        }
      }
    }
  }

  return didMutate;
}

type RecallActorContext = {
  actorId?: string;
  actorType?: "human" | "agent";
  chatType?: string;
};

function resolveRecallActorContext(params: {
  config?: OpenClawConfig;
  agentId?: string;
  sessionKey?: string;
}): RecallActorContext {
  const sessionKey = params.sessionKey?.trim();
  if (!params.config || !params.agentId || !sessionKey) {
    return {};
  }
  const storePath = resolveStorePath(params.config.session?.store, {
    agentId: params.agentId,
  });
  const store = loadSessionStore(storePath);
  const entry = store[sessionKey];
  if (!entry) {
    return {};
  }
  const channel =
    entry.origin?.provider?.trim() || entry.channel?.trim() || entry.lastChannel?.trim();
  const rawUserId =
    entry.origin?.from?.trim() || entry.deliveryContext?.to?.trim() || entry.lastTo?.trim();
  const actorId =
    channel && rawUserId && !rawUserId.includes(":") ? `${channel}:${rawUserId}` : rawUserId;
  return {
    actorId: actorId || undefined,
    actorType: actorId ? "human" : undefined,
    chatType: entry.chatType,
  };
}

function resolveRecallWindow(contextFiles: EmbeddedContextFile[]): {
  updatedAfter?: number;
  updatedBefore?: number;
} {
  let updatedAfter: number | undefined;
  let updatedBefore: number | undefined;
  const memoryDateRegex = /memory\/(\d{4}-\d{2}-\d{2})\.md$/i;
  for (const file of contextFiles) {
    const normalized = file.path.replace(/\\/g, "/");
    const match = normalized.match(memoryDateRegex);
    if (!match) {
      continue;
    }
    const date = new Date(`${match[1]}T00:00:00.000Z`);
    if (Number.isNaN(date.getTime())) {
      continue;
    }
    const start = date.getTime();
    const end = start + 24 * 60 * 60 * 1000 - 1;
    updatedAfter = updatedAfter ? Math.min(updatedAfter, start) : start;
    updatedBefore = updatedBefore ? Math.max(updatedBefore, end) : end;
  }
  if (updatedAfter || updatedBefore) {
    return { updatedAfter, updatedBefore };
  }
  const hasMemoryMd = contextFiles.some((file) =>
    file.path.toLowerCase().replace(/\\/g, "/").endsWith("/memory.md"),
  );
  if (hasMemoryMd) {
    return { updatedAfter: Date.now() - 30 * 24 * 60 * 60 * 1000 };
  }
  return {};
}

export async function buildDbRecallContext(params: {
  config?: OpenClawConfig;
  agentId?: string;
  sessionKey?: string;
  query: string;
  contextFiles: EmbeddedContextFile[];
}): Promise<EmbeddedContextFile | null> {
  const cleanedQuery = params.query.trim();
  if (!cleanedQuery || !params.config || !params.agentId) {
    return null;
  }
  const { manager } = await getMemorySearchManager({
    cfg: params.config,
    agentId: params.agentId,
  });
  if (!manager) {
    return null;
  }
  const actorContext = resolveRecallActorContext({
    config: params.config,
    agentId: params.agentId,
    sessionKey: params.sessionKey,
  });
  // For group chats, use sessionScope: "session" to prevent cross-session data leakage
  // This ensures ONLY session transcripts from this specific group chat are searched
  // Memory files are excluded to prevent private data leakage
  // For direct chats, use "actor" scope if actorId is available, otherwise "session"
  const isGroupChat = actorContext.chatType && actorContext.chatType !== "direct";
  const sessionScope = isGroupChat
    ? "session"
    : actorContext.actorId
      ? "actor"
      : "session";
  const { updatedAfter, updatedBefore } = resolveRecallWindow(params.contextFiles);
  
  // Log the search parameters for debugging
  const dbRecallLog = createSubsystemLogger("db-recall");
  dbRecallLog.debug(
    `buildDbRecallContext: sessionKey=${params.sessionKey} chatType=${actorContext.chatType ?? "unknown"} ` +
      `isGroupChat=${isGroupChat} sessionScope=${sessionScope} actorId=${actorContext.actorId ?? "none"}`,
  );
  
  const results = await manager.search(cleanedQuery, {
    mode: "hybrid",
    maxResults: 8,
    minScore: 0.15,
    sessionKey: params.sessionKey,
    sessionScope,
    actorId: actorContext.actorId,
    actorType: actorContext.actorType,
    updatedAfter,
    updatedBefore,
  });
  
  if (results && results.length > 0) {
    const sources = new Set(results.map((r) => r.source));
    dbRecallLog.debug(
      `buildDbRecallContext: found ${results.length} results from sources: ${Array.from(sources).join(", ")} ` +
        `sessionKey=${params.sessionKey}`,
    );
  }
  if (!results || results.length === 0) {
    return null;
  }
  const lines = [
    "# 🔍 DB Recall Context (Postgres Memory Search)",
    "",
    "This context was automatically retrieved from Postgres using hybrid vector/keyword search.",
    "Use memory_search and memory_recall tools for additional targeted queries.",
    "",
    `Query: ${cleanedQuery}`,
    updatedAfter || updatedBefore
      ? `Window: ${updatedAfter ?? "?"} - ${updatedBefore ?? "?"} (ms)`
      : "Window: none",
    "",
  ];
  for (const entry of results) {
    lines.push(
      `- ${entry.path}:${entry.startLine}-${entry.endLine} (${entry.source}) score=${entry.score.toFixed(
        3,
      )}`,
      `  ${entry.snippet.trim()}`,
      "",
    );
  }
  return {
    path: "memory/DB_RECALL.md",
    content: lines.join("\n").trim(),
  };
}

export async function runEmbeddedAttempt(
  params: EmbeddedRunAttemptParams,
): Promise<EmbeddedRunAttemptResult> {
  const resolvedWorkspace = resolveUserPath(params.workspaceDir);
  const prevCwd = process.cwd();
  const runAbortController = new AbortController();

  log.debug(
    `embedded run start: runId=${params.runId} sessionId=${params.sessionId} provider=${params.provider} model=${params.modelId} thinking=${params.thinkLevel} messageChannel=${params.messageChannel ?? params.messageProvider ?? "unknown"}`,
  );

  await fs.mkdir(resolvedWorkspace, { recursive: true });

  const sandboxSessionKey = params.sessionKey?.trim() || params.sessionId;
  const sandbox = await resolveSandboxContext({
    config: params.config,
    sessionKey: sandboxSessionKey,
    workspaceDir: resolvedWorkspace,
  });
  const effectiveWorkspace = sandbox?.enabled
    ? sandbox.workspaceAccess === "rw"
      ? resolvedWorkspace
      : sandbox.workspaceDir
    : resolvedWorkspace;
  await fs.mkdir(effectiveWorkspace, { recursive: true });

  let restoreSkillEnv: (() => void) | undefined;
  process.chdir(effectiveWorkspace);
  try {
    const shouldLoadSkillEntries = !params.skillsSnapshot || !params.skillsSnapshot.resolvedSkills;
    const skillEntries = shouldLoadSkillEntries
      ? loadWorkspaceSkillEntries(effectiveWorkspace)
      : [];
    restoreSkillEnv = params.skillsSnapshot
      ? applySkillEnvOverridesFromSnapshot({
          snapshot: params.skillsSnapshot,
          config: params.config,
        })
      : applySkillEnvOverrides({
          skills: skillEntries ?? [],
          config: params.config,
        });

    const skillsPrompt = resolveSkillsPromptForRun({
      skillsSnapshot: params.skillsSnapshot,
      entries: shouldLoadSkillEntries ? skillEntries : undefined,
      config: params.config,
      workspaceDir: effectiveWorkspace,
    });

    const sessionLabel = params.sessionKey ?? params.sessionId;
    const { bootstrapFiles: hookAdjustedBootstrapFiles, contextFiles } =
      await resolveBootstrapContextForRun({
        workspaceDir: effectiveWorkspace,
        config: params.config,
        sessionKey: params.sessionKey,
        sessionId: params.sessionId,
        warn: makeBootstrapWarn({ sessionLabel, warn: (message) => log.warn(message) }),
      });
    const { sessionAgentId: recallSessionAgentId } = resolveSessionAgentIds({
      sessionKey: params.sessionKey ?? params.sessionId,
      config: params.config,
    });
    // Sync session transcripts and memory files to Postgres on run start so memory_search/recall have fresh data even if the model never calls the tools this turn.
    if (params.config) {
      void getMemorySearchManager({
        cfg: params.config,
        agentId: recallSessionAgentId,
      }).then((r) => r.manager?.warmSession?.());
    }

    const recallContext = await buildDbRecallContext({
      config: params.config,
      agentId: recallSessionAgentId,
      sessionKey: params.sessionKey ?? params.sessionId,
      query: params.prompt,
      contextFiles,
    }).catch((err) => {
      log.warn(`db recall failed: ${String(err)}`);
      return null;
    });
    // Insert DB recall context early (after bootstrap files) to make it prominent
    const augmentedContextFiles = recallContext
      ? [recallContext, ...contextFiles]
      : contextFiles;
    const workspaceNotes = hookAdjustedBootstrapFiles.some(
      (file) => file.name === DEFAULT_BOOTSTRAP_FILENAME && !file.missing,
    )
      ? ["Reminder: commit your changes in this workspace after edits."]
      : undefined;

    const agentDir = params.agentDir ?? resolveOpenClawAgentDir();

    // Check if the model supports native image input
    const modelHasVision = params.model.input?.includes("image") ?? false;
    const toolsRaw = params.disableTools
      ? []
      : createOpenClawCodingTools({
          exec: {
            ...params.execOverrides,
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
          senderIsOwner: params.senderIsOwner,
          sessionKey: params.sessionKey ?? params.sessionId,
          agentDir,
          workspaceDir: effectiveWorkspace,
          config: params.config,
          abortSignal: runAbortController.signal,
          modelProvider: params.model.provider,
          modelId: params.modelId,
          modelAuthMode: resolveModelAuthMode(params.model.provider, params.config),
          currentChannelId: params.currentChannelId,
          currentThreadTs: params.currentThreadTs,
          replyToMode: params.replyToMode,
          hasRepliedRef: params.hasRepliedRef,
          modelHasVision,
          requireExplicitMessageTarget:
            params.requireExplicitMessageTarget ?? isSubagentSessionKey(params.sessionKey),
          disableMessageTool: params.disableMessageTool,
        });
    const tools = sanitizeToolsForGoogle({ tools: toolsRaw, provider: params.provider });
    logToolSchemasForGoogle({ tools, provider: params.provider });

    const machineName = await getMachineDisplayName();
    const runtimeChannel = normalizeMessageChannel(params.messageChannel ?? params.messageProvider);
    let runtimeCapabilities = runtimeChannel
      ? (resolveChannelCapabilities({
          cfg: params.config,
          channel: runtimeChannel,
          accountId: params.agentAccountId,
        }) ?? [])
      : undefined;
    if (runtimeChannel === "telegram" && params.config) {
      const inlineButtonsScope = resolveTelegramInlineButtonsScope({
        cfg: params.config,
        accountId: params.agentAccountId ?? undefined,
      });
      if (inlineButtonsScope !== "off") {
        if (!runtimeCapabilities) {
          runtimeCapabilities = [];
        }
        if (
          !runtimeCapabilities.some((cap) => String(cap).trim().toLowerCase() === "inlinebuttons")
        ) {
          runtimeCapabilities.push("inlineButtons");
        }
      }
    }
    const reactionGuidance =
      runtimeChannel && params.config
        ? (() => {
            if (runtimeChannel === "telegram") {
              const resolved = resolveTelegramReactionLevel({
                cfg: params.config,
                accountId: params.agentAccountId ?? undefined,
              });
              const level = resolved.agentReactionGuidance;
              return level ? { level, channel: "Telegram" } : undefined;
            }
            if (runtimeChannel === "signal") {
              const resolved = resolveSignalReactionLevel({
                cfg: params.config,
                accountId: params.agentAccountId ?? undefined,
              });
              const level = resolved.agentReactionGuidance;
              return level ? { level, channel: "Signal" } : undefined;
            }
            return undefined;
          })()
        : undefined;
    const { defaultAgentId, sessionAgentId } = resolveSessionAgentIds({
      sessionKey: params.sessionKey,
      config: params.config,
    });
    const sandboxInfo = buildEmbeddedSandboxInfo(sandbox, params.bashElevated);
    const reasoningTagHint = isReasoningTagProvider(params.provider);
    // Resolve channel-specific message actions for system prompt
    const channelActions = runtimeChannel
      ? listChannelSupportedActions({
          cfg: params.config,
          channel: runtimeChannel,
        })
      : undefined;
    const messageToolHints = runtimeChannel
      ? resolveChannelMessageToolHints({
          cfg: params.config,
          channel: runtimeChannel,
          accountId: params.agentAccountId,
        })
      : undefined;

    const defaultModelRef = resolveDefaultModelForAgent({
      cfg: params.config ?? {},
      agentId: sessionAgentId,
    });
    const defaultModelLabel = `${defaultModelRef.provider}/${defaultModelRef.model}`;
    const { runtimeInfo, userTimezone, userTime, userTimeFormat } = buildSystemPromptParams({
      config: params.config,
      agentId: sessionAgentId,
      workspaceDir: effectiveWorkspace,
      cwd: process.cwd(),
      runtime: {
        host: machineName,
        os: `${os.type()} ${os.release()}`,
        arch: os.arch(),
        node: process.version,
        model: `${params.provider}/${params.modelId}`,
        defaultModel: defaultModelLabel,
        channel: runtimeChannel,
        capabilities: runtimeCapabilities,
        channelActions,
      },
    });
    const isDefaultAgent = sessionAgentId === defaultAgentId;
    const promptMode = isSubagentSessionKey(params.sessionKey) ? "minimal" : "full";
    const docsPath = await resolveOpenClawDocsPath({
      workspaceDir: effectiveWorkspace,
      argv1: process.argv[1],
      cwd: process.cwd(),
      moduleUrl: import.meta.url,
    });
    const ttsHint = params.config ? buildTtsSystemPromptHint(params.config) : undefined;

    const appendPrompt = buildEmbeddedSystemPrompt({
      workspaceDir: effectiveWorkspace,
      defaultThinkLevel: params.thinkLevel,
      reasoningLevel: params.reasoningLevel ?? "off",
      extraSystemPrompt: params.extraSystemPrompt,
      ownerNumbers: params.ownerNumbers,
      reasoningTagHint,
      heartbeatPrompt: isDefaultAgent
        ? resolveHeartbeatPrompt(params.config?.agents?.defaults?.heartbeat?.prompt)
        : undefined,
      skillsPrompt,
      docsPath: docsPath ?? undefined,
      ttsHint,
      workspaceNotes,
      reactionGuidance,
      promptMode,
      runtimeInfo,
      messageToolHints,
      sandboxInfo,
      tools,
      modelAliasLines: buildModelAliasLines(params.config),
      userTimezone,
      userTime,
      userTimeFormat,
      contextFiles,
      memoryCitationsMode: params.config?.memory?.citations,
    });
    const systemPromptReport = buildSystemPromptReport({
      source: "run",
      generatedAt: Date.now(),
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
      provider: params.provider,
      model: params.modelId,
      workspaceDir: effectiveWorkspace,
      bootstrapMaxChars: resolveBootstrapMaxChars(params.config),
      sandbox: (() => {
        const runtime = resolveSandboxRuntimeStatus({
          cfg: params.config,
          sessionKey: params.sessionKey ?? params.sessionId,
        });
        return { mode: runtime.mode, sandboxed: runtime.sandboxed };
      })(),
      systemPrompt: appendPrompt,
      bootstrapFiles: hookAdjustedBootstrapFiles,
      injectedFiles: augmentedContextFiles,
      skillsPrompt,
      tools,
    });
    const systemPromptOverride = createSystemPromptOverride(appendPrompt);
    const systemPromptText = systemPromptOverride();

    const sessionLock = await acquireSessionWriteLock({
      sessionFile: params.sessionFile,
    });

    let sessionManager: ReturnType<typeof guardSessionManager> | undefined;
    let session: Awaited<ReturnType<typeof createAgentSession>>["session"] | undefined;
    try {
      await repairSessionFileIfNeeded({
        sessionFile: params.sessionFile,
        warn: (message) => log.warn(message),
      });
      const hadSessionFile = await fs
        .stat(params.sessionFile)
        .then(() => true)
        .catch(() => false);

      const transcriptPolicy = resolveTranscriptPolicy({
        modelApi: params.model?.api,
        provider: params.provider,
        modelId: params.modelId,
      });

      await prewarmSessionFile(params.sessionFile);
      sessionManager = guardSessionManager(SessionManager.open(params.sessionFile), {
        agentId: sessionAgentId,
        sessionKey: params.sessionKey,
        allowSyntheticToolResults: transcriptPolicy.allowSyntheticToolResults,
      });
      trackSessionManagerAccess(params.sessionFile);

      await prepareSessionManagerForRun({
        sessionManager,
        sessionFile: params.sessionFile,
        hadSessionFile,
        sessionId: params.sessionId,
        cwd: effectiveWorkspace,
      });

      const settingsManager = SettingsManager.create(effectiveWorkspace, agentDir);
      ensurePiCompactionReserveTokens({
        settingsManager,
        minReserveTokens: resolveCompactionReserveTokensFloor(params.config),
      });

      // Call for side effects (sets compaction/pruning runtime state)
      buildEmbeddedExtensionPaths({
        cfg: params.config,
        sessionManager,
        provider: params.provider,
        modelId: params.modelId,
        model: params.model,
      });

      const { builtInTools, customTools } = splitSdkTools({
        tools,
        sandboxEnabled: !!sandbox?.enabled,
      });

      // Add client tools (OpenResponses hosted tools) to customTools
      let clientToolCallDetected: { name: string; params: Record<string, unknown> } | null = null;
      const clientToolDefs = params.clientTools
        ? toClientToolDefinitions(
            params.clientTools,
            (toolName, toolParams) => {
              clientToolCallDetected = { name: toolName, params: toolParams };
            },
            {
              agentId: sessionAgentId,
              sessionKey: params.sessionKey,
            },
          )
        : [];

      const allCustomTools = [...customTools, ...clientToolDefs];

      ({ session } = await createAgentSession({
        cwd: resolvedWorkspace,
        agentDir,
        authStorage: params.authStorage,
        modelRegistry: params.modelRegistry,
        model: params.model,
        thinkingLevel: mapThinkingLevel(params.thinkLevel, params.provider, params.modelId),
        tools: builtInTools,
        customTools: allCustomTools,
        sessionManager,
        settingsManager,
      }));
      applySystemPromptOverrideToSession(session, systemPromptText);
      if (!session) {
        throw new Error("Embedded agent session missing");
      }
      const activeSession = session;
      const cacheTrace = createCacheTrace({
        cfg: params.config,
        env: process.env,
        runId: params.runId,
        sessionId: activeSession.sessionId,
        sessionKey: params.sessionKey,
        provider: params.provider,
        modelId: params.modelId,
        modelApi: params.model.api,
        workspaceDir: params.workspaceDir,
      });
      const anthropicPayloadLogger = createAnthropicPayloadLogger({
        env: process.env,
        runId: params.runId,
        sessionId: activeSession.sessionId,
        sessionKey: params.sessionKey,
        provider: params.provider,
        modelId: params.modelId,
        modelApi: params.model.api,
        workspaceDir: params.workspaceDir,
      });

      // Force a stable streamFn reference so vitest can reliably mock @mariozechner/pi-ai.
      activeSession.agent.streamFn = streamSimple;

      if (cacheTrace) {
        cacheTrace.recordStage("session:loaded", {
          messages: activeSession.messages,
          system: systemPromptText,
          note: "after session create",
        });
        activeSession.agent.streamFn = cacheTrace.wrapStreamFn(activeSession.agent.streamFn);
      }
      if (anthropicPayloadLogger) {
        activeSession.agent.streamFn = anthropicPayloadLogger.wrapStreamFn(
          activeSession.agent.streamFn,
        );
      }
      // Apply extra params (and GLM sanitization: prompt_cache_key, developer→system) last so
      // our wrapper is outermost and sanitizes the raw (model, context, options) before they reach the provider.
      applyExtraParamsToAgent(
        activeSession.agent,
        params.config,
        params.provider,
        params.modelId,
        params.streamParams,
        isGlmModel(params.provider, params.modelId)
          ? normalizeReasoningLevelForGlm(params.reasoningLevel ?? "off")
          : undefined,
      );

      try {
        const prior = await sanitizeSessionHistory({
          messages: activeSession.messages,
          modelApi: params.model.api,
          modelId: params.modelId,
          provider: params.provider,
          sessionManager,
          sessionId: params.sessionId,
          policy: transcriptPolicy,
        });
        cacheTrace?.recordStage("session:sanitized", { messages: prior });
        const validatedGemini = transcriptPolicy.validateGeminiTurns
          ? validateGeminiTurns(prior)
          : prior;
        const validated = transcriptPolicy.validateAnthropicTurns
          ? validateAnthropicTurns(validatedGemini)
          : validatedGemini;
        const limited = limitHistoryTurns(
          validated,
          getDmHistoryLimitFromSessionKey(params.sessionKey, params.config),
        );
        cacheTrace?.recordStage("session:limited", { messages: limited });
        if (limited.length > 0) {
          activeSession.agent.replaceMessages(limited);
        }
      } catch (err) {
        sessionManager.flushPendingToolResults?.();
        activeSession.dispose();
        throw err;
      }

      let aborted = Boolean(params.abortSignal?.aborted);
      let timedOut = false;
      const getAbortReason = (signal: AbortSignal): unknown =>
        "reason" in signal ? (signal as { reason?: unknown }).reason : undefined;
      const makeTimeoutAbortReason = (): Error => {
        const err = new Error("request timed out");
        err.name = "TimeoutError";
        return err;
      };
      const makeAbortError = (signal: AbortSignal): Error => {
        const reason = getAbortReason(signal);
        const err = reason ? new Error("aborted", { cause: reason }) : new Error("aborted");
        err.name = "AbortError";
        return err;
      };
      const abortRun = (isTimeout = false, reason?: unknown) => {
        aborted = true;
        if (isTimeout) {
          timedOut = true;
        }
        if (isTimeout) {
          const timeoutReason = reason ?? makeTimeoutAbortReason();
          log.warn(
            `embedded run aborting due to timeout: runId=${params.runId} sessionId=${params.sessionId} ` +
              `reason=${formatErrorMessage(timeoutReason)}`,
          );
          runAbortController.abort(timeoutReason);
        } else {
          runAbortController.abort(reason);
        }
        try {
          activeSession.abort();
        } catch (err) {
          log.warn(
            `embedded run abort() failed: runId=${params.runId} sessionId=${params.sessionId} error=${formatErrorMessage(err)}`,
          );
        }
      };
      const abortable = <T>(promise: Promise<T>): Promise<T> => {
        const signal = runAbortController.signal;
        if (signal.aborted) {
          return Promise.reject(makeAbortError(signal));
        }
        return new Promise<T>((resolve, reject) => {
          const onAbort = () => {
            signal.removeEventListener("abort", onAbort);
            reject(makeAbortError(signal));
          };
          signal.addEventListener("abort", onAbort, { once: true });
          promise.then(
            (value) => {
              signal.removeEventListener("abort", onAbort);
              resolve(value);
            },
            (err) => {
              signal.removeEventListener("abort", onAbort);
              reject(err);
            },
          );
        });
      };

      // Track event times via onAgentEvent callback for watchdog timer
      const originalOnAgentEvent = params.onAgentEvent;
      const eventTrackingOnAgentEvent = (evt: { stream: string; data: Record<string, unknown> }) => {
        lastEventTime = Date.now();
        originalOnAgentEvent?.(evt);
      };

      const subscription = subscribeEmbeddedPiSession({
        session: activeSession,
        runId: params.runId,
        verboseLevel: params.verboseLevel,
        reasoningMode: isGlmModel(params.provider, params.modelId)
          ? normalizeReasoningLevelForGlm(params.reasoningLevel ?? "off")
          : (params.reasoningLevel ?? "off"),
        toolResultFormat: params.toolResultFormat,
        shouldEmitToolResult: params.shouldEmitToolResult,
        shouldEmitToolOutput: params.shouldEmitToolOutput,
        onToolResult: params.onToolResult,
        onReasoningStream: params.onReasoningStream,
        onBlockReply: params.onBlockReply,
        onBlockReplyFlush: params.onBlockReplyFlush,
        blockReplyBreak: params.blockReplyBreak,
        blockReplyChunking: params.blockReplyChunking,
        onPartialReply: params.onPartialReply,
        onAssistantMessageStart: params.onAssistantMessageStart,
        onAgentEvent: eventTrackingOnAgentEvent,
        enforceFinalTag: params.enforceFinalTag,
      });

      const {
        assistantTexts,
        toolMetas,
        unsubscribe,
        waitForCompactionRetry,
        getMessagingToolSentTexts,
        getMessagingToolSentTargets,
        didSendViaMessagingTool,
        getLastToolError,
      } = subscription;

      const queueHandle: EmbeddedPiQueueHandle = {
        queueMessage: async (text: string) => {
          await activeSession.steer(text);
        },
        isStreaming: () => activeSession.isStreaming,
        isCompacting: () => subscription.isCompacting(),
        abort: abortRun,
      };
      setActiveEmbeddedRun(params.sessionId, queueHandle);

      let abortWarnTimer: NodeJS.Timeout | undefined;
      const isProbeSession = params.sessionId?.startsWith("probe-") ?? false;
      const abortTimer = setTimeout(
        () => {
          if (!isProbeSession) {
            log.warn(
              `embedded run timeout: runId=${params.runId} sessionId=${params.sessionId} timeoutMs=${params.timeoutMs} ` +
                `(this may indicate LM Studio errored during streaming and the stream hung)`,
            );
          }
          abortRun(true);
          if (!abortWarnTimer) {
            abortWarnTimer = setTimeout(() => {
              if (!activeSession.isStreaming) {
                return;
              }
              if (!isProbeSession) {
                log.warn(
                  `embedded run abort still streaming after 10s: runId=${params.runId} sessionId=${params.sessionId} ` +
                    `(stream may be hung - check LM Studio logs for errors)`,
                );
              }
            }, 10_000);
          }
        },
        Math.max(1, params.timeoutMs),
      );

      let messagesSnapshot: AgentMessage[] = [];
      let sessionIdUsed = activeSession.sessionId;
      const onAbort = () => {
        const reason = params.abortSignal ? getAbortReason(params.abortSignal) : undefined;
        const timeout = reason ? isTimeoutError(reason) : false;
        abortRun(timeout, reason);
      };
      if (params.abortSignal) {
        if (params.abortSignal.aborted) {
          onAbort();
        } else {
          params.abortSignal.addEventListener("abort", onAbort, {
            once: true,
          });
        }
      }

      // Get hook runner once for both before_agent_start and agent_end hooks
      const hookRunner = getGlobalHookRunner();

      let promptError: unknown = null;
      let watchdogInterval: NodeJS.Timeout | undefined;
      let lastEventTime = Date.now();
      try {
        const promptStartedAt = Date.now();

        // Run before_agent_start hooks to allow plugins to inject context
        let effectivePrompt = params.prompt;
        if (hookRunner?.hasHooks("before_agent_start")) {
          try {
            const hookResult = await hookRunner.runBeforeAgentStart(
              {
                prompt: params.prompt,
                messages: activeSession.messages,
              },
              {
                agentId: params.sessionKey?.split(":")[0] ?? "main",
                sessionKey: params.sessionKey,
                workspaceDir: params.workspaceDir,
                messageProvider: params.messageProvider ?? undefined,
              },
            );
            if (hookResult?.prependContext) {
              effectivePrompt = `${hookResult.prependContext}\n\n${params.prompt}`;
              log.debug(
                `hooks: prepended context to prompt (${hookResult.prependContext.length} chars)`,
              );
            }
          } catch (hookErr) {
            log.warn(`before_agent_start hook failed: ${String(hookErr)}`);
          }
        }

        log.debug(`embedded run prompt start: runId=${params.runId} sessionId=${params.sessionId}`);
        cacheTrace?.recordStage("prompt:before", {
          prompt: effectivePrompt,
          messages: activeSession.messages,
        });

        // Watchdog timer to detect hung streams (e.g., when LM Studio errors during streaming)
        // If no events are received for 30 seconds after prompt() starts, log a warning
        // Only abort if the stream is NOT active (isStreaming=false) AND no events for 60s
        // This handles slow models that buffer tokens before emitting events
        lastEventTime = Date.now();
        watchdogInterval = setInterval(() => {
          const timeSinceLastEvent = Date.now() - lastEventTime;
          const isStreaming = activeSession.isStreaming;
          
          // If stream is still active according to pi-ai, be lenient (model might be buffering)
          if (isStreaming && timeSinceLastEvent > 30_000) {
            const secondsSinceLastEvent = Math.round(timeSinceLastEvent / 1000);
            // Only warn if no events for 60s while streaming (slow model, not hung)
            if (timeSinceLastEvent > 60_000) {
              log.warn(
                `embedded run stream watchdog: runId=${params.runId} sessionId=${params.sessionId} ` +
                  `no events for ${secondsSinceLastEvent}s but stream still active (model may be buffering - check LM Studio logs)`,
              );
            }
            // Don't abort if stream is still active - wait for it to finish or timeout normally
            return;
          }
          
          // If stream is NOT active and no events for 30s, it's likely hung
          if (!isStreaming && timeSinceLastEvent > 30_000) {
            const secondsSinceLastEvent = Math.round(timeSinceLastEvent / 1000);
            log.warn(
              `embedded run stream watchdog: runId=${params.runId} sessionId=${params.sessionId} ` +
                `no events for ${secondsSinceLastEvent}s and stream not active (stream may be hung - check LM Studio logs for errors)`,
            );
            // If stream has been inactive for 60 seconds with no events, abort early
            if (timeSinceLastEvent > 60_000 && !aborted) {
              log.warn(
                `embedded run aborting early due to hung stream: runId=${params.runId} sessionId=${params.sessionId} ` +
                  `(no events for ${secondsSinceLastEvent}s and stream not active - LM Studio likely errored during streaming)`,
              );
              abortRun(true, new Error(`Stream hung: no events for ${secondsSinceLastEvent}s`));
            }
          }
        }, 10_000); // Check every 10 seconds

        // Repair orphaned trailing user messages so new prompts don't violate role ordering.
        // Do not remove the leaf user message when it matches the current prompt — that is the
        // message we're processing; removing it would drop the user's turn and prevent a reply.
        // Strip leading envelope(s) (e.g. "[Replied message - for context]", "[Discord Guild #general ...]")
        // so we compare message body only; envelope timestamps change every run and would otherwise differ.
        const stripEnvelopePrefix = (s: string): string => {
          let t = s.trim();
          for (;;) {
            const match = t.match(/^\[[^\]]*\]\s*/);
            if (!match) break;
            t = t.slice(match[0].length).trim();
          }
          return t;
        };
        const leafEntry = sessionManager.getLeafEntry();
        if (leafEntry?.type === "message" && leafEntry.message.role === "user") {
          // If leafEntry is a user message, it means the session ends with a user message
          // (no assistant response), indicating an incomplete run (e.g., interrupted/disconnected)
          const isIncompleteRun = true;
          
          const rawLeafText = extractTextFromMessage(leafEntry.message);
          const content = (leafEntry.message as unknown as Record<string, unknown>).content;
          let directText = "";
          if (Array.isArray(content) && content[0]) {
            const first = content[0] as Record<string, unknown>;
            if (typeof first.text === "string") {
              directText = first.text;
            }
          } else if (typeof content === "string") {
            directText = content;
          }
          const leafText =
            (typeof rawLeafText === "string" ? rawLeafText : "") || directText || "";
          // Normalize so minor differences (line endings, control/zero-width chars) don't cause false orphans.
          const normalizeForCompare = (s: string) =>
            s
              .replace(/\r\n|\r/g, "\n")
              .replace(/[\u200B-\u200D\uFEFF]/g, "")
              .trim()
              .replace(/\s+/g, " ");
          const bodyPrompt = stripEnvelopePrefix(effectivePrompt);
          const bodyLeaf = stripEnvelopePrefix(leafText);
          const normalizedPrompt = normalizeForCompare(bodyPrompt);
          const normalizedLeaf = normalizeForCompare(bodyLeaf);
          const exactMatch =
            (normalizedPrompt.length > 0 && normalizedLeaf === normalizedPrompt) ||
            (normalizedPrompt.length === 0 && normalizedLeaf.length === 0);
          const prefixMatch =
            normalizedPrompt.length > 0 &&
            ((normalizedLeaf.startsWith(normalizedPrompt) &&
              normalizedLeaf.slice(normalizedPrompt.length).trim() === "") ||
              (normalizedPrompt.startsWith(normalizedLeaf) &&
                normalizedPrompt.slice(normalizedLeaf.length).trim() === ""));
          const sameLength = normalizedPrompt.length === normalizedLeaf.length;
          const minLen = Math.min(normalizedPrompt.length, normalizedLeaf.length);
          let commonPrefixLen = 0;
          while (
            commonPrefixLen < minLen &&
            normalizedPrompt[commonPrefixLen] === normalizedLeaf[commonPrefixLen]
          ) {
            commonPrefixLen++;
          }
          const nearMatch =
            sameLength &&
            minLen > 0 &&
            commonPrefixLen >= Math.floor(minLen * 0.9);
          const isCurrentPrompt = exactMatch || prefixMatch || nearMatch;
          if (!isCurrentPrompt) {
            if (leafEntry.parentId) {
              sessionManager.branch(leafEntry.parentId);
            } else {
              sessionManager.resetLeaf();
            }
            const sessionContext = sessionManager.buildSessionContext();
            activeSession.agent.replaceMessages(sessionContext.messages);
            // Only warn if:
            // 1. Messages are similar (suggesting a true duplicate/retry), OR
            // 2. It's NOT an incomplete run (last message was assistant, so this is unexpected)
            // If it's an incomplete run with clearly different messages, it's expected (new message after interruption)
            const similarityRatio = minLen > 0 ? commonPrefixLen / minLen : 0;
            const shouldWarn = similarityRatio > 0.5 || !isIncompleteRun;
            if (shouldWarn) {
              log.warn(
                `Removed orphaned user message to prevent consecutive user turns. ` +
                  `runId=${params.runId} sessionId=${params.sessionId}` +
                  (isIncompleteRun ? " (incomplete run detected)" : ""),
              );
            }
            log.debug(
              `orphan repair: promptLen=${normalizedPrompt.length} leafLen=${normalizedLeaf.length} ` +
                `similarity=${Math.round(similarityRatio * 100)}% incompleteRun=${isIncompleteRun} ` +
                `prompt=${JSON.stringify(normalizedPrompt.slice(0, 80))} leaf=${JSON.stringify(normalizedLeaf.slice(0, 80))}`,
            );
          }
        }

        try {
          // Detect and load images referenced in the prompt for vision-capable models.
          // This eliminates the need for an explicit "view" tool call by injecting
          // images directly into the prompt when the model supports it.
          // Also scans conversation history to enable follow-up questions about earlier images.
          const imageResult = await detectAndLoadPromptImages({
            prompt: effectivePrompt,
            workspaceDir: effectiveWorkspace,
            model: params.model,
            existingImages: params.images,
            historyMessages: activeSession.messages,
            maxBytes: MAX_IMAGE_BYTES,
            // Enforce sandbox path restrictions when sandbox is enabled
            sandboxRoot: sandbox?.enabled ? sandbox.workspaceDir : undefined,
          });

          // Inject history images into their original message positions.
          // This ensures the model sees images in context (e.g., "compare to the first image").
          const didMutate = injectHistoryImagesIntoMessages(
            activeSession.messages,
            imageResult.historyImagesByIndex,
          );
          if (didMutate) {
            // Persist message mutations (e.g., injected history images) so we don't re-scan/reload.
            activeSession.agent.replaceMessages(activeSession.messages);
          }

          cacheTrace?.recordStage("prompt:images", {
            prompt: effectivePrompt,
            messages: activeSession.messages,
            note: `images: prompt=${imageResult.images.length} history=${imageResult.historyImagesByIndex.size}`,
          });

          const shouldTrackCacheTtl =
            params.config?.agents?.defaults?.contextPruning?.mode === "cache-ttl" &&
            isCacheTtlEligibleProvider(params.provider, params.modelId);
          if (shouldTrackCacheTtl) {
            appendCacheTtlTimestamp(sessionManager, {
              timestamp: Date.now(),
              provider: params.provider,
              modelId: params.modelId,
            });
          }

          // Only pass images option if there are actually images to pass
          // This avoids potential issues with models that don't expect the images parameter
          if (imageResult.images.length > 0) {
            await abortable(activeSession.prompt(effectivePrompt, { images: imageResult.images }));
          } else {
            await abortable(activeSession.prompt(effectivePrompt));
          }
        } catch (err) {
          promptError = err;
          const errMsg = formatErrorMessage(err);
          log.error(
            `embedded run prompt error: runId=${params.runId} sessionId=${params.sessionId} error=${errMsg}`,
          );
        } finally {
          log.debug(
            `embedded run prompt end: runId=${params.runId} sessionId=${params.sessionId} durationMs=${Date.now() - promptStartedAt}`,
          );
        }

        try {
          await waitForCompactionRetry();
        } catch (err) {
          if (isAbortError(err)) {
            if (!promptError) {
              promptError = err;
            }
          } else {
            throw err;
          }
        }

        messagesSnapshot = activeSession.messages.slice();
        sessionIdUsed = activeSession.sessionId;
        cacheTrace?.recordStage("session:after", {
          messages: messagesSnapshot,
          note: promptError ? "prompt error" : undefined,
        });
        anthropicPayloadLogger?.recordUsage(messagesSnapshot, promptError);

        // Run agent_end hooks to allow plugins to analyze the conversation
        // This is fire-and-forget, so we don't await
        if (hookRunner?.hasHooks("agent_end")) {
          hookRunner
            .runAgentEnd(
              {
                messages: messagesSnapshot,
                success: !aborted && !promptError,
                error: promptError ? describeUnknownError(promptError) : undefined,
                durationMs: Date.now() - promptStartedAt,
              },
              {
                agentId: params.sessionKey?.split(":")[0] ?? "main",
                sessionKey: params.sessionKey,
                workspaceDir: params.workspaceDir,
                messageProvider: params.messageProvider ?? undefined,
              },
            )
            .catch((err) => {
              log.warn(`agent_end hook failed: ${err}`);
            });
        }
      } finally {
        clearTimeout(abortTimer);
        if (abortWarnTimer) {
          clearTimeout(abortWarnTimer);
        }
        if (watchdogInterval) {
          clearInterval(watchdogInterval);
        }
        unsubscribe();
        clearActiveEmbeddedRun(params.sessionId, queueHandle);
        params.abortSignal?.removeEventListener?.("abort", onAbort);
        
        // Sync session transcripts to Postgres after run completes (messages are now written to disk)
        // This ensures new messages are indexed even if the agent doesn't call memory_search
        if (params.config) {
          void getMemorySearchManager({
            cfg: params.config,
            agentId: recallSessionAgentId,
          }).then((r) => {
            if (r.manager?.warmSession) {
              // Trigger async sync - session file should be written by now
              log.debug(`post-run memory sync: triggering warmSession for agent=${recallSessionAgentId}`);
              void r.manager.warmSession().catch((err) => {
                log.warn(`post-run memory sync failed: ${String(err)}`);
              });
            } else {
              log.debug(`post-run memory sync: no manager available for agent=${recallSessionAgentId}`);
            }
          }).catch((err) => {
            log.warn(`post-run memory sync manager lookup failed: ${String(err)}`);
          });
        }
      }

      const lastAssistant = messagesSnapshot
        .slice()
        .toReversed()
        .find((m) => m.role === "assistant");

      const toolMetasNormalized = toolMetas
        .filter(
          (entry): entry is { toolName: string; meta?: string } =>
            typeof entry.toolName === "string" && entry.toolName.trim().length > 0,
        )
        .map((entry) => ({ toolName: entry.toolName, meta: entry.meta }));

      return {
        aborted,
        timedOut,
        promptError,
        sessionIdUsed,
        systemPromptReport,
        messagesSnapshot,
        assistantTexts,
        toolMetas: toolMetasNormalized,
        lastAssistant,
        lastToolError: getLastToolError?.(),
        didSendViaMessagingTool: didSendViaMessagingTool(),
        messagingToolSentTexts: getMessagingToolSentTexts(),
        messagingToolSentTargets: getMessagingToolSentTargets(),
        cloudCodeAssistFormatError: Boolean(
          lastAssistant?.errorMessage && isCloudCodeAssistFormatError(lastAssistant.errorMessage),
        ),
        // Client tool call detected (OpenResponses hosted tools)
        clientToolCall: clientToolCallDetected ?? undefined,
      };
    } finally {
      // Always tear down the session (and release the lock) before we leave this attempt.
      sessionManager?.flushPendingToolResults?.();
      session?.dispose();
      await sessionLock.release();
    }
  } finally {
    restoreSkillEnv?.();
    process.chdir(prevCwd);
  }
}
