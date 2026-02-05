import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { runCommandWithTimeout } from "../process/exec.js";
import { isSubagentSessionKey } from "../routing/session-key.js";
import { resolveUserPath } from "../utils.js";
import {
  resolveImmutableBootstrapPath,
  resolveWorkspaceTemplateDir,
} from "./workspace-templates.js";

export function resolveDefaultAgentWorkspaceDir(
  env: NodeJS.ProcessEnv = process.env,
  homedir: () => string = os.homedir,
): string {
  const profile = env.OPENCLAW_PROFILE?.trim();
  if (profile && profile.toLowerCase() !== "default") {
    return path.join(homedir(), ".openclaw", `workspace-${profile}`);
  }
  return path.join(homedir(), ".openclaw", "workspace");
}

export const DEFAULT_AGENT_WORKSPACE_DIR = resolveDefaultAgentWorkspaceDir();
export const DEFAULT_AGENTS_FILENAME = "AGENTS.md";
export const DEFAULT_SOUL_FILENAME = "SOUL.md";
export const DEFAULT_SOUL_PLUS_FILENAME = "SOUL_PLUS.md";
export const DEFAULT_TOOLS_FILENAME = "TOOLS.md";
export const DEFAULT_IDENTITY_FILENAME = "IDENTITY.md";
export const DEFAULT_IDENTITY_PLUS_FILENAME = "IDENTITY_PLUS.md";
export const DEFAULT_USER_FILENAME = "USER.md";
export const DEFAULT_USER_PLUS_FILENAME = "USER_PLUS.md";
export const DEFAULT_HEARTBEAT_FILENAME = "HEARTBEAT.md";
export const DEFAULT_BOOTSTRAP_FILENAME = "BOOTSTRAP.md";
export const DEFAULT_WORKSPACE_RULES_FILENAME = "WORKSPACE_RULES.md";
export const DEFAULT_MEMORY_FILENAME = "MEMORY.md";
export const DEFAULT_MEMORY_ALT_FILENAME = "memory.md";

/**
 * Relative paths the agent may write/edit. SOUL.md, IDENTITY.md, USER.md, TOOLS.md are
 * immutable (read-only in workspace; TOOLS and code changes via PR only). This list cannot
 * be changed by the agent (it lives in OpenClaw source outside the workspace).
 */
export const ALLOWED_WRITABLE_WORKSPACE_FILES: readonly string[] = [
  "AGENTS.md",
  "BOOT.md",
  "BOOTSTRAP.md",
  "HEARTBEAT.md",
  "SOUL_PLUS.md",
  "IDENTITY_PLUS.md",
  "USER_PLUS.md",
  "MEMORY.md",
  "memory.md",
];

export function isAllowedWritablePath(relativePath: string): boolean {
  const normalized = relativePath.replace(/\\/g, "/").trim();
  if (ALLOWED_WRITABLE_WORKSPACE_FILES.includes(normalized)) {
    return true;
  }
  if (normalized.startsWith("memory/")) {
    return true;
  }
  return false;
}

function stripFrontMatter(content: string): string {
  if (!content.startsWith("---")) {
    return content;
  }
  const endIndex = content.indexOf("\n---", 3);
  if (endIndex === -1) {
    return content;
  }
  const start = endIndex + "\n---".length;
  let trimmed = content.slice(start);
  trimmed = trimmed.replace(/^\s+/, "");
  return trimmed;
}

async function loadTemplate(name: string): Promise<string> {
  const templateDir = await resolveWorkspaceTemplateDir();
  const templatePath = path.join(templateDir, name);
  try {
    const content = await fs.readFile(templatePath, "utf-8");
    return stripFrontMatter(content);
  } catch {
    throw new Error(
      `Missing workspace template: ${name} (${templatePath}). Ensure docs/reference/templates are packaged.`,
    );
  }
}

export type WorkspaceBootstrapFileName =
  | typeof DEFAULT_AGENTS_FILENAME
  | typeof DEFAULT_SOUL_FILENAME
  | typeof DEFAULT_SOUL_PLUS_FILENAME
  | typeof DEFAULT_TOOLS_FILENAME
  | typeof DEFAULT_IDENTITY_FILENAME
  | typeof DEFAULT_IDENTITY_PLUS_FILENAME
  | typeof DEFAULT_USER_FILENAME
  | typeof DEFAULT_USER_PLUS_FILENAME
  | typeof DEFAULT_HEARTBEAT_FILENAME
  | typeof DEFAULT_BOOTSTRAP_FILENAME
  | typeof DEFAULT_WORKSPACE_RULES_FILENAME
  | typeof DEFAULT_MEMORY_FILENAME
  | typeof DEFAULT_MEMORY_ALT_FILENAME;

export type WorkspaceBootstrapFile = {
  name: WorkspaceBootstrapFileName;
  path: string;
  content?: string;
  missing: boolean;
};

async function writeFileIfMissing(filePath: string, content: string) {
  try {
    await fs.writeFile(filePath, content, {
      encoding: "utf-8",
      flag: "wx",
    });
  } catch (err) {
    const anyErr = err as { code?: string };
    if (anyErr.code !== "EEXIST") {
      throw err;
    }
  }
}

/** Set file to read-only (0o444) so it cannot be overwritten by the agent. No-op on unsupported FS. */
async function makeReadOnlyIfExists(filePath: string): Promise<void> {
  try {
    await fs.chmod(filePath, 0o444);
  } catch (err) {
    const anyErr = err as { code?: string };
    if (anyErr.code === "ENOENT") {
      return;
    }
    // EINVAL or other on some FS (e.g. Windows); don't fail workspace creation
  }
}

async function hasGitRepo(dir: string): Promise<boolean> {
  try {
    await fs.stat(path.join(dir, ".git"));
    return true;
  } catch {
    return false;
  }
}

async function isGitAvailable(): Promise<boolean> {
  try {
    const result = await runCommandWithTimeout(["git", "--version"], { timeoutMs: 2_000 });
    return result.code === 0;
  } catch {
    return false;
  }
}

async function ensureGitRepo(dir: string, isBrandNewWorkspace: boolean) {
  if (!isBrandNewWorkspace) {
    return;
  }
  if (await hasGitRepo(dir)) {
    return;
  }
  if (!(await isGitAvailable())) {
    return;
  }
  try {
    await runCommandWithTimeout(["git", "init"], { cwd: dir, timeoutMs: 10_000 });
  } catch {
    // Ignore git init failures; workspace creation should still succeed.
  }
}

export async function ensureAgentWorkspace(params?: {
  dir?: string;
  ensureBootstrapFiles?: boolean;
}): Promise<{
  dir: string;
  agentsPath?: string;
  soulPath?: string;
  toolsPath?: string;
  identityPath?: string;
  userPath?: string;
  heartbeatPath?: string;
  bootstrapPath?: string;
}> {
  const rawDir = params?.dir?.trim() ? params.dir.trim() : DEFAULT_AGENT_WORKSPACE_DIR;
  const dir = resolveUserPath(rawDir);
  await fs.mkdir(dir, { recursive: true });

  if (!params?.ensureBootstrapFiles) {
    return { dir };
  }

  const agentsPath = path.join(dir, DEFAULT_AGENTS_FILENAME);
  const soulPath = path.join(dir, DEFAULT_SOUL_FILENAME);
  const toolsPath = path.join(dir, DEFAULT_TOOLS_FILENAME);
  const identityPath = path.join(dir, DEFAULT_IDENTITY_FILENAME);
  const userPath = path.join(dir, DEFAULT_USER_FILENAME);
  const heartbeatPath = path.join(dir, DEFAULT_HEARTBEAT_FILENAME);
  const bootstrapPath = path.join(dir, DEFAULT_BOOTSTRAP_FILENAME);
  const workspaceRulesPath = path.join(dir, DEFAULT_WORKSPACE_RULES_FILENAME);

  const isBrandNewWorkspace = await (async () => {
    const paths = [
      agentsPath,
      soulPath,
      toolsPath,
      identityPath,
      userPath,
      heartbeatPath,
      path.join(dir, DEFAULT_SOUL_PLUS_FILENAME),
      path.join(dir, DEFAULT_IDENTITY_PLUS_FILENAME),
      path.join(dir, DEFAULT_USER_PLUS_FILENAME),
    ];
    const existing = await Promise.all(
      paths.map(async (p) => {
        try {
          await fs.access(p);
          return true;
        } catch {
          return false;
        }
      }),
    );
    return existing.every((v) => !v);
  })();

  const agentsTemplate = await loadTemplate(DEFAULT_AGENTS_FILENAME);
  const soulTemplate = await loadTemplate(DEFAULT_SOUL_FILENAME);
  const toolsTemplate = await loadTemplate(DEFAULT_TOOLS_FILENAME);
  const identityTemplate = await loadTemplate(DEFAULT_IDENTITY_FILENAME);
  const userTemplate = await loadTemplate(DEFAULT_USER_FILENAME);
  const soulPlusTemplate = await loadTemplate(DEFAULT_SOUL_PLUS_FILENAME);
  const identityPlusTemplate = await loadTemplate(DEFAULT_IDENTITY_PLUS_FILENAME);
  const userPlusTemplate = await loadTemplate(DEFAULT_USER_PLUS_FILENAME);
  const heartbeatTemplate = await loadTemplate(DEFAULT_HEARTBEAT_FILENAME);
  const bootstrapTemplate = await loadTemplate(DEFAULT_BOOTSTRAP_FILENAME);
  const workspaceRulesTemplate = await loadTemplate(DEFAULT_WORKSPACE_RULES_FILENAME);

  await writeFileIfMissing(agentsPath, agentsTemplate);
  // When OPENCLAW_IMMUTABLE_DIR is set (Docker), SOUL/IDENTITY/USER/TOOLS/WORKSPACE_RULES live in the image read-only; do not write them to workspace.
  const useImmutableDir = Boolean(process.env.OPENCLAW_IMMUTABLE_DIR?.trim());
  if (!useImmutableDir) {
    await writeFileIfMissing(soulPath, soulTemplate);
    await writeFileIfMissing(toolsPath, toolsTemplate);
    await writeFileIfMissing(identityPath, identityTemplate);
    await writeFileIfMissing(userPath, userTemplate);
    await writeFileIfMissing(workspaceRulesPath, workspaceRulesTemplate);
  }
  await writeFileIfMissing(path.join(dir, DEFAULT_SOUL_PLUS_FILENAME), soulPlusTemplate);
  await writeFileIfMissing(path.join(dir, DEFAULT_IDENTITY_PLUS_FILENAME), identityPlusTemplate);
  await writeFileIfMissing(path.join(dir, DEFAULT_USER_PLUS_FILENAME), userPlusTemplate);
  await writeFileIfMissing(heartbeatPath, heartbeatTemplate);
  if (isBrandNewWorkspace) {
    await writeFileIfMissing(bootstrapPath, bootstrapTemplate);
  }
  // When not using immutable dir, make SOUL/IDENTITY/USER/TOOLS/WORKSPACE_RULES read-only in workspace so the agent cannot overwrite.
  if (!useImmutableDir) {
    await makeReadOnlyIfExists(soulPath);
    await makeReadOnlyIfExists(identityPath);
    await makeReadOnlyIfExists(userPath);
    await makeReadOnlyIfExists(toolsPath);
    await makeReadOnlyIfExists(workspaceRulesPath);
  }
  await ensureGitRepo(dir, isBrandNewWorkspace);

  return {
    dir,
    agentsPath,
    soulPath,
    toolsPath,
    identityPath,
    userPath,
    heartbeatPath,
    bootstrapPath,
  };
}

async function resolveMemoryBootstrapEntries(
  resolvedDir: string,
): Promise<Array<{ name: WorkspaceBootstrapFileName; filePath: string }>> {
  const candidates: WorkspaceBootstrapFileName[] = [
    DEFAULT_MEMORY_FILENAME,
    DEFAULT_MEMORY_ALT_FILENAME,
  ];
  const entries: Array<{ name: WorkspaceBootstrapFileName; filePath: string }> = [];
  for (const name of candidates) {
    const filePath = path.join(resolvedDir, name);
    try {
      await fs.access(filePath);
      entries.push({ name, filePath });
    } catch {
      // optional
    }
  }
  if (entries.length <= 1) {
    return entries;
  }

  const seen = new Set<string>();
  const deduped: Array<{ name: WorkspaceBootstrapFileName; filePath: string }> = [];
  for (const entry of entries) {
    let key = entry.filePath;
    try {
      key = await fs.realpath(entry.filePath);
    } catch {}
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    deduped.push(entry);
  }
  return deduped;
}

export async function loadWorkspaceBootstrapFiles(dir: string): Promise<WorkspaceBootstrapFile[]> {
  const resolvedDir = resolveUserPath(dir);

  const entries: Array<{
    name: WorkspaceBootstrapFileName;
    filePath: string;
  }> = [
    { name: DEFAULT_AGENTS_FILENAME, filePath: path.join(resolvedDir, DEFAULT_AGENTS_FILENAME) },
    { name: DEFAULT_SOUL_FILENAME, filePath: path.join(resolvedDir, DEFAULT_SOUL_FILENAME) },
    { name: DEFAULT_SOUL_PLUS_FILENAME, filePath: path.join(resolvedDir, DEFAULT_SOUL_PLUS_FILENAME) },
    { name: DEFAULT_TOOLS_FILENAME, filePath: path.join(resolvedDir, DEFAULT_TOOLS_FILENAME) },
    { name: DEFAULT_IDENTITY_FILENAME, filePath: path.join(resolvedDir, DEFAULT_IDENTITY_FILENAME) },
    { name: DEFAULT_IDENTITY_PLUS_FILENAME, filePath: path.join(resolvedDir, DEFAULT_IDENTITY_PLUS_FILENAME) },
    { name: DEFAULT_USER_FILENAME, filePath: path.join(resolvedDir, DEFAULT_USER_FILENAME) },
    { name: DEFAULT_USER_PLUS_FILENAME, filePath: path.join(resolvedDir, DEFAULT_USER_PLUS_FILENAME) },
    { name: DEFAULT_HEARTBEAT_FILENAME, filePath: path.join(resolvedDir, DEFAULT_HEARTBEAT_FILENAME) },
    { name: DEFAULT_BOOTSTRAP_FILENAME, filePath: path.join(resolvedDir, DEFAULT_BOOTSTRAP_FILENAME) },
    { name: DEFAULT_WORKSPACE_RULES_FILENAME, filePath: path.join(resolvedDir, DEFAULT_WORKSPACE_RULES_FILENAME) },
  ];

  entries.push(...(await resolveMemoryBootstrapEntries(resolvedDir)));

  const result: WorkspaceBootstrapFile[] = [];
  for (const entry of entries) {
    const readPath =
      (await resolveImmutableBootstrapPath(entry.name)) ?? entry.filePath;
    try {
      const content = await fs.readFile(readPath, "utf-8");
      result.push({
        name: entry.name,
        path: readPath,
        content,
        missing: false,
      });
    } catch {
      result.push({ name: entry.name, path: entry.filePath, missing: true });
    }
  }
  return result;
}

const SUBAGENT_BOOTSTRAP_ALLOWLIST = new Set([DEFAULT_AGENTS_FILENAME, DEFAULT_TOOLS_FILENAME]);

export function filterBootstrapFilesForSession(
  files: WorkspaceBootstrapFile[],
  sessionKey?: string,
): WorkspaceBootstrapFile[] {
  if (!sessionKey || !isSubagentSessionKey(sessionKey)) {
    return files;
  }
  return files.filter((file) => SUBAGENT_BOOTSTRAP_ALLOWLIST.has(file.name));
}
