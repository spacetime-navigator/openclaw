FROM node:22-bookworm

# Install Bun (required for build scripts)
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:${PATH}"

RUN corepack enable
RUN corepack prepare pnpm@10.23.0 --activate

WORKDIR /app

ARG OPENCLAW_DOCKER_APT_PACKAGES=""
RUN if [ -n "$OPENCLAW_DOCKER_APT_PACKAGES" ]; then \
      apt-get update && \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $OPENCLAW_DOCKER_APT_PACKAGES && \
      apt-get clean && \
      rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*; \
    fi

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml .npmrc ./
COPY ui/package.json ./ui/package.json
COPY patches ./patches
COPY scripts ./scripts

RUN pnpm install --frozen-lockfile

COPY . .
RUN OPENCLAW_A2UI_SKIP_MISSING=1 pnpm build
# Force pnpm for UI build (Bun may fail on ARM/Synology architectures)
ENV OPENCLAW_PREFER_PNPM=1
RUN pnpm ui:build
RUN chmod +x /app/scripts/docker-startup.sh

# Immutable workspace templates: SOUL, IDENTITY, USER, TOOLS, WORKSPACE_RULES live in image read-only.
# When OPENCLAW_IMMUTABLE_DIR is set (e.g. in Docker), the app reads these from here
# and does not write them to the workspace, so the agent cannot chmod or edit them.
RUN mkdir -p /openclaw-immutable && \
  cp docs/reference/templates/SOUL.md docs/reference/templates/IDENTITY.md \
     docs/reference/templates/USER.md docs/reference/templates/TOOLS.md \
     docs/reference/templates/WORKSPACE_RULES.md /openclaw-immutable/ && \
  chmod 444 /openclaw-immutable/*.md

# GitHub CLI for the GitHub skill (PR workflow). Agent can create PRs when GH_TOKEN/OPENCLAW_GITHUB_REPO are set.
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gh && \
  apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ENV NODE_ENV=production

# Allow non-root user to write temp files during runtime/tests.
RUN chown -R node:node /app

# Security hardening: Run as non-root user
# The node:22-bookworm image includes a 'node' user (uid 1000)
# This reduces the attack surface by preventing container escape via root privileges
USER node

# Start gateway server with default config.
# Binds to loopback (127.0.0.1) by default for security.
#
# For container platforms requiring external health checks:
#   1. Set OPENCLAW_GATEWAY_TOKEN or OPENCLAW_GATEWAY_PASSWORD env var
#   2. Override CMD: ["node","dist/index.js","gateway","--allow-unconfigured","--bind","lan"]
CMD ["node", "dist/index.js", "gateway", "--allow-unconfigured"]
