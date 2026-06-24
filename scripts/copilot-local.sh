#!/usr/bin/env bash
# copilot-local.sh — Bootstrap GitHub Copilot CLI to use this local llama-openai-server.
#
# USAGE:
#   ./scripts/copilot-local.sh [copilot-cli-args...]
#
# The script:
#   1. Reads HOST, PORT, MODEL_ID, and N_CTX from .env (with sane defaults).
#   2. Verifies the Copilot CLI is installed.
#   3. Checks whether the local server is reachable; offers to start it if not.
#   4. Exports COPILOT_PROVIDER_* env vars and execs the Copilot CLI.
#
# Copilot CLI BYOK env vars (set automatically):
#   COPILOT_PROVIDER_BASE_URL        — http://HOST:PORT/v1
#   COPILOT_PROVIDER_TYPE            — openai  (compatible with any OpenAI-style endpoint)
#   COPILOT_PROVIDER_API_KEY         — (empty; local server needs no auth)
#   COPILOT_PROVIDER_MODEL_ID        — claude-sonnet-4 (catalog alias for agent config)
#   COPILOT_PROVIDER_WIRE_MODEL      — MODEL_ID from .env (sent to the actual server)
#   COPILOT_PROVIDER_MAX_PROMPT_TOKENS  — N_CTX - MAX_OUTPUT_TOKENS
#   COPILOT_PROVIDER_MAX_OUTPUT_TOKENS  — 8192

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Helpers ────────────────────────────────────────────────────────────────

die()  { echo "[error] $*" >&2; exit 1; }
info() { echo "[info]  $*"; }
warn() { echo "[warn]  $*" >&2; }

# ── Parse .env ─────────────────────────────────────────────────────────────

ENV_FILE="${REPO_DIR}/.env"

parse_env_var() {
    local key="$1" default="$2"
    if [[ -f "${ENV_FILE}" ]]; then
        # Strip inline comments, leading/trailing whitespace, and quotes.
        local val
        val=$(grep -E "^[[:space:]]*${key}[[:space:]]*=" "${ENV_FILE}" \
            | tail -1 \
            | sed 's/^[^=]*=//; s/[[:space:]]*#.*$//; s/^[[:space:]]*//; s/[[:space:]]*$//; s/^["'"'"']//; s/["'"'"']$//')
        echo "${val:-${default}}"
    else
        echo "${default}"
    fi
}

SERVER_HOST=$(parse_env_var "HOST" "0.0.0.0")
SERVER_PORT=$(parse_env_var "PORT" "8000")
MODEL_ID=$(parse_env_var "MODEL_ID" "local-model")
N_CTX=$(parse_env_var "N_CTX" "32768")

# Token budget: reserve MAX_OUTPUT_TOKENS from the context window for output.
MAX_OUTPUT_TOKENS=8192
# If N_CTX is 0 (auto-detect mode) fall back to 32768 for budget calculation.
_ctx_for_budget=$(( N_CTX > 0 ? N_CTX : 32768 ))
MAX_PROMPT_TOKENS=$(( _ctx_for_budget - MAX_OUTPUT_TOKENS ))

# Copilot CLI connects from localhost; replace 0.0.0.0 with 127.0.0.1.
CONNECT_HOST="${SERVER_HOST}"
if [[ "${CONNECT_HOST}" == "0.0.0.0" ]]; then
    CONNECT_HOST="127.0.0.1"
fi

BASE_URL="http://${CONNECT_HOST}:${SERVER_PORT}/v1"
HEALTH_URL="http://${CONNECT_HOST}:${SERVER_PORT}/v1/health"

# ── Check Copilot CLI ──────────────────────────────────────────────────────

if ! command -v copilot &>/dev/null; then
    warn "GitHub Copilot CLI ('copilot') not found in PATH."
    warn "Install it with:  npm install -g @github/copilot"
    warn ""
    warn "Or via gh CLI (preview):  gh copilot"
    die  "Copilot CLI is required — install it and re-run this script."
fi

COPILOT_BIN="$(command -v copilot)"
info "Copilot CLI found: ${COPILOT_BIN}"

# ── Check / start local server ─────────────────────────────────────────────

check_server() {
    # Returns 0 if /v1/health responds with HTTP 200, 1 otherwise.
    curl --silent --fail --max-time 3 "${HEALTH_URL}" >/dev/null 2>&1
}

if check_server; then
    info "Local server is reachable at ${BASE_URL}"
else
    warn "Local server does not appear to be running (${HEALTH_URL} unreachable)."
    if [[ ! -f "${ENV_FILE}" ]]; then
        die ".env not found — copy .env.example to .env and configure it, then start the server."
    fi

    read -r -p "[prompt] Start the server now? [Y/n] " REPLY
    REPLY="${REPLY:-Y}"
    if [[ "${REPLY}" =~ ^[Yy]$ ]]; then
        info "Starting llama-openai-server in the background..."
        pushd "${REPO_DIR}" >/dev/null
        bash start.sh &
        SERVER_PID=$!
        popd >/dev/null

        info "Waiting for server to become ready (PID ${SERVER_PID})..."
        TIMEOUT=60
        ELAPSED=0
        while ! check_server; do
            sleep 2
            ELAPSED=$((ELAPSED + 2))
            if [[ ${ELAPSED} -ge ${TIMEOUT} ]]; then
                die "Server did not become ready within ${TIMEOUT}s. Check logs."
            fi
            echo -n "."
        done
        echo ""
        info "Server ready."
    else
        die "Server must be running before launching Copilot CLI in offline mode."
    fi
fi

# ── Export BYOK environment variables ─────────────────────────────────────

export COPILOT_PROVIDER_BASE_URL="${BASE_URL}"
export COPILOT_PROVIDER_TYPE="openai"
export COPILOT_PROVIDER_API_KEY=""           # no auth required for local server
# Use a catalog-known model alias so the CLI picks up the correct agent
# configuration (tool-calling format, prompting strategy).  The wire model
# is set separately to the actual model name your server expects.
export COPILOT_PROVIDER_MODEL_ID="claude-sonnet-4"
export COPILOT_PROVIDER_WIRE_MODEL="${MODEL_ID}"
export COPILOT_PROVIDER_MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS}"
export COPILOT_PROVIDER_MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS}"

info "Provider config:"
info "  COPILOT_PROVIDER_BASE_URL           = ${COPILOT_PROVIDER_BASE_URL}"
info "  COPILOT_PROVIDER_TYPE               = ${COPILOT_PROVIDER_TYPE}"
info "  COPILOT_PROVIDER_MODEL_ID           = ${COPILOT_PROVIDER_MODEL_ID}"
info "  COPILOT_PROVIDER_WIRE_MODEL         = ${COPILOT_PROVIDER_WIRE_MODEL}"
info "  COPILOT_PROVIDER_MAX_PROMPT_TOKENS  = ${COPILOT_PROVIDER_MAX_PROMPT_TOKENS}"
info "  COPILOT_PROVIDER_MAX_OUTPUT_TOKENS  = ${COPILOT_PROVIDER_MAX_OUTPUT_TOKENS}"
echo ""

# ── Launch Copilot CLI ─────────────────────────────────────────────────────

exec "${COPILOT_BIN}" "$@"
