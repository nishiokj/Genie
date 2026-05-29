#!/usr/bin/env bash
# Local dev launcher. Four moving parts:
#   1. substrate executioner host  (127.0.0.1:8765) — long-running host process
#   2. nova-service container       (127.0.0.1:9556) — Nova daemon w/ Codex auth
#                                    attached to the shared substrate env
#   3. agent service                (127.0.0.1:8001) — Python; talks to Nova
#                                    daemon via the nova-client TCP protocol
#   4. app service                  (127.0.0.1:8000) — UI + proxy

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_DIR="$ROOT/server"
SUBSTRATE_BIN="${SUBSTRATE_BIN:-}"
NOVA_REPO="${NOVA_REPO:-$ROOT/../agent}"
NOVA_CLIENT_PATH="${NOVA_CLIENT_PATH:-$NOVA_REPO/packages/clients/python}"
NOVA_IMAGE="${NOVA_IMAGE:-nova-service:local}"
NOVA_CONTAINER="${NOVA_CONTAINER:-nova-svc}"

export EXAMPLES_MOUNT="${EXAMPLES_MOUNT:-$ROOT/domains}"
export DOMAINS_DIR="${DOMAINS_DIR:-$ROOT/domains}"
export SUBSTRATE_HOST_URL="${SUBSTRATE_HOST_URL:-http://127.0.0.1:8765/}"
export SUBSTRATE_ENVIRONMENT_ID="${SUBSTRATE_ENVIRONMENT_ID:-interview_shared}"
export SUBSTRATE_WORKSPACE_ROOT="${SUBSTRATE_WORKSPACE_ROOT:-/tmp/nova-substrate-shared}"
export AGENT_URL="${AGENT_URL:-http://127.0.0.1:8001}"
export NOVA_HOST="${NOVA_HOST:-127.0.0.1}"
export NOVA_PORT="${NOVA_PORT:-9556}"

if [ ! -d "$NOVA_CLIENT_PATH" ]; then
  echo "[dev] nova-client python package not found at $NOVA_CLIENT_PATH" >&2
  exit 1
fi
if ! docker image inspect "$NOVA_IMAGE" >/dev/null 2>&1; then
  echo "[dev] docker image $NOVA_IMAGE not found." >&2
  echo "[dev] build it: (cd $NOVA_REPO && bun run --cwd packages/infra/harness-daemon build && docker build -f Dockerfile.nova-service -t $NOVA_IMAGE .)" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.11 || command -v python3.10 || command -v python3)}"
if [ ! -d "$SERVER_DIR/.venv" ]; then
  "$PYTHON_BIN" -m venv "$SERVER_DIR/.venv"
  "$SERVER_DIR/.venv/bin/pip" install -q --upgrade pip
  "$SERVER_DIR/.venv/bin/pip" install -q -r "$SERVER_DIR/agent/requirements.txt"
  "$SERVER_DIR/.venv/bin/pip" install -q -r "$SERVER_DIR/app/requirements.txt"
  "$SERVER_DIR/.venv/bin/pip" install -q -e "$NOVA_CLIENT_PATH"
fi
PY="$SERVER_DIR/.venv/bin/python"
if [ -z "$SUBSTRATE_BIN" ]; then
  SUBSTRATE_BIN="$("$PY" -c 'from substrate_runtime import binary_path; print(binary_path())')"
fi
if [ ! -x "$SUBSTRATE_BIN" ]; then
  echo "[dev] substrate runtime binary not found at $SUBSTRATE_BIN" >&2
  echo "[dev] install substrate-sdk/substrate-runtime or set SUBSTRATE_BIN explicitly" >&2
  exit 1
fi

echo "[dev] python:    $($PY --version)"
echo "[dev] substrate: $SUBSTRATE_BIN  -> $SUBSTRATE_HOST_URL"
echo "[dev] nova:      $NOVA_IMAGE  -> $NOVA_HOST:$NOVA_PORT"
echo "[dev] workspace: $SUBSTRATE_WORKSPACE_ROOT (env=$SUBSTRATE_ENVIRONMENT_ID)"

cleanup() {
  docker rm -f "$NOVA_CONTAINER" >/dev/null 2>&1 || true
  kill 0 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 1. substrate host
mkdir -p "$SUBSTRATE_WORKSPACE_ROOT"
"$SUBSTRATE_BIN" host --addr 127.0.0.1:8765 --state-dir /tmp/substrate-state &
until "$PYTHON_BIN" -c "
import socket, sys
s=socket.socket(); s.settimeout(0.5)
try: s.connect(('127.0.0.1', 8765)); s.close(); sys.exit(0)
except Exception: sys.exit(1)
" 2>/dev/null; do sleep 0.3; done

# 2. pre-create the shared substrate env (Nova reads it at daemon startup)
"$PY" -c "
import sys; sys.path.insert(0, '$SERVER_DIR/agent')
from substrate_host import ensure_shared_environment
info = ensure_shared_environment()
print('[dev] substrate env ready:', info.get('environment', info).get('id'))
"

# 3. nova-service container (attaches to the shared substrate env over HTTP)
docker rm -f "$NOVA_CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$NOVA_CONTAINER" --network host \
  -e NOVA_HOST=0.0.0.0 -e NOVA_PORT="$NOVA_PORT" -e NOVA_DAEMON_IDLE_TIMEOUT=0 \
  -e NOVA_TOOL_EXECUTION_BACKEND=substrate \
  -e NOVA_SUBSTRATE_HOST_BASE_URL="$SUBSTRATE_HOST_URL" \
  -e NOVA_SUBSTRATE_ENVIRONMENT_ID="$SUBSTRATE_ENVIRONMENT_ID" \
  -v "$HOME/.config/nova:/root/.config/nova:ro" \
  -v "$SUBSTRATE_WORKSPACE_ROOT:$SUBSTRATE_WORKSPACE_ROOT" \
  "$NOVA_IMAGE" \
  bun packages/infra/harness-daemon/dist/index.js --host 0.0.0.0 --port "$NOVA_PORT" --idle-timeout 0 --dangerous >/dev/null

# Wait for the bus to come up
until "$PY" -c "
import socket, sys
s=socket.socket(); s.settimeout(0.5)
try: s.connect(('$NOVA_HOST', $NOVA_PORT)); s.close(); sys.exit(0)
except Exception: sys.exit(1)
" 2>/dev/null; do sleep 0.3; done
echo "[dev] nova daemon listening"

# 4. agent + app
cd "$SERVER_DIR/agent" && "$PY" -m uvicorn main:app --host 127.0.0.1 --port 8001 &
cd "$SERVER_DIR/app"   && "$PY" -m uvicorn main:app --host 127.0.0.1 --port 8000 &

echo "[dev] UI: http://127.0.0.1:8000"
wait
