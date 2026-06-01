#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "../server/server.env" ]; then
  set -a
  . "../server/server.env"
  set +a
fi

if [ -f ".env.canvas" ]; then
  set -a
  . ".env.canvas"
  set +a
fi

PYTHON="${PYTHON:-python3}"
if [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
  PYTHON="$SCRIPT_DIR/venv/bin/python"
fi

exec "$PYTHON" canvas_agent_service.py
