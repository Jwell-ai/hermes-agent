#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"

if [ ! -d venv ]; then
  "$PYTHON" -m venv venv
fi

VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"

"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install -e .

printf '%s\n' "Canvas agent core is ready."
printf '%s\n' "Env is read from ../server/server.env. Use .env.canvas only for local overrides."
printf '%s\n' "Run: ./run-canvas-agent.sh"
