#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing $PYTHON. Follow the README setup first." >&2
  exit 1
fi

echo "Starting backend on http://localhost:8000 ..."
(cd "$DIR/backend" && "$PYTHON" -m uvicorn main:app --reload --host 0.0.0.0 --port 8000) &
BACKEND_PID=$!

echo "Starting frontend on http://localhost:5173 ..."
(cd "$DIR/frontend" && npm run dev -- --host 0.0.0.0) &
FRONTEND_PID=$!

# Kill both children on exit / Ctrl-C.
trap 'echo; echo "Stopping..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null' EXIT INT TERM

wait
