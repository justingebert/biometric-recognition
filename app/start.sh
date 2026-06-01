#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting backend on http://localhost:8000 ..."
(cd "$DIR/backend" && conda run --no-capture-output -n biometrics \
  uvicorn main:app --reload --port 8000) &
BACKEND_PID=$!

echo "Starting frontend on http://localhost:5173 ..."
(cd "$DIR/frontend" && npm run dev) &
FRONTEND_PID=$!

# Kill both children on exit / Ctrl-C.
trap 'echo; echo "Stopping..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null' EXIT INT TERM

wait
