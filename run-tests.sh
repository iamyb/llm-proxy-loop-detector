#!/usr/bin/env bash
# Run the regression tests inside the proxy container (which has orjson).
# Usage:
#   ./run-tests.sh            # all tests
#   ./run-tests.sh -v         # verbose
#   LLM_PROXY_CONTAINER=other ./run-tests.sh   # different container name
set -euo pipefail

CONTAINER="${LLM_PROXY_CONTAINER:-llm-proxy-loop-detector}"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "container '$CONTAINER' is not running; start it first:"
    echo "  docker compose up -d --build"
    exit 1
fi

exec docker exec "$CONTAINER" python3 /app/test_regression.py "$@"
