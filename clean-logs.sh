#!/usr/bin/env bash
# Manually shrink the log directory (the proxy also cleans up at startup).
# Usage:
#   ./clean-logs.sh [LOGDIR] [MAX_MB] [TARGET_MB]
# Defaults: LOGDIR=./logs  MAX_MB=4096  TARGET_MB=2048
#
# 1. delete session dirs older than 7 days
# 2. if total still > MAX_MB, delete oldest dirs until <= TARGET_MB
set -euo pipefail

LOGDIR="${1:-./logs}"
MAX_MB="${2:-4096}"
TARGET_MB="${3:-2048}"

if [[ ! -d "$LOGDIR" ]]; then
    echo "logdir '$LOGDIR' does not exist; nothing to do"
    exit 0
fi

# 1. age-based (7 days)
find "$LOGDIR" -mindepth 1 -maxdepth 1 -type d -mtime +7 -print0 \
    | while IFS= read -r -d '' d; do
        sz=$(du -sm "$d" | cut -f1)
        rm -rf "$d"
        echo "removed (age): $d (${sz}MB)"
    done

# 2. size-based fallback: oldest first until under TARGET_MB
total=$(du -sm "$LOGDIR" | cut -f1)
echo "total: ${total}MB (max ${MAX_MB}MB, target ${TARGET_MB}MB)"
if (( total > MAX_MB )); then
    while (( total > TARGET_MB )); do
        oldest=$(find "$LOGDIR" -mindepth 1 -maxdepth 1 -type d \
                 -printf '%T@ %p\n' | sort -n | head -1 | cut -d' ' -f2-)
        [[ -z "$oldest" ]] && break
        sz=$(du -sm "$oldest" | cut -f1)
        rm -rf "$oldest"
        total=$(( total - sz ))
        echo "removed (size): $oldest (${sz}MB), now ~${total}MB"
    done
fi
echo "done: ~${total}MB"
