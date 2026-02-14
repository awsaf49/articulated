#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_commit.sh <commit> -- <command...>

Example:
  scripts/run_commit.sh 1a2b3c4d -- \
    python -m articulated.estimation.train \
      --config articulated/configs/estimation/rnn.yaml

Notes:
  - Ensures a clean checkout of the requested commit.
  - Set ALLOW_DIRTY=1 to bypass the clean working tree check.
USAGE
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

COMMIT="$1"
shift

if [[ "${1:-}" != "--" ]]; then
  usage
  exit 2
fi
shift

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: not inside a git repository: $ROOT_DIR" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  if [[ "${ALLOW_DIRTY:-0}" != "1" ]]; then
    echo "Error: working tree has uncommitted changes." >&2
    echo "Commit/stash changes or set ALLOW_DIRTY=1 to proceed." >&2
    exit 1
  else
    echo "Warning: proceeding with dirty working tree (ALLOW_DIRTY=1)." >&2
  fi
fi

git fetch --all --prune

if ! git cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
  echo "Error: commit not found after fetch: $COMMIT" >&2
  exit 1
fi

git checkout --force "$COMMIT"

CHECKED_OUT="$(git rev-parse HEAD)"
if [[ "$CHECKED_OUT" != "$COMMIT" ]]; then
  echo "Warning: checked out $CHECKED_OUT (requested $COMMIT)." >&2
fi

git submodule update --init --recursive

export RUN_COMMIT="$CHECKED_OUT"
export RUN_COMMIT_SHORT="${CHECKED_OUT:0:8}"
export RUN_TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"

echo "Run commit: $RUN_COMMIT"
echo "Run timestamp (UTC): $RUN_TIMESTAMP"
echo "Command: $*"

exec "$@"
