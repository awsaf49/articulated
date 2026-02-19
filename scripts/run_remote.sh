#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_remote.sh -- <command...>

Example:
  scripts/run_remote.sh -- \
    python -m articulated.estimation.train \
      --config articulated/configs/estimation/rnn.yaml

Defaults:
  REMOTE_HOST=mllab07
  REMOTE_DIR=/home/huntae/workspace/articulated

Notes:
  - Requires a clean local working tree.
  - Requires local HEAD to be pushed to upstream.
  - Set ALLOW_DIRTY_LOCAL=1 to bypass the local clean check.
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

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
  if [[ "${ALLOW_DIRTY_LOCAL:-0}" != "1" ]]; then
    echo "Error: local working tree has uncommitted changes." >&2
    echo "Commit/stash changes or set ALLOW_DIRTY_LOCAL=1 to proceed." >&2
    exit 1
  else
    echo "Warning: proceeding with dirty local working tree (ALLOW_DIRTY_LOCAL=1)." >&2
  fi
fi

if ! git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
  echo "Error: no upstream configured for this branch." >&2
  echo "Set upstream with: git push -u origin <branch>" >&2
  exit 1
fi

ahead=$(git rev-list --count @{u}..HEAD)
if [[ "$ahead" -gt 0 ]]; then
  echo "Error: local branch has $ahead commit(s) not pushed." >&2
  echo "Push your commits with: git push" >&2
  exit 1
fi

COMMIT="$(git rev-parse HEAD)"
HOST="${REMOTE_HOST:-mllab07}"
REMOTE_DIR="${REMOTE_DIR:-/home/huntae/workspace/articulated}"

remote_cmd="cd \"$REMOTE_DIR\" && ./scripts/run_commit.sh \"$COMMIT\" --"
for arg in "$@"; do
  remote_cmd+=" $(printf '%q' "$arg")"
done

echo "Remote host: $HOST"
echo "Remote dir: $REMOTE_DIR"
echo "Commit: $COMMIT"

ssh "$HOST" "bash -lc $(printf '%q' "$remote_cmd")"
