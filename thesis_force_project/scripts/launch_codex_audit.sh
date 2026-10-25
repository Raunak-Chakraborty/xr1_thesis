#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

ROOT="/home/horus/robotics_thesis/xr1_force_exp"
LOGROOT="$ROOT/logs/codex"

MODE="${1:-new}"

case "$MODE" in
    new|resume)
        ;;
    *)
        echo "Usage: $0 [new|resume]"
        exit 2
        ;;
esac

test -s "$ROOT/AGENTS.md"
test -s "$ROOT/context/PROJECT_STATE.md"
test -s "$ROOT/context/SESSION_HANDOFF.md"

mkdir -p "$LOGROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"

SESSION_DIR="$LOGROOT/session_${STAMP}_${MODE}_$$"

mkdir -m 700 "$SESSION_DIR"

BEFORE="$SESSION_DIR/before.txt"
AFTER="$SESSION_DIR/after.txt"
TRANSCRIPT="$SESSION_DIR/terminal.typescript"

record_state() {
    local OUT="$1"

    {
        echo "Timestamp: $(date -Is)"
        echo "Codex: $(codex --version)"
        echo "Mode: $MODE"
        echo "Workspace: $ROOT"

        echo
        echo "===== XR-1 ====="

        git -C "$ROOT/xr1_thesis" branch --show-current
        git -C "$ROOT/xr1_thesis" rev-parse HEAD
        git -C "$ROOT/xr1_thesis" status --short

        echo
        echo "===== FORCEVLA REFERENCE ====="

        git -C "$ROOT/references/forcevla_thesis" \
            branch --show-current

        git -C "$ROOT/references/forcevla_thesis" \
            rev-parse HEAD

        git -C "$ROOT/references/forcevla_thesis" \
            status --short

        echo
        echo "===== EXPERIMENT WORKTREES ====="

        git -C "$ROOT/xr1_thesis" worktree list

        echo
        echo "===== XR-1 CHANGE SUMMARY ====="

        git -C "$ROOT/xr1_thesis" diff --stat
        git -C "$ROOT/xr1_thesis" diff --cached --stat

    } > "$OUT"
}

record_state "$BEFORE"

echo "===================================================="
echo "XR-1 FORCE — CODEX AUDIT SESSION"
echo "===================================================="

echo
echo "Workspace: $ROOT"
echo "Mode: $MODE"
echo "Sandbox: read-only"
echo "Approval policy: on-request"

echo
echo "Session directory:"
echo "$SESSION_DIR"

echo
echo "Read first:"
echo "  context/PROJECT_STATE.md"
echo "  context/SESSION_HANDOFF.md"

echo
echo "No source-code modifications are authorized."
echo

CODEX_ARGS=(
    codex
    -C "$ROOT"
    --sandbox read-only
    --ask-for-approval on-request
)

if [ "$MODE" = "resume" ]; then
    CODEX_ARGS+=(resume)
fi

printf -v CODEX_COMMAND '%q ' "${CODEX_ARGS[@]}"

set +e

script \
    --quiet \
    --flush \
    --return \
    --command "$CODEX_COMMAND" \
    "$TRANSCRIPT"

RESULT=$?

set -e

record_state "$AFTER"

{
    echo "Session: $SESSION_DIR"
    echo "Mode: $MODE"
    echo "Exit status: $RESULT"
    echo "Finished: $(date -Is)"
} > "$SESSION_DIR/session_summary.txt"

echo
echo "===================================================="
echo "CODEX SESSION ENDED"
echo "===================================================="

echo
echo "Exit status: $RESULT"
echo "Transcript: $TRANSCRIPT"
echo "Before:     $BEFORE"
echo "After:      $AFTER"

echo
echo "IMPORTANT:"
echo "Verify that PROJECT_STATE.md and SESSION_HANDOFF.md"
echo "reflect the work completed before starting a new session."

exit "$RESULT"
