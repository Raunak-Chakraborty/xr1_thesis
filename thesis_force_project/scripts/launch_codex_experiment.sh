#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

ROOT="/home/horus/robotics_thesis/xr1_force_exp"

EXPERIMENT="${1:-}"
PERMISSION="${2:-audit}"
SESSION_MODE="${3:-new}"

usage() {
    echo "Usage:"
    echo "  $0 <baseline|state|encoder|moe> <audit|develop> <new|resume>"
    echo
    echo "Examples:"
    echo "  $0 encoder audit new"
    echo "  $0 encoder develop new"
    echo "  $0 encoder develop resume"
}

case "$EXPERIMENT" in
    baseline|state|encoder|moe) ;;
    *)
        usage
        exit 2
        ;;
esac

case "$PERMISSION" in
    audit|develop) ;;
    *)
        usage
        exit 2
        ;;
esac

case "$SESSION_MODE" in
    new|resume) ;;
    *)
        usage
        exit 2
        ;;
esac

if [ "$#" -gt 3 ]; then
    usage
    exit 2
fi

WT="$ROOT/worktrees/xr1_force_$EXPERIMENT"
EXPECTED_BRANCH="thesis/xr1-force/$EXPERIMENT"

LOGROOT="$ROOT/logs/codex/$EXPERIMENT"

BASELINE="afe739d1c9aac9218036a13fee9f9eb182747159"

echo "======================================================"
echo "XR-1 FORCE — CODEX EXPERIMENT SESSION"
echo "======================================================"

echo
echo "Experiment: $EXPERIMENT"
echo "Branch:     $EXPECTED_BRANCH"
echo "Permission: $PERMISSION"
echo "Session:    $SESSION_MODE"

echo
echo "===== VERIFY PROJECT ====="

test -s "$ROOT/AGENTS.md"
test -s "$ROOT/context/PROJECT_STATE.md"
test -s "$ROOT/context/SESSION_HANDOFF.md"

test "$(realpath "$WT/AGENTS.md")" = \
    "$ROOT/AGENTS.md"

test "$(realpath "$WT/context")" = \
    "$ROOT/context"

test "$(git -C "$WT" rev-parse --show-toplevel)" = \
    "$WT"

test "$(git -C "$WT" branch --show-current)" = \
    "$EXPECTED_BRANCH"

test "$(git -C "$ROOT/xr1_thesis" branch --show-current)" = \
    "thesis/doosan-xr1"

test "$(git -C "$ROOT/xr1_thesis" rev-parse HEAD)" = \
    "$BASELINE"

test -z "$(git -C "$ROOT/xr1_thesis" status --porcelain)"

test "$(git -C "$ROOT/references/forcevla_thesis" \
    rev-parse HEAD)" = \
    "ef5ba4f5522d241fe72b559d2ccc331405f56b67"

test -z "$(git -C "$ROOT/references/forcevla_thesis" \
    status --porcelain)"

echo "PASS: Correct experiment and protected sources."

echo
echo "===== ACQUIRE PROJECT SESSION LOCK ====="

mkdir -p "$ROOT/logs/codex"
mkdir -p "$LOGROOT"

LOCK="$ROOT/logs/codex/.active_session.lock"

exec 9>"$LOCK"

if ! flock -n 9; then
    echo "ERROR: Another managed Codex session is active."
    echo "Finish that session before starting another."
    exit 1
fi

echo "PASS: Exclusive session lock acquired."

echo
echo "===== CREATE SESSION DIRECTORY ====="

SESSION_DIR="$(
    mktemp -d \
    "$LOGROOT/session_$(date +%Y%m%d_%H%M%S)_${SESSION_MODE}_XXXXXX"
)"

TRANSCRIPT="$SESSION_DIR/terminal.typescript"
BEFORE="$SESSION_DIR/before.txt"
AFTER="$SESSION_DIR/after.txt"
SUMMARY="$SESSION_DIR/session_summary.txt"

echo "Session directory:"
echo "$SESSION_DIR"

record_state() {
    local OUTPUT="$1"

    {
        echo "Timestamp: $(date -Is)"
        echo "Experiment: $EXPERIMENT"
        echo "Expected branch: $EXPECTED_BRANCH"
        echo "Permission: $PERMISSION"
        echo "Session mode: $SESSION_MODE"

        echo
        echo "===== CODEX VERSION ====="
        codex --version

        echo
        echo "===== EXPERIMENT GIT ROOT ====="
        git -C "$WT" rev-parse --show-toplevel

        echo
        echo "===== EXPERIMENT BRANCH ====="
        git -C "$WT" branch --show-current

        echo
        echo "===== EXPERIMENT HEAD ====="
        git -C "$WT" rev-parse HEAD

        echo
        echo "===== EXPERIMENT STATUS ====="
        git -C "$WT" status --short --untracked-files=all

        echo
        echo "===== TRACKED CHANGES ====="
        git -C "$WT" diff --stat
        git -C "$WT" diff --cached --stat

        echo
        echo "===== RECENT COMMIT ====="
        git -C "$WT" log -1 --format='%H%n%s'

        echo
        echo "===== PROJECT STATE HASH ====="
        sha256sum "$ROOT/context/PROJECT_STATE.md"

        echo
        echo "===== HANDOFF HASH ====="
        sha256sum "$ROOT/context/SESSION_HANDOFF.md"

        echo
        echo "===== ORIGINAL DOOSAN HEAD ====="
        git -C "$ROOT/xr1_thesis" rev-parse HEAD

        echo
        echo "===== FORCEVLA HEAD ====="
        git -C "$ROOT/references/forcevla_thesis" \
            rev-parse HEAD

    } > "$OUTPUT"
}

record_state "$BEFORE"

echo
echo "===== CONFIGURE CODEX ====="

CODEX_ARGS=(
    codex
    -C "$WT"
    --ask-for-approval on-request
)

if [ "$PERMISSION" = "audit" ]; then

    CODEX_ARGS+=(--sandbox read-only)

    echo "Sandbox: read-only"
    echo "Writable source directories: none"

else

    CODEX_ARGS+=(--sandbox workspace-write)

    # Shared context can be updated during development.
    # This does not grant write access to ForceVLA
    # or the original Doosan checkout.

    CODEX_ARGS+=(--add-dir "$ROOT/context")

    echo "Sandbox: workspace-write"
    echo "Writable project areas:"
    echo "  $WT"
    echo "  $ROOT/context"

fi

if [ "$SESSION_MODE" = "resume" ]; then
    CODEX_ARGS+=(resume)
else
    if [ "$PERMISSION" = "audit" ]; then

        INITIAL_PROMPT="Read AGENTS.md, context/PROJECT_STATE.md, \
context/SESSION_HANDOFF.md, and context/REPOSITORY_BASELINES.md. \
You are in the $EXPERIMENT experimental worktree. \
Confirm the current Git branch and summarize project status. \
Do not modify files. Do not begin architecture implementation. \
Wait for my next instruction."

    else

        INITIAL_PROMPT="Read AGENTS.md, context/PROJECT_STATE.md, \
context/SESSION_HANDOFF.md, and context/REPOSITORY_BASELINES.md. \
You are working only on experiment $EXPERIMENT. \
Confirm the Git branch, inspect pending changes, and explain \
the next approved task. Preserve existing changes. \
Do not start implementation until I give you a specific task. \
Keep context/PROJECT_STATE.md and \
context/SESSION_HANDOFF.md current after substantive work. \
Do not commit or push without my approval."

    fi

    CODEX_ARGS+=("$INITIAL_PROMPT")
fi

printf -v CODEX_COMMAND '%q ' "${CODEX_ARGS[@]}"

echo
echo "===== LAUNCH CODEX ====="

echo "Codex working directory:"
echo "$WT"

echo
echo "Transcript:"
echo "$TRANSCRIPT"

echo
echo "No automatic Git commits or pushes."
echo "No automatic browser launch."
echo

# Always record the final repository state when the
# launcher exits, including after a Codex error.

finish_session() {
    local EXIT_CODE=$?

    trap - EXIT
    set +e

    record_state "$AFTER"

    {
        echo "Experiment: $EXPERIMENT"
        echo "Branch: $EXPECTED_BRANCH"
        echo "Permission: $PERMISSION"
        echo "Session mode: $SESSION_MODE"
        echo "Exit status: $EXIT_CODE"
        echo "Finished: $(date -Is)"
        echo "Transcript: $TRANSCRIPT"
        echo "Before: $BEFORE"
        echo "After: $AFTER"
    } > "$SUMMARY"

    echo
    echo "======================================================"
    echo "CODEX SESSION FINISHED"
    echo "======================================================"

    echo "Exit status: $EXIT_CODE"
    echo "Session directory: $SESSION_DIR"

    echo
    echo "Review PROJECT_STATE.md and SESSION_HANDOFF.md"
    echo "before beginning another substantial task."

    exit "$EXIT_CODE"
}

trap finish_session EXIT

script \
    --quiet \
    --flush \
    --return \
    --command "$CODEX_COMMAND" \
    "$TRANSCRIPT"
