#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

ROOT="/home/horus/robotics_thesis/xr1_force_exp"

WT="$ROOT/worktrees/xr1_force_context"
PACKAGE="$WT/thesis_force_project"

BRANCH="thesis/xr1-force/context"

MODE="${1:-status}"

case "$MODE" in
    status|prepare)
        ;;
    *)
        echo "Usage: $0 [status|prepare]"
        exit 2
        ;;
esac

if [ "$#" -gt 1 ]; then
    echo "ERROR: Too many arguments."
    exit 2
fi

FILES=(
    "AGENTS.md"
    "README.md"

    "context/ARCHITECTURE.md"
    "context/CODEX_SESSION_OPERATIONS.md"
    "context/CONTEXT_INDEX.md"
    "context/DATA_CONTRACT.md"
    "context/DECISIONS.md"
    "context/EXPERIMENT_PLAN.md"
    "context/PROJECT_STATE.md"
    "context/REPOSITORY_BASELINES.md"
    "context/SESSION_HANDOFF.md"

    "scripts/launch_codex_audit.sh"
    "scripts/launch_codex_experiment.sh"
    "scripts/sync_project_context.sh"
)

echo "======================================================"
echo "XR-1 FORCE — PROJECT CONTEXT SYNCHRONIZATION"
echo "======================================================"

echo
echo "Mode: $MODE"

echo
echo "===== 1. ACQUIRE PROJECT LOCK ====="

mkdir -p "$ROOT/logs/codex"

exec 9>"$ROOT/logs/codex/.active_session.lock"

if ! flock -n 9; then
    echo "ERROR: A managed Codex session is active."
    echo "Finish it before synchronizing project context."
    exit 1
fi

echo "PASS: Exclusive project lock acquired."

echo
echo "===== 2. VERIFY CONTEXT WORKTREE ====="

test "$(git -C "$WT" branch --show-current)" = \
    "$BRANCH"

test "$(git -C "$WT" remote get-url origin)" = \
    "https://github.com/Raunak-Chakraborty/xr1_thesis.git"

echo "PASS: Correct context branch and repository."

echo
echo "===== 3. VERIFY REMOTE SYNCHRONIZATION ====="

LOCAL_HEAD="$(git -C "$WT" rev-parse HEAD)"

REMOTE_HEAD="$(
    git -C "$WT" ls-remote \
        origin "refs/heads/$BRANCH" |
        awk -v ref="refs/heads/$BRANCH" \
            '$2 == ref {print $1}'
)"

if [ -z "$REMOTE_HEAD" ]; then
    echo "ERROR: Remote context branch not found."
    exit 1
fi

echo "Local:  $LOCAL_HEAD"
echo "Remote: $REMOTE_HEAD"

if [ "$LOCAL_HEAD" != "$REMOTE_HEAD" ]; then
    echo "ERROR: Local and GitHub context commits differ."
    echo "Resolve this before preparing synchronization."
    exit 1
fi

echo "PASS: Context branch matches GitHub."

echo
echo "===== 4. VERIFY APPROVED SOURCE FILES ====="

for FILE in "${FILES[@]}"; do

    SOURCE="$ROOT/$FILE"
    TARGET="$PACKAGE/$FILE"

    if [ ! -f "$SOURCE" ] || [ -L "$SOURCE" ]; then
        echo "ERROR: Missing or linked source: $SOURCE"
        exit 1
    fi

    if [ -L "$TARGET" ]; then
        echo "ERROR: Destination is a symbolic link: $TARGET"
        exit 1
    fi

done

echo "PASS: All approved source files exist."

echo
echo "===== 5. COMPARE PROJECT FILES ====="

CHANGED=0

for FILE in "${FILES[@]}"; do

    SOURCE="$ROOT/$FILE"
    TARGET="$PACKAGE/$FILE"

    if [ ! -e "$TARGET" ]; then

        echo "NEW:     $FILE"
        CHANGED=$((CHANGED + 1))

    elif cmp -s "$SOURCE" "$TARGET"; then

        echo "MATCH:   $FILE"

    else

        echo "CHANGED: $FILE"
        CHANGED=$((CHANGED + 1))

    fi

done

echo
echo "Files requiring synchronization: $CHANGED"

if [ "$MODE" = "status" ]; then

    echo
    echo "===== CONTEXT WORKTREE STATUS ====="

    git -C "$WT" status --short --untracked-files=all

    echo
    echo "STATUS CHECK COMPLETE"
    echo "No files copied, committed, or pushed."

    exit 0
fi

echo
echo "===== 6. PREPARE APPROVED FILES ====="

# Do not overwrite unrelated changes already present
# in the context worktree.

for FILE in "${FILES[@]}"; do

    TARGET="$PACKAGE/$FILE"
    SOURCE="$ROOT/$FILE"

    if [ -e "$TARGET" ] &&
       ! cmp -s "$SOURCE" "$TARGET"; then

        if ! git -C "$WT" diff --quiet -- \
            "thesis_force_project/$FILE"; then

            echo "ERROR: Existing uncommitted change:"
            echo "$FILE"
            echo "Review before overwriting."
            exit 1
        fi

    fi

done

# Refuse unrelated tracked or untracked changes.

mapfile -t STATUS_LINES < <(
    git -C "$WT" status --porcelain --untracked-files=all
)

for LINE in "${STATUS_LINES[@]}"; do

    PATHNAME="${LINE:3}"

    APPROVED=0

    for FILE in "${FILES[@]}"; do
        if [ "$PATHNAME" = "thesis_force_project/$FILE" ]; then
            APPROVED=1
            break
        fi
    done

    if [ "$APPROVED" -ne 1 ]; then
        echo "ERROR: Unexpected context worktree change:"
        echo "$LINE"
        exit 1
    fi

done

for FILE in "${FILES[@]}"; do

    SOURCE="$ROOT/$FILE"
    TARGET="$PACKAGE/$FILE"

    if [ -e "$TARGET" ] && cmp -s "$SOURCE" "$TARGET"; then
        continue
    fi

    mkdir -p "$(dirname "$TARGET")"

    cp -p "$SOURCE" "$TARGET"

    echo "PREPARED: $FILE"

done

echo
echo "===== 7. REVIEW PREPARED CHANGES ====="

git -C "$WT" status --short --untracked-files=all

echo
echo "Tracked diff summary:"
git -C "$WT" diff --stat

echo
echo "======================================================"
echo "CONTEXT SYNCHRONIZATION PREPARED"
echo "======================================================"

echo
echo "No Git commit was created."
echo "No branch was pushed."
echo "Review changes before publication."
