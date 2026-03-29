#!/usr/bin/env bash
# version.sh — Single command to bump, tag, and release turboquant-torch.
#
# Usage:
#   ./scripts/version.sh              # show current version
#   ./scripts/version.sh patch        # 0.2.1 → 0.2.2
#   ./scripts/version.sh minor        # 0.2.1 → 0.3.0
#   ./scripts/version.sh major        # 0.2.1 → 1.0.0
#   ./scripts/version.sh 1.2.3        # set exact version
#   ./scripts/version.sh patch --dry  # preview without changing anything
#
# What it does:
#   1. Computes next version from latest git tag
#   2. Updates turboquant/__init__.py __version__
#   3. Commits the change
#   4. Creates git tag v{VERSION}
#   5. Prints push command (does NOT auto-push)
#
# Follows CLAUDE.md branching strategy:
#   - Run on staging for pre-release validation
#   - After CI green: merge staging→main, push tag from main

set -euo pipefail

INIT_FILE="turboquant/__init__.py"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── Parse args ────────────────────────────────────────────────────────────
BUMP="${1:-}"
DRY=false
[[ "${2:-}" == "--dry" ]] && DRY=true

# ── Get current version from latest git tag ───────────────────────────────
LATEST_TAG=$(git tag -l 'v*.*.*' | sort -V | tail -1)
if [[ -z "$LATEST_TAG" ]]; then
    CURRENT="0.0.0"
else
    CURRENT="${LATEST_TAG#v}"
fi

IFS='.' read -r CUR_MAJOR CUR_MINOR CUR_PATCH <<< "$CURRENT"

# ── Show current and exit if no arg ───────────────────────────────────────
if [[ -z "$BUMP" ]]; then
    INIT_VER=$(grep -oP '__version__\s*=\s*"\K[^"]+' "$INIT_FILE" 2>/dev/null || echo "?")
    echo "Current version"
    echo "  git tag:      v${CURRENT}"
    echo "  __init__.py:  ${INIT_VER}"
    echo "  branch:       $(git branch --show-current)"
    echo ""
    echo "Usage: $0 {patch|minor|major|X.Y.Z} [--dry]"
    exit 0
fi

# ── Compute next version ──────────────────────────────────────────────────
case "$BUMP" in
    patch)
        NEXT="${CUR_MAJOR}.${CUR_MINOR}.$((CUR_PATCH + 1))"
        ;;
    minor)
        NEXT="${CUR_MAJOR}.$((CUR_MINOR + 1)).0"
        ;;
    major)
        NEXT="$((CUR_MAJOR + 1)).0.0"
        ;;
    [0-9]*.*)
        NEXT="$BUMP"
        ;;
    *)
        echo "Error: unknown bump type '$BUMP'"
        echo "Usage: $0 {patch|minor|major|X.Y.Z} [--dry]"
        exit 1
        ;;
esac

# Validate format
if ! [[ "$NEXT" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: invalid version format '$NEXT' (expected X.Y.Z)"
    exit 1
fi

# Check tag doesn't already exist
if git tag -l "v${NEXT}" | grep -q .; then
    echo "Error: tag v${NEXT} already exists"
    exit 1
fi

# ── Preview ───────────────────────────────────────────────────────────────
BRANCH=$(git branch --show-current)
echo "Version bump: v${CURRENT} → v${NEXT}"
echo "  Branch:      ${BRANCH}"
echo "  __init__.py: __version__ = \"${NEXT}\""
echo "  Git tag:     v${NEXT}"

if $DRY; then
    echo ""
    echo "(dry run — no changes made)"
    exit 0
fi

# ── Guard: uncommitted changes (except __init__.py which we'll modify) ────
DIRTY=$(git diff --name-only HEAD 2>/dev/null | grep -v "$INIT_FILE" || true)
if [[ -n "$DIRTY" ]]; then
    echo ""
    echo "Warning: uncommitted changes detected:"
    echo "$DIRTY" | sed 's/^/  /'
    read -rp "Continue anyway? [y/N] " REPLY
    [[ "$REPLY" =~ ^[Yy]$ ]] || exit 1
fi

# ── Apply ─────────────────────────────────────────────────────────────────
# 1. Update __init__.py
sed -i "s/__version__ = \".*\"/__version__ = \"${NEXT}\"/" "$INIT_FILE"
echo ""
echo "✓ Updated $INIT_FILE"

# 2. Commit
git add "$INIT_FILE"
git commit -m "chore: bump version to v${NEXT}"
echo "✓ Committed"

# 3. Tag
git tag "v${NEXT}"
echo "✓ Tagged v${NEXT}"

# 4. Print next steps
echo ""
echo "Done! Next steps:"
if [[ "$BRANCH" == "staging" ]]; then
    echo "  1. git push origin staging"
    echo "  2. Wait for CI green"
    echo "  3. git checkout main && git merge staging"
    echo "  4. git push origin main --tags"
elif [[ "$BRANCH" == "main" ]]; then
    echo "  git push origin main --tags"
else
    echo "  git push origin ${BRANCH} --tags"
fi
