#!/usr/bin/env bash
# Assemble and push the Noah API to a Hugging Face Docker Space.
#
# The Space gets only what the request path needs: app/, the two trained-model
# artifacts, the FAQ cache and requirements.txt.  Training corpora, the legacy
# .joblib heads and the test suite stay out, which keeps the Space a few MB.
#
# Authenticate once, either way:
#   hf auth login                      # stores a token; nothing to pass here
#   export HF_TOKEN=hf_xxx             # or pass it in the environment
# Write tokens come from https://huggingface.co/settings/tokens
#
# The Space is created if it does not exist, so this is the only step.
#
# Usage
#   deploy/huggingface/publish.sh <hf-username> [space-name]

set -euo pipefail

HF_USER="${1:?usage: publish.sh <hf-username> [space-name]}"
SPACE="${2:-noah-payto-api}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND="$REPO_ROOT/noah_backend"
BUILD="$(mktemp -d -t noah-hf-space)"
trap 'rm -rf "$BUILD"' EXIT

echo "Assembling Space in $BUILD"
mkdir -p "$BUILD/trained_models" "$BUILD/rag"

cp -R "$BACKEND/app"                              "$BUILD/app"
cp    "$BACKEND/requirements.txt"                 "$BUILD/"
cp    "$BACKEND/trained_models/route.joblib" \
      "$BACKEND/trained_models/route_registry.json" "$BUILD/trained_models/"
cp    "$BACKEND/rag/web_faq_cache.json"           "$BUILD/rag/"
cp    "$REPO_ROOT/deploy/huggingface/Dockerfile"  "$BUILD/"
cp    "$REPO_ROOT/deploy/huggingface/README.md"   "$BUILD/"

find "$BUILD" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$BUILD" -name '*.pyc' -delete 2>/dev/null || true

# Hugging Face requires anything over 10 MB to be tracked with git-lfs.
oversized=$(find "$BUILD" -type f -size +10M)
if [ -n "$oversized" ]; then
  echo "ERROR: files over 10 MB need git-lfs before pushing:" >&2
  echo "$oversized" >&2
  exit 1
fi

echo "Space contents ($(du -sh "$BUILD" | cut -f1)):"
(cd "$BUILD" && find . -type f -not -path './.git/*' | sed 's|^\./|  |' | sort)

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1 set; assembled but not pushed. Copying to ./hf-space-preview"
  rm -rf "$REPO_ROOT/hf-space-preview"
  cp -R "$BUILD" "$REPO_ROOT/hf-space-preview"
  exit 0
fi

# Create the Space if it is not there yet. exist_ok makes this a no-op on
# every deploy after the first.
echo "Ensuring the Space exists..."
python3 - "$HF_USER/$SPACE" <<'PYEOF'
import sys
from huggingface_hub import create_repo
repo_id = sys.argv[1]
url = create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
print(f"  Space ready: {url}")
PYEOF

REMOTE="https://huggingface.co/spaces/$HF_USER/$SPACE"
if [ -n "${HF_TOKEN:-}" ]; then
  REMOTE="https://$HF_USER:$HF_TOKEN@huggingface.co/spaces/$HF_USER/$SPACE"
fi

cd "$BUILD"
git init -q -b main
git add -A
git -c user.email="noah@payto.local" -c user.name="Noah deploy" \
    commit -q -m "Deploy Noah API ($(cd "$REPO_ROOT" && git rev-parse --short HEAD))"
git remote add origin "$REMOTE"

echo "Pushing to https://huggingface.co/spaces/$HF_USER/$SPACE"
git push -f origin main

echo
echo "Done. The Space builds automatically; watch the log at:"
echo "  https://huggingface.co/spaces/$HF_USER/$SPACE"
echo "Once it is running:"
echo "  curl -s https://$HF_USER-$SPACE.hf.space/health"
