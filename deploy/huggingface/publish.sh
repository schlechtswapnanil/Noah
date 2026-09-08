#!/usr/bin/env bash
# Assemble and push the Noah API to a Hugging Face Docker Space.
#
# The Space gets only what the request path needs: app/, the two trained-model
# artifacts, the FAQ cache and requirements.txt.  Training corpora, the legacy
# .joblib heads and the test suite stay out, which keeps the Space a few MB.
#
# Prerequisites
#   1. Create the Space first (it must exist before you can push to it):
#        https://huggingface.co/new-space  ->  SDK: Docker  ->  Blank template
#   2. Have a write token: https://huggingface.co/settings/tokens
#
# Usage
#   HF_TOKEN=hf_xxx deploy/huggingface/publish.sh <hf-username> [space-name]
#
# Without HF_TOKEN, git will prompt; use your username and paste the token as
# the password (Hugging Face does not accept account passwords over git).

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
