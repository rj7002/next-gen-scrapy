#!/usr/bin/env bash
# Assemble the NGS Explorer Space (the package + the web service, nothing else) and push it to Hugging Face.
#   usage: deploy/huggingface/push.sh <hf-username>/<space-name> [--public | --existing]
# Needs `hf auth login` first. The Space is created private unless --public is given (new Docker Spaces on the free
# CPU tier need a PRO account; --existing reuses a Space you already have, replacing everything in it - its old
# files stay in the Space's git history).
set -euo pipefail
SPACE="${1:?usage: push.sh <user>/<space> [--public]}"
VIS="--private"; [[ "${2:-}" == "--public" ]] && VIS="--public"
EXISTING=0; [[ "${2:-}" == "--existing" ]] && EXISTING=1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$ROOT/deploy/huggingface"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp "$HERE"/{Dockerfile,README.md,requirements.txt,.dockerignore} "$STAGE"/
cp "$ROOT"/pyproject.toml "$ROOT"/LICENSE "$STAGE"/
rsync -a --exclude '__pycache__' "$ROOT/next_gen_scrapy" "$STAGE"/
# the web service, minus local-only state: caches (incl. Big Data Bowl tracking, which can't be redistributed),
# the old database, node_modules and builds, secrets
rsync -a --exclude '__pycache__' --exclude '.cache' --exclude 'matched' --exclude 'ngs.db' \
      --exclude 'web/node_modules' --exclude 'web/dist' --exclude '.env*' "$ROOT/service" "$STAGE"/

echo "staged $(du -sh "$STAGE" | cut -f1) in $STAGE"
if [[ $EXISTING == 1 ]]; then
  hf upload "$SPACE" "$STAGE" . --repo-type space --delete "*" --commit-message "Replace with NGS Explorer"
else
  hf repos create "$SPACE" --repo-type space --space-sdk docker $VIS --exist-ok
  hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy NGS Explorer"
fi
echo "pushed: https://huggingface.co/spaces/$SPACE (building takes a few minutes)"
