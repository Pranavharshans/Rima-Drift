#!/usr/bin/env bash
# Deploy the new-voices test app (space_voices/) to a PRIVATE Hugging Face Space, with its files:
#   scripts/deploy_voices_space.sh <space id> <ARDrift checkpoint> <MelBWE checkpoint> <voices.json>
# The checkpoints go into the Space repo itself; the script refuses a Space that is not private.
set -euo pipefail
cd "$(dirname "$0")/.."
SPACE=$1 MODEL=$2 BWE=$3 VOICES=$4
python -c "import sys; from huggingface_hub import HfApi; sys.exit(0 if HfApi().space_info('$SPACE').private else
'$SPACE is not private: refusing to deploy')"
STAGE=$(mktemp -d)
cp space_voices/app.py space_voices/requirements.txt space_voices/README.md "$STAGE/"
cp -r drifting_tts "$STAGE/"
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
cp "$MODEL" "$STAGE/ardrift_voices.pt"
cp "$BWE" "$STAGE/bwe.pt"
cp "$VOICES" "$STAGE/voices.json"
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy from github.com/kadirnar/drifting-tts@$(git rev-parse --short HEAD)"
rm -rf "$STAGE"
