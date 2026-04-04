#!/bin/bash
# Download trained DeBERTa v5 model from RunPod and integrate into project
# After download, updates models/detector symlink to point to detector_v5/
# Usage: bash scripts/download_runpod_v5.sh [RUNPOD_IP] [RUNPOD_PORT]

set -e

RUNPOD_IP="${1:-REPLACE_WITH_RUNPOD_IP}"
RUNPOD_PORT="${2:-22}"
SSH_KEY="$HOME/.runpod/ssh/RunPod-Key-Go"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="$PROJECT_DIR/models/detector_v5"
REMOTE_DIR="/workspace/detector_v5"

echo "=== Downloading DeBERTa v5 model from RunPod ==="

# Check training is complete
echo "Checking training status..."
STATUS=$(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no root@"$RUNPOD_IP" -p "$RUNPOD_PORT" \
  "tail -5 /workspace/training.log" 2>&1)

if echo "$STATUS" | grep -q "DONE\|Model saved\|Model at"; then
    echo "Training complete! Downloading model..."
else
    echo "Training not yet complete. Last output:"
    echo "$STATUS" | tail -3
    exit 1
fi

mkdir -p "$TARGET_DIR"

echo "Downloading model files..."
for file in model.safetensors config.json special_tokens_map.json spm.model; do
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$RUNPOD_PORT" \
        "root@${RUNPOD_IP}:${REMOTE_DIR}/${file}" \
        "$TARGET_DIR/" 2>&1
done

scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$RUNPOD_PORT" \
    "root@${RUNPOD_IP}:${REMOTE_DIR}/tokenizer*" \
    "$TARGET_DIR/" 2>&1

scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$RUNPOD_PORT" \
    "root@${RUNPOD_IP}:/workspace/training.log" \
    "$TARGET_DIR/training.log" 2>&1

# Update symlink: models/detector -> detector_v5
echo "Updating models/detector symlink..."
cd "$PROJECT_DIR/models"
rm -f detector
ln -s detector_v5 detector
echo "Symlink updated: detector -> detector_v5"

echo ""
echo "Model downloaded to: $TARGET_DIR"
ls -lh "$TARGET_DIR"
echo ""
echo "The detector symlink now points to v5."
echo "perplexity.py will auto-detect v5 without any config change."
echo "To revert: cd models && rm detector && ln -s detector_v4 detector"
