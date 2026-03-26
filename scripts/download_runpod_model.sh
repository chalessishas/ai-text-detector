#!/bin/bash
# Download trained DeBERTa model from RunPod and integrate into project
# Usage: bash scripts/download_runpod_model.sh

set -e

RUNPOD_IP="103.196.86.168"
RUNPOD_PORT="52279"
SSH_KEY="$HOME/.runpod/ssh/RunPod-Key-Go"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="$PROJECT_DIR/models/detector_v4"

echo "=== Downloading trained model from RunPod ==="

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

# Create target directory
mkdir -p "$TARGET_DIR"

# Download model files
echo "Downloading model files..."
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$RUNPOD_PORT" \
    "root@${RUNPOD_IP}:/workspace/detector_v4/model.safetensors" \
    "$TARGET_DIR/" 2>&1

scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$RUNPOD_PORT" \
    "root@${RUNPOD_IP}:/workspace/detector_v4/config.json" \
    "$TARGET_DIR/" 2>&1

scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$RUNPOD_PORT" \
    "root@${RUNPOD_IP}:/workspace/detector_v4/tokenizer*" \
    "$TARGET_DIR/" 2>&1

scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$RUNPOD_PORT" \
    "root@${RUNPOD_IP}:/workspace/detector_v4/special_tokens_map.json" \
    "$TARGET_DIR/" 2>&1

scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$RUNPOD_PORT" \
    "root@${RUNPOD_IP}:/workspace/detector_v4/spm.model" \
    "$TARGET_DIR/" 2>&1

# Download training log
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$RUNPOD_PORT" \
    "root@${RUNPOD_IP}:/workspace/training.log" \
    "$TARGET_DIR/training.log" 2>&1

echo ""
echo "Model downloaded to: $TARGET_DIR"
ls -lh "$TARGET_DIR"
echo ""
echo "To use this model, set: CLASSIFIER_PATH=$TARGET_DIR"
echo "Or replace models/detector/ with these files."
