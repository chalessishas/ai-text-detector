#!/bin/bash
# RunPod one-click setup for DeBERTa v5 training
# Usage: bash setup_runpod.sh
set -e

echo "=== DeBERTa v5 RunPod Setup ==="

# Install deps
pip install -q transformers datasets accelerate scikit-learn

# Check data files
for f in dataset_v4.jsonl dataset_adversarial_v4.jsonl train_runpod_v5.py; do
    if [ ! -f "/workspace/$f" ]; then
        echo "ERROR: /workspace/$f not found. Upload it first."
        exit 1
    fi
done

echo "All files present. Starting training..."
cd /workspace
python3 train_runpod_v5.py

echo ""
echo "=== Packaging model ==="
tar -czf detector_v5.tar.gz -C /workspace detector_v5/
ls -lh /workspace/detector_v5.tar.gz
echo "Done! Download detector_v5.tar.gz from the RunPod file browser."
