#!/bin/bash
set -e

DATA_DIR="${DATA_DIR:-/data}"
GATEWAY_PORT="${PORT:-8080}"
GITHUB_REPO="chalessishas/ai-text-detector"
RELEASE_TAG="data-v1"

echo "=== AI Text X-Ray Backend ==="
echo "Data dir: $DATA_DIR"
echo "Gateway port: $GATEWAY_PORT"

# ── Auto-download model data if missing ──

# DeBERTa model: prefer v5 (adversarial-trained), fall back to v4
# v5: adversarial retrain on 69K clean + 17K adversarial (RunPod 4090)
# v4: baseline retrain on 69K balanced dataset (97.6% eval, RunPod 4090)
V5_MARKER="$DATA_DIR/models/detector_v5/.v5_restored"
V4_MARKER="$DATA_DIR/models/detector/.v4_restored"

if [ -f "$V5_MARKER" ]; then
    echo "DeBERTa v5 model already present."
    CLASSIFIER_DIR="$DATA_DIR/models/detector_v5"
elif [ -f "$V4_MARKER" ]; then
    echo "DeBERTa v4 model already present (v5 not available)."
    CLASSIFIER_DIR="$DATA_DIR/models/detector"
else
    echo "Downloading DeBERTa v4 model..."
    rm -rf "$DATA_DIR/models/detector"
    mkdir -p "$DATA_DIR/models/detector"
    MODEL_FILE="detector.tar.gz"
    curl -sL "https://github.com/$GITHUB_REPO/releases/download/$RELEASE_TAG/$MODEL_FILE" \
        | tar xz -C "$DATA_DIR/models/detector/"
    touch "$DATA_DIR/models/detector/.v4_restored"
    echo "DeBERTa v4 model downloaded."
    CLASSIFIER_DIR="$DATA_DIR/models/detector"
fi

# FAISS corpus (future: add to GitHub Release or download from cloud storage)
if [ ! -f "$DATA_DIR/corpus/sentences.faiss" ]; then
    echo "FAISS corpus not found. Humanizer will NOT start."
    echo "To enable: upload sentences.faiss + sentences.jsonl to $DATA_DIR/corpus/"
    SKIP_HUMANIZER=1
fi

# ── Start services ──

echo "Starting detect server..."
HOST=0.0.0.0 PORT=5001 CLASSIFIER_PATH="$CLASSIFIER_DIR" \
    python scripts/perplexity.py &

if [ -z "$SKIP_HUMANIZER" ]; then
    echo "Starting humanizer server..."
    HOST=0.0.0.0 \
        python scripts/humanizer.py --port 5002 --corpus-dir "$DATA_DIR/corpus" &
fi

sleep 3

echo "Starting gateway on port $GATEWAY_PORT..."
PORT=$GATEWAY_PORT python scripts/gateway.py
