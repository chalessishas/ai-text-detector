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

# v3: Full retrain on 83K balanced data (97.7% eval acc)
# Trained from DeBERTa-v3-base on L4 GPU, includes cross-domain samples
MODEL_FILE="detector_v3.tar.gz"
MODEL_MARKER="$DATA_DIR/models/detector/.v3"
if [ ! -f "$MODEL_MARKER" ]; then
    echo "Downloading DeBERTa v3 model..."
    rm -rf "$DATA_DIR/models/detector"
    mkdir -p "$DATA_DIR/models/detector"
    curl -sL "https://github.com/$GITHUB_REPO/releases/download/$RELEASE_TAG/$MODEL_FILE" \
        | tar xz -C "$DATA_DIR/models/detector/"
    touch "$MODEL_MARKER"
    echo "DeBERTa v3 model downloaded."
fi

# FAISS corpus (future: add to GitHub Release or download from cloud storage)
if [ ! -f "$DATA_DIR/corpus/sentences.faiss" ]; then
    echo "FAISS corpus not found. Humanizer will NOT start."
    echo "To enable: upload sentences.faiss + sentences.jsonl to $DATA_DIR/corpus/"
    SKIP_HUMANIZER=1
fi

# ── Start services ──

echo "Starting detect server..."
HOST=0.0.0.0 PORT=5001 CLASSIFIER_PATH="$DATA_DIR/models/detector" \
    python scripts/perplexity.py &

if [ -z "$SKIP_HUMANIZER" ]; then
    echo "Starting humanizer server..."
    HOST=0.0.0.0 \
        python scripts/humanizer.py --port 5002 --corpus-dir "$DATA_DIR/corpus" &
fi

sleep 3

echo "Starting gateway on port $GATEWAY_PORT..."
PORT=$GATEWAY_PORT python scripts/gateway.py
