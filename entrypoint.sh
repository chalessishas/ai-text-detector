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

# v4: DeBERTa retrained on 69K balanced dataset (97.6% eval, RunPod 4090)
# Note: v1 had 72% red team; v4 has better eval but similar real-world limitations
MODEL_FILE="detector.tar.gz"
MODEL_MARKER="$DATA_DIR/models/detector/.v4_restored"
if [ ! -f "$MODEL_MARKER" ]; then
    echo "Downloading DeBERTa v4 model..."
    rm -rf "$DATA_DIR/models/detector"
    mkdir -p "$DATA_DIR/models/detector"
    curl -sL "https://github.com/$GITHUB_REPO/releases/download/$RELEASE_TAG/$MODEL_FILE" \
        | tar xz -C "$DATA_DIR/models/detector/"
    touch "$MODEL_MARKER"
    echo "DeBERTa v4 model downloaded."
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
