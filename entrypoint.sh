#!/bin/bash
set -e

DATA_DIR="${DATA_DIR:-/data}"
GATEWAY_PORT="${PORT:-8080}"

echo "=== AI Text X-Ray Backend ==="
echo "Data dir: $DATA_DIR"
echo "Gateway port: $GATEWAY_PORT"

# Check data
if [ ! -f "$DATA_DIR/models/detector/model.safetensors" ]; then
    echo "WARNING: DeBERTa model not found. Detection runs without classifier."
fi

if [ ! -f "$DATA_DIR/corpus/sentences.faiss" ]; then
    echo "WARNING: FAISS index not found. Humanizer will NOT start."
    SKIP_HUMANIZER=1
fi

# Start detect server on internal port 5001
echo "Starting detect server..."
HOST=0.0.0.0 PORT=5001 CLASSIFIER_PATH="$DATA_DIR/models/detector" \
    python scripts/perplexity.py &

# Start humanize server on internal port 5002
if [ -z "$SKIP_HUMANIZER" ]; then
    echo "Starting humanizer server..."
    HOST=0.0.0.0 \
        python scripts/humanizer.py --port 5002 --corpus-dir "$DATA_DIR/corpus" &
fi

# Wait for backends to start
sleep 3

# Start gateway on Railway's PORT (the only externally exposed port)
echo "Starting gateway on port $GATEWAY_PORT..."
PORT=$GATEWAY_PORT python scripts/gateway.py
