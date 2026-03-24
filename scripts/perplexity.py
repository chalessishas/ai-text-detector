#!/usr/bin/env python3
"""Persistent HTTP server for token-level features + DeBERTa classifier.

Perplexity backend priority:
  1. MLX + qwen3.5:4b (Apple Silicon, best signal separation)
  2. llama-cpp + llama3.2:1b (fallback)

When models/detector/ exists, also returns 4-class classification.
"""

import json
import math
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np

LABEL_NAMES = ["human", "ai", "ai_polished", "human_polished"]

# ── DeBERTa classifier (optional) ──────────────────────────────────────────

def load_classifier():
    """Load DeBERTa classifier from models/detector/ if available."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.environ.get(
        "CLASSIFIER_PATH",
        os.path.join(script_dir, "..", "models", "detector"),
    )

    if not os.path.isdir(model_dir):
        print(f"Classifier not found at {model_dir} -- running without it.", file=sys.stderr)
        return None, None

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.float()
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        print(f"Classifier loaded from {model_dir}", file=sys.stderr)
        return tokenizer, model
    except Exception as e:
        print(f"Failed to load classifier: {e}", file=sys.stderr)
        return None, None


def classify_text(tokenizer, model, text):
    """Run DeBERTa inference, return binary human/AI prediction.

    Model is 4-class internally but we collapse to binary:
      Human side = P(human) + P(human_polished)
      AI side    = P(ai) + P(ai_polished)
    """
    import torch

    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    ai_score = float(probs[1] + probs[2]) * 100  # P(ai) + P(ai_polished)
    human_score = float(probs[0] + probs[3]) * 100  # P(human) + P(human_polished)
    prediction = "ai" if ai_score > 50 else "human"

    return {
        "prediction": prediction,
        "ai_score": round(ai_score, 1),
        "human_score": round(human_score, 1),
        "confidence": round(max(ai_score, human_score), 1),
        # Keep 4-class detail available for future use
        "_4class": {name: round(float(probs[i]), 4) for i, name in enumerate(LABEL_NAMES)},
    }


# ── Perplexity model (MLX preferred, llama-cpp fallback) ──────────────────

MLX_MODEL_ID = "mlx-community/Qwen3.5-4B-4bit"

def load_model():
    """Try MLX qwen3.5:4b first (3x better signal), fall back to llama-cpp."""
    # Try MLX (Apple Silicon, needs mlx_lm)
    try:
        import mlx_lm
        import mlx.core as mx
        print(f"Loading MLX model {MLX_MODEL_ID}...", file=sys.stderr)
        model, tokenizer = mlx_lm.load(MLX_MODEL_ID)
        test_tokens = tokenizer.encode("test")
        _ = model(mx.array([test_tokens]))
        print(f"Perplexity model: MLX {MLX_MODEL_ID}", file=sys.stderr)
        return ("mlx", model, tokenizer)
    except Exception as e:
        print(f"MLX not available ({e}), trying llama-cpp...", file=sys.stderr)

    # Fallback: llama-cpp llama3.2:1b
    model_path = os.environ.get(
        "MODEL_PATH",
        os.path.expanduser(
            "~/.ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
        ),
    )
    if not os.path.exists(model_path):
        print(f"No perplexity model found. Token analysis disabled.", file=sys.stderr)
        return None
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4, logits_all=True, verbose=False)
        print(f"Perplexity model: llama-cpp llama3.2:1b", file=sys.stderr)
        return ("llama", llm)
    except Exception as e:
        print(f"Failed to load perplexity model: {e}", file=sys.stderr)
        return None


def compute_features(model_tuple, text):
    """Dispatch to MLX or llama-cpp backend."""
    if model_tuple is None:
        return {"tokens": []}
    backend = model_tuple[0]
    if backend == "mlx":
        return _compute_mlx(model_tuple[1], model_tuple[2], text)
    return _compute_llama(model_tuple[1], text)


def _compute_mlx(model, tokenizer, text):
    """Token-level features via MLX (qwen3.5:4b)."""
    import mlx.core as mx

    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return {"error": "Text too short for analysis"}
    if len(tokens) > 2048:
        tokens = tokens[:2048]

    x = mx.array([tokens])
    logits = model(x)
    # mx → numpy conversion: force float32 to avoid PEP 3118 buffer mismatch
    probs_mx = mx.softmax(logits[0], axis=-1)
    mx.eval(probs_mx)
    probs_all = np.array(probs_mx.astype(mx.float32))

    results = []
    for i in range(1, len(tokens)):
        actual_id = tokens[i]
        actual_prob = float(probs_all[i - 1, actual_id])
        logprob = math.log(max(actual_prob, 1e-20))
        rank = int(np.sum(probs_all[i - 1] > actual_prob)) + 1

        p = probs_all[i - 1]
        valid = p > 1e-10
        entropy = float(-np.sum(p[valid] * np.log(p[valid])))

        token_str = tokenizer.decode([actual_id])
        results.append({"token": token_str, "logprob": logprob, "rank": rank, "entropy": entropy})

    return {"tokens": results}


def _compute_llama(model, text):
    """Token-level features via llama-cpp (llama3.2:1b fallback)."""
    token_ids = model.tokenize(text.encode("utf-8"), add_bos=True)
    n_tokens = len(token_ids)

    if n_tokens < 2:
        return {"error": "Text too short for analysis"}
    if n_tokens > 2048:
        token_ids = token_ids[:2048]
        n_tokens = 2048

    stderr_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        model.reset()
        model.eval(token_ids)
    finally:
        os.dup2(stderr_fd, 2)
        os.close(devnull)
        os.close(stderr_fd)

    results = []
    for i in range(1, n_tokens):
        logits = np.array(model.scores[i - 1], dtype=np.float64)
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)

        actual_id = token_ids[i]
        actual_prob = float(probs[actual_id])
        logprob = float(np.log(max(actual_prob, 1e-20)))
        rank = int(np.sum(probs > actual_prob)) + 1

        valid = probs > 1e-10
        entropy = float(-np.sum(probs[valid] * np.log(probs[valid])))

        token_bytes = model.detokenize([actual_id])
        token_str = token_bytes.decode("utf-8", errors="replace")
        results.append({"token": token_str, "logprob": logprob, "rank": rank, "entropy": entropy})

    return {"tokens": results}


def compute_diveye_features(logprobs):
    """Compute DivEye surprisal diversity features (IBM, TMLR 2026).

    10 features from token-level surprisal that capture how "bumpy" the
    predictability pattern is. Human text has more diversity in surprisal
    than AI text, which tends to be smoother.
    """
    surprisal = np.array([-lp for lp in logprobs])
    if len(surprisal) < 5:
        return {}

    from scipy.stats import skew, kurtosis

    # Distributional features
    s_mean = float(np.mean(surprisal))
    s_std = float(np.std(surprisal))
    s_var = float(np.var(surprisal))
    s_skew = float(skew(surprisal))
    s_kurt = float(kurtosis(surprisal))

    # First-order differences (volatility)
    diff1 = np.diff(surprisal)
    d1_mean = float(np.mean(diff1)) if len(diff1) > 0 else 0.0
    d1_std = float(np.std(diff1)) if len(diff1) > 0 else 0.0

    # Second-order differences (acceleration of surprisal changes)
    diff2 = np.diff(diff1) if len(diff1) > 1 else np.array([0.0])
    d2_var = float(np.var(diff2)) if len(diff2) > 0 else 0.0

    # Entropy of second-order distribution
    if len(diff2) > 5:
        hist, _ = np.histogram(diff2, bins=20, density=True)
        hist = hist[hist > 0]
        d2_entropy = float(-np.sum(hist * np.log(hist + 1e-10)))
    else:
        d2_entropy = 0.0

    # Autocorrelation of second-order
    if len(diff2) > 2:
        d2_autocorr = float(np.corrcoef(diff2[:-1], diff2[1:])[0, 1])
        if np.isnan(d2_autocorr):
            d2_autocorr = 0.0
    else:
        d2_autocorr = 0.0

    return {
        "surprisal_mean": round(s_mean, 3),
        "surprisal_std": round(s_std, 3),
        "surprisal_var": round(s_var, 3),
        "surprisal_skew": round(s_skew, 3),
        "surprisal_kurtosis": round(s_kurt, 3),
        "diff1_mean": round(d1_mean, 3),
        "diff1_std": round(d1_std, 3),
        "diff2_var": round(d2_var, 3),
        "diff2_entropy": round(d2_entropy, 3),
        "diff2_autocorr": round(d2_autocorr, 3),
    }


def compute_specdetect_energy(logprobs):
    """Compute SpecDetect DFT total energy (AAAI 2026 Oral).

    Human text has higher spectral energy in its surprisal sequence
    than AI text. A single DFT feature achieves SOTA performance.
    """
    if len(logprobs) < 10:
        return 0.0
    surprisal = np.array([-lp for lp in logprobs])
    # Remove DC component (mean) to focus on variation
    surprisal = surprisal - np.mean(surprisal)
    # DFT and compute total energy (Parseval's theorem)
    fft = np.fft.rfft(surprisal)
    energy = float(np.sum(np.abs(fft) ** 2) / len(surprisal))
    return round(energy, 3)


def compute_perplexity_score(tokens):
    """Compute aggregate stats from token data, run LR classifier for AI probability."""
    if not tokens:
        return None
    logprobs = [t["logprob"] for t in tokens]
    ranks = [t["rank"] for t in tokens]
    entropies = [t["entropy"] for t in tokens]

    ppl = math.exp(-sum(logprobs) / len(logprobs))
    log_ppl = math.log(max(ppl, 1e-5))
    top10 = sum(1 for r in ranks if r <= 10) / len(ranks) * 100
    top1 = sum(1 for r in ranks if r == 1) / len(ranks) * 100
    mean_ent = sum(entropies) / len(entropies)
    ent_std = float(np.std(entropies))

    # DivEye surprisal diversity features
    diveye = compute_diveye_features(logprobs)

    # SpecDetect DFT total energy
    spec_energy = compute_specdetect_energy(logprobs)

    result = {
        "perplexity": round(ppl, 2),
        "top10_pct": round(top10, 1),
        "top1_pct": round(top1, 1),
        "mean_entropy": round(mean_ent, 2),
        "entropy_std": round(ent_std, 2),
        "diveye": diveye,
        "specdetect_energy": spec_energy,
    }

    # LR classifier trained on qwen3.5:4b features (models/perplexity_lr.pkl)
    lr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "perplexity_lr.pkl")
    if os.path.exists(lr_path):
        try:
            import pickle
            with open(lr_path, "rb") as f:
                lr_data = pickle.load(f)
            lr_model = lr_data["model"]
            features = np.array([[log_ppl, top10, mean_ent, top1, ent_std]])
            prob = lr_model.predict_proba(features)[0]
            result["lr_ai_probability"] = round(float(prob[1]) * 100, 1)
            result["lr_prediction"] = "ai" if prob[1] > 0.5 else "human"
        except Exception:
            pass

    return result


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        text = body.get("text", "").strip()

        if not text:
            result = {"error": "No text provided"}
        else:
            try:
                if self.server.model_tuple:
                    result = compute_features(self.server.model_tuple, text)
                else:
                    result = {"tokens": []}
                # Aggregate perplexity stats + LR AI probability
                result["perplexity_stats"] = compute_perplexity_score(result.get("tokens", []))
                # DeBERTa binary classification
                clf_tok = self.server.classifier_tokenizer
                clf_mod = self.server.classifier_model
                if clf_tok and clf_mod:
                    result["classification"] = classify_text(clf_tok, clf_mod, text)
                # Fused score: DeBERTa + Perplexity LR + PPL rule boost
                deb_ai = result.get("classification", {}).get("ai_score", 50)
                lr_ai = (result.get("perplexity_stats") or {}).get("lr_ai_probability", 50)
                ppl_val = (result.get("perplexity_stats") or {}).get("perplexity", 20)

                # Adaptive weighting: when PPL is very low (strong AI signal)
                # but DeBERTa misses (model memorization blind spot), boost PPL weight.
                # Normal: 60% DeBERTa + 40% LR
                # Low PPL override: if PPL < 8 and DeBERTa < 20%, use 30/70 split
                if ppl_val < 8 and deb_ai < 20:
                    fused = deb_ai * 0.3 + lr_ai * 0.7
                else:
                    fused = deb_ai * 0.6 + lr_ai * 0.4

                # Length gating: short texts (<150 words) have higher false positive
                # rates across all detectors. Raise threshold for short texts.
                word_count = len(text.split())
                if word_count < 150:
                    threshold = 65  # Require higher confidence for short texts
                elif word_count < 300:
                    threshold = 55
                else:
                    threshold = 50

                result["fused"] = {
                    "ai_score": round(fused, 1),
                    "prediction": "ai" if fused > threshold else "human",
                    "confidence": round(max(fused, 100 - fused), 1),
                    "word_count": word_count,
                    "threshold": threshold,
                }
            except Exception as e:
                result = {"error": str(e)}

        response = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        pass


class PerplexityServer(HTTPServer):
    allow_reuse_address = True

    def __init__(self, addr, handler, model_tuple, classifier_tokenizer=None, classifier_model=None):
        super().__init__(addr, handler)
        self.model_tuple = model_tuple
        self.classifier_tokenizer = classifier_tokenizer
        self.classifier_model = classifier_model


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("Loading perplexity model...", file=sys.stderr)
    model_tuple = load_model()
    print("Loading classifier...", file=sys.stderr)
    clf_tokenizer, clf_model = load_classifier()
    print(f"Server running at http://127.0.0.1:{port}", file=sys.stderr)
    server = PerplexityServer(("127.0.0.1", port), Handler, model_tuple, clf_tokenizer, clf_model)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", file=sys.stderr)
