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
import re
import sys
from collections import Counter
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np

LABEL_NAMES = ["human", "ai", "ai_polished", "human_polished"]

# AI Vocabulary overuse detection (GPTZero-style)
AI_VOCAB = {
    "furthermore", "moreover", "additionally", "consequently",
    "nevertheless", "nonetheless", "notably", "essentially",
    "fundamentally", "inherently", "ultimately", "crucial",
    "significant", "comprehensive", "robust", "innovative",
    "diverse", "dynamic", "transformative", "unprecedented",
    "multifaceted", "nuanced", "pivotal", "leverage",
    "facilitate", "enhance", "foster", "underscore",
    "navigate", "streamline", "optimize", "delve",
    "encompass", "utilize", "harness", "bolster",
    "mitigate", "exacerbate",
}

def compute_ai_vocab(text):
    """Count AI-overused vocabulary density."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 20:
        return 0.0, []
    hits = [(w, c) for w in AI_VOCAB if (c := Counter(words).get(w, 0)) > 0]
    total = sum(c for _, c in hits)
    density = (total / len(words)) * 100
    return round(density, 2), sorted(hits, key=lambda x: -x[1])[:5]

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


TEMPERATURE = float(os.environ.get("DEBERTA_TEMPERATURE", "2.0"))

def classify_text(tokenizer, model, text):
    """Run DeBERTa inference with temperature scaling + logit gap confidence.

    Temperature scaling (Guo et al. 2017) softens overconfident outputs.
    Logit gap measures true model uncertainty (more reliable than softmax).
    """
    import torch

    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    # Temperature scaling: soften extreme 0/100 outputs
    scaled_logits = logits / TEMPERATURE
    probs = torch.softmax(scaled_logits, dim=-1).cpu().numpy()[0]
    raw_logits = logits.cpu().numpy()[0]

    ai_score = float(probs[1] + probs[2]) * 100
    human_score = float(probs[0] + probs[3]) * 100

    # Logit gap: raw measure of model certainty (immune to softmax saturation)
    ai_logit = float(raw_logits[1] + raw_logits[2])
    human_logit = float(raw_logits[0] + raw_logits[3])
    logit_gap = abs(ai_logit - human_logit)

    # Uncertain when logit gap is small (model genuinely doesn't know)
    is_uncertain = logit_gap < 2.0
    if is_uncertain:
        prediction = "uncertain"
    else:
        prediction = "ai" if ai_score > 50 else "human"

    return {
        "prediction": prediction,
        "ai_score": round(ai_score, 1),
        "human_score": round(human_score, 1),
        "confidence": round(max(ai_score, human_score), 1),
        "logit_gap": round(logit_gap, 2),
        "is_uncertain": is_uncertain,
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

    # Fallback: llama-cpp — try qwen3:4b first (better signal), then llama3.2:1b
    qwen3_path = os.path.expanduser(
        "~/.ollama/models/blobs/sha256-3e4cb14174460404e7a233e531675303b2fbf7749c02f91864fe311ab6344e4f"
    )
    llama_path = os.path.expanduser(
        "~/.ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
    )
    model_path = os.environ.get("MODEL_PATH", "")
    if not model_path:
        if os.path.exists(qwen3_path):
            model_path = qwen3_path
            model_name = "qwen3:4b"
        elif os.path.exists(llama_path):
            model_path = llama_path
            model_name = "llama3.2:1b"
        else:
            print("No perplexity model found. Token analysis disabled.", file=sys.stderr)
            return None
    else:
        model_name = "custom"
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4, logits_all=True, verbose=False)
        print(f"Perplexity model: llama-cpp {model_name}", file=sys.stderr)
        return ("llama", llm)
    except Exception as e:
        print(f"Failed to load perplexity model: {e}", file=sys.stderr)
        return None


# ── Binoculars (dual-model detection, Hans et al. ICML 2024) ─────────────

BINOCULARS_OBSERVER_PATH = os.path.expanduser(
    "~/.ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
)  # llama3.2:1b
BINOCULARS_PERFORMER_PATH = os.path.expanduser(
    "~/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff"
)  # llama3.2:3b


def load_binoculars():
    """Load observer (1b) + performer (3b) for Binoculars scoring."""
    if not os.path.exists(BINOCULARS_OBSERVER_PATH) or not os.path.exists(BINOCULARS_PERFORMER_PATH):
        print("Binoculars: missing model(s), disabled.", file=sys.stderr)
        return None
    try:
        from llama_cpp import Llama
        observer = Llama(model_path=BINOCULARS_OBSERVER_PATH, n_ctx=512, n_threads=4, logits_all=True, verbose=False)
        performer = Llama(model_path=BINOCULARS_PERFORMER_PATH, n_ctx=512, n_threads=4, logits_all=True, verbose=False)
        print("Binoculars: llama3.2:1b (observer) + llama3.2:3b (performer) loaded.", file=sys.stderr)
        return (observer, performer)
    except Exception as e:
        print(f"Binoculars load failed: {e}", file=sys.stderr)
        return None


def compute_binoculars(observer, performer, text):
    """Compute Binoculars score: log_ppl(observer) / cross_ppl(observer←performer).

    Low score (~0.5-0.8) = likely AI, high score (~0.9-1.2+) = likely human.
    Threshold from paper: 0.9015 (low-FPR mode).
    """
    tokens = observer.tokenize(text.encode())
    if len(tokens) < 10:
        return None

    # Limit to 512 tokens
    tokens = tokens[:512]

    # Get observer logits
    observer.reset()
    observer.eval(tokens)
    obs_logits = np.array(observer.scores[:len(tokens)-1])  # (seq_len-1, vocab)

    # Get performer logits
    performer.reset()
    performer.eval(tokens)
    perf_logits = np.array(performer.scores[:len(tokens)-1])

    # Ensure same shape
    min_vocab = min(obs_logits.shape[1], perf_logits.shape[1])
    obs_logits = obs_logits[:, :min_vocab]
    perf_logits = perf_logits[:, :min_vocab]

    # Observer log-perplexity: how surprised is observer by the actual next tokens
    from scipy.special import log_softmax
    obs_log_probs = log_softmax(obs_logits, axis=-1)
    actual_next_tokens = tokens[1:]
    obs_token_logprobs = []
    for i, tok in enumerate(actual_next_tokens):
        if tok < min_vocab:
            obs_token_logprobs.append(obs_log_probs[i, tok])
        else:
            obs_token_logprobs.append(-10.0)  # unknown token penalty
    log_ppl = -np.mean(obs_token_logprobs)

    # Cross-perplexity: observer evaluates performer's predictions
    # For each position, use performer's predicted distribution, score it with observer's model
    perf_log_probs = log_softmax(perf_logits, axis=-1)

    # cross_ppl = avg(-sum(performer_prob * observer_logprob)) over positions
    cross_entropy_per_pos = []
    for i in range(len(actual_next_tokens)):
        # Performer's probability distribution at position i
        perf_probs = np.exp(perf_log_probs[i])
        # Observer's log probabilities at position i
        obs_lp = obs_log_probs[i]
        # Cross entropy: -sum(p_performer * log_p_observer)
        # Only compute over top-k tokens for efficiency
        top_k = 100
        top_indices = np.argpartition(perf_probs, -top_k)[-top_k:]
        ce = -np.sum(perf_probs[top_indices] * obs_lp[top_indices])
        cross_entropy_per_pos.append(ce)

    cross_ppl = np.mean(cross_entropy_per_pos)

    # Binoculars score
    if cross_ppl < 1e-6:
        return None
    score = log_ppl / cross_ppl

    return {
        "score": round(float(score), 4),
        "log_ppl": round(float(log_ppl), 3),
        "cross_ppl": round(float(cross_ppl), 3),
        "ai_probability": round(float(max(0, min(100, (1.0 - score) * 100))), 1),
    }


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


def compute_min_window_ppl(logprobs, window=32):
    """Sliding window minimum perplexity.

    AI text has consistently low PPL across all windows.
    Human text may have low-PPL segments but also high-PPL segments.
    min-window-PPL captures the "most AI-like" segment of the text.
    """
    if len(logprobs) < window:
        return math.exp(-sum(logprobs) / len(logprobs)) if logprobs else 999
    min_ppl = float('inf')
    for i in range(len(logprobs) - window + 1):
        w = logprobs[i:i + window]
        ppl = math.exp(-sum(w) / len(w))
        if ppl < min_ppl:
            min_ppl = ppl
    return round(min_ppl, 2)


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

    # Sliding window min-PPL (captures most AI-like segment)
    min_window_ppl = compute_min_window_ppl(logprobs, window=32)

    result = {
        "perplexity": round(ppl, 2),
        "min_window_ppl": min_window_ppl,
        "top10_pct": round(top10, 1),
        "top1_pct": round(top1, 1),
        "mean_entropy": round(mean_ent, 2),
        "entropy_std": round(ent_std, 2),
        "diveye": diveye,
        "specdetect_energy": spec_energy,
    }

    # LR classifier: prefer v2 (16 features + scaler) over v1 (5 features)
    # pickle is used here for sklearn model files we generated ourselves
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    lr_v2_path = os.path.join(models_dir, "perplexity_lr_v2.pkl")
    lr_v1_path = os.path.join(models_dir, "perplexity_lr.pkl")

    if os.path.exists(lr_v2_path):
        try:
            import pickle
            with open(lr_v2_path, "rb") as f:
                lr_data = pickle.load(f)
            lr_pipeline = lr_data["model"]  # Pipeline(StandardScaler + LR)
            # Build 16-feature vector: 5 basic + 10 DivEye + 1 SpecDetect
            features_v2 = np.array([[
                log_ppl, top10, mean_ent, top1, ent_std,
                diveye.get("surprisal_mean", 0), diveye.get("surprisal_std", 0),
                diveye.get("surprisal_var", 0), diveye.get("surprisal_skew", 0),
                diveye.get("surprisal_kurtosis", 0),
                diveye.get("diff1_mean", 0), diveye.get("diff1_std", 0),
                diveye.get("diff2_var", 0), diveye.get("diff2_entropy", 0),
                diveye.get("diff2_autocorr", 0),
                spec_energy,
            ]])
            prob = lr_pipeline.predict_proba(features_v2)[0]
            result["lr_ai_probability"] = round(float(prob[1]) * 100, 1)
            result["lr_prediction"] = "ai" if prob[1] > 0.5 else "human"
            result["lr_version"] = "v2"
        except Exception:
            pass
    elif os.path.exists(lr_v1_path):
        try:
            import pickle
            with open(lr_v1_path, "rb") as f:
                lr_data = pickle.load(f)
            lr_model = lr_data["model"]
            features = np.array([[log_ppl, top10, mean_ent, top1, ent_std]])
            prob = lr_model.predict_proba(features)[0]
            result["lr_ai_probability"] = round(float(prob[1]) * 100, 1)
            result["lr_prediction"] = "ai" if prob[1] > 0.5 else "human"
            result["lr_version"] = "v1"
        except Exception:
            pass

    return result


def preprocess_for_detection(text):
    """Clean text before detection to counter known adversarial bypasses.

    Handles: emoji injection, dialogue formatting, code blocks, short sentence splitting.
    Returns cleaned text suitable for PPL/LR/DeBERTa analysis.
    """
    import re
    import unicodedata

    # 0. Unicode homoglyph normalization — map lookalike chars to ASCII
    # Greek/Cyrillic letters that look identical to Latin ones
    CONFUSABLES = {
        "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0396": "Z",
        "\u0397": "H", "\u0399": "I", "\u039A": "K", "\u039C": "M",
        "\u039D": "N", "\u039F": "O", "\u03A1": "P", "\u03A4": "T",
        "\u03A5": "Y", "\u03A7": "X",
        "\u03B1": "a", "\u03B5": "e", "\u03B9": "i", "\u03BF": "o",
        "\u03C1": "p", "\u03C5": "u",
        # Cyrillic
        "\u0410": "A", "\u0412": "B", "\u0415": "E", "\u041A": "K",
        "\u041C": "M", "\u041D": "H", "\u041E": "O", "\u0420": "P",
        "\u0421": "C", "\u0422": "T", "\u0423": "Y", "\u0425": "X",
        "\u0430": "a", "\u0435": "e", "\u043E": "o", "\u0440": "p",
        "\u0441": "c", "\u0443": "y", "\u0445": "x",
    }
    text = "".join(CONFUSABLES.get(ch, ch) for ch in text)

    # 1. Strip emoji and other Unicode symbols that inflate PPL
    # Covers emoticons, dingbats, symbols, pictographs, flags, etc.
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U0000FE00-\U0000FE0F"  # variation selectors
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols extended-A
        "\U00002600-\U000026FF"  # misc symbols
        "\U0000200D"             # zero width joiner
        "\U00003030\U0000303D"   # wavy dash, part alternation mark
        "]+", flags=re.UNICODE
    )
    cleaned = emoji_pattern.sub(" ", text)

    # 2. Strip dialogue markers — convert "She said X" / "He replied Y" to just X, Y
    # Remove common dialogue tags
    cleaned = re.sub(
        r'["\u201c\u201d]([^"\u201c\u201d]{10,})["\u201c\u201d]\s*'
        r'(?:,?\s*(?:she|he|they|I)\s+(?:said|asked|replied|continued|added|noted|explained|whispered|shouted|exclaimed|responded|answered|remarked|observed|commented|muttered)\.?\s*)',
        r'\1. ', cleaned, flags=re.IGNORECASE
    )
    # Also handle: she said, "..."
    cleaned = re.sub(
        r'(?:she|he|they|I)\s+(?:said|asked|replied|continued|added|noted|explained)\s*[,:]?\s*["\u201c\u201d]([^"\u201c\u201d]{10,})["\u201c\u201d]\.?\s*',
        r'\1. ', cleaned, flags=re.IGNORECASE
    )
    # Remove remaining quotation marks from dialogue
    cleaned = re.sub(r'["\u201c\u201d]', '', cleaned)
    # Remove action tags like "She nodded thoughtfully."
    cleaned = re.sub(
        r'\b(?:She|He|They)\s+(?:nodded|smiled|laughed|sighed|shrugged|paused|frowned|grinned)\b[^.]*\.\s*',
        '', cleaned
    )

    # 3. Strip markdown formatting
    cleaned = re.sub(r'^#{1,6}\s+', '', cleaned, flags=re.MULTILINE)  # headers
    cleaned = re.sub(r'^\s*[-*]\s+', '', cleaned, flags=re.MULTILINE)  # bullet points
    cleaned = re.sub(r'`[^`]+`', '', cleaned)  # inline code
    cleaned = re.sub(r'```[\s\S]*?```', '', cleaned)  # code blocks

    # 4. Merge very short sentences (< 5 words) with neighbors
    # This counters the "short sentence splitting" attack
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    merged = []
    buffer = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s.split()) < 5 and buffer:
            buffer += " " + s
        elif len(s.split()) < 5 and not buffer:
            buffer = s
        else:
            if buffer:
                merged.append(buffer + " " + s)
                buffer = ""
            else:
                merged.append(s)
    if buffer:
        if merged:
            merged[-1] += " " + buffer
        else:
            merged.append(buffer)
    cleaned = " ".join(merged)

    # 5. Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        text = body.get("text", "").strip()

        if not text:
            result = {"error": "No text provided"}
        else:
            try:
                # Preprocess to counter adversarial bypasses
                cleaned_text = preprocess_for_detection(text)
                analysis_text = cleaned_text if len(cleaned_text) > 20 else text

                if self.server.model_tuple:
                    result = compute_features(self.server.model_tuple, analysis_text)
                else:
                    result = {"tokens": []}
                # Binoculars dual-model scoring
                if self.server.binoculars:
                    try:
                        bino = compute_binoculars(
                            self.server.binoculars[0],
                            self.server.binoculars[1],
                            analysis_text,
                        )
                        if bino:
                            result["binoculars"] = bino
                    except Exception as e:
                        print(f"Binoculars error: {e}", file=sys.stderr)
                # Aggregate perplexity stats + LR AI probability
                result["perplexity_stats"] = compute_perplexity_score(result.get("tokens", []))
                # DeBERTa binary classification
                clf_tok = self.server.classifier_tokenizer
                clf_mod = self.server.classifier_model
                if clf_tok and clf_mod:
                    result["classification"] = classify_text(clf_tok, clf_mod, analysis_text)
                # AI Vocab analysis (use original text for vocab, cleaned for scoring)
                ai_vocab_density, ai_vocab_hits = compute_ai_vocab(analysis_text)
                result["ai_vocab"] = {
                    "density": ai_vocab_density,
                    "matches": ai_vocab_hits,
                }

                # Ensemble: PPL (model-agnostic) + DeBERTa (domain-specific) + AI Vocab
                # Validated: 91% OOD accuracy on 22-text benchmark.
                deb_ai = result.get("classification", {}).get("ai_score", 50)
                ppl_stats = result.get("perplexity_stats") or {}
                has_ppl = ppl_stats is not None and "perplexity" in ppl_stats
                ppl_val = ppl_stats.get("perplexity", 20)
                top10 = ppl_stats.get("top10_pct", 80)
                mean_ent = ppl_stats.get("mean_entropy", 2.5)
                lr_ai = ppl_stats.get("lr_ai_probability", 50)

                # Binoculars signal (0-100 scale, higher = more AI-like)
                # NOTE: Currently disabled in fusion — llama3.2 1b/3b pair lacks
                # separation (scores overlap for AI/human). Kept for API exposure.
                bino = result.get("binoculars")
                has_bino = False  # Disabled until calibrated with better model pair
                if bino is not None and "score" in (bino or {}):
                    bino_score = bino["score"]
                    # Map binoculars score to 0-100: <0.7 = very AI (95), 0.7-0.9 = AI (75),
                    # 0.9-1.1 = uncertain (50), >1.1 = human (20), >1.3 = very human (5)
                    if bino_score < 0.7:
                        bino_ai = 95
                    elif bino_score < 0.85:
                        bino_ai = 80
                    elif bino_score < 0.95:
                        bino_ai = 60
                    elif bino_score < 1.1:
                        bino_ai = 40
                    elif bino_score < 1.3:
                        bino_ai = 20
                    else:
                        bino_ai = 5
                else:
                    bino_ai = 50  # neutral if unavailable

                word_count = len(analysis_text.split())

                # DeBERTa-only fast path: no perplexity model → use raw DeBERTa score
                clf_data = result.get("classification", {})
                is_uncertain = clf_data.get("is_uncertain", False)

                if not has_ppl:
                    fused = deb_ai
                    if is_uncertain:
                        fused = 50  # genuinely uncertain → report 50%
                        signal_source = f"deberta_uncertain(gap={clf_data.get('logit_gap', '?')})"
                    else:
                        signal_source = f"deberta_only(deb={deb_ai:.0f})"
                    ppl_ai_signal = False
                    ppl_human_signal = False
                else:
                    # Multi-signal weighted vote
                    min_ppl = ppl_stats.get("min_window_ppl", ppl_val)
                    ent_std = ppl_stats.get("entropy_std", 2.0)

                    # PPL scoring — tightened thresholds calibrated for llama3.2:1b
                    # llama3.2:1b observed ranges: AI ppl 5-9, human casual 15-30+
                    # Overlap zone: 9-15 (formal/academic text)
                    if ppl_val < 6 and top10 > 88:
                        ppl_score = 95  # very likely AI
                    elif ppl_val < 8 and top10 > 85:
                        ppl_score = 80  # likely AI
                    elif ppl_val > 25 and top10 < 75:
                        ppl_score = 10  # very likely human
                    elif ppl_val > 18 and top10 < 80:
                        ppl_score = 20  # likely human
                    elif ppl_val > 12:
                        ppl_score = 35  # lean human (overlap zone)
                    elif ppl_val < 10 and top10 > 82:
                        ppl_score = 65  # lean AI but not confident
                    else:
                        ppl_score = 50  # genuinely uncertain

                    if min_ppl < 3.5:
                        ppl_score = max(ppl_score, 80)

                    # Statistical text features (typo-resistant, no model needed)
                    # Extended stylometric analysis inspired by StyloAI (31 features)
                    sentences_list = [s.strip() for s in analysis_text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
                    stat_score = 50  # default neutral
                    if len(sentences_list) >= 3:
                        words_all = analysis_text.lower().split()
                        word_count_stat = len(words_all)

                        # === Feature Group 1: Sentence-level ===
                        sent_lens = [len(s.split()) for s in sentences_list]
                        mean_len = sum(sent_lens) / len(sent_lens)
                        len_var = sum((l - mean_len) ** 2 for l in sent_lens) / len(sent_lens)
                        len_cv = (len_var ** 0.5) / max(mean_len, 1)

                        # === Feature Group 2: Vocabulary richness (typo-resistant) ===
                        # These measure lexical diversity — typos create NEW unique words,
                        # so they actually INCREASE these metrics (making text look MORE human)
                        from collections import Counter
                        word_freq = Counter(words_all)
                        unique_words = len(word_freq)
                        ttr = unique_words / max(word_count_stat, 1)  # type-token ratio
                        hapax = sum(1 for w, c in word_freq.items() if c == 1)  # words appearing once
                        hapax_ratio = hapax / max(unique_words, 1)
                        # Yule's K — vocabulary richness measure, robust to text length
                        freq_spectrum = Counter(word_freq.values())
                        m1 = sum(word_freq.values())
                        m2 = sum(i * i * freq_spectrum[i] for i in freq_spectrum)
                        yules_k = 10000 * (m2 - m1) / max(m1 * m1, 1) if m1 > 0 else 0

                        # === Feature Group 3: Function word analysis (very hard to fake) ===
                        # Function words are structure words — their distribution is deeply ingrained
                        # in writing style and very hard to consciously manipulate
                        function_words = {
                            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
                            'should', 'may', 'might', 'can', 'could', 'must', 'of', 'in', 'to',
                            'for', 'with', 'on', 'at', 'by', 'from', 'as', 'into', 'through',
                            'during', 'before', 'after', 'above', 'below', 'between', 'under',
                            'that', 'which', 'who', 'whom', 'this', 'these', 'those', 'it',
                            'its', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both',
                            'either', 'neither', 'if', 'then', 'else', 'when', 'while', 'although',
                        }
                        fw_count = sum(1 for w in words_all if w in function_words)
                        fw_ratio = fw_count / max(word_count_stat, 1)

                        # === Feature Group 4: Transition/discourse markers (AI signature) ===
                        transition_words = {'furthermore', 'moreover', 'additionally', 'consequently',
                                          'nevertheless', 'however', 'therefore', 'thus', 'hence',
                                          'accordingly', 'subsequently', 'in conclusion', 'in summary',
                                          'it is important to note', 'it is worth noting',
                                          'significantly', 'notably', 'fundamentally', 'specifically',
                                          'particularly', 'essentially', 'ultimately', 'increasingly'}
                        words_lower = analysis_text.lower()
                        tw_count = sum(1 for tw in transition_words if tw in words_lower)
                        tw_density = tw_count / max(len(sentences_list), 1)

                        # === Feature Group 5: Punctuation & formatting ===
                        punct_types = set()
                        for ch in analysis_text:
                            if ch in '—–-…();:!?"\'':
                                punct_types.add(ch)
                        punct_div = len(punct_types)
                        # Contraction usage (AI rarely uses contractions)
                        contractions = ["n't", "'m", "'re", "'ve", "'ll", "'d", "'s"]
                        contraction_count = sum(analysis_text.lower().count(c) for c in contractions)
                        contraction_rate = contraction_count / max(len(sentences_list), 1)

                        # === Feature Group 6: Sentence starters diversity ===
                        # AI tends to start sentences with similar patterns
                        starters = [s.split()[0].lower() if s.split() else '' for s in sentences_list]
                        unique_starters = len(set(starters))
                        starter_diversity = unique_starters / max(len(starters), 1)

                        # === Feature Group 7: Average word length distribution ===
                        avg_word_len = sum(len(w) for w in words_all) / max(word_count_stat, 1)
                        # AI tends toward medium-length words; humans use more short + long extremes
                        short_words = sum(1 for w in words_all if len(w) <= 3) / max(word_count_stat, 1)
                        long_words = sum(1 for w in words_all if len(w) >= 8) / max(word_count_stat, 1)

                        # === Scoring: weighted signal accumulation ===
                        ai_signals = 0
                        human_signals = 0

                        # Sentence length uniformity
                        if len_cv < 0.25: ai_signals += 2
                        elif len_cv < 0.35: ai_signals += 1
                        elif len_cv > 0.45: human_signals += 2
                        elif len_cv > 0.38: human_signals += 1

                        # Transition word density (AI uses many)
                        if tw_density > 0.35: ai_signals += 2
                        elif tw_density > 0.20: ai_signals += 1
                        elif tw_density < 0.10: human_signals += 1

                        # Punctuation diversity (high = human)
                        if punct_div >= 5: human_signals += 2
                        elif punct_div >= 3: human_signals += 1

                        # Vocabulary richness — AI has lower hapax ratio & Yule's K
                        if hapax_ratio < 0.55: ai_signals += 1  # AI reuses words more
                        elif hapax_ratio > 0.70: human_signals += 1
                        if yules_k < 80: ai_signals += 1  # AI: low vocabulary richness
                        elif yules_k > 140: human_signals += 1

                        # Function word ratio — AI tends toward 0.42-0.48, human more variable
                        if 0.42 <= fw_ratio <= 0.48: ai_signals += 1
                        elif fw_ratio < 0.38 or fw_ratio > 0.52: human_signals += 1

                        # Contractions — AI almost never uses them
                        if contraction_rate > 0.3: human_signals += 2  # strong human signal
                        elif contraction_rate > 0.1: human_signals += 1
                        elif contraction_rate == 0 and word_count_stat > 50: ai_signals += 1

                        # Sentence starter diversity
                        if starter_diversity < 0.5: ai_signals += 1  # AI repetitive starts
                        elif starter_diversity > 0.8: human_signals += 1

                        # Short word ratio — humans use more informal short words
                        if short_words > 0.45: human_signals += 1
                        elif short_words < 0.30: ai_signals += 1

                        total = ai_signals + human_signals
                        if total == 0:
                            stat_score = 50
                        else:
                            ratio = ai_signals / total
                            stat_score = int(10 + ratio * 80)  # range: 10-90

                    has_lr = lr_ai != 50  # LR model actually loaded

                    # Consensus override: when 2+ strong signals agree, trust them
                    all_signals = [deb_ai, ppl_score, stat_score, lr_ai]
                    if has_bino:
                        all_signals.append(bino_ai)
                    strong_ai = sum(1 for s in all_signals if s > 80)
                    strong_human = sum(1 for s in [100-s for s in all_signals] if s > 80)

                    if has_lr and strong_ai >= 2 and strong_human == 0:
                        # Multiple signals strongly agree AI — consensus overrides dissent
                        scores = all_signals
                        fused = sum(scores) / len(scores)  # simple average when consensus
                        bino_str = f",bino={bino_ai}" if has_bino else ""
                        signal_source = f"consensus_ai(deb={deb_ai:.0f},ppl={ppl_score},stat={stat_score},lr={lr_ai:.0f}{bino_str})"
                    elif has_lr and strong_human >= 2 and strong_ai == 0:
                        scores = [deb_ai, ppl_score, stat_score, lr_ai]
                        fused = sum(scores) / 4
                        bino_str = f",bino={bino_ai}" if has_bino else ""
                        signal_source = f"consensus_human(deb={deb_ai:.0f},ppl={ppl_score},stat={stat_score},lr={lr_ai:.0f}{bino_str})"
                    elif has_lr and lr_ai > 85:
                        # LR confident AI — trust LR but keep DeBERTa voice
                        fused = lr_ai * 0.35 + deb_ai * 0.30 + ppl_score * 0.20 + stat_score * 0.15
                        signal_source = f"lr_confident(lr={lr_ai:.0f},deb={deb_ai:.0f},ppl={ppl_score},stat={stat_score})"
                    elif has_lr and lr_ai < 15:
                        # LR confident human — but don't let LR alone override DeBERTa
                        if deb_ai > 80 and stat_score > 70:
                            # DeBERTa AND stat both say AI → don't let LR/PPL dominate
                            fused = lr_ai * 0.15 + deb_ai * 0.35 + ppl_score * 0.15 + stat_score * 0.35
                            signal_source = f"deb_stat_override(lr={lr_ai:.0f},deb={deb_ai:.0f},ppl={ppl_score},stat={stat_score})"
                        elif deb_ai > 80:
                            fused = lr_ai * 0.25 + deb_ai * 0.25 + ppl_score * 0.30 + stat_score * 0.20
                            signal_source = f"lr_confident(lr={lr_ai:.0f},deb={deb_ai:.0f},ppl={ppl_score},stat={stat_score})"
                        else:
                            fused = lr_ai * 0.35 + deb_ai * 0.25 + ppl_score * 0.25 + stat_score * 0.15
                            signal_source = f"lr_confident(lr={lr_ai:.0f},deb={deb_ai:.0f},ppl={ppl_score},stat={stat_score})"
                    elif has_lr:
                        # Default vote — but handle DeBERTa unreliability
                        if ppl_score >= 70 and lr_ai >= 60 and deb_ai < 20:
                            # PPL+LR say AI but DeBERTa says human → DeBERTa wrong, demote it
                            fused = lr_ai * 0.30 + deb_ai * 0.10 + ppl_score * 0.35 + stat_score * 0.25
                            signal_source = f"ppl_lr_override(lr={lr_ai:.0f},deb={deb_ai:.0f},ppl={ppl_score},stat={stat_score})"
                        elif ppl_score <= 30 and lr_ai <= 30 and deb_ai > 80:
                            # PPL+LR say human but DeBERTa says AI → DeBERTa wrong, demote it
                            fused = lr_ai * 0.30 + deb_ai * 0.10 + ppl_score * 0.35 + stat_score * 0.25
                            signal_source = f"ppl_lr_override(lr={lr_ai:.0f},deb={deb_ai:.0f},ppl={ppl_score},stat={stat_score})"
                        else:
                            fused = lr_ai * 0.25 + deb_ai * 0.30 + ppl_score * 0.25 + stat_score * 0.20
                            signal_source = f"vote(lr={lr_ai:.0f},deb={deb_ai:.0f},ppl={ppl_score},stat={stat_score})"
                    else:
                        # No LR model: redistribute weight to DeBERTa + PPL + stats
                        # When DeBERTa is high but other signals disagree, dampen DeBERTa
                        deb_weight = 0.40
                        ppl_weight = 0.35
                        stat_weight = 0.25
                        if deb_ai > 80 and ppl_score < 50 and stat_score < 50:
                            # DeBERTa says AI but PPL+stats say human → conflict, trust PPL+stats more
                            deb_weight = 0.25
                            ppl_weight = 0.40
                            stat_weight = 0.35
                        elif deb_ai > 80 and (ppl_score < 50 or stat_score < 40):
                            # Partial disagreement → moderate dampening
                            deb_weight = 0.30
                            ppl_weight = 0.40
                            stat_weight = 0.30
                        fused = deb_ai * deb_weight + ppl_score * ppl_weight + stat_score * stat_weight
                        signal_source = f"vote_no_lr(deb={deb_ai:.0f},ppl={ppl_score},stat={stat_score})"

                    ppl_ai_signal = ppl_score >= 80
                    ppl_human_signal = ppl_score <= 20
                if word_count < 50:
                    # Too short for reliable detection — force uncertain
                    fused = 50
                    signal_source = f"too_short({word_count}w)"
                    threshold = 50
                elif word_count < 150:
                    threshold = 65
                elif word_count < 300:
                    threshold = 55
                else:
                    threshold = 50

                # Determine prediction with uncertainty zone
                if fused > threshold + 8:
                    prediction = "ai"
                elif fused < threshold - 8:
                    prediction = "human"
                else:
                    prediction = "uncertain"

                result["fused"] = {
                    "ai_score": round(fused, 1),
                    "prediction": prediction,
                    "confidence": round(max(fused, 100 - fused), 1),
                    "word_count": word_count,
                    "threshold": threshold,
                    "signal_source": signal_source,
                    "ppl_ai_signal": ppl_ai_signal,
                    "ppl_human_signal": ppl_human_signal,
                }

                # ── Sliding window analysis (segment-level detection) ────
                # Detects sandwich/hybrid attacks by scoring 3-sentence windows
                if has_ppl and len(sentences_list) >= 6:
                    window_size = 3
                    stride = 2
                    segment_scores = []
                    for wi in range(0, len(sentences_list) - window_size + 1, stride):
                        window_sents = sentences_list[wi:wi + window_size]
                        window_text = ". ".join(window_sents) + "."
                        # Quick per-window stat score
                        w_words = window_text.lower().split()
                        w_wc = len(w_words)
                        if w_wc < 10:
                            segment_scores.append({"start": wi, "end": wi + window_size, "ai_score": 50})
                            continue
                        # Sentence length CV for window
                        w_lens = [len(s.split()) for s in window_sents]
                        w_mean = sum(w_lens) / len(w_lens)
                        w_var = sum((l - w_mean) ** 2 for l in w_lens) / len(w_lens)
                        w_cv = (w_var ** 0.5) / max(w_mean, 1)
                        # Transition words in window
                        w_tw = sum(1 for tw in transition_words if tw in window_text.lower())
                        w_tw_d = w_tw / len(window_sents)
                        # Contractions in window
                        w_contr = sum(window_text.lower().count(c) for c in ["n't", "'m", "'re", "'ve", "'ll", "'d"])
                        w_contr_r = w_contr / len(window_sents)
                        # Simple window score
                        w_ai = 0
                        w_hu = 0
                        if w_cv < 0.25: w_ai += 2
                        elif w_cv > 0.45: w_hu += 2
                        if w_tw_d > 0.3: w_ai += 2
                        elif w_tw_d < 0.1: w_hu += 1
                        if w_contr_r > 0.2: w_hu += 2
                        elif w_contr_r == 0: w_ai += 1
                        w_total = w_ai + w_hu
                        if w_total == 0:
                            w_score = 50
                        else:
                            w_score = int(10 + (w_ai / w_total) * 80)
                        # Blend with DeBERTa if available (DeBERTa sees full text, not window)
                        # Just use stat-based score for windows
                        segment_scores.append({
                            "start": wi,
                            "end": wi + window_size,
                            "sentences": [s[:60] + "..." if len(s) > 60 else s for s in window_sents],
                            "ai_score": w_score,
                            "features": {"cv": round(w_cv, 3), "tw": round(w_tw_d, 2), "contr": round(w_contr_r, 2)},
                        })
                    # Flag segments with high AI score
                    max_seg = max((s["ai_score"] for s in segment_scores), default=50)
                    min_seg = min((s["ai_score"] for s in segment_scores), default=50)
                    result["segment_analysis"] = {
                        "segments": segment_scores,
                        "max_ai_score": max_seg,
                        "min_ai_score": min_seg,
                        "variance": max_seg - min_seg,
                        "sandwich_risk": max_seg - min_seg > 30,
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

    def __init__(self, addr, handler, model_tuple, classifier_tokenizer=None, classifier_model=None, binoculars=None):
        super().__init__(addr, handler)
        self.model_tuple = model_tuple
        self.classifier_tokenizer = classifier_tokenizer
        self.classifier_model = classifier_model
        self.binoculars = binoculars


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5001))
    print("Loading perplexity model...", file=sys.stderr)
    model_tuple = load_model()
    print("Loading classifier...", file=sys.stderr)
    clf_tokenizer, clf_model = load_classifier()
    print("Loading Binoculars...", file=sys.stderr)
    binoculars = load_binoculars()
    print(f"Server running at http://{host}:{port}", file=sys.stderr)
    server = PerplexityServer((host, port), Handler, model_tuple, clf_tokenizer, clf_model, binoculars)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", file=sys.stderr)
