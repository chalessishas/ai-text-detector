#!/usr/bin/env python3
"""Generate 4-class labeled dataset using 23 real LLM APIs.

Classes:
  0 = Pure Human      (from corpus)
  1 = Pure AI         (23 models × 6 prompt styles × 3 temps × 3 lengths)
  2 = AI + polished   (AI output → another model rewrites it)
  3 = Human + polished (human text → AI polishes/improves it)

Features:
  - Checkpoint resume: reads existing output, skips completed entries
  - Incremental writes: appends each result as it completes
  - Per-model temperature limits from config
  - Auto-disables models after repeated failures
  - Parallel generation of all classes

Usage:
    python3 scripts/generate_dataset.py [--output dataset.jsonl] [--target-per-class 17500]
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Load .env
env_path = Path(__file__).parent.parent / ".env.local"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from dataset_config import (
    PROVIDERS, PROMPT_STYLES, PERSONAS, TOPICS,
    TEMPERATURES, LENGTHS, LABELS, DEFAULT_MAX_TEMP,
)


# --- Model health tracker ---

class ModelTracker:
    """Track per-model failures and auto-disable after threshold."""

    def __init__(self, max_consecutive_failures: int = 5):
        self.max_failures = max_consecutive_failures
        self.consecutive_failures: dict[str, int] = defaultdict(int)
        self.disabled: set[str] = set()
        self.success_count: dict[str, int] = defaultdict(int)
        self.failure_count: dict[str, int] = defaultdict(int)

    def record_success(self, model: str):
        self.consecutive_failures[model] = 0
        self.success_count[model] += 1

    def record_failure(self, model: str):
        self.consecutive_failures[model] += 1
        self.failure_count[model] += 1
        if self.consecutive_failures[model] >= self.max_failures:
            self.disabled.add(model)
            print(f"  DISABLED {model}: {self.max_failures} consecutive failures", file=sys.stderr)

    def is_available(self, model: str) -> bool:
        return model not in self.disabled

    def summary(self) -> str:
        lines = []
        all_models = set(self.success_count) | set(self.failure_count)
        for m in sorted(all_models):
            s, f = self.success_count[m], self.failure_count[m]
            status = "DISABLED" if m in self.disabled else "ok"
            lines.append(f"    {m}: {s} ok / {f} err [{status}]")
        return "\n".join(lines)


tracker = ModelTracker()


# --- Checkpoint ---

def load_checkpoint(output_path: str) -> tuple[list[dict], dict[int, int]]:
    """Load existing dataset entries and count per label."""
    existing = []
    counts = defaultdict(int)
    if not os.path.exists(output_path):
        return existing, counts
    with open(output_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"  WARNING: skipping corrupted line {i + 1} in checkpoint", file=sys.stderr)
                continue
            existing.append(entry)
            counts[entry["label"]] += 1
    return existing, counts


# --- Incremental writer ---

class IncrementalWriter:
    """Thread-safe append-only JSONL writer."""

    def __init__(self, path: str):
        self.path = path
        self.lock = asyncio.Lock()
        self.written = 0

    async def write(self, entry: dict):
        async with self.lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.written += 1


# --- Unified API callers ---

async def call_openai_compatible(
    session, base_url: str, api_key: str, model: str,
    prompt: str, temperature: float, max_tokens: int,
) -> Optional[str]:
    import aiohttp

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if "gpt-5" in model or "gpt-4o" in model:
        payload["max_completion_tokens"] = max_tokens
    elif "GLM" in model or "glm" in model:
        # GLM: skip max_tokens, add max_length in meta if needed
        pass
    else:
        payload["max_tokens"] = max_tokens

    url = f"{base_url}/chat/completions"

    try:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status == 429:
                print(f"  RATE-LIMIT {model}: 429", file=sys.stderr)
                return None
            if resp.status != 200:
                error = await resp.text()
                print(f"  ERROR {model}: {resp.status} {error[:120]}", file=sys.stderr)
                return None
            data = await resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  ERROR {model}: {e}", file=sys.stderr)
        return None


async def call_anthropic(
    session, api_key: str, model: str,
    prompt: str, temperature: float, max_tokens: int,
) -> Optional[str]:
    import aiohttp

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        async with session.post(
            "https://api.anthropic.com/v1/messages",
            json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status == 429:
                print(f"  RATE-LIMIT {model}: 429", file=sys.stderr)
                return None
            if resp.status != 200:
                error = await resp.text()
                print(f"  ERROR {model}: {resp.status} {error[:120]}", file=sys.stderr)
                return None
            data = await resp.json()
            return data["content"][0]["text"]
    except Exception as e:
        print(f"  ERROR {model}: {e}", file=sys.stderr)
        return None


async def call_gemini(
    session, api_key: str, model: str,
    prompt: str, temperature: float, max_tokens: int,
    is_thinking: bool = False,
) -> Optional[str]:
    import aiohttp

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )
    gen_config = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
    }
    if is_thinking:
        gen_config["thinkingConfig"] = {"thinkingBudget": 1024}
        gen_config["maxOutputTokens"] = max(max_tokens, 4000)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_config,
    }

    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status == 429:
                print(f"  RATE-LIMIT {model}: 429", file=sys.stderr)
                return None
            if resp.status != 200:
                error = await resp.text()
                print(f"  ERROR {model}: {resp.status} {error[:120]}", file=sys.stderr)
                return None
            data = await resp.json()
            parts = data["candidates"][0].get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part:
                    return part["text"]
            return None
    except Exception as e:
        print(f"  ERROR {model}: {e}", file=sys.stderr)
        return None


def strip_preamble(text: str) -> str:
    """Remove markdown headers, 'Here's the polished...' preamble, and '---' dividers."""
    lines = text.split("\n")
    cleaned = []
    skip_next_blank = False
    for line in lines:
        stripped = line.strip()
        # Skip markdown headers
        if stripped.startswith("#"):
            skip_next_blank = True
            continue
        # Skip '---' dividers
        if re.match(r'^-{3,}$', stripped):
            skip_next_blank = True
            continue
        # Skip preamble lines like "Here's the polished text:" but not "Here in the US..."
        if re.match(r"^(Here'?s?\s+(the|is|my|your|a|an?\s)|Below is|The following is|Polished version|Rewritten version|Here you go)", stripped, re.IGNORECASE):
            skip_next_blank = True
            continue
        if skip_next_blank and stripped == "":
            skip_next_blank = False
            continue
        skip_next_blank = False
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def get_model_temperature(provider: str, model: str, requested: float) -> float:
    """Resolve actual temperature respecting per-model limits.

    - fixed_temp: model only accepts this exact value (e.g. gpt-5 = 1.0)
    - max_temp: clamp requested to this ceiling
    - otherwise: pass through
    """
    model_cfg = PROVIDERS[provider]["models"].get(model, {})
    if "fixed_temp" in model_cfg:
        return model_cfg["fixed_temp"]
    return min(requested, model_cfg.get("max_temp", DEFAULT_MAX_TEMP))


async def generate_text(
    session, provider: str, model: str,
    prompt: str, temperature: float, max_tokens: int,
) -> Optional[str]:
    """Route to the correct API caller, respecting per-model temp limits."""
    if not tracker.is_available(model):
        return None

    config = PROVIDERS[provider]
    api_key = os.environ.get(config["env_key"], "")
    if not api_key:
        return None

    temp = get_model_temperature(provider, model, temperature)

    for attempt in range(3):
        if provider == "anthropic":
            result = await call_anthropic(session, api_key, model, prompt, temp, max_tokens)
        elif provider == "google":
            is_thinking = PROVIDERS[provider]["models"].get(model, {}).get("thinking", False)
            result = await call_gemini(session, api_key, model, prompt, temp, max_tokens, is_thinking)
        else:
            result = await call_openai_compatible(session, config["base_url"], api_key, model, prompt, temp, max_tokens)

        if result is not None:
            tracker.record_success(model)
            return result

        tracker.record_failure(model)
        if not tracker.is_available(model):
            return None

        wait = (attempt + 1) * 5 + random.uniform(0, 3)
        await asyncio.sleep(wait)

    return None


# --- Prompt builder ---

def build_prompt(style: str, topic: str, length_key: str, source_text: str = "") -> str:
    length_words = LENGTHS[length_key]["words"]
    template = PROMPT_STYLES[style]

    if style == "persona":
        persona = random.choice(PERSONAS)
        return template.format(topic=topic, length=length_words, persona=persona)
    elif style == "rewrite":
        if not source_text:
            return PROMPT_STYLES["direct"].format(topic=topic, length=length_words)
        return template.format(source_text=source_text[:2000])
    else:
        return template.format(topic=topic, length=length_words)


# --- Human text loader ---

def load_human_texts(corpus_dir: str, count: int) -> list[str]:
    """Load coherent human-written passages from corpus.

    Uses consecutive sentence windows (not random shuffle) to preserve
    article coherence. Passages are selected from random starting positions
    across the corpus files.
    """
    raw_dir = os.path.join(corpus_dir, "raw")

    # Read sentences per file, keeping order within each file
    file_sents: list[list[str]] = []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(raw_dir, fname), "r", errors="ignore") as f:
            sents = [line.strip() for line in f if len(line.strip().split()) >= 10]
        if sents:
            file_sents.append(sents)

    # Build passages using consecutive sentence windows
    paragraphs: list[str] = []
    attempts = 0
    max_attempts = count * 10

    while len(paragraphs) < count and attempts < max_attempts:
        attempts += 1
        # Pick a random file
        sents = random.choice(file_sents)
        # Pick a random window size and starting position
        n_sents = random.choice([12, 18, 25, 35, 50])
        if len(sents) < n_sents:
            continue
        start = random.randint(0, len(sents) - n_sents)
        para = " ".join(sents[start : start + n_sents])
        word_count = len(para.split())
        if 300 <= word_count <= 1500:
            paragraphs.append(para)

    random.shuffle(paragraphs)
    return paragraphs[:count]


# --- Available models helper ---

def get_available_models(tier_filter: Optional[set] = None) -> list[tuple[str, str]]:
    """Return (provider, model) pairs that have API keys set."""
    available = []
    for provider, config in PROVIDERS.items():
        api_key = os.environ.get(config["env_key"], "")
        if not api_key:
            continue
        for model, model_cfg in config["models"].items():
            if tier_filter and model_cfg["tier"] not in tier_filter:
                continue
            available.append((provider, model))
    return available


# --- Generation workers ---

async def generate_ai_texts(
    session, target: int, human_texts: list[str], writer: IncrementalWriter,
) -> list[dict]:
    """Generate Pure AI texts from all available models."""
    available = get_available_models()
    if not available:
        print("ERROR: No API keys found. Set them in .env.local", file=sys.stderr)
        return []

    print(f"  Available models: {len(available)}", file=sys.stderr)
    for p, m in available:
        print(f"    {p}/{m}", file=sys.stderr)

    styles = list(PROMPT_STYLES.keys())

    results = []
    completed = 0
    provider_semaphores = {p: asyncio.Semaphore(3) for p in PROVIDERS}
    global_semaphore = asyncio.Semaphore(12)

    BATCH_SIZE = 50

    async def run_one():
        nonlocal completed
        if len(results) >= target:
            return

        candidates = [(p, m) for p, m in available if tracker.is_available(m)]
        if not candidates:
            return
        provider, model = random.choice(candidates)

        style = random.choice(styles)
        topic = random.choice(TOPICS)
        temp = random.choice(TEMPERATURES)
        length_key = random.choice(list(LENGTHS.keys()))
        source_text = random.choice(human_texts) if style == "rewrite" and human_texts else ""
        prompt = build_prompt(style, topic, length_key, source_text)
        max_tokens = LENGTHS[length_key]["target_tokens"]

        async with global_semaphore, provider_semaphores[provider]:
            text = await generate_text(session, provider, model, prompt, temp, max_tokens)

        completed += 1
        if completed % 100 == 0:
            print(f"  AI: {len(results)}/{target} done ({completed} attempted)", file=sys.stderr)

        if text:
            text = strip_preamble(text)
        if text and len(text.split()) >= 200 and len(results) < target:
            entry = {
                "text": text,
                "label": 1,
                "label_name": "ai",
                "model": model,
                "provider": provider,
                "style": style,
                "topic": topic,
                "temperature": temp,
                "length": length_key,
                "tier": PROVIDERS[provider]["models"][model]["tier"],
            }
            results.append(entry)
            await writer.write(entry)

    while len(results) < target:
        candidates = [(p, m) for p, m in available if tracker.is_available(m)]
        if not candidates:
            print("  All models disabled, stopping AI generation", file=sys.stderr)
            break
        batch = [run_one() for _ in range(BATCH_SIZE)]
        await asyncio.gather(*batch)

    return results


async def generate_polished_texts(
    session, ai_texts: list[dict], human_texts: list[str],
    target_ai_polished: int, target_human_polished: int,
    writer: IncrementalWriter,
) -> tuple[list[dict], list[dict]]:
    """Generate AI+polished and Human+polished concurrently."""
    available = get_available_models(tier_filter={"mid", "budget"})

    if not available:
        print("  WARNING: No mid/budget models for polishing, using rule-based", file=sys.stderr)
        ai_p, h_p = _rule_based_polish(ai_texts, human_texts, target_ai_polished, target_human_polished)
        for entry in ai_p + h_p:
            await writer.write(entry)
        return ai_p, h_p

    ai_polished = []
    human_polished = []
    semaphore = asyncio.Semaphore(20)
    completed = 0
    total = target_ai_polished + target_human_polished

    async def polish_one(text: str, label: int, label_name: str, original_model: str = ""):
        nonlocal completed
        candidates = [(p, m) for p, m in available if tracker.is_available(m)]
        if not candidates:
            return None
        provider, model = random.choice(candidates)

        if label == 2:
            prompt = (
                "Rewrite the following text to sound more natural and human-like. "
                "Change sentence structures, vary word choices, but keep the same meaning. "
                "Output ONLY the rewritten text, no headers or commentary.\n\n"
                f"{text[:3000]}"
            )
        else:
            prompt = (
                "Polish and improve the following text. Fix any grammar issues, "
                "improve clarity and flow, but keep the original voice and meaning. "
                "Output ONLY the polished text, no headers or commentary.\n\n"
                f"{text[:3000]}"
            )

        max_tokens = max(1500, len(text.split()) * 2)
        async with semaphore:
            result = await generate_text(session, provider, model, prompt, 0.7, max_tokens)
            completed += 1
            if completed % 100 == 0:
                print(f"  Polish: {completed}/{total} "
                      f"(ai_p={len(ai_polished)}, h_p={len(human_polished)})", file=sys.stderr)
            return result, model, provider

    BATCH_SIZE = 100

    async def run_ai_polished():
        while len(ai_polished) < target_ai_polished and ai_texts:
            candidates = [(p, m) for p, m in available if tracker.is_available(m)]
            if not candidates:
                print("  All polishing models disabled, stopping ai_polished", file=sys.stderr)
                break
            batch_samples = [random.choice(ai_texts) for _ in range(BATCH_SIZE)]
            batch_tasks = [
                polish_one(s["text"], 2, "ai_polished", s.get("model", ""))
                for s in batch_samples
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for orig, result in zip(batch_samples, batch_results):
                if isinstance(result, Exception) or result is None:
                    continue
                polished, model, provider = result
                if polished:
                    polished = strip_preamble(polished)
                if polished and len(polished.split()) >= 200 and len(ai_polished) < target_ai_polished:
                    entry = {
                        "text": polished,
                        "label": 2,
                        "label_name": "ai_polished",
                        "original_model": orig.get("model", ""),
                        "polisher_model": model,
                        "polisher_provider": provider,
                    }
                    ai_polished.append(entry)
                    await writer.write(entry)

    async def run_human_polished():
        while len(human_polished) < target_human_polished and human_texts:
            candidates = [(p, m) for p, m in available if tracker.is_available(m)]
            if not candidates:
                print("  All polishing models disabled, stopping human_polished", file=sys.stderr)
                break
            batch_samples = [random.choice(human_texts) for _ in range(BATCH_SIZE)]
            batch_tasks = [
                polish_one(s, 3, "human_polished")
                for s in batch_samples
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for orig_text, result in zip(batch_samples, batch_results):
                if isinstance(result, Exception) or result is None:
                    continue
                polished, model, provider = result
                if polished:
                    polished = strip_preamble(polished)
                if polished and len(polished.split()) >= 200 and len(human_polished) < target_human_polished:
                    entry = {
                        "text": polished,
                        "label": 3,
                        "label_name": "human_polished",
                        "polisher_model": model,
                        "polisher_provider": provider,
                    }
                    human_polished.append(entry)
                    await writer.write(entry)

    # Run both classes in parallel
    await asyncio.gather(run_ai_polished(), run_human_polished())

    return ai_polished, human_polished


def _rule_based_polish(ai_texts, human_texts, target_ai, target_human):
    synonyms = {
        'important': 'significant', 'significant': 'important',
        'show': 'demonstrate', 'demonstrate': 'show',
        'increase': 'rise', 'rise': 'increase',
        'help': 'assist', 'assist': 'help',
        'use': 'utilize', 'utilize': 'use',
        'many': 'numerous', 'numerous': 'many',
        'often': 'frequently', 'frequently': 'often',
        'good': 'excellent', 'bad': 'poor',
        'problem': 'challenge', 'method': 'approach',
    }

    def swap(text, rate=0.2):
        words = text.split()
        out = []
        for w in words:
            clean = w.lower().strip('.,;:!?()"\'')
            if clean in synonyms and random.random() < rate:
                repl = synonyms[clean]
                if w[0].isupper():
                    repl = repl.capitalize()
                trailing = ''.join(c for c in reversed(w) if c in '.,;:!?()"\'')[::-1]
                out.append(repl + trailing)
            else:
                out.append(w)
        return ' '.join(out)

    ai_polished = []
    for s in random.choices(ai_texts, k=target_ai):
        ai_polished.append({"text": swap(s["text"], 0.3), "label": 2, "label_name": "ai_polished"})

    human_polished = []
    for s in random.choices(human_texts, k=target_human):
        human_polished.append({"text": swap(s, 0.15), "label": 3, "label_name": "human_polished"})

    return ai_polished, human_polished


# --- Main ---

async def main():
    parser = argparse.ArgumentParser(description="Generate 4-class dataset")
    parser.add_argument("--output", default="dataset.jsonl")
    parser.add_argument("--target-per-class", type=int, default=17500)
    parser.add_argument("--corpus-dir", default="corpus")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoint, start fresh")
    args = parser.parse_args()

    n = args.target_per_class
    output_path = Path(__file__).parent.parent / args.output

    print(f"Target: {n} per class ({n * 4} total)", file=sys.stderr)

    # --- Checkpoint resume ---
    existing = []
    counts = defaultdict(int)
    existing_ai_texts = []

    if not args.fresh and output_path.exists():
        existing, counts = load_checkpoint(str(output_path))
        existing_ai_texts = [e for e in existing if e["label"] == 1]
        # Offset seed by checkpoint size to avoid regenerating same combos
        random.seed(42 + len(existing))
        print(f"\nCheckpoint loaded from {output_path}:", file=sys.stderr)
        for label_id, label_name in LABELS.items():
            print(f"  {label_name}: {counts[label_id]}/{n}", file=sys.stderr)
        if all(counts[i] >= n for i in range(4)):
            print("All targets met. Nothing to do.", file=sys.stderr)
            return
    else:
        random.seed(42)
        if output_path.exists() and args.fresh:
            backup = str(output_path) + f".bak.{int(time.time())}"
            os.rename(str(output_path), backup)
            print(f"Backed up existing dataset to {backup}", file=sys.stderr)

    need_human = max(0, n - counts[0])
    need_ai = max(0, n - counts[1])
    need_ai_polished = max(0, n - counts[2])
    need_human_polished = max(0, n - counts[3])

    print(f"\nRemaining: human={need_human}, ai={need_ai}, "
          f"ai_polished={need_ai_polished}, human_polished={need_human_polished}", file=sys.stderr)

    writer = IncrementalWriter(str(output_path))

    # --- Class 0: Pure Human (local, no API) ---
    if need_human > 0:
        print("\n[1/4] Loading Pure Human texts...", file=sys.stderr)
        human_texts = load_human_texts(args.corpus_dir, need_human + n)  # extra for polishing
        for text in human_texts[:need_human]:
            await writer.write({"text": text, "label": 0, "label_name": "human"})
        print(f"  Wrote {min(need_human, len(human_texts))} human paragraphs", file=sys.stderr)
    else:
        print("\n[1/4] Human texts: already met target", file=sys.stderr)
        human_texts = load_human_texts(args.corpus_dir, n)  # still need for polishing

    import aiohttp

    async with aiohttp.ClientSession() as session:
        # --- Class 1: Pure AI ---
        if need_ai > 0:
            print(f"\n[2/4] Generating {need_ai} Pure AI texts...", file=sys.stderr)
            new_ai_texts = await generate_ai_texts(session, need_ai, human_texts, writer)
            all_ai_texts = existing_ai_texts + new_ai_texts
            print(f"  Got {len(new_ai_texts)} new AI texts (total: {len(all_ai_texts)})", file=sys.stderr)
        else:
            print("\n[2/4] AI texts: already met target", file=sys.stderr)
            all_ai_texts = existing_ai_texts

        # --- Class 2 & 3: Polished (run concurrently) ---
        if need_ai_polished > 0 or need_human_polished > 0:
            print(f"\n[3/4] Generating polished texts (ai_polished={need_ai_polished}, "
                  f"human_polished={need_human_polished})...", file=sys.stderr)

            if not all_ai_texts and need_ai_polished > 0:
                print("  WARNING: No AI texts available for ai_polished, skipping", file=sys.stderr)
                need_ai_polished = 0

            ai_polished, human_polished = await generate_polished_texts(
                session, all_ai_texts, human_texts,
                need_ai_polished, need_human_polished, writer,
            )
            print(f"  AI+polished: {len(ai_polished)}", file=sys.stderr)
            print(f"  Human+polished: {len(human_polished)}", file=sys.stderr)
        else:
            print("\n[3/4] Polished texts: already met targets", file=sys.stderr)

    # --- Summary ---
    _, final_counts = load_checkpoint(str(output_path))
    total = sum(final_counts.values())
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Dataset: {output_path}", file=sys.stderr)
    print(f"Total: {total}", file=sys.stderr)
    for label_id, label_name in LABELS.items():
        c = final_counts[label_id]
        status = "✓" if c >= n else f"({n - c} short)"
        print(f"  {label_name}: {c}/{n} {status}", file=sys.stderr)

    print(f"\nModel health:", file=sys.stderr)
    print(tracker.summary(), file=sys.stderr)
    print(f"Written this run: {writer.written}", file=sys.stderr)


if __name__ == "__main__":
    try:
        import aiohttp
    except ImportError:
        print("Installing aiohttp...", file=sys.stderr)
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "-q"])

    asyncio.run(main())
