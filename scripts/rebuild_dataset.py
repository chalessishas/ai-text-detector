#!/usr/bin/env python3
"""Rebuild dataset.jsonl: replace human & human_polished with quality data.

Keeps existing ai (label=1) and ai_polished (label=2) intact.
Regenerates:
  - human (label=0): coherent passages from corpus (consecutive window sampling)
  - human_polished (label=3): mix of API polish (DeepSeek/Qwen) + local paraphrase (spaCy)

Usage:
    python3 scripts/rebuild_dataset.py [--output dataset.jsonl] [--target 17500]
    python3 scripts/rebuild_dataset.py --local-only  # skip API, 100% local paraphrase
"""

from __future__ import annotations

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

sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import load_human_texts
from local_paraphraser import paraphrase as local_paraphrase

# ── API config (DeepSeek + Qwen only) ──

API_MODELS = []

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if DEEPSEEK_KEY:
    API_MODELS.append({
        "provider": "deepseek",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": DEEPSEEK_KEY,
    })

QWEN_KEY = os.environ.get("QWEN_API_KEY", "")
if QWEN_KEY:
    for model in ["qwen-turbo"]:
        API_MODELS.append({
            "provider": "qwen",
            "model": model,
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": QWEN_KEY,
        })

POLISH_PROMPT = """Rewrite the following text to improve clarity, flow, and readability.
Keep the same meaning and content. Fix any grammar issues.
Make it sound more polished and professional, but keep it natural.
Do NOT add new information. Do NOT add a title or introduction.
Just output the rewritten text directly.

Text:
{text}"""


def strip_preamble(text: str) -> str:
    lines = text.strip().split("\n")
    skip = [r"^(here|below|sure|certainly|of course)", r"^(rewritten|revised|polished)"]
    while lines:
        first = lines[0].strip().lower()
        if any(re.match(p, first) for p in skip) or first.endswith(":"):
            lines.pop(0)
        else:
            break
    return "\n".join(lines).strip()


async def polish_via_api(session, text: str, config: dict) -> Optional[str]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}",
    }
    body = {
        "model": config["model"],
        "temperature": 0.7,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": POLISH_PROMPT.format(text=text)}],
    }
    try:
        url = f"{config['base_url']}/chat/completions"
        async with session.post(url, json=body, headers=headers, timeout=60) as resp:
            if resp.status != 200:
                err = await resp.text()
                print(f"  API {resp.status}: {err[:100]}", file=sys.stderr)
                return None
            data = await resp.json()
        result = data["choices"][0]["message"]["content"]
        return strip_preamble(result)
    except Exception as e:
        print(f"  API error ({config['provider']}/{config['model']}): {e}", file=sys.stderr)
        return None


async def main():
    import argparse
    import aiohttp

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dataset.jsonl")
    parser.add_argument("--target", type=int, default=17500)
    parser.add_argument("--corpus-dir", default="corpus")
    parser.add_argument("--local-only", action="store_true", help="100% local paraphrase, no API")
    parser.add_argument("--local-ratio", type=float, default=0.4,
                        help="Fraction of human_polished done by local paraphraser (default 0.4)")
    args = parser.parse_args()

    target = args.target
    output_path = Path(__file__).parent.parent / args.output
    local_ratio = 1.0 if args.local_only else args.local_ratio

    if not args.local_only and not API_MODELS:
        print("No DeepSeek/Qwen API keys found. Use --local-only or set keys in .env.local", file=sys.stderr)
        sys.exit(1)

    # ── Step 1: Keep existing ai + ai_polished ──
    kept = []
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry["label"] in (1, 2):
                    kept.append(entry)

    ai_count = sum(1 for e in kept if e["label"] == 1)
    ap_count = sum(1 for e in kept if e["label"] == 2)
    print(f"Keeping: ai={ai_count}, ai_polished={ap_count}", file=sys.stderr)

    # ── Step 2: Generate coherent human texts ──
    print(f"\n[1/3] Generating {target} coherent human texts...", file=sys.stderr)
    random.seed(42)
    extra = 5000  # extra source texts for polishing
    human_texts = load_human_texts(args.corpus_dir, target + extra)
    human_entries = [{"text": t, "label": 0, "label_name": "human"} for t in human_texts[:target]]
    print(f"  {len(human_entries)} human passages", file=sys.stderr)
    print(f"  Sample: {human_texts[0][:100]}...", file=sys.stderr)

    # ── Step 3: Generate human_polished (local + API in parallel) ──
    n_local = int(target * local_ratio)
    n_api = target - n_local
    print(f"\n[2/3] human_polished: {n_local} local + {n_api} API = {target} total", file=sys.stderr)

    source_texts = human_texts[:target + extra]
    local_sources = source_texts[:n_local + 1000]
    api_sources = source_texts[n_local:n_local + n_api + 2000]

    local_results = []
    api_results = []

    # Run local paraphrase in a thread (CPU-bound) while API runs async
    import concurrent.futures

    def run_local_batch():
        results = []
        for i, text in enumerate(local_sources):
            if len(results) >= n_local:
                break
            rate = random.uniform(0.2, 0.45)
            polished = local_paraphrase(text, synonym_rate=rate)
            if polished and len(polished.split()) >= 200:
                results.append({
                    "text": polished,
                    "label": 3,
                    "label_name": "human_polished",
                    "polisher_model": "local_spacy_paraphraser",
                    "polisher_provider": "local",
                })
            if len(results) % 500 == 0 and len(results) > 0:
                print(f"    local: {len(results)}/{n_local}", file=sys.stderr)
        print(f"  Local done: {len(results)}", file=sys.stderr)
        return results

    async def run_api_batch():
        if n_api <= 0 or not API_MODELS:
            return []
        results = []
        failures = 0
        semaphore = asyncio.Semaphore(20)  # 20 concurrent API calls

        async def process_one(text):
            nonlocal failures
            async with semaphore:
                config = random.choice(API_MODELS)
                result = await polish_via_api(session, text, config)
                if result and len(result.split()) >= 200:
                    return {
                        "text": result,
                        "label": 3,
                        "label_name": "human_polished",
                        "polisher_model": config["model"],
                        "polisher_provider": config["provider"],
                    }
                else:
                    failures += 1
                    return None

        async with aiohttp.ClientSession() as session:
            # Process in chunks of 100 to report progress
            for chunk_start in range(0, len(api_sources), 100):
                if len(results) >= n_api:
                    break
                chunk = api_sources[chunk_start:chunk_start + 100]
                tasks = [process_one(text) for text in chunk]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in batch_results:
                    if isinstance(r, dict) and len(results) < n_api:
                        results.append(r)
                print(f"    API: {len(results)}/{n_api} (failures: {failures})", file=sys.stderr)

        print(f"  API done: {len(results)} (failures: {failures})", file=sys.stderr)
        return results

    # Run both in parallel: local in thread pool, API async
    print("  Starting local + API in parallel...", file=sys.stderr)
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        local_future = loop.run_in_executor(executor, run_local_batch)
        api_future = run_api_batch()
        local_results, api_results = await asyncio.gather(local_future, api_future)

    human_polished_entries = local_results + api_results

    print(f"\n[3/3] Writing dataset...", file=sys.stderr)

    # ── Step 4: Write ──
    if output_path.exists():
        backup = str(output_path) + f".bak.{int(time.time())}"
        os.rename(str(output_path), backup)
        print(f"  Backed up to {backup}", file=sys.stderr)

    all_entries = kept + human_entries + human_polished_entries[:target]
    random.shuffle(all_entries)

    with open(output_path, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    counts = defaultdict(int)
    for e in all_entries:
        counts[e["label"]] += 1
    labels = ["human", "ai", "ai_polished", "human_polished"]
    print(f"\nDataset: {output_path}", file=sys.stderr)
    print(f"Total: {len(all_entries)}", file=sys.stderr)
    for i in range(4):
        print(f"  {labels[i]}: {counts[i]}", file=sys.stderr)

    # Polisher distribution
    polisher_counts = defaultdict(int)
    for e in all_entries:
        if e["label"] == 3:
            polisher_counts[e.get("polisher_provider", "unknown")] += 1
    if polisher_counts:
        print(f"\nhuman_polished sources:", file=sys.stderr)
        for provider, count in sorted(polisher_counts.items()):
            print(f"  {provider}: {count}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
