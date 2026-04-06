#!/usr/bin/env python3
"""Fast parallel adversarial AI text generation via DeepSeek API.

Takes existing AI text from dataset_v6.jsonl and applies 7 attack types
to create adversarial variants. 500 concurrent workers.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    python3 scripts/generate_adversarial_parallel.py --target 100000 --workers 500
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

SEED = 42
PROJECT_DIR = Path(__file__).parent.parent
DATASET = PROJECT_DIR / "dataset_v6.jsonl"

ATTACKS = {
    "back_translation": {
        "system": "You are a translator.",
        "prompt": "Translate the following English text to Chinese, then translate back to English. Only output the final English version.\n\nText: {text}",
        "weight": 0.15,
    },
    "paraphrase": {
        "system": "You are a writing assistant.",
        "prompt": "Paraphrase the following text to sound more natural and human. Keep the same meaning but change the wording significantly. Output only the paraphrased text.\n\nText: {text}",
        "weight": 0.20,
    },
    "casual_injection": {
        "system": "You are a college student rewriting text for a class.",
        "prompt": "Rewrite this text casually, like a college student would write it. Add contractions and informal phrasing. Output only the rewritten text.\n\nText: {text}",
        "weight": 0.15,
    },
    "synonym_substitution": {
        "system": "You are an editor.",
        "prompt": "Rewrite by replacing at least 40% of content words with synonyms. Keep sentence structure. Output only the rewritten text.\n\nText: {text}",
        "weight": 0.10,
    },
    "contraction_expansion": {
        "system": "You are a copy editor.",
        "prompt": "Rewrite using contractions wherever possible. Add casual transitions like 'anyway', 'so yeah'. Output only the rewritten text.\n\nText: {text}",
        "weight": 0.10,
    },
    "sandwich": {
        "system": "You are a writing assistant.",
        "prompt": "Sandwich the following AI text between a personal anecdote opening (2-3 sentences) and a casual reflection closing (2-3 sentences). Keep the middle mostly unchanged. Output the full combined text.\n\nText: {text}",
        "weight": 0.15,
    },
    "homoglyph_typo": {
        "system": "You are a text processor.",
        "prompt": "Modify the text: 1) Replace 5-10 letters with similar Unicode chars 2) Add 3-5 realistic typos 3) Randomly capitalize 2-3 words. Output only the modified text.\n\nText: {text}",
        "weight": 0.15,
    },
}


def text_hash(text):
    return hashlib.md5(text[:200].encode("utf-8", errors="replace")).hexdigest()


def apply_attack(client, source_text, attack_type):
    attack = ATTACKS[attack_type]
    prompt = attack["prompt"].format(text=source_text[:2000])
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": attack["system"]},
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
            temperature=0.7,
        )
        text = resp.choices[0].message.content.strip()
        text = " ".join(text.split())
        if len(text.split()) < 30:
            return None
        return {
            "text": text,
            "label": 1,
            "source": "deepseek_adversarial",
            "domain": "adversarial",
            "attack_type": attack_type,
        }
    except Exception as e:
        if "rate" in str(e).lower():
            time.sleep(2)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=500)
    args = parser.parse_args()

    random.seed(SEED)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # Load existing hashes + AI source texts
    seen = set()
    ai_texts = []
    print("Loading dataset...", file=sys.stderr)
    with open(DATASET) as f:
        for line in f:
            try:
                d = json.loads(line)
                seen.add(text_hash(d.get("text", "")))
                if d.get("label") == 1 and d.get("source", "").startswith("deepseek_clean"):
                    ai_texts.append(d["text"])
            except json.JSONDecodeError:
                pass

    if not ai_texts:
        # Fallback: use any AI text
        with open(DATASET) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("label") == 1:
                        ai_texts.append(d["text"])
                except json.JSONDecodeError:
                    pass

    print(f"  {len(seen):,} existing samples, {len(ai_texts):,} AI source texts", file=sys.stderr)
    random.shuffle(ai_texts)

    # Build task queue
    attack_types = list(ATTACKS.keys())
    tasks = []
    for i in range(args.target):
        at = attack_types[i % len(attack_types)]
        src = ai_texts[i % len(ai_texts)]
        tasks.append((src, at))

    random.shuffle(tasks)
    cost = (len(tasks) * 500 / 1e6) * 0.07 + (len(tasks) * 400 / 1e6) * 0.28
    print(f"\nGenerating {len(tasks):,} adversarial samples with {args.workers} workers", file=sys.stderr)
    print(f"  Estimated cost: ${cost:.2f}", file=sys.stderr)

    generated = []
    failed = 0
    batch = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for i, (src, at) in enumerate(tasks):
            f = pool.submit(apply_attack, client, src, at)
            futures[f] = i

        for f in as_completed(futures):
            result = f.result()
            done = len(generated) + failed

            if result is None:
                failed += 1
            else:
                h = text_hash(result["text"])
                if h not in seen:
                    seen.add(h)
                    generated.append(result)
                    batch.append(result)

            if len(batch) >= 50:
                with open(DATASET, "a") as out:
                    for s in batch:
                        out.write(json.dumps(s, ensure_ascii=False) + "\n")
                batch = []

            if done % 100 == 0:
                print(f"  {done:,}/{len(tasks):,} done, "
                      f"{len(generated):,} generated, {failed} failed",
                      file=sys.stderr)

    if batch:
        with open(DATASET, "a") as out:
            for s in batch:
                out.write(json.dumps(s, ensure_ascii=False) + "\n")

    attack_counts = Counter(s["attack_type"] for s in generated)
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"ADVERSARIAL GENERATION COMPLETE", file=sys.stderr)
    print(f"  Generated: {len(generated):,}", file=sys.stderr)
    print(f"  Failed: {failed}", file=sys.stderr)
    for at, c in attack_counts.most_common():
        print(f"    {at}: {c:,}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()
