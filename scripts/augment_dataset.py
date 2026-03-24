#!/usr/bin/env python3
"""Augment existing dataset with new genres using DeepSeek API.

Generates AI samples for the 14 new prompt styles that are in dataset_config.py
but missing from the actual 70K dataset. Uses only DeepSeek (the only available API key).

Also extracts human + AI samples from the RAID benchmark across 11 domains.

Usage:
    python3 scripts/augment_dataset.py --mode generate  # Generate new AI samples
    python3 scripts/augment_dataset.py --mode raid       # Extract from RAID
    python3 scripts/augment_dataset.py --mode both       # Do both
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

# Load .env
env_path = Path(__file__).parent.parent / ".env.local"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from dataset_config import PROMPT_STYLES, TOPICS, LENGTHS

# --- Config ---

OUTPUT_FILE = Path(__file__).parent.parent / "dataset_augmented.jsonl"

# Styles already in the 70K dataset
EXISTING_STYLES = {"direct", "academic", "casual", "persona", "anti_detect", "anti_detect_v2", "rewrite"}

# New styles to generate
NEW_STYLES = {k: v for k, v in PROMPT_STYLES.items() if k not in EXISTING_STYLES}

# New topics (indices 20+) not in original dataset
NEW_TOPICS = TOPICS[20:]  # STEM, business, history, medicine, daily life, arts, law

# Also sample some original topics to maintain balance
ORIGINAL_TOPICS = TOPICS[:20]

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


async def generate_with_deepseek(prompt: str, temperature: float = 0.7):
    """Call DeepSeek API to generate text."""
    import httpx

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        return None

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(
                f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": DEEPSEEK_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 2000,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  DeepSeek error: {e}", file=sys.stderr)
            return None


async def generate_new_styles(target_per_style: int = 200):
    """Generate AI samples for each new prompt style."""
    print(f"\n=== Generating {len(NEW_STYLES)} new styles × {target_per_style} samples ===")
    print(f"Styles: {list(NEW_STYLES.keys())}")

    generated = 0
    errors = 0

    with open(OUTPUT_FILE, "a") as f:
        for style_name, style_template in NEW_STYLES.items():
            print(f"\n--- Style: {style_name} ---")
            style_count = 0

            # Mix old and new topics
            all_topics = NEW_TOPICS + random.sample(ORIGINAL_TOPICS, min(10, len(ORIGINAL_TOPICS)))
            random.shuffle(all_topics)

            for topic in all_topics:
                if style_count >= target_per_style:
                    break

                for length_name, length_info in LENGTHS.items():
                    if style_count >= target_per_style:
                        break

                    # Build prompt
                    prompt = style_template.format(
                        topic=topic,
                        length=length_info["words"],
                        persona=random.choice([
                            "software engineer", "data scientist", "product manager",
                            "journalist", "professor", "consultant"
                        ]),
                        source_text="N/A"  # For rewrite style
                    )

                    temp = random.choice([0.3, 0.7, 1.0])
                    text = await generate_with_deepseek(prompt, temperature=temp)

                    if text and len(text) > 100:
                        entry = {
                            "text": text,
                            "label": 1,
                            "label_name": "ai",
                            "model": DEEPSEEK_MODEL,
                            "provider": "deepseek",
                            "style": style_name,
                            "topic": topic,
                            "temperature": temp,
                            "length": length_name,
                            "tier": "mid",
                        }
                        f.write(json.dumps(entry) + "\n")
                        f.flush()
                        style_count += 1
                        generated += 1

                        if generated % 10 == 0:
                            ts = time.strftime("%H:%M:%S")
                            print(f"  [{ts}] Generated {generated} total ({style_name}: {style_count}/{target_per_style})")
                    else:
                        errors += 1

                    # Rate limit: ~60 req/min for DeepSeek
                    await asyncio.sleep(1.0)

    print(f"\n=== Done: {generated} generated, {errors} errors ===")
    return generated


def extract_raid_samples(target_per_domain: int = 500):
    """Extract balanced human + AI samples from RAID across all domains."""
    from datasets import load_dataset

    print(f"\n=== Extracting RAID samples ({target_per_domain} per domain) ===")

    ds = load_dataset("liamdugan/raid", split="train", streaming=True)

    domain_counts = {}
    extracted = 0

    with open(OUTPUT_FILE, "a") as f:
        for ex in ds:
            domain = ex["domain"]
            model = ex["model"]
            attack = ex["attack"]

            # Only take clean (no attack) samples
            if attack != "none":
                continue

            # Track per-domain counts
            if domain not in domain_counts:
                domain_counts[domain] = {"human": 0, "ai": 0}

            is_human = model == "human"
            key = "human" if is_human else "ai"

            # Balance: half human, half AI per domain
            half = target_per_domain // 2
            if domain_counts[domain][key] >= half:
                continue

            text = ex.get("generation", "")
            if not text or len(text) < 50:
                continue

            entry = {
                "text": text,
                "label": 0 if is_human else 1,
                "label_name": "human" if is_human else "ai",
                "model": model if not is_human else "",
                "provider": "raid",
                "style": f"raid_{domain}",
                "topic": ex.get("title", domain),
                "temperature": 0,
                "length": "medium",
                "tier": "external",
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()

            domain_counts[domain][key] += 1
            extracted += 1

            if extracted % 100 == 0:
                ts = time.strftime("%H:%M:%S")
                print(f"  [{ts}] Extracted {extracted} total")
                for d, c in sorted(domain_counts.items()):
                    print(f"    {d}: human={c['human']}, ai={c['ai']}")

            # Check if all domains are full
            all_full = True
            for d, c in domain_counts.items():
                if c["human"] < half or c["ai"] < half:
                    all_full = False
                    break
            if all_full and len(domain_counts) >= 8:
                break

    print(f"\n=== RAID extraction done: {extracted} samples ===")
    for d, c in sorted(domain_counts.items()):
        print(f"  {d}: human={c['human']}, ai={c['ai']}")

    return extracted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "raid", "both"], default="both")
    parser.add_argument("--target-per-style", type=int, default=200)
    parser.add_argument("--target-per-domain", type=int, default=500)
    args = parser.parse_args()

    print(f"Output file: {OUTPUT_FILE}")
    print(f"Mode: {args.mode}")

    if args.mode in ("generate", "both"):
        asyncio.run(generate_new_styles(args.target_per_style))

    if args.mode in ("raid", "both"):
        extract_raid_samples(args.target_per_domain)


if __name__ == "__main__":
    main()
