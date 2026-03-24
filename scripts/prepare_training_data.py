#!/usr/bin/env python3
"""Prepare merged + noised training dataset for DeBERTa retraining.

Merges:
  1. Original dataset.jsonl (70K, 4-class)
  2. Augmented data (new genres from DeepSeek + RAID extracts)

Applies data noising (DEFACTIFY 2025 strategy):
  - 10% of words replaced with random junk (3-8 chars)
  - Creates a noised copy alongside clean data

Outputs:
  - dataset_merged.jsonl (clean)
  - dataset_merged_noised.jsonl (with 10% junk injection)

Run: python3 scripts/prepare_training_data.py
"""

import json
import os
import random
import string
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
ORIGINAL = PROJECT_DIR / "dataset.jsonl"
AUGMENTED = PROJECT_DIR / "dataset_augmented.jsonl"
RAID_EXTRACT = PROJECT_DIR / "dataset_raid_extract.jsonl"
OUTPUT_CLEAN = PROJECT_DIR / "dataset_merged.jsonl"
OUTPUT_NOISED = PROJECT_DIR / "dataset_merged_noised.jsonl"

NOISE_RATIO = 0.10  # 10% of words replaced with junk


def random_junk(min_len=3, max_len=8):
    """Generate a random junk word (3-8 lowercase chars)."""
    length = random.randint(min_len, max_len)
    return "".join(random.choices(string.ascii_lowercase, k=length))


def inject_noise(text, ratio=NOISE_RATIO):
    """Replace `ratio` fraction of words with random junk."""
    words = text.split()
    if len(words) < 10:
        return text

    n_replace = max(1, int(len(words) * ratio))
    indices = random.sample(range(len(words)), n_replace)
    for idx in indices:
        words[idx] = random_junk()
    return " ".join(words)


def load_jsonl(path):
    """Load JSONL file, return list of dicts."""
    entries = []
    if not path.exists():
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def normalize_label(entry):
    """Ensure entry has correct label and label_name fields.

    Augmented data may only have 'ai' (label=1) or 'human' (label=0).
    Map to our 4-class scheme.
    """
    label_name = entry.get("label_name", "")
    label = entry.get("label", -1)

    # Validate known labels
    valid = {"human": 0, "ai": 1, "ai_polished": 2, "human_polished": 3}
    if label_name in valid:
        entry["label"] = valid[label_name]
        return entry

    # Fallback by numeric label
    reverse = {0: "human", 1: "ai", 2: "ai_polished", 3: "human_polished"}
    if label in reverse:
        entry["label_name"] = reverse[label]
        return entry

    return None  # Unknown, skip


def main():
    print("=" * 60)
    print("Prepare Training Data: Merge + Noise")
    print("=" * 60)

    # Load original
    print(f"\nLoading original: {ORIGINAL}")
    original = load_jsonl(ORIGINAL)
    print(f"  {len(original)} entries")

    # Load augmented
    print(f"Loading augmented: {AUGMENTED}")
    augmented = load_jsonl(AUGMENTED)
    print(f"  {len(augmented)} entries")

    # Load RAID extract
    print(f"Loading RAID extract: {RAID_EXTRACT}")
    raid = load_jsonl(RAID_EXTRACT)
    print(f"  {len(raid)} entries")

    # Normalize labels
    merged = []
    skipped = 0
    for entry in original + augmented + raid:
        normalized = normalize_label(entry)
        if normalized and len(normalized.get("text", "")) > 50:
            merged.append(normalized)
        else:
            skipped += 1

    print(f"\nMerged: {len(merged)} entries ({skipped} skipped)")

    # Deduplicate by text
    seen = set()
    deduped = []
    for entry in merged:
        text_key = entry["text"][:200]  # First 200 chars as key
        if text_key not in seen:
            seen.add(text_key)
            deduped.append(entry)
    print(f"After dedup: {len(deduped)} entries ({len(merged) - len(deduped)} duplicates)")
    merged = deduped

    # Stats
    label_counts = Counter(e["label_name"] for e in merged)
    style_counts = Counter(e.get("style", "(none)") for e in merged)
    provider_counts = Counter(e.get("provider", "(none)") for e in merged)

    print(f"\n--- Label distribution ---")
    for k, v in label_counts.most_common():
        print(f"  {k}: {v}")

    print(f"\n--- Styles (top 15 of {len(style_counts)}) ---")
    for k, v in style_counts.most_common(15):
        print(f"  {k}: {v}")

    print(f"\n--- Providers ---")
    for k, v in provider_counts.most_common():
        print(f"  {k}: {v}")

    # Write clean merged
    print(f"\nWriting clean: {OUTPUT_CLEAN}")
    random.shuffle(merged)
    with open(OUTPUT_CLEAN, "w") as f:
        for entry in merged:
            f.write(json.dumps(entry) + "\n")
    size_mb = OUTPUT_CLEAN.stat().st_size / (1024 * 1024)
    print(f"  {len(merged)} entries, {size_mb:.0f} MB")

    # Write noised version
    print(f"\nWriting noised ({NOISE_RATIO*100:.0f}% junk injection): {OUTPUT_NOISED}")
    noised = []
    for entry in merged:
        noised_entry = entry.copy()
        noised_entry["text"] = inject_noise(entry["text"], NOISE_RATIO)
        noised.append(noised_entry)

    random.shuffle(noised)
    with open(OUTPUT_NOISED, "w") as f:
        for entry in noised:
            f.write(json.dumps(entry) + "\n")
    size_mb = OUTPUT_NOISED.stat().st_size / (1024 * 1024)
    print(f"  {len(noised)} entries, {size_mb:.0f} MB")

    # Verify noise (compare before shuffling noised)
    print(f"\n--- Noise verification (3 random samples) ---")
    for i in random.sample(range(len(merged)), min(3, len(merged))):
        orig_words = merged[i]["text"].split()
        noised_text = inject_noise(merged[i]["text"], NOISE_RATIO)
        noised_words = noised_text.split()
        diffs = sum(1 for a, b in zip(orig_words, noised_words) if a != b)
        total = min(len(orig_words), len(noised_words))
        print(f"  Sample {i}: {diffs}/{total} words changed ({diffs/max(total,1)*100:.0f}%)")

    print(f"\n{'='*60}")
    print(f"Done. Files ready for training:")
    print(f"  Clean:  {OUTPUT_CLEAN}")
    print(f"  Noised: {OUTPUT_NOISED}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
