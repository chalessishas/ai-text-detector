#!/usr/bin/env python3
"""Prepare academic human text data for DeBERTa training.

Reads downloaded human text from scripts/data/ and formats it for
merging into the main training dataset.

Sources:
  - arXiv abstracts (STEM academic writing)
  - Student essays (education domain, ASAP-style)
  - HC3 human answers (Q&A domain)

Each entry becomes label=0 (human) in the training set.

For each human entry, we also generate a prompt that can be used
to create an AI counterpart (label=1) with generate_dataset.py.

Usage:
    python3 scripts/prepare_academic_data.py
"""

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT = PROJECT_DIR / "dataset_academic_human.jsonl"
PROMPTS_OUTPUT = SCRIPT_DIR / "academic_prompts.jsonl"

# Target: balanced across sources, max 5000 per source
MAX_PER_SOURCE = 5000
MIN_TEXT_LENGTH = 200  # chars
MAX_TEXT_LENGTH = 8000  # chars


def load_arxiv():
    """Load arXiv abstracts from downloaded data."""
    path = DATA_DIR / "arxiv_human.jsonl"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return []

    entries = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            text = d.get("text", "").strip()
            if len(text) < MIN_TEXT_LENGTH or len(text) > MAX_TEXT_LENGTH:
                continue
            entries.append({
                "text": text,
                "label": 0,
                "label_name": "human",
                "source": "arxiv",
                "domain": d.get("domain", "academic"),
            })

    if len(entries) > MAX_PER_SOURCE:
        entries = random.sample(entries, MAX_PER_SOURCE)

    print(f"  arXiv: {len(entries)} entries")
    return entries


def load_student_essays():
    """Load student essays from downloaded data."""
    path = DATA_DIR / "student_essays_human.jsonl"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return []

    entries = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            text = d.get("text", "").strip()
            if len(text) < MIN_TEXT_LENGTH or len(text) > MAX_TEXT_LENGTH:
                continue
            entries.append({
                "text": text,
                "label": 0,
                "label_name": "human",
                "source": "student_essay",
                "domain": d.get("domain", "education"),
            })

    if len(entries) > MAX_PER_SOURCE:
        entries = random.sample(entries, MAX_PER_SOURCE)

    print(f"  Student essays: {len(entries)} entries")
    return entries


def load_hc3():
    """Load HC3 human answers from downloaded data."""
    path = DATA_DIR / "hc3_human.jsonl"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return []

    entries = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            text = d.get("text", "").strip()
            if len(text) < MIN_TEXT_LENGTH or len(text) > MAX_TEXT_LENGTH:
                continue
            entries.append({
                "text": text,
                "label": 0,
                "label_name": "human",
                "source": "hc3",
                "domain": d.get("domain", "qa"),
            })

    if len(entries) > MAX_PER_SOURCE:
        entries = random.sample(entries, MAX_PER_SOURCE)

    print(f"  HC3: {len(entries)} entries")
    return entries


def extract_prompts(entries):
    """Extract writing prompts from human text for AI generation.

    For each human text, create a prompt that could generate
    similar content, enabling paired human/AI training data.
    """
    prompts = []
    for entry in entries:
        text = entry["text"]
        # Use first sentence as topic hint
        first_sent = text.split(".")[0].strip()
        if len(first_sent) < 20:
            first_sent = " ".join(text.split()[:30])

        prompts.append({
            "prompt": f"Write a {entry['domain']} essay or passage about: {first_sent}",
            "source_domain": entry["domain"],
            "source": entry["source"],
            "target_length": "medium" if len(text) < 2000 else "long",
        })

    return prompts


def main():
    random.seed(42)

    print("Loading academic human text data...")

    all_entries = []
    all_entries.extend(load_arxiv())
    all_entries.extend(load_student_essays())
    all_entries.extend(load_hc3())

    if not all_entries:
        print("ERROR: No data found in scripts/data/. Run download scripts first.")
        sys.exit(1)

    # Shuffle
    random.shuffle(all_entries)

    # Write human entries
    with open(OUTPUT, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_entries)} human entries to {OUTPUT}")

    # Extract prompts for AI generation
    prompts = extract_prompts(all_entries)
    with open(PROMPTS_OUTPUT, "w") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Wrote {len(prompts)} AI generation prompts to {PROMPTS_OUTPUT}")

    # Summary
    source_counts = Counter(e["source"] for e in all_entries)
    domain_counts = Counter(e["domain"] for e in all_entries)
    print(f"\nBy source: {dict(source_counts)}")
    print(f"By domain: {dict(domain_counts)}")


if __name__ == "__main__":
    main()
