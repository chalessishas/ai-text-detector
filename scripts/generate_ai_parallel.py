#!/usr/bin/env python3
"""Fast parallel AI text generation via DeepSeek API.

Generates AI text samples concurrently (10 workers) to fill domain gaps
in dataset_v6.jsonl. Much faster than the sequential build_dataset_v6.py.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    python3 scripts/generate_ai_parallel.py --target 30000
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

SEED = 42
PROJECT_DIR = Path(__file__).parent.parent
DATASET = PROJECT_DIR / "dataset_v6.jsonl"
MIN_WORDS = 80
MAX_WORDS = 500

STYLES = {
    "standard": "Write a {length}-word passage about: {topic}. Write naturally.",
    "formal": "Write a formal {length}-word passage about: {topic}. Professional language.",
    "casual": "Write a casual {length}-word passage about: {topic}. Like explaining to a friend.",
    "academic": "Write a scholarly {length}-word passage analyzing: {topic}. Academic tone.",
    "creative": "Write a creative {length}-word passage about: {topic}. Vivid language.",
}

# Domains that need more AI coverage (weak or missing in human data)
SUPPLEMENTARY_TOPICS = {
    "legal": [
        "contract law principles", "intellectual property rights", "employment law",
        "criminal defense procedures", "constitutional amendments", "tort liability",
        "corporate governance", "international trade law", "privacy regulations",
        "antitrust enforcement",
    ],
    "patent": [
        "semiconductor manufacturing process", "gene therapy delivery methods",
        "renewable energy storage systems", "autonomous vehicle navigation",
        "quantum computing error correction", "biodegradable polymer synthesis",
        "neural network optimization techniques", "water purification membrane design",
    ],
    "religious": [
        "meditation practices across traditions", "religious architecture symbolism",
        "comparative theology perspectives", "sacred texts interpretation methods",
        "pilgrimage traditions worldwide", "interfaith dialogue approaches",
    ],
    "philosophy": [
        "ethics of artificial intelligence", "consciousness and free will",
        "social contract theory modern applications", "existentialism daily life",
        "philosophy of science methods", "political philosophy distributive justice",
    ],
    "textbook": [
        "introduction to microeconomics", "organic chemistry reaction mechanisms",
        "cell biology mitosis and meiosis", "linear algebra vector spaces",
        "psychology cognitive development", "environmental science climate systems",
        "world history industrial revolution", "statistics hypothesis testing",
    ],
    "advertising": [
        "luxury watch brand marketing", "sustainable fashion campaign",
        "tech startup product launch", "real estate listing description",
        "restaurant opening promotion", "fitness app user acquisition",
    ],
    "translation": [
        "literary translation challenges", "technical manual translation",
        "legal document translation standards", "poetry translation theory",
    ],
}


def word_count(text):
    return len(text.split())


def text_hash(text):
    return hashlib.md5(text[:200].encode("utf-8", errors="replace")).hexdigest()


def load_existing_hashes():
    hashes = set()
    if DATASET.exists():
        with open(DATASET) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    hashes.add(text_hash(d.get("text", "")))
                except json.JSONDecodeError:
                    pass
    return hashes


def load_topics_from_human():
    topics = defaultdict(list)
    if DATASET.exists():
        with open(DATASET) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("label") == 0:
                        text = d["text"]
                        # Extract first 2 sentences as topic
                        sentences = text.split(".")
                        topic = ". ".join(sentences[:2]).strip()
                        words = topic.split()
                        if len(words) > 50:
                            topic = " ".join(words[:50])
                        if len(topic) > 20:
                            topics[d.get("domain", "mixed")].append(topic)
                except json.JSONDecodeError:
                    pass
    return topics


def generate_one(client, topic, style, domain):
    template = STYLES[style]
    length = random.randint(MIN_WORDS, MAX_WORDS)
    prompt = template.format(topic=topic, length=length)
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful writing assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
            temperature=random.choice([0.3, 0.7, 1.0]),
        )
        text = resp.choices[0].message.content.strip()
        text = " ".join(text.split())
        if word_count(text) < MIN_WORDS // 2:
            return None
        return {
            "text": text,
            "label": 1,
            "source": "deepseek_clean",
            "domain": domain,
            "prompt_style": style,
        }
    except Exception as e:
        if "rate" in str(e).lower():
            time.sleep(3)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=30000)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    random.seed(SEED)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    print("Loading existing hashes...", file=sys.stderr)
    seen = load_existing_hashes()
    print(f"  {len(seen):,} existing samples", file=sys.stderr)

    print("Loading topics from human samples...", file=sys.stderr)
    domain_topics = load_topics_from_human()
    # Add supplementary topics for weak domains
    for domain, topics in SUPPLEMENTARY_TOPICS.items():
        domain_topics[domain].extend(topics)

    total_topics = sum(len(v) for v in domain_topics.values())
    print(f"  {total_topics:,} topics across {len(domain_topics)} domains", file=sys.stderr)

    # Build task queue — distribute across domains proportionally
    tasks = []
    styles = list(STYLES.keys())
    domains = list(domain_topics.keys())
    per_domain = max(1, args.target // len(domains))

    for domain in domains:
        topics = domain_topics[domain]
        if not topics:
            continue
        n = min(per_domain, len(topics) * 3)
        for i in range(n):
            topic = topics[i % len(topics)]
            style = styles[i % len(styles)]
            tasks.append((topic, style, domain))

    random.shuffle(tasks)
    tasks = tasks[:args.target]
    print(f"\nGenerating {len(tasks):,} AI samples with {args.workers} workers...", file=sys.stderr)

    # Estimate cost
    cost = (len(tasks) * 80 / 1e6) * 0.07 + (len(tasks) * 400 / 1e6) * 0.28
    print(f"  Estimated cost: ${cost:.2f}", file=sys.stderr)

    generated = []
    failed = 0
    batch = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for i, (topic, style, domain) in enumerate(tasks):
            f = pool.submit(generate_one, client, topic, style, domain)
            futures[f] = (i, domain)

        for f in as_completed(futures):
            idx, domain = futures[f]
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

            # Write in batches of 50
            if len(batch) >= 50:
                with open(DATASET, "a") as out:
                    for s in batch:
                        out.write(json.dumps(s, ensure_ascii=False) + "\n")
                batch = []

            if done % 50 == 0:
                print(f"  {done:,}/{len(tasks):,} done, "
                      f"{len(generated):,} generated, {failed} failed",
                      file=sys.stderr)

    # Flush remaining
    if batch:
        with open(DATASET, "a") as out:
            for s in batch:
                out.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Summary
    domain_counts = Counter(s["domain"] for s in generated)
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"AI GENERATION COMPLETE", file=sys.stderr)
    print(f"  Generated: {len(generated):,}", file=sys.stderr)
    print(f"  Failed: {failed}", file=sys.stderr)
    print(f"  Domains: {len(domain_counts)}", file=sys.stderr)
    for d, c in domain_counts.most_common(20):
        print(f"    {d}: {c:,}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()
