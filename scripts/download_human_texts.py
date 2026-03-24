"""Download human-written text datasets for AI detector training.

Task 1: arXiv abstracts (STEM academic writing) - gfissore/arxiv-abstracts-2021
Task 2: Student essays (education domain) - ivypanda-essays
Task 3: HC3 human answers - Hello-SimpleAI/HC3 (raw jsonl)

Usage: /usr/bin/python3 scripts/download_human_texts.py
"""

import json
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

random.seed(42)


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records to {path}")


# ─── Task 1: arXiv abstracts ───────────────────────────────────────────

def download_arxiv():
    print("\n=== Task 1: arXiv abstracts ===")
    out_path = os.path.join(DATA_DIR, "arxiv_human.jsonl")

    from datasets import load_dataset
    print("  Loading gfissore/arxiv-abstracts-2021 (streaming)...")
    ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)

    TARGET = 5000
    category_buckets = {}
    count = 0

    for item in ds:
        text = item.get("abstract", "").strip()
        if not text or len(text) < 100:
            continue

        cats_raw = item.get("categories", "[]")
        if isinstance(cats_raw, str):
            try:
                cats_list = json.loads(cats_raw.replace("'", '"'))
            except:
                cats_list = [cats_raw]
        elif isinstance(cats_raw, list):
            cats_list = cats_raw
        else:
            cats_list = ["unknown"]

        primary_cat = cats_list[0] if cats_list else "unknown"
        cat_prefix = primary_cat.split(".")[0] if "." in primary_cat else primary_cat

        # Extract year from arXiv ID (e.g., "2101.12345" -> 2021)
        aid = item.get("id", "")
        year = None
        if aid and "." in aid:
            try:
                prefix = aid.split(".")[0]
                if len(prefix) == 4 and prefix.isdigit():
                    y = int(prefix[:2])
                    year = 2000 + y if y < 50 else 1900 + y
            except (ValueError, IndexError):
                pass

        if cat_prefix not in category_buckets:
            category_buckets[cat_prefix] = []
        category_buckets[cat_prefix].append({
            "text": text,
            "source": "arxiv",
            "domain": primary_cat,
            "year": year
        })
        count += 1

        if count % 10000 == 0:
            print(f"    Collected {count} abstracts across {len(category_buckets)} categories...")

        if count >= TARGET * 3:
            break

    if count == 0:
        print("  ERROR: No records collected")
        return False

    # Sample evenly across categories
    records = []
    per_cat = max(1, TARGET // len(category_buckets))
    for cat, items in sorted(category_buckets.items()):
        sample_size = min(per_cat, len(items))
        records.extend(random.sample(items, sample_size))

    # Fill remaining from largest buckets
    if len(records) < TARGET:
        remaining = TARGET - len(records)
        used_texts = set(r["text"][:50] for r in records)
        pool = [r for items in category_buckets.values()
                for r in items if r["text"][:50] not in used_texts]
        if pool:
            records.extend(random.sample(pool, min(remaining, len(pool))))

    random.shuffle(records)
    records = records[:TARGET]
    save_jsonl(records, out_path)

    cats_summary = {}
    for r in records:
        c = r["domain"].split(".")[0] if "." in r["domain"] else r["domain"]
        cats_summary[c] = cats_summary.get(c, 0) + 1
    top_cats = dict(sorted(cats_summary.items(), key=lambda x: -x[1])[:20])
    print(f"  Category distribution ({len(cats_summary)} categories): {top_cats}")
    return True


# ─── Task 2: Student essays ───────────────────────────────────────────

def download_student_essays():
    print("\n=== Task 2: Student essays ===")
    out_path = os.path.join(DATA_DIR, "student_essays_human.jsonl")

    from datasets import load_dataset

    all_records = []

    # Source 1: essays-with-instructions
    try:
        print("  Loading ChristophSchuhmann/essays-with-instructions...")
        ds = load_dataset("ChristophSchuhmann/essays-with-instructions", split="train")
        for item in ds:
            text = item.get("essay", item.get("text", "")).strip()
            if text and len(text) > 50:
                all_records.append({
                    "text": text,
                    "source": "essays-with-instructions",
                    "domain": "education"
                })
        print(f"    Got {len(all_records)} from essays-with-instructions")
    except Exception as e:
        print(f"    essays-with-instructions failed: {e}")

    # Source 2: ivypanda essays (streaming, grab up to 10k)
    try:
        print("  Loading qwedsacf/ivypanda-essays (streaming)...")
        ds = load_dataset("qwedsacf/ivypanda-essays", split="train", streaming=True)
        ivy_count = 0
        for item in ds:
            text = item.get("TEXT", item.get("text", "")).strip()
            if text and len(text) > 100:
                all_records.append({
                    "text": text,
                    "source": "ivypanda",
                    "domain": "education"
                })
                ivy_count += 1
            if ivy_count >= 10000:
                break
        print(f"    Got {ivy_count} from ivypanda")
    except Exception as e:
        print(f"    ivypanda failed: {e}")

    # Source 3: HuggingFaceH4/no_robots (human-written responses)
    try:
        print("  Loading HuggingFaceH4/no_robots...")
        ds = load_dataset("HuggingFaceH4/no_robots", split="train")
        nr_count = 0
        for item in ds:
            messages = item.get("messages", [])
            for msg in messages:
                if msg.get("role") == "assistant":
                    text = msg.get("content", "").strip()
                    if text and len(text) > 100:
                        all_records.append({
                            "text": text,
                            "source": "no_robots",
                            "domain": "education",
                            "category": item.get("category", "")
                        })
                        nr_count += 1
        print(f"    Got {nr_count} from no_robots")
    except Exception as e:
        print(f"    no_robots failed: {e}")

    if all_records:
        random.shuffle(all_records)
        save_jsonl(all_records, out_path)
        sources = {}
        for r in all_records:
            s = r["source"]
            sources[s] = sources.get(s, 0) + 1
        print(f"  Source distribution: {sources}")
        return True

    print("  WARNING: Could not download student essays from any source")
    return False


# ─── Task 3: HC3 human answers ────────────────────────────────────────

def download_hc3():
    print("\n=== Task 3: HC3 human answers ===")
    out_path = os.path.join(DATA_DIR, "hc3_human.jsonl")

    from huggingface_hub import hf_hub_download

    configs = ["all", "finance", "medicine", "open_qa", "wiki_csai", "reddit_eli5"]
    records = []

    for config in configs:
        fname = f"{config}.jsonl"
        try:
            print(f"  Downloading {fname}...")
            path = hf_hub_download("Hello-SimpleAI/HC3", fname, repo_type="dataset")
            with open(path, encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    question = item.get("question", "")
                    human_raw = item.get("human_answers", "[]")

                    # Parse human_answers (may be string or list)
                    if isinstance(human_raw, str):
                        try:
                            human_answers = json.loads(human_raw)
                        except:
                            human_answers = [human_raw] if human_raw else []
                    elif isinstance(human_raw, list):
                        human_answers = human_raw
                    else:
                        human_answers = []

                    for ans in human_answers:
                        text = ans.strip() if isinstance(ans, str) else ""
                        if text and len(text) > 30:
                            records.append({
                                "text": text,
                                "source": "hc3",
                                "domain": config,
                                "question": question[:200]
                            })
            print(f"    {config}: running total {len(records)} human answers")
        except Exception as e:
            print(f"    {config} failed: {e}")

    if records:
        random.shuffle(records)
        save_jsonl(records, out_path)

        domain_dist = {}
        for r in records:
            d = r["domain"]
            domain_dist[d] = domain_dist.get(d, 0) + 1
        print(f"  Domain distribution: {domain_dist}")
        return True

    print("  WARNING: Could not download HC3 dataset")
    return False


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Human Text Dataset Downloader")
    print("=" * 60)

    results = {}
    results["arxiv"] = download_arxiv()
    results["student_essays"] = download_student_essays()
    results["hc3"] = download_hc3()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    files = {
        "arxiv": "arxiv_human.jsonl",
        "student_essays": "student_essays_human.jsonl",
        "hc3": "hc3_human.jsonl"
    }

    for name, fname in files.items():
        path = os.path.join(DATA_DIR, fname)
        if results.get(name) and os.path.exists(path):
            size = os.path.getsize(path)
            with open(path) as f:
                count = sum(1 for _ in f)
            print(f"  {name}: {count} records ({size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  {name}: FAILED")

    all_ok = all(results.values())
    print(f"\nOverall: {'All datasets downloaded successfully!' if all_ok else 'Some datasets failed'}")
    sys.exit(0 if all_ok else 1)
