#!/usr/bin/env python3
"""Build dataset_v6 supplement: Chinese multi-family AI samples (Path C).

Implements Path C from 2026-04-17 Research Loop 02:18
(docs/research/2026-04-17-research-loop-0218-deepseek-viability.md).

Target: ~24K samples each × 5 families = ~120K Chinese AI samples to
complement the existing DeepSeek-heavy dataset_v6.jsonl. After merge the
AI side becomes 60% DeepSeek + 40% other families — better cross-family
generalization for Chinese-market AI detection.

Cost estimate (500 chars × 120K samples ≈ 60M tokens):
    gpt4o-mini     ~$10 @ $0.15/M in + $0.60/M out
    claude-haiku   ~$48 @ $1/M in + $5/M out
    qwen3-max      ~$2  @ ~$0.02/M (Alibaba)
    ernie-5        ~$4  @ $0.10/M (Baidu)
    kimi-k2.5      ~$5  @ $0.15/M (Moonshot)
    Total          ~$70

Usage:
    python3 scripts/build_dataset_v6_supplement.py --vendor gpt4o-mini --target 24000
    python3 scripts/build_dataset_v6_supplement.py --vendor claude-haiku --target 24000
    python3 scripts/build_dataset_v6_supplement.py --vendor qwen3-max --target 24000
    python3 scripts/build_dataset_v6_supplement.py --vendor ernie-5 --target 24000
    python3 scripts/build_dataset_v6_supplement.py --vendor kimi-k2.5 --target 24000
    python3 scripts/build_dataset_v6_supplement.py --merge  # append all into dataset_v6.jsonl

Requires: pip install openai anthropic python-dotenv tqdm
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore
        return it

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_CHARS = 100   # Chinese uses char count, not word count
MAX_CHARS = 500

PROJECT_DIR = Path(__file__).parent.parent
ENV_PATH = PROJECT_DIR / ".env.local"
OUTPUT_DIR = PROJECT_DIR / "supplement_v6_shards"
FINAL_DATASET = PROJECT_DIR / "dataset_v6.jsonl"

# Chinese prompt templates — each produces a passage of ~{length} Chinese characters.
AI_PROMPT_STYLES_ZH = {
    "standard": (
        "请写一篇约{length}字的短文，主题如下。语言自然流畅，内容翔实。\n\n主题：{topic}"
    ),
    "formal": (
        "请就以下主题撰写一篇约{length}字的正式文章，语言专业，结构清晰，逻辑严谨。\n\n主题：{topic}"
    ),
    "casual": (
        "请用随意口语的风格写一篇约{length}字的短文，就像在和朋友聊天一样，自然随意。\n\n主题：{topic}"
    ),
    "academic": (
        "请以学术分析的语气撰写一篇约{length}字的文章，客观探讨以下主题，引用一般性研究结论。\n\n主题：{topic}"
    ),
    "creative": (
        "请用创意写作的方式写一篇约{length}字的文章，语言生动形象，句式富于变化，富有感染力。\n\n主题：{topic}"
    ),
}

# Fallback topics used when dataset_v6.jsonl is unavailable.
FALLBACK_TOPICS = [
    "人工智能对现代教育的影响",
    "城市化进程中的环境保护问题",
    "社交媒体与青少年心理健康",
    "传统文化与现代生活方式的融合",
    "气候变化对农业生产的挑战",
    "远程办公的优势与挑战",
    "电子支付普及的社会影响",
    "老龄化社会的医疗保障问题",
    "网络谣言的传播与危害",
    "创业精神在经济发展中的作用",
    "阅读习惯变化与数字出版",
    "食品安全与消费者权益保护",
]

# ---------------------------------------------------------------------------
# Vendor configurations
# ---------------------------------------------------------------------------

# sdk_type distinguishes OpenAI-compatible endpoints from Anthropic's native SDK.
# Do NOT treat these uniformly in get_client():
#   sdk_type="openai"    → openai.OpenAI(base_url=cfg["base_url"], api_key=...)
#   sdk_type="anthropic" → anthropic.Anthropic(api_key=...)  (different method names + schema)
#
# Quality caveat: Expect asymmetric Chinese generation quality across vendors —
# different training corpora, RL objectives, and tokenizers produce stylistically
# distinct outputs even on identical prompts. After generation, sample ~50 outputs
# per vendor. If any vendor produces obviously low-quality or translated-looking
# Chinese, exclude that shard or regenerate with a different model_id — mixing
# low-quality samples risks teaching the detector vendor-specific stylistic
# artifacts rather than generic AI-text signals.
VENDORS: Dict[str, Dict] = {
    "gpt4o-mini": {
        "sdk_type": "openai",
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "model_id": "gpt-4o-mini",
        "source_tag": "gpt4o_mini_chinese",
    },
    "claude-haiku": {
        "sdk_type": "anthropic",  # NOT OpenAI-compatible — use anthropic SDK
        "base_url": None,         # anthropic SDK handles endpoint internally
        "env_key": "ANTHROPIC_API_KEY",
        "model_id": "claude-haiku-4-5-20251001",
        "source_tag": "claude_haiku_chinese",
    },
    "qwen3-max": {
        "sdk_type": "openai",  # Dashscope exposes OpenAI-compat mode
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "model_id": "qwen-max",
        "source_tag": "qwen3_max_chinese",
    },
    "ernie-5": {
        "sdk_type": "openai",  # Baidu Qianfan v2 exposes OpenAI-compat
        "base_url": "https://qianfan.baidubce.com/v2",
        "env_key": "ERNIE_API_KEY",
        "model_id": "ernie-speed-128k",
        "source_tag": "ernie5_chinese",
    },
    "kimi-k2.5": {
        "sdk_type": "openai",  # Moonshot exposes OpenAI-compat
        "base_url": "https://api.moonshot.cn/v1",
        "env_key": "MOONSHOT_API_KEY",
        "model_id": "moonshot-v1-8k",
        "source_tag": "kimi_k25_chinese",
    },
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def text_hash(text: str) -> str:
    return hashlib.md5(text[:200].encode("utf-8", errors="replace")).hexdigest()


def char_count(text: str) -> int:
    """Count Chinese characters + other non-space chars (not split()-based)."""
    return len(text.replace(" ", ""))


def clean_text(text: str) -> str:
    return " ".join(text.split()).strip()


def extract_topic_from_text(text: str) -> str:
    """Extract a rough topic from a text (first 2 sentences or ~50 chars)."""
    sentences = text.split("。")
    topic = "。".join(sentences[:2]).strip()
    if len(topic) > 80:
        topic = topic[:80]
    return topic + ("。" if topic and not topic.endswith("。") else "")


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def get_client(vendor: str):
    """Return an initialized API client for the chosen vendor.

    Handles the sdk_type split: Anthropic uses its own SDK; all others use
    the OpenAI SDK pointed at the vendor's base_url.
    """
    cfg = VENDORS[vendor]
    if load_dotenv:
        load_dotenv(ENV_PATH)

    api_key_val = os.environ.get(cfg["env_key"])
    if not api_key_val:
        print(f"ERROR: {cfg['env_key']} not set in environment or .env.local", file=sys.stderr)
        sys.exit(1)

    if cfg["sdk_type"] == "anthropic":
        try:
            import anthropic as ant
        except ImportError:
            print("ERROR: pip install anthropic (required for claude-haiku)", file=sys.stderr)
            sys.exit(1)
        return ant.Anthropic(api_key=api_key_val)

    # Default: OpenAI-compatible
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key_val, base_url=cfg["base_url"])


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------


def generate_sample(
    vendor: str,
    client,
    topic: str,
    style: str,
    domain: str,
) -> Optional[dict]:
    """Generate one Chinese AI sample. Returns None on failure after 3 retries.

    Return schema matches dataset_v6.jsonl:
        {"text": str, "label": 1, "source": source_tag, "domain": str, "prompt_style": str}
    """
    cfg = VENDORS[vendor]
    prompt_template = AI_PROMPT_STYLES_ZH[style]
    target_length = random.randint(MIN_CHARS, MAX_CHARS)
    prompt = prompt_template.format(topic=topic, length=target_length)

    for attempt in range(3):
        try:
            if cfg["sdk_type"] == "anthropic":
                response = client.messages.create(
                    model=cfg["model_id"],
                    max_tokens=900,
                    temperature=random.choice([0.3, 0.7, 1.0]),
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()
            else:
                response = client.chat.completions.create(
                    model=cfg["model_id"],
                    messages=[
                        {"role": "system", "content": "你是一个有帮助的写作助手。"},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=900,
                    temperature=random.choice([0.3, 0.7, 1.0]),
                )
                text = response.choices[0].message.content.strip()

            text = clean_text(text)
            if char_count(text) >= MIN_CHARS // 2:
                return {
                    "text": text,
                    "label": 1,
                    "source": cfg["source_tag"],
                    "domain": domain,
                    "prompt_style": style,
                }
        except Exception as e:
            wait = 2 * (attempt + 1)
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(wait)
            elif attempt == 2:
                return None

    return None


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


def load_topics_from_dataset() -> List[tuple]:
    """Load (topic, domain) pairs from existing dataset_v6.jsonl human samples."""
    if not FINAL_DATASET.exists():
        return []
    topics = []
    with open(FINAL_DATASET) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if d.get("label") == 0:
                    topic = extract_topic_from_text(d["text"])
                    if len(topic) > 10:
                        topics.append((topic, d.get("domain", "mixed")))
            except json.JSONDecodeError:
                continue
    return topics


def run_vendor(vendor: str, target: int) -> None:
    """Generate `target` Chinese AI samples for the given vendor."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    shard_path = OUTPUT_DIR / f"{vendor}.jsonl"

    # Resume: count already-written samples
    already_written = 0
    seen_hashes: set = set()
    if shard_path.exists():
        with open(shard_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    seen_hashes.add(text_hash(d.get("text", "")))
                    already_written += 1
                except json.JSONDecodeError:
                    continue
        print(f"[{vendor}] Resuming: {already_written}/{target} already written")

    remaining = target - already_written
    if remaining <= 0:
        print(f"[{vendor}] Already complete ({already_written} samples). Skipping.")
        return

    # Load topics
    topics = load_topics_from_dataset()
    if topics:
        print(f"[{vendor}] Loaded {len(topics):,} topics from dataset_v6.jsonl")
    else:
        topics = [(t, "general") for t in FALLBACK_TOPICS]
        print(f"[{vendor}] dataset_v6.jsonl not found — using {len(topics)} fallback topics")

    client = get_client(vendor)
    styles = list(AI_PROMPT_STYLES_ZH.keys())
    failures = 0
    max_failures = 100
    written = 0

    print(f"[{vendor}] Generating {remaining:,} samples → {shard_path}")

    with open(shard_path, "a") as out_f:
        for i in tqdm(range(remaining), desc=vendor):
            topic_text, domain = topics[i % len(topics)]
            style = styles[i % len(styles)]

            sample = generate_sample(vendor, client, topic_text, style, domain)
            if sample is None:
                failures += 1
                if failures >= max_failures:
                    print(f"\n[{vendor}] Aborting: {max_failures} consecutive failures",
                          file=sys.stderr)
                    break
                continue

            failures = 0
            h = text_hash(sample["text"])
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

            if i % 10 == 0:
                time.sleep(0.05)

    total = already_written + written
    print(f"[{vendor}] Done: {written:,} new + {already_written:,} existing = {total:,} total")


# ---------------------------------------------------------------------------
# Merge shards
# ---------------------------------------------------------------------------


def merge_shards() -> None:
    """Append all supplement shards into dataset_v6.jsonl."""
    shards = sorted(OUTPUT_DIR.glob("*.jsonl")) if OUTPUT_DIR.exists() else []
    if not shards:
        print("ERROR: no shards found in supplement_v6_shards/", file=sys.stderr)
        sys.exit(1)

    print(f"Merging {len(shards)} shards into {FINAL_DATASET}")
    total = 0
    with open(FINAL_DATASET, "a") as out:
        for shard in shards:
            count = 0
            with open(shard) as inp:
                for line in inp:
                    out.write(line)
                    count += 1
                    total += 1
            print(f"  {shard.name}: {count:,} samples")
    print(f"Appended {total:,} samples total to {FINAL_DATASET}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", choices=list(VENDORS.keys()),
                        help="Which vendor to generate samples for")
    parser.add_argument("--target", type=int, default=24000,
                        help="Number of samples per vendor (default 24000)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge generated shards into dataset_v6.jsonl")
    args = parser.parse_args()

    if args.merge:
        merge_shards()
    elif args.vendor:
        run_vendor(args.vendor, args.target)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
