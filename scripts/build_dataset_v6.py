#!/usr/bin/env python3
"""Build dataset_v6.jsonl: 1M sample AI text detection training dataset.

Massively expanded human text diversity (~40 sources covering 60+ text types)
plus matched AI text via DeepSeek API (clean + adversarial).

Target: 600K human (label 0) + 300K clean AI (label 1) + 100K adversarial AI (label 1) = 1M total.

Usage:
    # Full pipeline (all 3 parts)
    python3 scripts/build_dataset_v6.py --part all

    # Human text only (no API calls)
    python3 scripts/build_dataset_v6.py --part human

    # Clean AI generation only (requires DEEPSEEK_API_KEY)
    python3 scripts/build_dataset_v6.py --part ai

    # Adversarial AI generation only (requires DEEPSEEK_API_KEY + human samples)
    python3 scripts/build_dataset_v6.py --part adversarial

    # Quick test (1000 samples total)
    python3 scripts/build_dataset_v6.py --part all --max-samples 1000

    # Resume interrupted run (auto-detects existing output)
    python3 scripts/build_dataset_v6.py --part human

Requirements:
    pip install datasets openai python-dotenv tqdm
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: pip install datasets", file=sys.stderr)
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
MIN_WORDS = 100
MAX_WORDS = 500
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT = PROJECT_DIR / "dataset_v6.jsonl"
V4_PATH = PROJECT_DIR / "dataset_v4.jsonl"
PROGRESS_FILE = PROJECT_DIR / "dataset_v6_progress.json"
ENV_PATH = PROJECT_DIR / ".env.local"

HUMAN_TARGET = 600_000
AI_CLEAN_TARGET = 300_000
AI_ADVERSARIAL_TARGET = 100_000

# DeepSeek pricing (per 1M tokens) -- as of 2026-04
DEEPSEEK_OUTPUT_PRICE_PER_1M = 0.28   # USD, deepseek-chat
DEEPSEEK_INPUT_PRICE_PER_1M = 0.07
AVG_OUTPUT_TOKENS = 400   # ~300 words
AVG_INPUT_TOKENS = 80
# Adversarial uses longer input (human text + instruction)
ADV_AVG_INPUT_TOKENS = 500
ADV_AVG_OUTPUT_TOKENS = 400

# ---------------------------------------------------------------------------
# Human text source definitions (~40 sources, 600K target)
# ---------------------------------------------------------------------------

HUMAN_SOURCES = [
    # --- Daily Communication ---
    {
        "name": "daily_dialog",
        "hf_path": "li2017dailydialog/daily_dialog",
        "hf_name": None,
        "text_field": "dialog",
        "domain": "chat",
        "target": 10_000,
        "split": "train",
    },
    {
        "name": "enron_emails",
        "hf_path": "corbt/enron-emails",
        "hf_name": None,
        "text_field": "text",
        "domain": "email",
        "target": 30_000,
        "split": "train",
    },
    {
        "name": "pushshift_reddit",
        "hf_path": "fddemarco/pushshift-reddit",
        "hf_name": None,
        "text_field": "selftext",
        "domain": "social_media",
        "target": 30_000,
        "split": "train",
    },
    {
        "name": "openwebtext",
        "hf_path": "Skylion007/openwebtext",
        "hf_name": None,
        "text_field": "text",
        "domain": "web_reddit",
        "target": 30_000,
        "split": "train",
    },
    {
        "name": "stackoverflow",
        "hf_path": "mikex86/stackoverflow-posts",
        "hf_name": None,
        "text_field": "Body",
        "domain": "forum",
        "target": 20_000,
        "split": "train",
    },

    # --- Creative Writing ---
    {
        "name": "pg19_literature",
        "hf_path": "deepmind/pg19",
        "hf_name": None,
        "text_field": "text",
        "domain": "literature",
        "target": 30_000,
        "split": "train",
    },
    {
        "name": "gutenberg_english",
        "hf_path": "sedthh/gutenberg_english",
        "hf_name": None,
        "text_field": "text",
        "domain": "literature",
        "target": 15_000,
        "split": "train",
    },
    {
        "name": "gutenberg_poetry",
        "hf_path": "biglam/gutenberg-poetry-corpus",
        "hf_name": None,
        "text_field": "s",
        "domain": "poetry",
        "target": 10_000,
        "split": "train",
        "min_words_override": 20,  # poetry lines are short, aggregate them
        "aggregate_lines": 10,     # combine 10 lines into one sample
    },
    {
        "name": "moviesum_screenplays",
        "hf_path": "rohitsaxena/MovieSum",
        "hf_name": None,
        "text_field": "script",
        "domain": "screenplay",
        "target": 5_000,
        "split": "train",
    },
    {
        "name": "genius_lyrics",
        "hf_path": "brunokreiner/genius-lyrics",
        "hf_name": None,
        "text_field": "lyrics",
        "domain": "lyrics",
        "target": 10_000,
        "split": "train",
        "min_words_override": 30,
    },
    {
        "name": "blog_authorship",
        "hf_path": "barilan/blog_authorship_corpus",
        "hf_name": None,
        "text_field": "text",
        "domain": "blog_diary",
        "target": 20_000,
        "split": "train",
    },
    {
        "name": "ivypanda_essays",
        "hf_path": "qwedsacf/ivypanda-essays",
        "hf_name": None,
        "text_field": "text",
        "domain": "creative_nonfiction",
        "target": 10_000,
        "split": "train",
    },

    # --- Academic Writing ---
    {
        "name": "asap_student_essays",
        "hf_path": "TasfiaS/ASAP-AES",
        "hf_name": None,
        "text_field": "essay",
        "domain": "student_essay",
        "target": 10_000,
        "split": "train",
    },
    {
        "name": "arxiv_abstracts",
        "hf_path": "gfissore/arxiv-abstracts-2021",
        "hf_name": None,
        "text_field": "abstract",
        "domain": "academic",
        "target": 30_000,
        "split": "train",
    },
    {
        "name": "scientific_papers_arxiv",
        "hf_path": "armanc/scientific_papers",
        "hf_name": "arxiv",
        "text_field": "article",
        "domain": "academic",
        "target": 20_000,
        "split": "train",
    },
    {
        "name": "scientific_papers_pubmed",
        "hf_path": "armanc/scientific_papers",
        "hf_name": "pubmed",
        "text_field": "article",
        "domain": "medical_academic",
        "target": 15_000,
        "split": "train",
    },

    # --- News / Media ---
    {
        "name": "cnn_dailymail",
        "hf_path": "abisee/cnn_dailymail",
        "hf_name": "3.0.0",
        "text_field": "article",
        "domain": "news",
        "target": 30_000,
        "split": "train",
    },
    {
        "name": "newsroom",
        "hf_path": "lil-lab/newsroom",
        "hf_name": None,
        "text_field": "text",
        "domain": "news_longform",
        "target": 20_000,
        "split": "train",
    },
    {
        "name": "news_category_huffpost",
        "hf_path": "Fraser/news-category-dataset",
        "hf_name": None,
        "text_field": "short_description",
        "domain": "news_headline",
        "target": 15_000,
        "split": "train",
        "min_words_override": 15,
    },
    {
        "name": "ag_news",
        "hf_path": "fancyzhx/ag_news",
        "hf_name": None,
        "text_field": "text",
        "domain": "news",
        "target": 20_000,
        "split": "train",
    },
    {
        "name": "conversational_weather",
        "hf_path": "GEM/conversational_weather",
        "hf_name": None,
        "text_field": "user_query",
        "domain": "weather",
        "target": 2_000,
        "split": "train",
        "min_words_override": 10,
    },

    # --- Business / Professional ---
    {
        "name": "product_descriptions_ads",
        "hf_path": "llm-wizard/Product-Descriptions-and-Ads",
        "hf_name": None,
        "text_field": "text",
        "domain": "marketing",
        "target": 10_000,
        "split": "train",
    },
    {
        "name": "amazon_product_desc",
        "hf_path": "Ateeqq/Amazon-Product-Description",
        "hf_name": None,
        "text_field": "description",
        "domain": "product_description",
        "target": 15_000,
        "split": "train",
    },
    {
        "name": "amazon_reviews",
        "hf_path": "McAuley-Lab/Amazon-Reviews-2023",
        "hf_name": "raw_review_All_Beauty",
        "text_field": "text",
        "domain": "reviews",
        "target": 20_000,
        "split": "full",
        "trust_remote_code": True,
    },
    {
        "name": "yelp_reviews",
        "hf_path": "Yelp/yelp_review_full",
        "hf_name": None,
        "text_field": "text",
        "domain": "reviews",
        "target": 15_000,
        "split": "train",
    },
    {
        "name": "cover_letters",
        "hf_path": "ShashiVish/cover-letter-dataset",
        "hf_name": None,
        "text_field": "text",
        "domain": "cover_letter",
        "target": 5_000,
        "split": "train",
    },
    {
        "name": "sec_filings",
        "hf_path": "PleIAs/SEC",
        "hf_name": None,
        "text_field": "text",
        "domain": "financial",
        "target": 15_000,
        "split": "train",
    },
    {
        "name": "meeting_transcripts",
        "hf_path": "lytang/MeetingBank-transcript",
        "hf_name": None,
        "text_field": "transcript",
        "domain": "meeting",
        "target": 5_000,
        "split": "train",
    },
    {
        "name": "newswire_historical",
        "hf_path": "dell-research-harvard/newswire",
        "hf_name": None,
        "text_field": "article",
        "domain": "press_release",
        "target": 10_000,
        "split": "train",
    },

    # --- Legal / Government ---
    {
        "name": "pile_of_law_contracts",
        "hf_path": "pile-of-law/pile-of-law",
        "hf_name": "atticus_contracts",
        "text_field": "text",
        "domain": "legal_contract",
        "target": 10_000,
        "split": "train",
    },
    {
        "name": "pile_of_law_court",
        "hf_path": "pile-of-law/pile-of-law",
        "hf_name": "courtlistener_opinions",
        "text_field": "text",
        "domain": "legal_court",
        "target": 10_000,
        "split": "train",
    },
    {
        "name": "pile_of_law_gov",
        "hf_path": "pile-of-law/pile-of-law",
        "hf_name": "federal_register",
        "text_field": "text",
        "domain": "government",
        "target": 10_000,
        "split": "train",
    },
    {
        "name": "congressional_speeches",
        "hf_path": "Eugleo/us-congressional-speeches",
        "hf_name": None,
        "text_field": "speech",
        "domain": "political_speech",
        "target": 10_000,
        "split": "train",
    },
    {
        "name": "hupd_patents",
        "hf_path": "HUPD/hupd",
        "hf_name": None,
        "text_field": "abstract",
        "domain": "patent",
        "target": 15_000,
        "split": "train",
    },

    # --- Medical / Health ---
    {
        "name": "pubmed_abstracts",
        "hf_path": "ncbi/pubmed",
        "hf_name": None,
        "text_field": "MedlineCitation.Article.Abstract.AbstractText",
        "domain": "medical",
        "target": 15_000,
        "split": "train",
    },
    {
        "name": "clinical_trials",
        "hf_path": "domenicrosati/clinical_trial_texts",
        "hf_name": None,
        "text_field": "text",
        "domain": "clinical",
        "target": 10_000,
        "split": "train",
    },
    {
        "name": "synthetic_clinical_notes",
        "hf_path": "starmpcc/Asclepius-Synthetic-Clinical-Notes",
        "hf_name": None,
        "text_field": "note",
        "domain": "clinical_notes",
        "target": 5_000,
        "split": "train",
    },
    {
        "name": "med_dataset",
        "hf_path": "Med-dataset/Med_Dataset",
        "hf_name": None,
        "text_field": "text",
        "domain": "patient_education",
        "target": 5_000,
        "split": "train",
    },

    # --- Technical / Engineering ---
    {
        "name": "medium_articles",
        "hf_path": "fabiochiu/medium-articles",
        "hf_name": None,
        "text_field": "text",
        "domain": "tech_blog",
        "target": 15_000,
        "split": "train",
    },

    # --- Religious / Philosophy ---
    {
        "name": "bible",
        "hf_path": "bible-nlp/biblenlp-corpus",
        "hf_name": None,
        "text_field": "text",
        "domain": "religious",
        "target": 5_000,
        "split": "train",
        "min_words_override": 20,
        "aggregate_lines": 5,
    },
    {
        "name": "philpapers",
        "hf_path": "malteos/philpapers-2023-10-28",
        "hf_name": None,
        "text_field": "abstract",
        "domain": "philosophy",
        "target": 10_000,
        "split": "train",
    },

    # --- Lifestyle ---
    {
        "name": "recipe_nlg",
        "hf_path": "mbien/recipe_nlg",
        "hf_name": None,
        "text_field": "directions",
        "domain": "recipe",
        "target": 10_000,
        "split": "train",
    },
    {
        "name": "tripadvisor_reviews",
        "hf_path": "argilla/tripadvisor-hotel-reviews",
        "hf_name": None,
        "text_field": "text",
        "domain": "travel",
        "target": 10_000,
        "split": "train",
    },
    {
        "name": "fitness_qa",
        "hf_path": "its-myrto/fitness-question-answers",
        "hf_name": None,
        "text_field": "answer",
        "domain": "fitness",
        "target": 3_000,
        "split": "train",
        "min_words_override": 30,
    },

    # --- Education ---
    {
        "name": "openstax_textbooks",
        "hf_path": "crumb/openstax-text",
        "hf_name": None,
        "text_field": "text",
        "domain": "textbook",
        "target": 15_000,
        "split": "train",
    },
    {
        "name": "exams_mcq",
        "hf_path": "mhardalov/exams",
        "hf_name": None,
        "text_field": "question",
        "domain": "exam",
        "target": 5_000,
        "split": "train",
        "min_words_override": 20,
    },

    # --- Historical ---
    {
        "name": "american_speeches",
        "hf_path": "owahltinez/speaker-recognition-american-rhetoric",
        "hf_name": None,
        "text_field": "text",
        "domain": "historical_speech",
        "target": 5_000,
        "split": "train",
    },
    {
        "name": "foiarchive_declassified",
        "hf_path": "HistoryLab/foiarchive",
        "hf_name": None,
        "text_field": "text",
        "domain": "historical_gov",
        "target": 10_000,
        "split": "train",
    },
    {
        "name": "blbooks_british_library",
        "hf_path": "TheBritishLibrary/blbooks",
        "hf_name": None,
        "text_field": "text",
        "domain": "historical_books",
        "target": 10_000,
        "split": "train",
    },
]

# Computed total human target from sources
_HUMAN_SOURCE_TARGET_SUM = sum(s["target"] for s in HUMAN_SOURCES)

# ---------------------------------------------------------------------------
# AI generation prompt styles (Part 2: Clean AI)
# ---------------------------------------------------------------------------

AI_PROMPT_STYLES = {
    "standard": (
        "Write a {length}-word passage about the following topic. "
        "Write naturally and informatively.\n\nTopic: {topic}"
    ),
    "formal": (
        "Write a formal, well-structured {length}-word passage about the following topic. "
        "Use professional language and clear organization.\n\nTopic: {topic}"
    ),
    "casual": (
        "Write a casual, conversational {length}-word passage about this topic. "
        "Write like you're explaining to a friend.\n\nTopic: {topic}"
    ),
    "academic": (
        "Write a scholarly {length}-word passage analyzing the following topic. "
        "Use academic tone, cite general findings, maintain objectivity.\n\nTopic: {topic}"
    ),
    "creative": (
        "Write a creative, engaging {length}-word passage about this topic. "
        "Use vivid language, varied sentence structure, and narrative techniques.\n\nTopic: {topic}"
    ),
}

# ---------------------------------------------------------------------------
# Adversarial attack types (Part 3)
# ---------------------------------------------------------------------------

ADVERSARIAL_ATTACKS = {
    "back_translation": {
        "system": "You are a translator.",
        "prompt": (
            "Translate the following English text to Chinese, then translate your Chinese "
            "version back to English. Only output the final English version, nothing else.\n\n"
            "Text: {text}"
        ),
        "weight": 0.15,  # fraction of adversarial budget
    },
    "paraphrase": {
        "system": "You are a writing assistant.",
        "prompt": (
            "Paraphrase the following text to sound more natural and human. "
            "Keep the same meaning but change the wording significantly. "
            "Output only the paraphrased text.\n\nText: {text}"
        ),
        "weight": 0.20,
    },
    "casual_injection": {
        "system": "You are a college student rewriting text for a class.",
        "prompt": (
            "Rewrite this text casually, like a college student would write it for a class "
            "discussion post. Add filler words, contractions, and informal phrasing. "
            "Output only the rewritten text.\n\nText: {text}"
        ),
        "weight": 0.15,
    },
    "synonym_substitution": {
        "system": "You are an editor.",
        "prompt": (
            "Rewrite the following text by replacing at least 40%% of the content words "
            "(nouns, verbs, adjectives, adverbs) with synonyms. Keep the sentence structure "
            "the same. Output only the rewritten text.\n\nText: {text}"
        ),
        "weight": 0.10,
    },
    "contraction_expansion": {
        "system": "You are a copy editor.",
        "prompt": (
            "Rewrite the following text using contractions wherever possible (e.g., "
            "'do not' -> 'don't', 'it is' -> 'it's'). Also add some sentence fragments "
            "and casual transitions like 'anyway', 'so yeah', 'honestly'. "
            "Output only the rewritten text.\n\nText: {text}"
        ),
        "weight": 0.10,
    },
    "sandwich": {
        "system": "You are a writing assistant.",
        "prompt": (
            "Take the following AI-generated text and sandwich it between human-sounding "
            "opening and closing paragraphs. The opening should be a personal anecdote or "
            "opinion (2-3 sentences). The closing should be a casual reflection (2-3 sentences). "
            "Keep the middle text mostly unchanged. Output the full combined text.\n\nText: {text}"
        ),
        "weight": 0.15,
    },
    "homoglyph_typo": {
        "system": "You are a text processor.",
        "prompt": (
            "Rewrite the following text with these modifications:\n"
            "1. Replace 5-10 random letters with visually similar Unicode characters "
            "(e.g., 'a' -> '\u0430', 'e' -> '\u0435', 'o' -> '\u043e')\n"
            "2. Add 3-5 realistic typos (transposed letters, missing letters, double letters)\n"
            "3. Randomly capitalize 2-3 words mid-sentence\n"
            "Output only the modified text.\n\nText: {text}"
        ),
        "weight": 0.15,
    },
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def text_hash(text):
    # type: (str) -> str
    """MD5 hash of first 200 chars for dedup."""
    return hashlib.md5(text[:200].encode("utf-8", errors="replace")).hexdigest()


def word_count(text):
    # type: (str) -> int
    return len(text.split())


def is_english_heuristic(text):
    # type: (str) -> bool
    """Fast heuristic: >70% ASCII letters among alpha chars."""
    if not text:
        return False
    alpha = sum(1 for c in text[:500] if c.isalpha())
    if alpha == 0:
        return False
    ascii_alpha = sum(1 for c in text[:500] if c.isascii() and c.isalpha())
    return (ascii_alpha / alpha) > 0.70


def clean_text(text):
    # type: (str) -> str
    """Basic cleaning: collapse whitespace, strip."""
    text = " ".join(text.split())
    return text.strip()


def extract_text(example, text_field):
    # type: (dict, str) -> List[str]
    """Extract text from a dataset example, handling nested fields.

    Returns a list because some fields (HC3, ELI5, dialog) contain multiple items.
    """
    parts = text_field.split(".")
    value = example  # type: Any
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list) and part.isdigit():
            value = value[int(part)]
        else:
            return []
        if value is None:
            return []

    if isinstance(value, list):
        # For dialog-type fields, join list items into one text
        if all(isinstance(v, str) for v in value):
            joined = " ".join(v.strip() for v in value if v and v.strip())
            return [joined] if joined else []
        return [str(v) for v in value if v]
    return [str(value)] if value else []


def load_progress():
    # type: () -> Dict[str, Any]
    """Load progress tracking file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {
        "completed_sources": [],
        "human_count": 0,
        "ai_clean_count": 0,
        "ai_adversarial_count": 0,
    }


def save_progress(progress):
    # type: (Dict[str, Any]) -> None
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def estimate_api_cost(num_samples, avg_input=AVG_INPUT_TOKENS, avg_output=AVG_OUTPUT_TOKENS):
    # type: (int, int, int) -> float
    """Estimate DeepSeek API cost."""
    input_cost = (num_samples * avg_input / 1_000_000) * DEEPSEEK_INPUT_PRICE_PER_1M
    output_cost = (num_samples * avg_output / 1_000_000) * DEEPSEEK_OUTPUT_PRICE_PER_1M
    return input_cost + output_cost


def print_cost_estimate(label, num_samples, avg_input=AVG_INPUT_TOKENS, avg_output=AVG_OUTPUT_TOKENS):
    # type: (str, int, int, int) -> None
    """Print detailed API cost estimate."""
    cost = estimate_api_cost(num_samples, avg_input, avg_output)
    input_tokens_m = num_samples * avg_input / 1_000_000
    output_tokens_m = num_samples * avg_output / 1_000_000
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  {label}", file=sys.stderr)
    print(f"  Samples to generate: {num_samples:,}", file=sys.stderr)
    print(f"  Estimated API cost: ${cost:.2f}", file=sys.stderr)
    print(f"    Input:  {input_tokens_m:.1f}M tokens "
          f"x ${DEEPSEEK_INPUT_PRICE_PER_1M}/M = "
          f"${input_tokens_m * DEEPSEEK_INPUT_PRICE_PER_1M:.2f}", file=sys.stderr)
    print(f"    Output: {output_tokens_m:.1f}M tokens "
          f"x ${DEEPSEEK_OUTPUT_PRICE_PER_1M}/M = "
          f"${output_tokens_m * DEEPSEEK_OUTPUT_PRICE_PER_1M:.2f}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)


def get_deepseek_client():
    # type: () -> Optional[Any]
    """Initialize DeepSeek API client."""
    if OpenAI is None:
        print("ERROR: pip install openai (required for AI generation)", file=sys.stderr)
        return None

    if load_dotenv:
        load_dotenv(ENV_PATH)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not found in environment or .env.local",
              file=sys.stderr)
        return None

    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def write_samples_to_file(samples, output_path, mode="a"):
    # type: (List[Dict], Path, str) -> int
    """Write samples to JSONL file. Returns count written."""
    written = 0
    with open(output_path, mode) as f:
        for entry in samples:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1
    return written


def load_existing_hashes(output_path):
    # type: (Path) -> Tuple[Set[str], int]
    """Load hashes from existing output file for dedup/resume."""
    seen_hashes = set()  # type: Set[str]
    existing_count = 0
    if output_path.exists():
        print(f"\nResuming: loading hashes from existing {output_path.name}...",
              file=sys.stderr)
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    h = text_hash(d.get("text", ""))
                    seen_hashes.add(h)
                    existing_count += 1
                except json.JSONDecodeError:
                    continue
        print(f"  Found {existing_count:,} existing samples "
              f"({len(seen_hashes):,} unique hashes)", file=sys.stderr)
    return seen_hashes, existing_count


# ---------------------------------------------------------------------------
# Part 1: Human text collection
# ---------------------------------------------------------------------------


def collect_human_source(source, seen_hashes, max_per_source):
    # type: (Dict, Set[str], int) -> List[Dict]
    """Collect human text samples from a single HuggingFace dataset source."""
    name = source["name"]
    target = min(source["target"], max_per_source) if max_per_source else source["target"]
    min_words = source.get("min_words_override", MIN_WORDS)
    aggregate_n = source.get("aggregate_lines", 0)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"[{name}] Collecting up to {target:,} samples...", file=sys.stderr)
    print(f"  Dataset: {source['hf_path']}", file=sys.stderr)

    samples = []  # type: List[Dict]
    skipped_short = 0
    skipped_dup = 0
    skipped_lang = 0
    examined = 0
    line_buffer = []  # type: List[str]

    try:
        kwargs = {
            "path": source["hf_path"],
            "split": source["split"],
            "streaming": True,
        }  # type: Dict[str, Any]
        if source.get("hf_name"):
            kwargs["name"] = source["hf_name"]
        if source.get("trust_remote_code"):
            kwargs["trust_remote_code"] = True

        ds = load_dataset(**kwargs)

        for example in ds:
            if len(samples) >= target:
                break

            texts = extract_text(example, source["text_field"])
            for raw_text in texts:
                if len(samples) >= target:
                    break

                raw_text = clean_text(raw_text)
                examined += 1

                # Aggregate short lines (poetry, bible verses, etc.)
                if aggregate_n > 0:
                    line_buffer.append(raw_text)
                    if len(line_buffer) < aggregate_n:
                        continue
                    raw_text = " ".join(line_buffer)
                    line_buffer = []

                wc = word_count(raw_text)
                if wc < min_words:
                    skipped_short += 1
                    continue

                # Truncate long texts to a random window
                if wc > MAX_WORDS:
                    words = raw_text.split()
                    target_wc = random.randint(MIN_WORDS, MAX_WORDS)
                    if len(words) > target_wc:
                        start = random.randint(0, len(words) - target_wc)
                        raw_text = " ".join(words[start:start + target_wc])

                if not is_english_heuristic(raw_text):
                    skipped_lang += 1
                    continue

                h = text_hash(raw_text)
                if h in seen_hashes:
                    skipped_dup += 1
                    continue
                seen_hashes.add(h)

                samples.append({
                    "text": raw_text,
                    "label": 0,
                    "source": name,
                    "domain": source["domain"],
                })

                if len(samples) % 5_000 == 0:
                    print(f"  ... {len(samples):,}/{target:,} collected "
                          f"(examined {examined:,})", file=sys.stderr)

    except Exception as e:
        print(f"  ERROR loading {name}: {e}", file=sys.stderr)
        print(f"  Continuing with {len(samples):,} samples collected so far.",
              file=sys.stderr)

    print(f"  [{name}] Done: {len(samples):,} samples "
          f"(examined {examined:,}, "
          f"skip: {skipped_short:,} short, "
          f"{skipped_dup:,} dup, {skipped_lang:,} non-en)", file=sys.stderr)

    return samples


def run_part_human(seen_hashes, progress, max_total, output_path):
    # type: (Set[str], Dict, Optional[int], Path) -> List[Dict]
    """Part 1: Collect human text from all sources."""
    print(f"\n{'='*60}", file=sys.stderr)
    print("PART 1: HUMAN TEXT COLLECTION", file=sys.stderr)
    print(f"  Sources: {len(HUMAN_SOURCES)}", file=sys.stderr)
    print(f"  Source targets sum: {_HUMAN_SOURCE_TARGET_SUM:,}", file=sys.stderr)
    print(f"  Pipeline target: {HUMAN_TARGET:,}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    all_samples = []  # type: List[Dict]
    completed = set(progress.get("completed_sources", []))

    human_target = HUMAN_TARGET
    if max_total:
        human_target = int(max_total * 0.6)  # 60% of max is human
    max_per_source = (human_target // len(HUMAN_SOURCES) + 1)

    for source in HUMAN_SOURCES:
        if source["name"] in completed:
            print(f"[{source['name']}] Already completed, skipping.", file=sys.stderr)
            continue

        if len(all_samples) >= human_target:
            print(f"Reached human target ({human_target:,}), stopping.", file=sys.stderr)
            break

        remaining = human_target - len(all_samples)
        effective_max = min(max_per_source, remaining)

        samples = collect_human_source(source, seen_hashes, effective_max)
        all_samples.extend(samples)

        # Write incrementally
        if samples:
            write_samples_to_file(samples, output_path)

        progress["completed_sources"].append(source["name"])
        progress["human_count"] = len(all_samples)
        save_progress(progress)

    # Summary
    print(f"\nHuman collection summary:", file=sys.stderr)
    domain_counts = defaultdict(int)  # type: Dict[str, int]
    for s in all_samples:
        domain_counts[s["domain"]] += 1
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {domain}: {count:,}", file=sys.stderr)
    print(f"  TOTAL: {len(all_samples):,}", file=sys.stderr)

    return all_samples


# ---------------------------------------------------------------------------
# Part 2: Clean AI text generation
# ---------------------------------------------------------------------------


def extract_topic_from_text(text):
    # type: (str) -> str
    """Extract a rough topic from human text (first 2 sentences or 50 words)."""
    sentences = text.split(".")
    topic = ". ".join(sentences[:2]).strip()
    words = topic.split()
    if len(words) > 50:
        topic = " ".join(words[:50])
    return topic + ("." if not topic.endswith(".") else "")


def generate_single_ai_sample(client, topic, style, domain):
    # type: (Any, str, str, str) -> Optional[Dict]
    """Generate a single AI text sample via DeepSeek API."""
    prompt_template = AI_PROMPT_STYLES[style]
    target_length = random.randint(MIN_WORDS, MAX_WORDS)
    prompt = prompt_template.format(topic=topic, length=target_length)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful writing assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
            temperature=random.choice([0.3, 0.7, 1.0]),
        )
        text = response.choices[0].message.content.strip()
        text = clean_text(text)

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
            time.sleep(2)
        return None


def load_v4_ai_samples(seen_hashes):
    # type: (Set[str]) -> List[Dict]
    """Load existing AI samples from dataset_v4.jsonl."""
    if not V4_PATH.exists():
        print("  dataset_v4.jsonl not found, skipping.", file=sys.stderr)
        return []

    samples = []  # type: List[Dict]
    with open(V4_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            if d.get("label") not in (1, 2):
                continue

            text = d.get("text", "")
            h = text_hash(text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            samples.append({
                "text": text,
                "label": 1,
                "source": "v4_{}".format(d.get("model", "unknown")),
                "domain": d.get("topic", "mixed"),
                "prompt_style": d.get("style", "unknown"),
            })

    print(f"  Loaded {len(samples):,} AI samples from dataset_v4.jsonl", file=sys.stderr)
    return samples


def run_part_ai_clean(seen_hashes, progress, max_total, output_path):
    # type: (Set[str], Dict, Optional[int], Path) -> List[Dict]
    """Part 2: Generate clean AI text via DeepSeek API."""
    print(f"\n{'='*60}", file=sys.stderr)
    print("PART 2: CLEAN AI TEXT GENERATION", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    client = get_deepseek_client()
    if client is None:
        return []

    # Load v4 AI samples
    print("Loading existing AI samples from dataset_v4.jsonl...", file=sys.stderr)
    v4_samples = load_v4_ai_samples(seen_hashes)

    ai_target = AI_CLEAN_TARGET
    if max_total:
        ai_target = int(max_total * 0.3)  # 30% of max
    remaining_target = ai_target - len(v4_samples)

    if remaining_target <= 0:
        print(f"Already have {len(v4_samples):,} AI samples, target met.", file=sys.stderr)
        if v4_samples:
            write_samples_to_file(v4_samples, output_path)
        return v4_samples

    print_cost_estimate(
        "CLEAN AI GENERATION COST ESTIMATE",
        remaining_target,
        AVG_INPUT_TOKENS,
        AVG_OUTPUT_TOKENS,
    )

    # Load human samples from output file for topic extraction
    topic_pool = []  # type: List[Tuple[str, str]]
    if output_path.exists():
        print("Loading human samples for topic extraction...", file=sys.stderr)
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get("label") == 0:
                        topic = extract_topic_from_text(d["text"])
                        if len(topic) > 20:
                            topic_pool.append((topic, d.get("domain", "mixed")))
                except json.JSONDecodeError:
                    continue
        print(f"  Extracted {len(topic_pool):,} topics from human samples", file=sys.stderr)

    if not topic_pool:
        print("ERROR: No topics available. Run --part human first.", file=sys.stderr)
        return v4_samples

    random.shuffle(topic_pool)
    styles = list(AI_PROMPT_STYLES.keys())
    ai_samples = []  # type: List[Dict]
    failures = 0
    max_failures = 100
    batch_buffer = []  # type: List[Dict]

    # Write v4 samples first
    if v4_samples:
        write_samples_to_file(v4_samples, output_path)

    print(f"Generating {remaining_target:,} clean AI samples via DeepSeek API...",
          file=sys.stderr)

    for i in tqdm(range(remaining_target), desc="Clean AI generation", file=sys.stderr):
        topic_text, domain = topic_pool[i % len(topic_pool)]
        style = styles[i % len(styles)]

        result = generate_single_ai_sample(client, topic_text, style, domain)

        if result is None:
            failures += 1
            if failures >= max_failures:
                print(f"\n  Aborting: {max_failures} consecutive API failures.",
                      file=sys.stderr)
                break
            continue

        failures = 0
        h = text_hash(result["text"])
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        ai_samples.append(result)
        batch_buffer.append(result)

        # Periodic flush to disk
        if len(batch_buffer) >= 1000:
            write_samples_to_file(batch_buffer, output_path)
            batch_buffer = []
            progress["ai_clean_count"] = len(v4_samples) + len(ai_samples)
            save_progress(progress)

        # Rate limiting
        if i % 10 == 0:
            time.sleep(0.1)

    # Flush remaining
    if batch_buffer:
        write_samples_to_file(batch_buffer, output_path)

    total_ai = len(v4_samples) + len(ai_samples)
    print(f"\nClean AI generation complete: {len(ai_samples):,} new + "
          f"{len(v4_samples):,} from v4 = {total_ai:,} total", file=sys.stderr)

    progress["ai_clean_count"] = total_ai
    save_progress(progress)

    return v4_samples + ai_samples


# ---------------------------------------------------------------------------
# Part 3: Adversarial AI generation
# ---------------------------------------------------------------------------


def generate_adversarial_sample(client, source_text, attack_type):
    # type: (Any, str, str) -> Optional[Dict]
    """Apply an adversarial attack to AI-generated text via DeepSeek API."""
    attack = ADVERSARIAL_ATTACKS[attack_type]
    prompt = attack["prompt"].format(text=source_text)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": attack["system"]},
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip()
        text = clean_text(text)

        if word_count(text) < MIN_WORDS // 3:
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


def run_part_adversarial(seen_hashes, progress, max_total, output_path):
    # type: (Set[str], Dict, Optional[int], Path) -> List[Dict]
    """Part 3: Generate adversarial AI text (attacks on clean AI text)."""
    print(f"\n{'='*60}", file=sys.stderr)
    print("PART 3: ADVERSARIAL AI TEXT GENERATION", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    client = get_deepseek_client()
    if client is None:
        return []

    adv_target = AI_ADVERSARIAL_TARGET
    if max_total:
        adv_target = int(max_total * 0.1)  # 10% of max

    existing_adv = progress.get("ai_adversarial_count", 0)
    remaining = adv_target - existing_adv
    if remaining <= 0:
        print(f"Already have {existing_adv:,} adversarial samples, target met.",
              file=sys.stderr)
        return []

    print_cost_estimate(
        "ADVERSARIAL AI GENERATION COST ESTIMATE",
        remaining,
        ADV_AVG_INPUT_TOKENS,
        ADV_AVG_OUTPUT_TOKENS,
    )

    # Load clean AI samples from output file as source material
    ai_source_texts = []  # type: List[str]
    if output_path.exists():
        print("Loading clean AI samples as adversarial source material...", file=sys.stderr)
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get("label") == 1 and d.get("source", "").startswith("deepseek_clean"):
                        ai_source_texts.append(d["text"])
                except json.JSONDecodeError:
                    continue

    # If no clean AI text yet, also look for v4 AI and any label=1
    if len(ai_source_texts) < 1000:
        print("  Not enough clean AI samples. Also loading any AI-labeled text...",
              file=sys.stderr)
        if output_path.exists():
            with open(output_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        if d.get("label") == 1:
                            ai_source_texts.append(d["text"])
                    except json.JSONDecodeError:
                        continue

    if not ai_source_texts:
        # Fall back: generate fresh AI text to attack
        print("  No AI source text found. Generating fresh AI text for attacks...",
              file=sys.stderr)
        # Load human topics
        human_topics = []  # type: List[Tuple[str, str]]
        if output_path.exists():
            with open(output_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        if d.get("label") == 0:
                            topic = extract_topic_from_text(d["text"])
                            if len(topic) > 20:
                                human_topics.append((topic, d.get("domain", "mixed")))
                    except json.JSONDecodeError:
                        continue

        if not human_topics:
            print("ERROR: No source material. Run --part human and --part ai first.",
                  file=sys.stderr)
            return []

        # Generate minimal AI source texts
        print("  Generating 5000 AI source texts for adversarial attacks...", file=sys.stderr)
        random.shuffle(human_topics)
        styles = list(AI_PROMPT_STYLES.keys())
        for i in tqdm(range(min(5000, len(human_topics))), desc="Source AI gen", file=sys.stderr):
            topic_text, domain = human_topics[i]
            style = styles[i % len(styles)]
            result = generate_single_ai_sample(client, topic_text, style, domain)
            if result:
                ai_source_texts.append(result["text"])
            if i % 10 == 0:
                time.sleep(0.1)

    if not ai_source_texts:
        print("ERROR: Could not obtain any AI source text.", file=sys.stderr)
        return []

    random.shuffle(ai_source_texts)
    print(f"  Using {len(ai_source_texts):,} AI source texts for adversarial attacks",
          file=sys.stderr)

    # Distribute across attack types by weight
    attack_types = list(ADVERSARIAL_ATTACKS.keys())
    attack_counts = {}  # type: Dict[str, int]
    for at in attack_types:
        attack_counts[at] = int(remaining * ADVERSARIAL_ATTACKS[at]["weight"])
    # Distribute remainder
    allocated = sum(attack_counts.values())
    for at in attack_types:
        if allocated >= remaining:
            break
        attack_counts[at] += 1
        allocated += 1

    print(f"  Attack distribution:", file=sys.stderr)
    for at, count in attack_counts.items():
        print(f"    {at}: {count:,}", file=sys.stderr)

    adv_samples = []  # type: List[Dict]
    failures = 0
    max_failures = 100
    batch_buffer = []  # type: List[Dict]
    source_idx = 0

    for attack_type, count in attack_counts.items():
        print(f"\n  Running attack: {attack_type} ({count:,} samples)...", file=sys.stderr)

        for i in tqdm(range(count), desc=attack_type, file=sys.stderr):
            source_text = ai_source_texts[source_idx % len(ai_source_texts)]
            source_idx += 1

            result = generate_adversarial_sample(client, source_text, attack_type)

            if result is None:
                failures += 1
                if failures >= max_failures:
                    print(f"\n  Aborting {attack_type}: {max_failures} consecutive failures.",
                          file=sys.stderr)
                    break
                continue

            failures = 0
            h = text_hash(result["text"])
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            adv_samples.append(result)
            batch_buffer.append(result)

            # Periodic flush
            if len(batch_buffer) >= 500:
                write_samples_to_file(batch_buffer, output_path)
                batch_buffer = []
                progress["ai_adversarial_count"] = existing_adv + len(adv_samples)
                save_progress(progress)

            # Rate limiting
            if i % 10 == 0:
                time.sleep(0.1)

    # Flush remaining
    if batch_buffer:
        write_samples_to_file(batch_buffer, output_path)

    total = existing_adv + len(adv_samples)
    print(f"\nAdversarial generation complete: {len(adv_samples):,} new samples",
          file=sys.stderr)

    # Summary by attack type
    attack_summary = defaultdict(int)  # type: Dict[str, int]
    for s in adv_samples:
        attack_summary[s.get("attack_type", "unknown")] += 1
    for at, c in sorted(attack_summary.items(), key=lambda x: -x[1]):
        print(f"  {at}: {c:,}", file=sys.stderr)

    progress["ai_adversarial_count"] = total
    save_progress(progress)

    return adv_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build dataset_v6.jsonl: 1M AI detection training dataset"
    )
    parser.add_argument(
        "--part",
        type=str,
        choices=["human", "ai", "adversarial", "all"],
        default="all",
        help="Which part to run: human, ai, adversarial, or all (default: all)",
    )
    parser.add_argument(
        "--human-only",
        action="store_true",
        help="(Legacy) Same as --part human",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit total samples (for testing). E.g. --max-samples 1000",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: {})".format(OUTPUT),
    )
    args = parser.parse_args()

    # Legacy flag support
    if args.human_only:
        args.part = "human"

    random.seed(SEED)
    output_path = Path(args.output) if args.output else OUTPUT
    max_total = args.max_samples  # type: Optional[int]

    print("=" * 60, file=sys.stderr)
    print("BUILD DATASET V6", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Output: {output_path}", file=sys.stderr)
    print(f"  Part: {args.part}", file=sys.stderr)
    print(f"  Max samples: {max_total or 'unlimited'}", file=sys.stderr)
    print(f"  Targets: {HUMAN_TARGET:,} human + {AI_CLEAN_TARGET:,} clean AI "
          f"+ {AI_ADVERSARIAL_TARGET:,} adversarial AI = "
          f"{HUMAN_TARGET + AI_CLEAN_TARGET + AI_ADVERSARIAL_TARGET:,} total",
          file=sys.stderr)
    print(f"  Human sources: {len(HUMAN_SOURCES)}", file=sys.stderr)
    print(f"  AI prompt styles: {len(AI_PROMPT_STYLES)}", file=sys.stderr)
    print(f"  Adversarial attack types: {len(ADVERSARIAL_ATTACKS)}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Load progress / resume state
    progress = load_progress()
    seen_hashes, existing_count = load_existing_hashes(output_path)

    run_human = args.part in ("human", "all")
    run_ai = args.part in ("ai", "all")
    run_adversarial = args.part in ("adversarial", "all")

    human_samples = []  # type: List[Dict]
    ai_samples = []  # type: List[Dict]
    adv_samples = []  # type: List[Dict]

    # --- Part 1: Human ---
    if run_human:
        human_samples = run_part_human(seen_hashes, progress, max_total, output_path)

    # --- Part 2: Clean AI ---
    if run_ai:
        ai_samples = run_part_ai_clean(seen_hashes, progress, max_total, output_path)

    # --- Part 3: Adversarial AI ---
    if run_adversarial:
        adv_samples = run_part_adversarial(seen_hashes, progress, max_total, output_path)

    # --- Final summary ---
    new_count = len(human_samples) + len(ai_samples) + len(adv_samples)
    total = existing_count + new_count

    print(f"\n{'='*60}", file=sys.stderr)
    print("DATASET V6 BUILD COMPLETE", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  New samples this run: {new_count:,}", file=sys.stderr)
    print(f"    Human: {len(human_samples):,}", file=sys.stderr)
    print(f"    Clean AI: {len(ai_samples):,}", file=sys.stderr)
    print(f"    Adversarial AI: {len(adv_samples):,}", file=sys.stderr)
    print(f"  Previously existing: {existing_count:,}", file=sys.stderr)
    print(f"  Total in file: {total:,}", file=sys.stderr)
    print(f"  Output: {output_path}", file=sys.stderr)

    # Domain distribution for this run
    if new_count > 0:
        all_new = human_samples + ai_samples + adv_samples
        label_counts = defaultdict(int)  # type: Dict[int, int]
        domain_counts = defaultdict(int)  # type: Dict[str, int]
        for s in all_new:
            label_counts[s["label"]] += 1
            domain_counts[s["domain"]] += 1

        print(f"\nBy label:", file=sys.stderr)
        label_names = {0: "human", 1: "ai"}
        for label in sorted(label_counts):
            name = label_names.get(label, "label_{}".format(label))
            count = label_counts[label]
            pct = count / len(all_new) * 100
            print(f"  {name}: {count:,} ({pct:.1f}%)", file=sys.stderr)

        print(f"\nBy domain (top 20):", file=sys.stderr)
        for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"  {domain}: {count:,}", file=sys.stderr)

    # Clean up progress on full successful run
    if args.part == "all" and not max_total:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
            print("\nProgress file cleaned up.", file=sys.stderr)

    print(f"\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
