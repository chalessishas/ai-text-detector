"""Generate adversarial training samples from standard AI text.

Applies known bypass techniques (typos, casual tone, dialogue wrapping, emoji,
short sentences, homoglyphs, first-person injection) to AI-generated text to
create "hard negative" samples for adversarial-aware classifier training.

All output samples are labeled as AI (label=1) — they ARE AI text, just disguised.

Usage:
  python3.11 scripts/generate_adversarial_data.py --input dataset.jsonl --output dataset_adversarial.jsonl

Or standalone (uses built-in samples):
  python3.11 scripts/generate_adversarial_data.py --output dataset_adversarial.jsonl
"""
import json
import random
import re
import argparse
import os

random.seed(42)

# ── Attack functions ──────────────────────────────────────────────────────

def attack_typos(text: str, rate: float = 0.08) -> str:
    """Introduce random typos at given rate."""
    words = text.split()
    result = []
    for w in words:
        if len(w) > 3 and random.random() < rate:
            # Random character deletion, swap, or insertion
            op = random.choice(["delete", "swap", "insert"])
            pos = random.randint(1, len(w) - 2)
            if op == "delete":
                w = w[:pos] + w[pos+1:]
            elif op == "swap" and pos < len(w) - 1:
                w = w[:pos] + w[pos+1] + w[pos] + w[pos+2:]
            elif op == "insert":
                w = w[:pos] + random.choice("abcdefghijklmnopqrstuvwxyz") + w[pos:]
        result.append(w)
    return " ".join(result)


def attack_casual_tone(text: str) -> str:
    """Inject casual markers into formal AI text."""
    # Add filler words at sentence boundaries
    fillers = [
        "like, ", "honestly, ", "basically, ", "you know, ",
        "I mean, ", "so yeah, ", "right? ", "tbh ", "ngl ",
    ]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for i, s in enumerate(sentences):
        if i > 0 and random.random() < 0.4:
            s = random.choice(fillers) + s[0].lower() + s[1:]
        result.append(s)
    # Replace some formal words
    replacements = {
        "Furthermore": "Plus",
        "Additionally": "Also",
        "unprecedented": "crazy",
        "fundamentally": "basically",
        "significantly": "really",
        "demonstrates": "shows",
        "implementation": "setup",
        "utilize": "use",
        "facilitate": "help with",
        "consequently": "so",
        "Nevertheless": "But still",
    }
    text_out = " ".join(result)
    for formal, casual in replacements.items():
        text_out = text_out.replace(formal, casual)
        text_out = text_out.replace(formal.lower(), casual.lower())
    return text_out


def attack_dialogue(text: str) -> str:
    """Wrap AI content in dialogue format."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    speakers = [
        ('she asked.', 'he replied.', 'she continued.', 'he added.', 'she noted.'),
        ('"', '" '),
    ]
    result = []
    tags = speakers[0]
    for i, s in enumerate(sentences):
        s = s.strip()
        if not s:
            continue
        tag = tags[i % len(tags)]
        result.append(f'"{s}" {tag}')
    result.append("She nodded thoughtfully.")
    return " ".join(result)


def attack_emoji(text: str) -> str:
    """Insert emojis between sentences."""
    emojis = ["🤖", "📊", "🔬", "💡", "🌍", "🚀", "💪", "🧠", "✅", "🔥", "📈", "🎯"]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for s in sentences:
        result.append(s)
        if random.random() < 0.6:
            result.append(random.choice(emojis))
    return " ".join(result)


def attack_short_sentences(text: str) -> str:
    """Break text into very short sentences."""
    # Split on commas and conjunctions too
    parts = re.split(r'[,;]\s+|\.\s+|\band\b|\bbut\b|\bwhich\b|\bthat\b', text)
    result = []
    for p in parts:
        p = p.strip().rstrip('.,;')
        if len(p) > 10:
            result.append(p + ".")
    return "\n".join(result)


def attack_homoglyphs(text: str, rate: float = 0.3) -> str:
    """Replace some Latin chars with Greek/Cyrillic lookalikes."""
    mapping = {
        'a': '\u0430', 'e': '\u0435', 'o': '\u043e', 'p': '\u0440',
        'c': '\u0441', 'T': '\u0422', 'H': '\u041d', 'M': '\u041c',
        'A': '\u0410', 'O': '\u041e', 'B': '\u0412',
    }
    result = []
    for ch in text:
        if ch in mapping and random.random() < rate:
            result.append(mapping[ch])
        else:
            result.append(ch)
    return "".join(result)


def attack_first_person(text: str) -> str:
    """Add first-person framing to AI text."""
    prefixes = [
        "I remember when I first learned about how ",
        "In my experience, ",
        "I've always found it fascinating that ",
        "From what I've seen, ",
        "I think it's worth noting that ",
    ]
    suffixes = [
        " This really resonated with me personally.",
        " I found this particularly interesting.",
        " At least that's been my experience.",
        " But that's just my perspective.",
    ]
    # Lowercase first char after prefix
    text_lower = text[0].lower() + text[1:] if text else text
    return random.choice(prefixes) + text_lower + random.choice(suffixes)


def attack_contractions(text: str) -> str:
    """Add contractions to formal AI text (AI rarely uses them)."""
    replacements = [
        ("it is", "it's"), ("that is", "that's"), ("there is", "there's"),
        ("do not", "don't"), ("does not", "doesn't"), ("did not", "didn't"),
        ("have not", "haven't"), ("has not", "hasn't"), ("will not", "won't"),
        ("cannot", "can't"), ("could not", "couldn't"), ("would not", "wouldn't"),
        ("they are", "they're"), ("we are", "we're"), ("you are", "you're"),
        ("I am", "I'm"), ("he is", "he's"), ("she is", "she's"),
        ("It is", "It's"), ("That is", "That's"), ("There is", "There's"),
        ("Do not", "Don't"), ("Does not", "Doesn't"),
    ]
    result = text
    for formal, contracted in replacements:
        result = result.replace(formal, contracted)
    return result


def attack_code_injection(text: str) -> str:
    """Mix code snippets into AI text."""
    code_snippets = [
        "The function `process_data()` handles this. ",
        "Using `numpy.array()` for vectorized operations, ",
        "The `sklearn.preprocessing` module normalizes features. ",
        "With `model.fit(X_train, y_train)`, ",
    ]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for i, s in enumerate(sentences):
        if i > 0 and i < len(sentences) - 1 and random.random() < 0.3:
            result.append(random.choice(code_snippets))
        result.append(s)
    return " ".join(result)


# All attack functions
ATTACKS = {
    "typo": attack_typos,
    "casual": attack_casual_tone,
    "dialogue": attack_dialogue,
    "emoji": attack_emoji,
    "short_sent": attack_short_sentences,
    "homoglyph": attack_homoglyphs,
    "first_person": attack_first_person,
    "contraction": attack_contractions,
    "code_inject": attack_code_injection,
}

# ── Built-in AI text samples ─────────────────────────────────────────────

BUILTIN_AI_TEXTS = [
    "The rapid advancement of artificial intelligence has fundamentally transformed how we approach complex problem-solving in modern society. Machine learning algorithms now process vast amounts of data with unprecedented efficiency, enabling breakthroughs in healthcare, finance, and scientific research.",
    "Climate change represents one of the most pressing challenges facing humanity in the 21st century. Rising global temperatures, driven primarily by anthropogenic greenhouse gas emissions, have led to significant alterations in weather patterns, sea levels, and biodiversity.",
    "The importance of mental health awareness in modern society cannot be overstated. As our understanding of psychological well-being continues to evolve, it becomes increasingly clear that mental health is just as crucial as physical health.",
    "Education serves as the cornerstone of societal progress and individual empowerment. Through the acquisition of knowledge and the development of critical thinking skills, individuals are better equipped to navigate the complexities of the modern world.",
    "The global economy has undergone a remarkable transformation in recent decades, driven by technological innovation, globalization, and shifting demographic patterns. These interconnected forces have created both unprecedented opportunities and significant challenges.",
    "Sustainable development has emerged as a critical paradigm for addressing the complex interplay between economic growth, social equity, and environmental preservation.",
    "The evolution of social media has profoundly impacted how individuals communicate, consume information, and form social connections. These digital platforms have democratized content creation while simultaneously raising concerns.",
    "Blockchain technology represents a paradigm shift in how we conceptualize trust, transparency, and decentralized governance. By leveraging cryptographic principles and distributed consensus mechanisms, blockchain has the potential to revolutionize industries.",
    "Remote work has become an increasingly prevalent feature of the modern workplace, accelerated by the global pandemic and enabled by advances in communication technology.",
    "The healthcare industry is experiencing a period of unprecedented transformation, driven by technological innovation, changing patient expectations, and evolving regulatory frameworks.",
    "Artificial neural networks, inspired by the biological structure of the human brain, have demonstrated remarkable capabilities in pattern recognition, natural language processing, and decision-making tasks.",
    "The field of renewable energy has witnessed remarkable advancements in recent years, with solar and wind technologies achieving cost parity with traditional fossil fuels in many markets.",
    "Digital literacy has become an essential skill in the modern world, encompassing the ability to effectively navigate, evaluate, and create information using digital technologies.",
    "The intersection of ethics and artificial intelligence presents complex challenges that require careful consideration from policymakers, technologists, and society at large.",
    "Space exploration continues to captivate the human imagination and drive scientific discovery. Recent missions to Mars, advances in satellite technology, and the emergence of private space companies have ushered in a new era.",
]


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial AI text samples")
    parser.add_argument("--input", help="Input JSONL with AI texts (label=1)")
    parser.add_argument("--output", default="dataset_adversarial.jsonl")
    parser.add_argument("--attacks-per-text", type=int, default=3, help="Number of random attacks per text")
    args = parser.parse_args()

    # Load AI texts
    ai_texts = []
    if args.input and os.path.exists(args.input):
        with open(args.input) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("label") in (1, 2):  # AI or AI-polished
                    ai_texts.append(entry["text"])
        print(f"Loaded {len(ai_texts)} AI texts from {args.input}")
    else:
        ai_texts = BUILTIN_AI_TEXTS
        print(f"Using {len(ai_texts)} built-in AI texts")

    # Generate adversarial samples
    attack_names = list(ATTACKS.keys())
    samples = []

    for text in ai_texts:
        # Apply random subset of attacks
        selected = random.sample(attack_names, min(args.attacks_per_text, len(attack_names)))
        for attack_name in selected:
            attack_fn = ATTACKS[attack_name]
            try:
                adversarial = attack_fn(text)
                if len(adversarial.strip()) > 20:
                    samples.append({
                        "text": adversarial,
                        "label": 1,  # Still AI! Just disguised
                        "attack": attack_name,
                        "source": "adversarial_augmentation",
                    })
            except Exception as e:
                print(f"  Warning: {attack_name} failed: {e}")

    # Write output
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.output)
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Stats
    attack_counts = {}
    for s in samples:
        attack_counts[s["attack"]] = attack_counts.get(s["attack"], 0) + 1

    print(f"\nGenerated {len(samples)} adversarial samples → {output_path}")
    print("Attack distribution:")
    for name, count in sorted(attack_counts.items()):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
