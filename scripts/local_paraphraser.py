"""Local QuillBot-style paraphraser using spaCy + NLTK WordNet.

No API calls. Applies these transforms per sentence:
  1. Synonym replacement (WordNet-based, POS-aware)
  2. Sentence restructuring (active↔passive, clause reordering)
  3. Connector swapping (Furthermore→Additionally, etc.)
  4. Minor simplification (remove hedging, shorten phrases)

Usage:
    from local_paraphraser import paraphrase
    result = paraphrase("Your text here.")
"""

import random
import re
from typing import Optional

import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Try NLTK WordNet, fall back to built-in synonyms
try:
    from nltk.corpus import wordnet as wn
    # Trigger download if needed
    wn.synsets("test")
    HAS_WORDNET = True
except Exception:
    try:
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        from nltk.corpus import wordnet as wn
        HAS_WORDNET = True
    except Exception:
        HAS_WORDNET = False

# ── Built-in synonym map (fallback + supplement) ──

SYNONYM_MAP = {
    # Verbs
    "show": "demonstrate", "demonstrate": "illustrate",
    "use": "utilize", "utilize": "employ", "employ": "use",
    "help": "assist", "assist": "support", "support": "aid",
    "make": "create", "create": "produce", "produce": "generate",
    "get": "obtain", "obtain": "acquire", "acquire": "get",
    "give": "provide", "provide": "supply", "supply": "offer",
    "find": "discover", "discover": "identify", "identify": "determine",
    "start": "begin", "begin": "commence", "commence": "initiate",
    "end": "conclude", "conclude": "finish", "finish": "complete",
    "change": "alter", "alter": "modify", "modify": "adjust",
    "increase": "rise", "rise": "grow", "grow": "expand",
    "decrease": "decline", "decline": "diminish", "diminish": "reduce",
    "reduce": "lower", "lower": "decrease",
    "need": "require", "require": "necessitate",
    "allow": "enable", "enable": "permit", "permit": "allow",
    "cause": "lead to", "prevent": "hinder",
    "include": "encompass", "encompass": "comprise",
    "suggest": "indicate", "indicate": "imply",
    "affect": "influence", "influence": "impact",
    "improve": "enhance", "enhance": "strengthen",
    "maintain": "preserve", "preserve": "sustain",
    "develop": "establish", "establish": "build",
    "consider": "examine", "examine": "evaluate",
    # Adjectives
    "important": "significant", "significant": "crucial",
    "crucial": "essential", "essential": "vital",
    "good": "excellent", "excellent": "outstanding",
    "bad": "poor", "poor": "inadequate",
    "big": "large", "large": "substantial", "substantial": "considerable",
    "small": "minor", "minor": "slight",
    "many": "numerous", "numerous": "various",
    "different": "diverse", "diverse": "varied",
    "new": "novel", "novel": "innovative",
    "old": "traditional", "traditional": "conventional",
    "fast": "rapid", "rapid": "swift",
    "slow": "gradual", "gradual": "steady",
    "hard": "difficult", "difficult": "challenging",
    "easy": "simple", "simple": "straightforward",
    "clear": "evident", "evident": "apparent",
    "common": "widespread", "widespread": "prevalent",
    "main": "primary", "primary": "principal",
    "recent": "latest", "latest": "current",
    # Adverbs
    "often": "frequently", "frequently": "regularly",
    "also": "additionally", "additionally": "moreover",
    "however": "nevertheless", "nevertheless": "nonetheless",
    "therefore": "consequently", "consequently": "thus",
    "especially": "particularly", "particularly": "notably",
    "usually": "typically", "typically": "generally",
    "quickly": "rapidly", "rapidly": "swiftly",
    "slowly": "gradually",
    "really": "truly", "truly": "genuinely",
    # Nouns
    "problem": "issue", "issue": "challenge", "challenge": "obstacle",
    "way": "method", "method": "approach", "approach": "strategy",
    "part": "component", "component": "element",
    "result": "outcome", "outcome": "consequence",
    "area": "field", "field": "domain",
    "group": "category", "category": "class",
    "study": "research", "research": "investigation",
    "effect": "impact", "impact": "influence",
    "goal": "objective", "objective": "aim",
    "reason": "factor", "factor": "cause",
    "idea": "concept", "concept": "notion",
    "example": "instance", "instance": "case",
    "feature": "characteristic", "characteristic": "attribute",
    "advantage": "benefit", "benefit": "merit",
}

# ── Connector/transition swaps ──

CONNECTOR_SWAPS = {
    "furthermore": "additionally",
    "additionally": "moreover",
    "moreover": "what is more",
    "however": "nevertheless",
    "nevertheless": "nonetheless",
    "nonetheless": "even so",
    "therefore": "as a result",
    "as a result": "consequently",
    "consequently": "thus",
    "thus": "hence",
    "in addition": "besides",
    "besides": "apart from that",
    "for example": "for instance",
    "for instance": "as an illustration",
    "in conclusion": "to sum up",
    "to sum up": "in summary",
    "on the other hand": "conversely",
    "conversely": "in contrast",
    "in contrast": "on the other hand",
    "meanwhile": "in the meantime",
    "although": "even though",
    "even though": "despite the fact that",
    "because": "since",
    "since": "as",
    "while": "whereas",
    "whereas": "while",
    "it is important to note that": "it should be noted that",
    "it should be noted that": "notably",
    "it is worth noting that": "it bears mentioning that",
}

# ── Hedging phrases to simplify ──

HEDGING_REMOVALS = [
    (r"\bit is important to note that\b", ""),
    (r"\bit should be noted that\b", ""),
    (r"\bit is worth mentioning that\b", ""),
    (r"\bin order to\b", "to"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bin spite of the fact that\b", "despite"),
    (r"\bat this point in time\b", "now"),
    (r"\bfor the purpose of\b", "to"),
    (r"\bin the event that\b", "if"),
    (r"\bhas the ability to\b", "can"),
    (r"\bis able to\b", "can"),
    (r"\ba large number of\b", "many"),
    (r"\ba significant amount of\b", "much"),
]


def _get_wordnet_synonym(word: str, pos_tag: str) -> Optional[str]:
    """Get a WordNet synonym respecting POS."""
    if not HAS_WORDNET:
        return None

    pos_map = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}
    wn_pos = pos_map.get(pos_tag)
    if not wn_pos:
        return None

    synsets = wn.synsets(word, pos=wn_pos)
    candidates = set()
    for syn in synsets[:3]:
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower() and len(name) > 2:
                candidates.add(name)

    if not candidates:
        return None
    return random.choice(list(candidates))


def _swap_synonym(token, rate: float = 0.25) -> str:
    """Try to replace a token with a synonym."""
    if random.random() > rate:
        return token.text

    word = token.text.lower()
    pos = token.pos_

    # Skip short words, stop words, proper nouns, punctuation
    if len(word) < 4 or token.is_stop or token.is_punct or pos == "PROPN":
        return token.text

    # Try built-in map first
    if word in SYNONYM_MAP:
        replacement = SYNONYM_MAP[word]
        if token.text[0].isupper():
            replacement = replacement.capitalize()
        return replacement

    # Try WordNet
    wn_syn = _get_wordnet_synonym(word, pos)
    if wn_syn:
        if token.text[0].isupper():
            wn_syn = wn_syn.capitalize()
        return wn_syn

    return token.text


def _swap_connectors(text: str) -> str:
    """Replace transition words/phrases."""
    lower = text.lower()
    for old, new in CONNECTOR_SWAPS.items():
        pattern = r"\b" + re.escape(old) + r"\b"
        match = re.search(pattern, lower)
        if match:
            start, end = match.start(), match.end()
            original = text[start:end]
            # Preserve capitalization
            if original[0].isupper():
                new = new[0].upper() + new[1:]
            text = text[:start] + new + text[end:]
            break  # One swap per call to keep it natural
    return text


def _simplify_hedging(text: str) -> str:
    """Remove verbose hedging phrases."""
    for pattern, replacement in HEDGING_REMOVALS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE, count=1)
    # Clean up double spaces
    text = re.sub(r"  +", " ", text).strip()
    # Fix sentence start after removal
    text = re.sub(r"^\s*,\s*", "", text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _restructure_sentence(doc) -> Optional[str]:
    """Try active→passive or clause reordering."""
    # Find subject-verb-object pattern for active→passive
    subj = None
    verb = None
    obj = None

    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass") and not subj:
            subj = token
        if token.pos_ == "VERB" and token.dep_ == "ROOT":
            verb = token
        if token.dep_ in ("dobj", "attr") and not obj:
            obj = token

    if not (subj and verb and obj):
        return None

    # Only restructure ~30% of eligible sentences
    if random.random() > 0.3:
        return None

    # Simple clause reordering: move prepositional phrase to front
    for token in doc:
        if token.dep_ == "prep" and token.head == verb:
            prep_span = doc[token.i:token.head.i]
            if len(prep_span) >= 3:
                prep_text = doc[token.i:].text.split(",")[0]
                main_text = doc[:token.i].text.rstrip(" ,")
                return f"{prep_text}, {main_text}."

    return None


def paraphrase_sentence(text: str, synonym_rate: float = 0.25) -> str:
    """Paraphrase a single sentence using local transforms."""
    # Step 1: Simplify hedging
    text = _simplify_hedging(text)

    # Step 2: Swap connectors
    text = _swap_connectors(text)

    # Step 3: Parse and do synonym replacement
    doc = nlp(text)

    # Step 4: Try restructuring (30% chance)
    restructured = _restructure_sentence(doc)
    if restructured:
        # Re-parse the restructured version for synonym replacement
        doc = nlp(restructured)

    # Step 5: Synonym replacement
    tokens = []
    for token in doc:
        if token.is_punct or token.is_space:
            tokens.append(token.text)
        else:
            tokens.append(_swap_synonym(token, rate=synonym_rate))

    result = ""
    for i, token in enumerate(doc):
        if i > 0 and not token.is_punct and token.text not in ("'s", "n't", "'t", "'re", "'ve", "'ll", "'d"):
            result += " "
        result += tokens[i]

    # Clean up
    result = re.sub(r"  +", " ", result).strip()
    return result


def paraphrase(text: str, synonym_rate: float = 0.25) -> str:
    """Paraphrase full text, sentence by sentence."""
    doc = nlp(text)
    sentences = list(doc.sents)

    paraphrased = []
    for sent in sentences:
        sent_text = sent.text.strip()
        if len(sent_text.split()) < 5:
            paraphrased.append(sent_text)
        else:
            paraphrased.append(paraphrase_sentence(sent_text, synonym_rate))

    return " ".join(paraphrased)


if __name__ == "__main__":
    test = (
        "The rapid advancement of technology has fundamentally transformed "
        "the landscape of modern education. In today's interconnected world, "
        "students have unprecedented access to a wealth of information and "
        "resources that were previously unavailable. Furthermore, the integration "
        "of artificial intelligence into educational platforms has created new "
        "opportunities for personalized learning experiences. It is important "
        "to note that these technological innovations not only enhance the "
        "learning process but also prepare students for the demands of an "
        "increasingly digital workforce."
    )
    print("ORIGINAL:")
    print(test)
    print()

    random.seed(42)
    result = paraphrase(test, synonym_rate=0.3)
    print("PARAPHRASED:")
    print(result)
