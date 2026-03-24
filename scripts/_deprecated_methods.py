"""
⚠️ DEPRECATED — 以下四种 humanizer 方法经验证无效。

phrase / collocation / noise / splice 都是在 AI 原文上做局部替换，
AI 检测器仍然能识别出来。保留代码仅供参考，不再在生产中使用。

有效的方法（corpus, structure, transplant, inject, harvest, remix, anchor）
全部以人类语料库文本为底，见 humanizer.py。
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict

# ── Constants (only used by deprecated methods) ──────────────────────────────

AI_SYNONYMS = {
    'utilize': ['use', 'employ', 'rely on'],
    'implement': ['put in place', 'carry out', 'adopt'],
    'significant': ['big', 'major', 'real', 'notable'],
    'demonstrate': ['show', 'reveal', 'illustrate'],
    'facilitate': ['help', 'enable', 'make possible'],
    'comprehensive': ['thorough', 'complete', 'full'],
    'fundamental': ['basic', 'core', 'key', 'central'],
    'transform': ['change', 'reshape', 'alter', 'shift'],
    'remarkable': ['surprising', 'notable', 'striking', 'unusual'],
    'substantial': ['large', 'major', 'considerable', 'sizable'],
    'innovative': ['new', 'novel', 'fresh', 'creative'],
    'enhance': ['improve', 'boost', 'strengthen', 'lift'],
    'leverage': ['use', 'tap into', 'draw on', 'harness'],
    'optimize': ['improve', 'refine', 'streamline'],
    'paradigm': ['model', 'framework', 'approach'],
    'methodology': ['method', 'approach', 'technique'],
    'subsequently': ['then', 'later', 'after that', 'next'],
    'furthermore': ['also', 'besides', 'on top of that'],
    'additionally': ['also', 'plus', 'as well'],
    'consequently': ['as a result', 'so', 'therefore'],
    'nevertheless': ['still', 'even so', 'yet', 'all the same'],
    'approximately': ['about', 'roughly', 'around', 'close to'],
    'predominantly': ['mostly', 'mainly', 'largely', 'chiefly'],
    'revolutionized': ['changed', 'reshaped', 'upended', 'redefined'],
    'particularly': ['especially', 'notably', 'above all'],
    'extremely': ['very', 'highly', 'really'],
    'crucial': ['key', 'vital', 'essential', 'critical'],
    'diverse': ['varied', 'wide-ranging', 'mixed'],
    'robust': ['strong', 'solid', 'sturdy', 'resilient'],
    'overarching': ['broad', 'main', 'central', 'guiding'],
    'delve': ['look into', 'explore', 'dig into', 'examine'],
    'multifaceted': ['complex', 'layered', 'many-sided'],
    'holistic': ['whole', 'complete', 'integrated'],
    'underscore': ['highlight', 'stress', 'emphasize', 'point to'],
    'landscape': ['scene', 'field', 'terrain', 'picture'],
    'realm': ['area', 'field', 'domain', 'sphere'],
    'pivotal': ['key', 'central', 'decisive'],
    'foster': ['encourage', 'promote', 'support', 'nurture'],
    'bolster': ['strengthen', 'support', 'boost', 'shore up'],
    'hinder': ['block', 'slow', 'hamper', 'hold back'],
    'mitigate': ['reduce', 'lessen', 'ease', 'soften'],
    'exacerbate': ['worsen', 'intensify', 'aggravate'],
    'proliferation': ['spread', 'growth', 'rise', 'surge'],
    'encompasses': ['includes', 'covers', 'takes in'],
    'necessitate': ['require', 'demand', 'call for'],
    'transformative': ['game-changing', 'ground-breaking', 'radical'],
    'imperative': ['essential', 'urgent', 'pressing'],
    'detrimental': ['harmful', 'damaging', 'bad'],
    'commenced': ['started', 'began', 'kicked off'],
    'resided': ['lived', 'stayed', 'settled'],
    'pertaining': ['about', 'related to', 'concerning'],
    'aforementioned': ['earlier', 'previous', 'above'],
    'notwithstanding': ['despite', 'even though', 'regardless of'],
}

TRANSITION_PHRASES = [
    'In fact,', 'As it turns out,', 'Interestingly,', 'To be sure,',
    'Of course,', 'Naturally,', 'Indeed,', 'That said,', 'Still,',
    'Even so,', 'At the same time,', 'In practice,', 'In reality,',
    'As a matter of fact,', 'Truth be told,', 'By all accounts,',
    'On balance,', 'In hindsight,', 'For better or worse,',
]

GENERIC_VERBS = frozenset({
    'be', 'have', 'do', 'make', 'get', 'go', 'come', 'take', 'give',
    'say', 'know', 'think', 'see', 'want', 'use', 'find', 'tell',
    'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call', 'need',
    'keep', 'let', 'begin', 'show', 'hear', 'play', 'run', 'move',
    'like', 'live', 'believe', 'hold', 'bring', 'happen', 'write',
    'provide', 'sit', 'stand', 'lose', 'pay', 'meet', 'include',
    'continue', 'set', 'learn', 'change', 'lead', 'understand',
    'watch', 'follow', 'stop', 'create', 'speak', 'read', 'allow',
    'add', 'spend', 'grow', 'open', 'walk', 'win', 'offer', 'remember',
    'love', 'consider', 'appear', 'buy', 'wait', 'serve', 'die',
    'send', 'expect', 'build', 'stay', 'fall', 'reach', 'kill',
    'remain', 'suggest', 'raise', 'pass', 'sell', 'require', 'report',
})


# ── Helpers (imported from humanizer.py at call time) ────────────────────────

def match_case(source: str, template: str) -> str:
    if not template or not source:
        return source
    if template[0].isupper() and source[0].islower():
        return source[0].upper() + source[1:]
    if template[0].islower() and source[0].isupper():
        return source[0].lower() + source[1:]
    return source


# ── Deprecated method implementations ────────────────────────────────────────

def method_phrase(nlp, src_sent: str, matches: list[dict]) -> dict:
    """Swap non-subject noun chunks with human alternatives."""
    src_doc = nlp(src_sent)
    src_chunks = [c for c in src_doc.noun_chunks
                  if c.root.dep_ not in ('nsubj', 'nsubjpass')]
    if not src_chunks:
        src_chunks = list(src_doc.noun_chunks)

    match_chunks = defaultdict(list)
    all_match_chunks = []
    for m in matches[:15]:
        m_doc = nlp(m['text'])
        for chunk in m_doc.noun_chunks:
            match_chunks[chunk.root.dep_].append(chunk.text)
            all_match_chunks.append(chunk.text)

    result = src_sent
    swaps = []
    used_replacements = set()
    for chunk in src_chunks:
        role = chunk.root.dep_
        candidates = [c for c in match_chunks.get(role, [])
                      if c.lower() != chunk.text.lower()
                      and c.lower() not in used_replacements]
        if not candidates:
            candidates = [c for c in all_match_chunks
                          if c.lower() != chunk.text.lower()
                          and c.lower() not in used_replacements]
        if not candidates:
            continue
        most_common = Counter(candidates).most_common(1)[0][0]
        used_replacements.add(most_common.lower())
        most_common = match_case(most_common, chunk.text)
        result = result.replace(chunk.text, most_common, 1)
        swaps.append({'from': chunk.text, 'to': most_common})

    if not swaps and src_chunks and all_match_chunks:
        chunk = max(src_chunks, key=lambda c: len(c.text))
        replacement = all_match_chunks[0]
        replacement = match_case(replacement, chunk.text)
        result = result.replace(chunk.text, replacement, 1)
        swaps.append({'from': chunk.text, 'to': replacement})

    return {'text': result, 'swaps': swaps}


def method_collocation(nlp, src_sent: str, verb_for_noun: dict, adj_for_noun: dict) -> dict:
    """Replace verb+noun / adj+noun pairs with corpus-frequency alternatives."""
    from humanizer import conjugate_verb

    doc = nlp(src_sent)
    result = src_sent
    swaps = []

    for token in doc:
        if (token.pos_ == 'VERB'
                and token.dep_ not in ('aux', 'auxpass')
                and token.lemma_.lower() not in GENERIC_VERBS):
            for child in token.children:
                if child.dep_ not in ('dobj', 'attr'):
                    continue
                noun_lemma = re.sub(r'[^a-z]', '', child.lemma_.lower())
                if noun_lemma not in verb_for_noun:
                    continue
                counts = verb_for_noun[noun_lemma]
                verb_lemma = re.sub(r'[^a-z]', '', token.lemma_.lower())
                top = counts.most_common(5)
                top_verbs = [v for v, _ in top]
                if verb_lemma in top_verbs[:3]:
                    continue
                best_verb = top_verbs[0]
                if best_verb == verb_lemma or len(best_verb) < 2:
                    continue
                conjugated = conjugate_verb(best_verb, token.tag_)
                conjugated = match_case(conjugated, token.text)
                result = result.replace(token.text, conjugated, 1)
                swaps.append({'from': token.text, 'to': conjugated})

        if token.pos_ == 'ADJ' and token.dep_ == 'amod':
            noun_lemma = re.sub(r'[^a-z]', '', token.head.lemma_.lower())
            if noun_lemma not in adj_for_noun:
                continue
            counts = adj_for_noun[noun_lemma]
            adj_lemma = re.sub(r'[^a-z]', '', token.lemma_.lower())
            top = counts.most_common(5)
            top_adjs = [a for a, _ in top]
            if adj_lemma in top_adjs[:3]:
                continue
            best_adj = top_adjs[0]
            if best_adj == adj_lemma or len(best_adj) < 2:
                continue
            replacement = match_case(best_adj, token.text)
            result = result.replace(token.text, replacement, 1)
            swaps.append({'from': token.text, 'to': replacement})

    if not swaps:
        rng = random.Random(hashlib.md5(src_sent.encode()).hexdigest())
        for token in doc:
            word = token.text.lower()
            if word in AI_SYNONYMS:
                replacement = rng.choice(AI_SYNONYMS[word])
                replacement = match_case(replacement, token.text)
                result = result.replace(token.text, replacement, 1)
                swaps.append({'from': token.text, 'to': replacement})
                break
        if not swaps:
            for token in doc:
                if token.pos_ == 'ADJ' and len(token.text) > 3:
                    noun_lemma = re.sub(r'[^a-z]', '', token.head.lemma_.lower())
                    if noun_lemma in adj_for_noun:
                        top = adj_for_noun[noun_lemma].most_common(1)
                        if top and top[0][0] != token.lemma_.lower():
                            replacement = match_case(top[0][0], token.text)
                            result = result.replace(token.text, replacement, 1)
                            swaps.append({'from': token.text, 'to': replacement})
                            break

    return {'text': result, 'swaps': swaps}


def method_noise(src_sent: str) -> dict:
    """Inject human statistical patterns (transitions, synonym swaps)."""
    rng = random.Random(hashlib.md5(src_sent.encode()).hexdigest())
    result = src_sent

    phrase = rng.choice(TRANSITION_PHRASES)
    lower = result.lower()
    has_transition = any(lower.startswith(t.lower().rstrip(','))
                        for t in TRANSITION_PHRASES)
    if not has_transition:
        result = phrase + ' ' + result[0].lower() + result[1:]

    for ai_word, alternatives in AI_SYNONYMS.items():
        pattern = re.compile(r'\b' + re.escape(ai_word) + r'\b', re.IGNORECASE)
        match = pattern.search(result)
        if match:
            replacement = rng.choice(alternatives)
            original = match.group()
            replacement = match_case(replacement, original)
            result = result[:match.start()] + replacement + result[match.end():]

    return {'text': result}


def method_splice(nlp, src_sent: str, matches: list[dict]) -> dict:
    """Keep subject+verb, attach second half from semantic match."""
    src_doc = nlp(src_sent)

    root = None
    for token in src_doc:
        if token.dep_ == 'ROOT':
            root = token
            break

    if root is not None:
        split_idx = root.i + 1
        for child in sorted(root.rights, key=lambda t: t.i):
            if child.dep_ in ('prt', 'neg') and child.i == split_idx:
                split_idx = child.i + 1
            else:
                break

        if split_idx < len(src_doc) - 2:
            first_half = src_doc[:split_idx].text

            for m in matches[:20]:
                m_doc = nlp(m['text'])
                m_root = None
                for t in m_doc:
                    if t.dep_ == 'ROOT':
                        m_root = t
                        break
                if m_root is None:
                    continue

                m_split = m_root.i + 1
                for child in sorted(m_root.rights, key=lambda t: t.i):
                    if child.dep_ in ('prt', 'neg') and child.i == m_split:
                        m_split = child.i + 1
                    else:
                        break

                if m_split >= len(m_doc) - 2:
                    continue

                second_half = m_doc[m_split:].text
                if len(second_half.split()) < 3:
                    continue

                combined = first_half + ' ' + second_half
                if not combined.rstrip().endswith(('.', '!', '?')):
                    combined = combined.rstrip() + '.'
                return {
                    'text': combined,
                    'splitPoint': len(first_half.split()),
                }

    src_words = src_sent.split()
    mid = len(src_words) // 2
    if mid < 2:
        mid = 2
    first_half = ' '.join(src_words[:mid])
    if matches:
        m_words = matches[0]['text'].split()
        m_mid = len(m_words) // 2
        second_half = ' '.join(m_words[m_mid:])
        combined = first_half + ' ' + second_half
        if not combined.rstrip().endswith(('.', '!', '?')):
            combined = combined.rstrip() + '.'
        return {'text': combined, 'splitPoint': mid}

    return {'text': src_sent, 'splitPoint': 0}
