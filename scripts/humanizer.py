#!/usr/bin/env python3
"""HTTP server for humanizing AI-generated text.

Six humanization methods:
  - Corpus:      semantically similar human sentence, unmodified
  - Structure:   best template from semantic matches → fill with morphology
  - Phrase:      swap non-subject noun chunks with human alternatives
  - Collocation: replace verb+noun / adj+noun pairs with corpus-frequency alternatives
  - Noise:       inject human statistical patterns (transitions, synonym swaps)
  - Splice:      keep subject+verb, attach second half from semantic match

Usage:
    python3 scripts/humanizer.py [--port 5002] [--corpus-dir corpus]
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

import faiss
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer


# ── Memory-efficient sentence storage ────────────────────────────────────────

class SentenceStore:
    """Disk-backed sentence access via byte-offset index into JSONL file.

    Drop-in replacement for a Python list of strings.
    Memory: ~400MB (offset array) instead of ~8GB (50M Python strings).
    Access: seek + readline per sentence, < 1ms on SSD.
    """

    def __init__(self, jsonl_path: str):
        self._path = jsonl_path
        self._file = open(jsonl_path, 'r', encoding='utf-8')
        self._offsets = self._load_or_build_offsets()

    def _load_or_build_offsets(self) -> np.ndarray:
        cache_path = self._path + '.offsets.npy'
        jsonl_mtime = os.path.getmtime(self._path)

        if (os.path.exists(cache_path)
                and os.path.getmtime(cache_path) >= jsonl_mtime):
            print("  Loading cached offset index...", file=sys.stderr)
            return np.load(cache_path)

        print("  Building offset index (first time only)...", file=sys.stderr)
        offsets = []
        with open(self._path, 'rb') as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                offsets.append(pos)
        arr = np.array(offsets, dtype=np.int64)
        np.save(cache_path, arr)
        print(f"  Offset index: {len(arr):,} entries, "
              f"{arr.nbytes / (1024*1024):.0f}MB", file=sys.stderr)
        return arr

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = range(*key.indices(len(self)))
            return [self._read_one(i) for i in indices]
        return self._read_one(key)

    def _read_one(self, idx: int) -> str:
        self._file.seek(int(self._offsets[idx]))
        line = self._file.readline()
        return json.loads(line)['text']

    def __del__(self):
        if hasattr(self, '_file') and self._file and not self._file.closed:
            self._file.close()


# ── Constants ────────────────────────────────────────────────────────────────

IRREGULAR_PAST = {
    'be': 'was', 'have': 'had', 'do': 'did', 'go': 'went', 'get': 'got',
    'make': 'made', 'say': 'said', 'take': 'took', 'come': 'came', 'see': 'saw',
    'know': 'knew', 'give': 'gave', 'find': 'found', 'think': 'thought',
    'tell': 'told', 'become': 'became', 'show': 'showed', 'leave': 'left',
    'feel': 'felt', 'bring': 'brought', 'begin': 'began', 'keep': 'kept',
    'hold': 'held', 'write': 'wrote', 'stand': 'stood', 'lose': 'lost',
    'pay': 'paid', 'meet': 'met', 'run': 'ran', 'set': 'set', 'learn': 'learned',
    'lead': 'led', 'understand': 'understood', 'grow': 'grew', 'draw': 'drew',
    'break': 'broke', 'spend': 'spent', 'build': 'built', 'fall': 'fell',
    'send': 'sent', 'hit': 'hit', 'put': 'put', 'cut': 'cut', 'read': 'read',
    'sit': 'sat', 'speak': 'spoke', 'rise': 'rose', 'drive': 'drove',
    'buy': 'bought', 'win': 'won', 'teach': 'taught', 'eat': 'ate',
    'catch': 'caught', 'choose': 'chose', 'fight': 'fought', 'throw': 'threw',
}

IRREGULAR_PARTICIPLE = {
    'be': 'been', 'have': 'had', 'do': 'done', 'go': 'gone', 'get': 'gotten',
    'make': 'made', 'say': 'said', 'take': 'taken', 'come': 'come', 'see': 'seen',
    'know': 'known', 'give': 'given', 'find': 'found', 'think': 'thought',
    'tell': 'told', 'become': 'become', 'show': 'shown', 'leave': 'left',
    'feel': 'felt', 'bring': 'brought', 'begin': 'begun', 'keep': 'kept',
    'hold': 'held', 'write': 'written', 'stand': 'stood', 'lose': 'lost',
    'pay': 'paid', 'meet': 'met', 'run': 'run', 'set': 'set', 'learn': 'learned',
    'lead': 'led', 'understand': 'understood', 'grow': 'grown', 'draw': 'drawn',
    'break': 'broken', 'spend': 'spent', 'build': 'built', 'fall': 'fallen',
    'send': 'sent', 'hit': 'hit', 'put': 'put', 'cut': 'cut', 'read': 'read',
    'sit': 'sat', 'speak': 'spoken', 'rise': 'risen', 'drive': 'driven',
    'buy': 'bought', 'win': 'won', 'teach': 'taught', 'eat': 'eaten',
    'catch': 'caught', 'choose': 'chosen', 'fight': 'fought', 'throw': 'thrown',
}

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


# ── Helpers ──────────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if len(s.strip().split()) >= 3]


def conjugate_verb(lemma: str, target_tag: str) -> str:
    """Conjugate a verb lemma to match a target POS tag (VB/VBD/VBG/VBN/VBP/VBZ)."""
    if target_tag in ('VB', 'VBP'):
        return lemma
    if target_tag == 'VBZ':
        if lemma == 'be': return 'is'
        if lemma == 'have': return 'has'
        if lemma == 'do': return 'does'
        if lemma == 'go': return 'goes'
        if lemma.endswith(('s', 'sh', 'ch', 'x', 'z')):
            return lemma + 'es'
        if lemma.endswith('y') and len(lemma) > 1 and lemma[-2] not in 'aeiou':
            return lemma[:-1] + 'ies'
        return lemma + 's'
    if target_tag == 'VBD':
        if lemma in IRREGULAR_PAST:
            return IRREGULAR_PAST[lemma]
        if lemma.endswith('e'):
            return lemma + 'd'
        if lemma.endswith('y') and len(lemma) > 1 and lemma[-2] not in 'aeiou':
            return lemma[:-1] + 'ied'
        if (3 <= len(lemma) <= 5 and lemma[-1] not in 'aeiouwy'
                and lemma[-2] in 'aeiou' and lemma[-3] not in 'aeiou'):
            return lemma + lemma[-1] + 'ed'
        return lemma + 'ed'
    if target_tag == 'VBN':
        if lemma in IRREGULAR_PARTICIPLE:
            return IRREGULAR_PARTICIPLE[lemma]
        return conjugate_verb(lemma, 'VBD')
    if target_tag == 'VBG':
        if lemma == 'be': return 'being'
        if lemma == 'have': return 'having'
        if lemma.endswith('ie'):
            return lemma[:-2] + 'ying'
        if lemma.endswith('e') and not lemma.endswith('ee'):
            return lemma[:-1] + 'ing'
        if (3 <= len(lemma) <= 5 and lemma[-1] not in 'aeiouwy'
                and lemma[-2] in 'aeiou' and lemma[-3] not in 'aeiou'):
            return lemma + lemma[-1] + 'ing'
        return lemma + 'ing'
    return lemma


def pluralize_noun(word: str) -> str:
    if word.endswith(('s', 'sh', 'ch', 'x', 'z')):
        return word + 'es'
    if word.endswith('y') and len(word) > 1 and word[-2] not in 'aeiou':
        return word[:-1] + 'ies'
    if word.endswith('f'):
        return word[:-1] + 'ves'
    if word.endswith('fe'):
        return word[:-2] + 'ves'
    return word + 's'


def match_noun_form(source_word: str, template_tag: str) -> str:
    """Match singular/plural form of source word to template's tag."""
    if template_tag in ('NNS', 'NNPS') and not source_word.endswith('s'):
        return pluralize_noun(source_word)
    if template_tag in ('NN', 'NNP') and source_word.endswith('s'):
        if source_word.endswith('ies'):
            return source_word[:-3] + 'y'
        if source_word.endswith('ves'):
            return source_word[:-3] + 'f'
        if source_word.endswith('es') and source_word[-3] in 'shchxz':
            return source_word[:-2]
        if source_word.endswith('s') and not source_word.endswith('ss'):
            return source_word[:-1]
    return source_word


def match_case(source: str, template: str) -> str:
    """Match the casing pattern of template onto source."""
    if not template or not source:
        return source
    if template[0].isupper() and source[0].islower():
        return source[0].upper() + source[1:]
    if template[0].islower() and source[0].isupper():
        return source[0].lower() + source[1:]
    return source


# ── Humanizer ────────────────────────────────────────────────────────────────

class Humanizer:
    def __init__(self, corpus_dir: str, model_name: str = 'all-MiniLM-L6-v2'):
        index_path = os.path.join(corpus_dir, 'sentences.faiss')
        meta_path = os.path.join(corpus_dir, 'sentences.jsonl')

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            print(f"Corpus not found at {corpus_dir}/", file=sys.stderr)
            print("Run: python3 scripts/build_corpus.py", file=sys.stderr)
            sys.exit(1)

        print("Loading sentence transformer...", file=sys.stderr)
        self.model = SentenceTransformer(model_name)

        print("Loading FAISS index...", file=sys.stderr)
        self.index = faiss.read_index(index_path)
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = 64

        print("Loading sentence metadata...", file=sys.stderr)
        self.sentences = SentenceStore(meta_path)

        print("Loading spaCy...", file=sys.stderr)
        self.nlp = spacy.load('en_core_web_sm')

        print("Building indexes (structure + collocations)...", file=sys.stderr)
        self.struct_index, self.verb_for_noun, self.adj_for_noun = self._build_indexes()
        print(f"  {len(self.struct_index)} structures, "
              f"{len(self.verb_for_noun)} verb collocations, "
              f"{len(self.adj_for_noun)} adj collocations", file=sys.stderr)

        print(f"Ready. {len(self.sentences)} sentences in corpus.", file=sys.stderr)

    # ── Embedding & search ───────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True).astype(np.float32)

    def _semantic_search(self, query_embeddings: np.ndarray, k: int = 20) -> list[list[dict]]:
        scores, indices = self.index.search(query_embeddings, k)
        results = []
        for i in range(len(query_embeddings)):
            matches = []
            for j in range(k):
                idx = int(indices[i][j])
                if idx >= 0:
                    matches.append({'text': self.sentences[idx], 'score': float(scores[i][j])})
            results.append(matches)
        return results

    # ── Index building (structure + collocations in one pass) ────────────────

    def _structure_key(self, doc) -> str:
        core_deps = {'nsubj', 'nsubjpass', 'ROOT', 'dobj', 'pobj', 'attr',
                     'prep', 'pcomp', 'acomp', 'xcomp', 'ccomp', 'advcl'}
        return '|'.join(f"{t.pos_}:{t.dep_}" for t in doc if t.dep_ in core_deps)

    def _build_indexes(self) -> tuple[dict, dict, dict]:
        """Build structure index and collocation dictionaries in one spaCy pass."""
        sample_size = min(len(self.sentences), 50000)
        struct_idx = {}
        verb_for_noun = defaultdict(Counter)
        adj_for_noun = defaultdict(Counter)

        for start in range(0, sample_size, 1000):
            batch = self.sentences[start:start + 1000]
            docs = list(self.nlp.pipe(batch, disable=['ner']))
            for j, doc in enumerate(docs):
                # Structure index
                key = self._structure_key(doc)
                if key:
                    struct_idx.setdefault(key, []).append(start + j)

                # Collocations (strip non-alpha from lemmas to avoid curly quotes)
                for token in doc:
                    if (token.pos_ == 'VERB'
                            and token.dep_ not in ('aux', 'auxpass')
                            and token.lemma_.lower() not in GENERIC_VERBS):
                        verb_lemma = re.sub(r'[^a-z]', '', token.lemma_.lower())
                        if len(verb_lemma) < 2:
                            continue
                        for child in token.children:
                            if child.dep_ in ('dobj', 'attr'):
                                noun_lemma = re.sub(r'[^a-z]', '', child.lemma_.lower())
                                if len(noun_lemma) >= 2:
                                    verb_for_noun[noun_lemma][verb_lemma] += 1
                    if token.pos_ == 'ADJ' and token.dep_ == 'amod':
                        adj_lemma = re.sub(r'[^a-z]', '', token.lemma_.lower())
                        noun_lemma = re.sub(r'[^a-z]', '', token.head.lemma_.lower())
                        if len(adj_lemma) >= 2 and len(noun_lemma) >= 2:
                            adj_for_noun[noun_lemma][adj_lemma] += 1

        return struct_idx, dict(verb_for_noun), dict(adj_for_noun)

    # ── Utils ────────────────────────────────────────────────────────────────

    def _is_length_compatible(self, src: str, tpl: str, tolerance: float = 0.4) -> bool:
        src_words = len(src.split())
        tpl_words = len(tpl.split())
        if src_words == 0:
            return False
        ratio = tpl_words / src_words
        return (1 - tolerance) <= ratio <= (1 + tolerance)

    # ── Method 1: Corpus (semantic match, unmodified) ────────────────────────

    def _method_corpus(self, src_sent: str, matches: list[dict]) -> dict:
        # Prefer length-compatible match above threshold
        for m in matches:
            if m['score'] >= 0.55 and self._is_length_compatible(src_sent, m['text']):
                return {'text': m['text'], 'score': round(m['score'], 3)}
        # Fallback: best match regardless of length/threshold
        if matches:
            best = matches[0]
            return {'text': best['text'], 'score': round(best['score'], 3)}
        return {'text': src_sent, 'score': 0}

    # ── Method 2: Structure (find human sentence with identical grammar) ────

    def _method_structure(self, src_sent: str, matches: list[dict]) -> dict:
        """Find a corpus sentence with the same syntactic structure, unmodified."""
        src_doc = self.nlp(src_sent)
        src_key = self._structure_key(src_doc)
        if not src_key:
            if matches:
                return {'text': matches[0]['text'], 'score': 0}
            return {'text': src_sent, 'score': 0}

        src_parts = src_key.split('|')

        # Exact structure match from index
        exact_indices = self.struct_index.get(src_key, [])
        for idx in exact_indices:
            sent = self.sentences[idx]
            if self._is_length_compatible(src_sent, sent, tolerance=0.5):
                return {'text': sent, 'score': 1.0}
        # Return first exact match even if length doesn't match
        if exact_indices:
            return {'text': self.sentences[exact_indices[0]], 'score': 1.0}

        # Fuzzy: find closest structure (off by at most 1-2 elements)
        best_sent = None
        best_sim = 0
        for struct_key, indices in self.struct_index.items():
            other = struct_key.split('|')
            if abs(len(src_parts) - len(other)) > 2:
                continue
            max_len = max(len(src_parts), len(other))
            common = sum(1 for a, b in zip(src_parts, other) if a == b)
            sim = common / max_len if max_len > 0 else 0
            if sim > best_sim:
                for idx in indices[:3]:
                    sent = self.sentences[idx]
                    if self._is_length_compatible(src_sent, sent, tolerance=0.5):
                        best_sim = sim
                        best_sent = sent
                        break
                if not best_sent and indices:
                    best_sim = sim
                    best_sent = self.sentences[indices[0]]

        if best_sent:
            return {'text': best_sent, 'score': round(best_sim, 3)}

        # Fallback
        if matches:
            return {'text': matches[0]['text'], 'score': 0}
        return {'text': src_sent, 'score': 0}

    # ── Method 3: Entity Transplant (corpus + swap entities/numbers) ─────

    def _method_transplant(self, src_sent: str, corpus_sent: str) -> dict:
        """Take corpus sentence, replace only named entities and numbers with source's."""
        src_doc = self.nlp(src_sent)
        corp_doc = self.nlp(corpus_sent)

        # Collect source entities by label
        src_ents_by_label = defaultdict(list)
        for ent in src_doc.ents:
            src_ents_by_label[ent.label_].append(ent.text)
        src_nums = [t.text for t in src_doc if t.like_num or t.pos_ == 'NUM']

        result = corpus_sent
        swaps = []

        # Replace corpus entities with source entities of same type
        used_labels = defaultdict(int)
        for ent in corp_doc.ents:
            label = ent.label_
            idx = used_labels[label]
            if label in src_ents_by_label and idx < len(src_ents_by_label[label]):
                replacement = src_ents_by_label[label][idx]
                if replacement.lower() != ent.text.lower():
                    result = result.replace(ent.text, replacement, 1)
                    swaps.append({'from': ent.text, 'to': replacement})
                used_labels[label] = idx + 1

        # Replace corpus numbers with source numbers
        corp_nums = [t for t in corp_doc if t.like_num or t.pos_ == 'NUM']
        for i, corp_num in enumerate(corp_nums):
            if i < len(src_nums) and src_nums[i] != corp_num.text:
                result = result.replace(corp_num.text, src_nums[i], 1)
                swaps.append({'from': corp_num.text, 'to': src_nums[i]})

        return {'text': result, 'swaps': swaps}

    # ── Method 4: Fact Injection (corpus + replace objects with source's) ──

    def _method_inject(self, src_sent: str, corpus_sent: str) -> dict:
        """Take corpus sentence, replace dobj/pobj with source's objects."""
        src_doc = self.nlp(src_sent)
        corp_doc = self.nlp(corpus_sent)

        # Collect source object chunks
        src_objects = {}
        for chunk in src_doc.noun_chunks:
            if chunk.root.dep_ in ('dobj', 'pobj', 'attr', 'oprd'):
                src_objects.setdefault(chunk.root.dep_, []).append(chunk.text)

        # Build replacement map for corpus objects
        replacements = {}
        used = {}
        swaps = []
        for chunk in corp_doc.noun_chunks:
            role = chunk.root.dep_
            if role not in ('dobj', 'pobj', 'attr', 'oprd'):
                continue
            idx = used.get(role, 0)
            if role in src_objects and idx < len(src_objects[role]):
                replacement = src_objects[role][idx]
                words = replacement.split()
                if words:
                    words[-1] = match_noun_form(words[-1], chunk.root.tag_)
                    replacement = ' '.join(words)
                replacements[(chunk.start, chunk.end)] = replacement
                swaps.append({'from': chunk.text, 'to': replacement})
                used[role] = idx + 1

        # Reconstruct
        sorted_repls = sorted(replacements.items(), key=lambda x: x[0][0])
        result = []
        i = 0
        repl_idx = 0
        while i < len(corp_doc):
            if repl_idx < len(sorted_repls) and i == sorted_repls[repl_idx][0][0]:
                start, end = sorted_repls[repl_idx][0]
                repl = sorted_repls[repl_idx][1]
                repl = match_case(repl, corp_doc[start].text)
                result.append(repl)
                result.append(corp_doc[end - 1].whitespace_)
                i = end
                repl_idx += 1
            else:
                result.append(corp_doc[i].text_with_ws)
                i += 1

        return {'text': ''.join(result).strip(), 'swaps': swaps}

    # ── Method 5: Clause Harvest (combine clauses from multiple matches) ──

    def _method_harvest(self, src_sent: str, matches: list[dict]) -> dict:
        """Pick the best human clause for each clause in the source sentence."""
        # Split source into clauses
        clause_splits = re.split(
            r',\s+(?:and |but |or |yet |while |although |though |because )?|;\s+',
            src_sent)
        clauses = [c.strip() for c in clause_splits if len(c.strip().split()) >= 3]

        # Collect all human clauses from top matches
        all_human_clauses = []
        for m in matches[:15]:
            m_clauses = re.split(r',\s+|;\s+', m['text'])
            for c in m_clauses:
                c = c.strip()
                if len(c.split()) >= 3:
                    all_human_clauses.append(c)

        if not all_human_clauses:
            return {'text': matches[0]['text'] if matches else src_sent}

        if len(clauses) < 2:
            # Single clause: find best matching human clause by word overlap
            src_words = set(src_sent.lower().split())
            best = max(all_human_clauses,
                       key=lambda c: len(set(c.lower().split()) & src_words))
            text = best if best.rstrip().endswith(('.', '!', '?')) else best.rstrip() + '.'
            return {'text': text}

        # Multiple clauses: find best human clause for each
        result_clauses = []
        used = set()
        for clause in clauses:
            src_words = set(clause.lower().split())
            best = None
            best_overlap = -1
            best_idx = -1
            for i, hc in enumerate(all_human_clauses):
                if i in used:
                    continue
                overlap = len(set(hc.lower().split()) & src_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = hc
                    best_idx = i
            if best:
                result_clauses.append(best)
                used.add(best_idx)

        combined = ', '.join(result_clauses)
        if not combined.rstrip().endswith(('.', '!', '?')):
            combined = combined.rstrip() + '.'
        return {'text': combined}

    # ── Method 6: Corpus Remix (phrase-level fragments from multiple matches) ─

    def _method_remix(self, src_sent: str, matches: list[dict]) -> dict:
        """Build a new sentence from short human phrases harvested across matches."""
        src_doc = self.nlp(src_sent)

        # Extract source noun chunks + verb phrases as "slots" to fill
        src_slots = []
        for chunk in src_doc.noun_chunks:
            src_slots.append({
                'start': chunk.start, 'end': chunk.end,
                'text': chunk.text, 'dep': chunk.root.dep_, 'type': 'chunk',
            })
        for t in src_doc:
            if t.pos_ == 'VERB' and t.dep_ not in ('aux', 'auxpass'):
                already = any(s['start'] <= t.i < s['end'] for s in src_slots)
                if not already:
                    src_slots.append({
                        'start': t.i, 'end': t.i + 1,
                        'text': t.text, 'dep': t.dep_, 'type': 'verb',
                    })
        src_slots.sort(key=lambda s: s['start'])

        # Collect human phrases from matches, grouped by dep role
        human_chunks = defaultdict(list)
        human_verbs = defaultdict(list)
        for m in matches[:15]:
            m_doc = self.nlp(m['text'])
            for chunk in m_doc.noun_chunks:
                human_chunks[chunk.root.dep_].append(chunk.text)
            for t in m_doc:
                if t.pos_ == 'VERB' and t.dep_ not in ('aux', 'auxpass'):
                    human_verbs[t.dep_].append(t.text)

        # Build replacement map
        replacements = {}
        used_chunks = set()
        used_verbs = set()
        swaps = []
        for slot in src_slots:
            if slot['type'] == 'chunk':
                candidates = [c for c in human_chunks.get(slot['dep'], [])
                              if c.lower() not in used_chunks]
                if not candidates:
                    # Try any role
                    for role, chunks in human_chunks.items():
                        candidates = [c for c in chunks if c.lower() not in used_chunks]
                        if candidates:
                            break
                if candidates:
                    pick = Counter(candidates).most_common(1)[0][0]
                    used_chunks.add(pick.lower())
                    pick = match_case(pick, slot['text'])
                    replacements[(slot['start'], slot['end'])] = pick
                    swaps.append({'from': slot['text'], 'to': pick})
            elif slot['type'] == 'verb':
                candidates = [v for v in human_verbs.get(slot['dep'], [])
                              if v.lower() not in used_verbs
                              and v.lower() != slot['text'].lower()]
                if candidates:
                    pick = Counter(candidates).most_common(1)[0][0]
                    used_verbs.add(pick.lower())
                    pick = match_case(pick, slot['text'])
                    replacements[(slot['start'], slot['end'])] = pick
                    swaps.append({'from': slot['text'], 'to': pick})

        # Reconstruct: keep source's function words, replace content words
        sorted_repls = sorted(replacements.items(), key=lambda x: x[0][0])
        result = []
        i = 0
        repl_idx = 0
        while i < len(src_doc):
            if repl_idx < len(sorted_repls) and i == sorted_repls[repl_idx][0][0]:
                start, end = sorted_repls[repl_idx][0]
                repl = sorted_repls[repl_idx][1]
                result.append(repl)
                result.append(src_doc[end - 1].whitespace_)
                i = end
                repl_idx += 1
            else:
                result.append(src_doc[i].text_with_ws)
                i += 1

        return {'text': ''.join(result).strip(), 'swaps': swaps}

    # ── Method 7: Anchor + Append (corpus sentence + short factual tail) ─────

    def _method_anchor(self, src_sent: str, corpus_sent: str) -> dict:
        """Use corpus sentence as anchor, append key facts from source as a tail."""
        src_doc = self.nlp(src_sent)

        # Extract factual fragments: named entities, numbers, key noun chunks
        facts = []
        for ent in src_doc.ents:
            facts.append(ent.text)
        for t in src_doc:
            if (t.like_num or t.pos_ == 'NUM') and t.text not in facts:
                facts.append(t.text)
        # Also grab unique dobj/pobj not already in facts
        for chunk in src_doc.noun_chunks:
            if chunk.root.dep_ in ('dobj', 'pobj', 'attr'):
                if chunk.text not in facts and chunk.text.lower() not in corpus_sent.lower():
                    facts.append(chunk.text)

        if not facts:
            # No specific facts to append — at least grab the most "specific" chunk
            chunks = sorted(src_doc.noun_chunks, key=lambda c: len(c.text), reverse=True)
            for c in chunks:
                if c.text.lower() not in corpus_sent.lower():
                    facts.append(c.text)
                    break

        if not facts:
            return {'text': corpus_sent, 'swaps': []}

        # Build a short tail from the facts
        tail = ', '.join(facts[:3])

        # Clean up corpus sentence ending
        base = corpus_sent.rstrip()
        if base.endswith('.'):
            base = base[:-1]

        result = f"{base} — {tail}."
        swaps = [{'from': '(appended)', 'to': tail}]
        return {'text': result, 'swaps': swaps}

    # ⚠️ phrase / collocation / noise / splice 已移至 _deprecated_methods.py

    # ── Main humanize ────────────────────────────────────────────────────────

    def humanize(self, text: str, top_k: int = 20) -> dict:
        sentences = split_sentences(text)
        if not sentences:
            return {'error': 'No valid sentences found in input'}

        embeddings = self._embed(sentences)
        semantic_results = self._semantic_search(embeddings, k=top_k)

        details = []
        for src_sent, matches in zip(sentences, semantic_results):
            detail = self._process_sentence(src_sent, matches)
            details.append(detail)

        # Default output: pick first available method per sentence
        output_parts = []
        preference = ('structure', 'corpus', 'transplant', 'inject', 'harvest', 'remix', 'anchor')
        for d in details:
            methods = d['methods']
            picked = None
            for key in preference:
                if key in methods and methods[key] and methods[key].get('text'):
                    picked = methods[key]['text']
                    break
            output_parts.append(picked or d['original'])

        return {
            'humanized': ' '.join(output_parts),
            'sentenceCount': len(details),
            'details': details,
        }

    def _process_sentence(self, src_sent: str, matches: list[dict]) -> dict:
        # Corpus match is shared by corpus, transplant, inject
        corpus_result = self._method_corpus(src_sent, matches)
        corpus_text = corpus_result['text']

        return {
            'original': src_sent,
            'methods': {
                'corpus': corpus_result,
                'structure': self._method_structure(src_sent, matches),
                'transplant': self._method_transplant(src_sent, corpus_text),
                'inject': self._method_inject(src_sent, corpus_text),
                'harvest': self._method_harvest(src_sent, matches),
                'remix': self._method_remix(src_sent, matches),
                'anchor': self._method_anchor(src_sent, corpus_text),
                # ⚠️ 以下四种方法经验证无效（在原文上做局部替换，AI检测器仍可识别）。
                # 保留代码仅供参考，不再生成输出。
                # 'phrase': self._method_phrase(src_sent, matches),
                # 'collocation': self._method_collocation(src_sent),
                # 'noise': self._method_noise(src_sent),
                # 'splice': self._method_splice(src_sent, matches),
            },
        }


# ── HTTP Server ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length))
        text = body.get('text', '').strip()
        top_k = body.get('topK', 20)

        if not text:
            self._respond({'error': 'No text provided'})
            return

        result = self.server.humanizer.humanize(text, top_k=top_k)
        self._respond(result)

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def _respond(self, data: dict, status: int = 200):
        response = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(response)

    def _cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def log_message(self, format, *args):
        pass


class HumanizerServer(HTTPServer):
    allow_reuse_address = True

    def __init__(self, addr, handler, humanizer):
        super().__init__(addr, handler)
        self.humanizer = humanizer


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5002)
    parser.add_argument('--corpus-dir', default='corpus')
    args = parser.parse_args()

    humanizer = Humanizer(args.corpus_dir)
    print(f"Humanizer server running at http://127.0.0.1:{args.port}", file=sys.stderr)

    server = HumanizerServer(('127.0.0.1', args.port), Handler, humanizer)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", file=sys.stderr)
