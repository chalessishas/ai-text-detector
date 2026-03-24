# Verified Human-Written Academic Text Sources

Research for AI text detector training data. Goal: 20,000+ high-confidence human-written samples covering diverse academic disciplines.

**Key principle:** Pre-2022 text is strongly preferred because widespread LLM use (ChatGPT: Nov 2022, GPT-3 API: Jun 2020) introduces contamination risk. Pre-2020 is ideal.

---

## Current Project Sources (build_corpus_colab.py)

Our existing corpus pipeline already uses 5 sources with these weights:

| Source | Weight | Notes |
|--------|--------|-------|
| C4 (allenai/c4) | 30% | General web text, filtered |
| Wikipedia (wikitext-103) | 20% | Encyclopedia articles |
| CC-News | 20% | News articles, pre-2019 only |
| CNN/DailyMail | 15% | News summaries |
| Project Gutenberg | 15% | Classic literature |

**Gap:** Heavy on news/web/literature, weak on academic writing (research papers, student essays, textbook-style content). This matters because our detector targets academic contexts.

---

## Source-by-Source Analysis

### 1. C4 (Colossal Clean Crawled Corpus)

**Already in use (30% of corpus)**

- **Access:** `load_dataset('allenai/c4', 'en', split='train', streaming=True)` via HuggingFace. Also on TensorFlow Datasets and raw download from AllenAI.
- **Volume:** ~365 million pages (~156 billion tokens). Effectively unlimited for our needs.
- **Domain:** General web text. Cleaned from Common Crawl April 2019 snapshot. Some academic content mixed in, but not categorized.
- **License:** ODC-BY (attribution required). Based on Common Crawl data (no explicit license, treated as fair use for research).
- **Human confidence:** HIGH for pre-2020 snapshot. The April 2019 crawl predates widespread LLM use. Some GPT-2 contamination theoretically possible but negligible.
- **Limitations:** Not academic-specific. Quality varies widely. Deduplication applied but not perfect.

**Verdict:** Good as backfill. Not a solution for academic domain coverage.

---

### 2. Wikipedia

**Already in use (20% via wikitext-103)**

- **Access:** Multiple options:
  - `wikitext-103-v1` on HuggingFace (current approach) — only ~100M tokens
  - Full Wikipedia dump: `dumps.wikimedia.org` — XML format, monthly releases
  - HuggingFace `wikipedia` dataset — preprocessed, one split per language
  - **Pre-ChatGPT snapshot:** Academic Torrents has a May 2022 Kiwix dump specifically preserved as a pre-LLM reference point
- **Volume:** ~6.7 million English articles. wikitext-103 is a small subset (~28K articles).
- **Domain:** Broad encyclopedia coverage including STEM, humanities, social sciences. Category system allows filtering to academic topics.
- **License:** CC BY-SA 3.0. Free for any use with attribution and share-alike.
- **Human confidence:** HIGH for pre-2022 dumps. Wikipedia had strict editorial oversight pre-ChatGPT. Post-2023 articles have known AI contamination issues.
- **Limitations:** Encyclopedia style, not research paper style. Wikitext-103 is too small; should upgrade to full dump or larger HuggingFace split.

**Recommendation:** Switch from wikitext-103 to the full `wikipedia` HuggingFace dataset. Use the `20220301.en` snapshot for maximum pre-LLM confidence. Filter by academic categories (Science, Mathematics, History, etc.) for domain relevance.

---

### 3. arXiv Abstracts (pre-2022)

**NOT currently used. HIGH PRIORITY addition.**

- **Access:**
  - **Kaggle dataset** (Cornell University): ~1.7M+ articles with metadata + abstracts in JSON. Free download, no AWS costs. URL: `kaggle.com/datasets/Cornell-University/arxiv`
  - **HuggingFace:** `arxiv-community/arxiv_dataset` — preprocessed
  - **OAI-PMH API:** `export.arxiv.org` — bulk metadata harvesting, free
  - **S3 bulk download:** Full PDFs on AWS S3 (requester-pays, ~$100 for 1.1TB). Overkill for abstracts only.
- **Volume:** ~2.3M+ papers total. Pre-2022 subset: ~1.8M papers. Each abstract is 100-300 words. **At least 1.8 million abstracts available pre-2022.**
- **Domain:** Physics, Mathematics, Computer Science, Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, Economics. Strongest in STEM.
- **License:** Metadata and abstracts are freely available. Full text has mixed licenses per paper.
- **Human confidence:** VERY HIGH for pre-2020. arXiv papers are peer-reviewed or at least author-submitted academic work. No AI writing tools existed at scale before 2020. Pre-2022 is still very safe.
- **Limitations:** STEM-heavy. Humanities/social sciences underrepresented. Abstract style is specific (dense, jargon-heavy) — may not generalize to student essays.

**Recommendation:** Use the Kaggle dataset. Filter by `update_date < 2022-01-01`. Sample across all categories for diversity. Target 20,000-50,000 abstracts. This is the single best source for verified human academic writing.

---

### 4. PubMed Abstracts

**NOT currently used. HIGH PRIORITY addition.**

- **Access:**
  - **NCBI FTP baseline:** `ftp.ncbi.nlm.nih.gov/pubmed/baseline/` — annual XML dump of all citations
  - **HuggingFace:** `ncbi/pubmed` dataset — ~360GB, preprocessed
  - **E-utilities API:** Programmatic access, rate-limited (3 req/sec without API key, 10 with)
  - **Kaggle:** `bonhart/pubmed-abstracts` — smaller curated subset
  - **Bulk download tool:** `erilu/pubmed-abstract-compiler` (Python, keyword-based)
- **Volume:** 36M+ citations total. ~21M have English abstracts. Pre-2022: ~19M+ abstracts available.
- **Domain:** Biomedical and life sciences. Covers medicine, biology, chemistry, pharmacology, public health, neuroscience, genetics. Strongest in health/bio.
- **License:** NLM provides data freely for research. Individual abstract copyright varies by publisher, but metadata + abstracts are generally considered fair use for research/ML training.
- **Human confidence:** VERY HIGH. Peer-reviewed medical literature. Pre-2022 PubMed abstracts were written entirely by human researchers. Even post-2022 contamination is lower than web text because of editorial standards.
- **Limitations:** Narrow domain (biomedical only). Dense scientific writing style. May not represent student-level writing.

**Recommendation:** Use HuggingFace `ncbi/pubmed` or the Kaggle subset. Filter `year < 2022`. Sample 10,000-20,000 abstracts across MeSH categories for domain diversity within biomedicine.

---

### 5. JSTOR Data for Research (DfR)

**NOT currently used. MEDIUM PRIORITY.**

- **Access:**
  - **DfR portal:** `jstor.org/dfr/` — request datasets up to 25,000 documents per request. Returns word counts, n-grams, and metadata (NOT full text).
  - **Early Journal Content (EJC):** Pre-1923 US / pre-1870 international articles. Includes full-text OCR. Public domain.
  - **Full text:** Available only via NDA agreement with JSTOR for approved research.
- **Volume:** 12M+ articles in JSTOR total. EJC subset: ~500K articles.
- **Domain:** EXCELLENT diversity. Covers all academic disciplines — humanities, social sciences, natural sciences, law, economics, literature, philosophy. This is the broadest academic source.
- **License:** EJC is public domain. Modern articles require DfR agreement (non-commercial research). Full text requires separate NDA.
- **Human confidence:** VERY HIGH. Published journal articles, all pre-LLM for the EJC subset (pre-1923/1870). Even modern JSTOR articles pre-2022 are human-written.
- **Limitations:** Full text access is restricted. DfR only provides n-grams, not running text. EJC full text is OCR quality (noisy). The NDA process adds friction.

**Recommendation:** Low ROI for effort. The EJC full text is pre-1923 and stylistically very different from modern academic writing. The DfR n-gram data is not useful for our sentence-level detector. Skip unless we specifically need humanities coverage and are willing to go through the NDA process.

---

### 6. Project Gutenberg

**Already in use (15% of corpus)**

- **Access:** `load_dataset('sedthh/gutenberg_english', split='train', streaming=True)` via HuggingFace. Also: direct mirror download, rsync, or `gutenbergr` R package.
- **Volume:** ~70,000 English-language books.
- **Domain:** Literature, philosophy, history, science (all pre-copyright). Strongest in 19th-20th century fiction and nonfiction.
- **License:** Public domain in the US (pre-1928). International rights vary.
- **Human confidence:** EXTREMELY HIGH. All works predate computers entirely.
- **Limitations:** Writing style is archaic. Not representative of modern academic writing at all. Useful for "definitely human" baseline but not for academic domain matching.

**Verdict:** Keep at 15% for human baseline diversity, but don't increase. The style gap hurts more than helps for academic detection.

---

### 7. Reddit Academic Subreddits (via Pushshift)

**NOT currently used. LOW-MEDIUM PRIORITY.**

- **Access:**
  - **Pushshift archive:** Full Reddit data 2005-2022 available via Academic Torrents and `files.pushshift.io`. Zstandard-compressed NDJSON files.
  - **Per-subreddit files:** Top 20,000 subreddits available as separate downloads on Academic Torrents, so you can download only academic subs.
  - Relevant subreddits: r/AskHistorians, r/AskScience, r/ExplainLikeImFive, r/philosophy, r/economics, r/science, r/AcademicPhilosophy, r/AskSocialScience
- **Volume:** Billions of comments/posts total. Academic subreddits: likely 1-5M substantial posts pre-2022.
- **Domain:** Varies by subreddit. r/AskHistorians has high-quality long-form writing. r/AskScience has expert-level explanations. Mix of formal and informal.
- **License:** Reddit content is user-generated. Reddit's ToS grants them a license; research use is a gray area post-2023 API changes. Pushshift archive was created when this was more permissible.
- **Human confidence:** HIGH for pre-2022. Reddit posts before ChatGPT were human-written. Some GPT-3 generated content possible in 2021-2022 but rare on academic subreddits.
- **Limitations:** Informal writing style (even on academic subs). Quality varies enormously. Requires heavy filtering. Legal gray area.

**Recommendation:** Good supplementary source for informal academic discussion. Use r/AskHistorians and r/AskScience pre-2022 data. These have strict moderation and long-form answers that resemble academic explanations. Target 5,000-10,000 high-quality posts (filter by score > 10, length > 200 words).

---

### 8. OpenStax Textbooks

**NOT currently used. LOW PRIORITY.**

- **Access:** Free PDF/EPUB/web download from `openstax.org`. No API for bulk text extraction. Would need PDF parsing or web scraping.
- **Volume:** ~90 textbooks. Each book is 500-1000 pages. Total: ~45,000-90,000 pages of content.
- **Domain:** Undergraduate-level STEM and social sciences. Subjects include: Biology, Chemistry, Physics, Calculus, Statistics, Psychology, Sociology, Economics, US History, Anatomy.
- **License:** CC BY 4.0 (most books) or CC BY-NC-SA (Calculus). Free to use, adapt, and redistribute with attribution.
- **Human confidence:** EXTREMELY HIGH. Peer-reviewed textbooks written by named professors, published by Rice University. Zero AI contamination.
- **Limitations:** Only ~90 books — limited volume. Textbook style is pedagogical, different from research papers or essays. PDF extraction introduces noise. No bulk download API.

**Recommendation:** Nice-to-have but low ROI. The volume is too small to justify the extraction effort. If we need textbook-style content, better to sample from Wikipedia's academic articles (similar explanatory style, much more volume).

---

### 9. Student Essay Datasets (ASAP, ETS)

**NOT currently used. MEDIUM PRIORITY for domain matching.**

- **ASAP (Automated Student Assessment Prize):**
  - Access: Kaggle competition dataset (`kaggle.com/c/asap-aes`). Free download with Kaggle account.
  - Volume: ~17,450 essays (original ASAP). ASAP 2.0: ~24,000 essays.
  - Domain: 7th-10th grade US students. 8 different prompts covering various topics.
  - License: Released for the Kaggle competition. Research use permitted.
  - Human confidence: VERY HIGH. Standardized test essays collected pre-2012. Zero AI contamination.
  - Limitation: Middle/high school level writing, not college or graduate.

- **ETS Corpus of Non-Native Written English (TOEFL11):**
  - Access: Linguistic Data Consortium (LDC). Requires LDC membership or purchase ($150 for non-members).
  - Volume: 12,100 essays from TOEFL iBT test-takers.
  - Domain: English language proficiency essays from 11 L1 backgrounds.
  - License: LDC license — research use, non-redistributable.
  - Human confidence: VERY HIGH. Handwritten/typed under exam conditions pre-2013.
  - Limitation: Non-native English. Specific essay prompts. Paid access.

- **ICNALE (International Corpus Network of Asian Learners of English):**
  - Access: Free registration at `language.sakura.ne.jp/icnale/`
  - Volume: ~10,000 essays from Asian English learners + native speakers
  - Domain: Argumentative essays on standardized topics
  - Human confidence: VERY HIGH (exam conditions)

**Recommendation:** ASAP is the most accessible and relevant. 17K-24K essays of student writing is exactly the domain our detector targets. Download from Kaggle. ASAP 2.0 is even better with 24K essays and demographic data.

---

### 10. News Archives (CC-News, CNN/DailyMail)

**Already in use (35% combined)**

- **CC-News:** ~708K English articles (2017-2019 on HuggingFace). Our pipeline already filters to pre-2019.
- **CNN/DailyMail:** ~300K articles with summaries. All pre-2017 collection.
- **Additional option — All The News:** Kaggle dataset with 143K articles from 15 publications (2016-2017).
- **Human confidence:** VERY HIGH. Pre-2019 news articles were human-written.

**Verdict:** Already well-represented. No need to add more news sources.

---

### 11. Additional Sources Worth Noting

#### Semantic Scholar Open Research Corpus (S2ORC)
- **Access:** AllenAI provides 81.1M papers with structured full text. Available via HuggingFace or direct download.
- **Volume:** 81.1M papers, 8.1M with full text parsed.
- **Domain:** All academic disciplines (broader than arXiv).
- **License:** CC BY-NC 2.0. Non-commercial use.
- **Human confidence:** VERY HIGH (published research).
- **Worth investigating** as a complement to arXiv for humanities/social science coverage.

#### HC3 (Human ChatGPT Comparison Corpus)
- **Access:** HuggingFace `Hello-SimpleAI/HC3`
- **Volume:** ~37K human answers to 27K questions (reddit, medicine, finance, law domains)
- **Already labeled** as human vs. AI — directly usable for detector training.
- **Human confidence:** HIGH (curated from known human sources).

#### RAID Benchmark
- **Access:** `github.com/liamdugan/raid`
- **Volume:** 6M+ texts spanning 11 LLMs and 8 domains
- **Already formatted** for AI detection evaluation. Includes human baseline texts.

#### GPABench
- **Volume:** 600K samples (human-written + GPT-written abstracts in CS, physics, humanities)
- **Pre-labeled** human vs. AI. Directly usable.

---

## Recommended Acquisition Plan

Priority order for adding new sources to reach 20K+ diverse academic samples:

| Priority | Source | Target Samples | Effort | Domain Gap Filled |
|----------|--------|---------------|--------|-------------------|
| 1 | **arXiv abstracts** (Kaggle) | 20,000 | LOW (JSON download + filter) | STEM research |
| 2 | **PubMed abstracts** (HuggingFace) | 10,000 | LOW (streaming dataset) | Biomedical |
| 3 | **ASAP 2.0 essays** (Kaggle) | 24,000 | LOW (CSV download) | Student writing |
| 4 | **HC3 human answers** | 37,000 | LOW (HuggingFace) | Mixed academic Q&A |
| 5 | **Wikipedia full dump** (upgrade) | 20,000 | MEDIUM (larger dataset) | General academic |
| 6 | **Reddit academic subs** | 5,000 | MEDIUM (Pushshift filter) | Informal academic |
| 7 | **S2ORC** | 10,000 | MEDIUM (large download) | Humanities/social sci |

**Total achievable with priorities 1-4 alone: ~91,000 verified human samples.**

### Implementation Notes

1. **arXiv + PubMed** together give excellent STEM coverage with minimal effort (both available as streaming HuggingFace datasets or Kaggle downloads).
2. **ASAP essays** are the most domain-relevant source — actual student writing is exactly what our detector will encounter in the real world.
3. **HC3** is pre-labeled, saving annotation effort.
4. All Priority 1-4 sources can be acquired in a single Python script session, no web scraping needed.
5. Pre-2022 filtering should be applied to all sources. Pre-2020 is ideal.
6. Deduplicate across sources using sentence-level hashing (already implemented in `build_corpus_colab.py`).

---

*Research conducted 2026-03-24. Sources verified via web search and project codebase analysis.*
