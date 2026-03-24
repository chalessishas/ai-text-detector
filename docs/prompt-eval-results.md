# Prompt Evaluation Results

Generated: 2026-03-21T19:44:58.566Z
Model: deepseek-chat
Runs per article: 3

## Summary

**128/144** checks passed (**88.9%**)

## Results by Article

### 01-poor-essay.md (poor) — 10/12

| # | Criterion | Result | Detail |
|---|-----------|--------|--------|
| 1 | 1. Honesty | FAIL | Issue counts: 3, 3, 2. Need ≥3 each. |
| 2 | 2. Rewrite | PASS | All suggestion/issue annotations have rewrite |
| 3 | 3. No empty praise | PASS | No empty praise found |
| 4 | 4. Good specificity | PASS | All good annotations cite specific text |
| 5 | 5. Question quality | PASS | All questions end with ? |
| 6 | 6. Liz Lerman order | PASS | Liz Lerman order correct |
| 7 | 7. Suppression fires | PASS | Suppression fired correctly |
| 8 | 8. Suppression no false-fire | PASS | N/A (not article 11) |
| 9 | 9. Score calibration | PASS | Poor article scores calibrated |
| 10 | 10. Message length | PASS | All messages ≥ 15 words |
| 11 | 11. Good minimum | PASS | ≥ 2 good annotations present |
| 12 | 12. Score stability | FAIL | conventions: spread=15 (35,50,40) |

### 02-poor-creative.md (poor) — 11/12

| # | Criterion | Result | Detail |
|---|-----------|--------|--------|
| 1 | 1. Honesty | FAIL | Issue counts: 2, 2, 2. Need ≥3 each. |
| 2 | 2. Rewrite | PASS | All suggestion/issue annotations have rewrite |
| 3 | 3. No empty praise | PASS | No empty praise found |
| 4 | 4. Good specificity | PASS | All good annotations cite specific text |
| 5 | 5. Question quality | PASS | All questions end with ? |
| 6 | 6. Liz Lerman order | PASS | Liz Lerman order correct |
| 7 | 7. Suppression fires | PASS | Suppression fired correctly |
| 8 | 8. Suppression no false-fire | PASS | N/A (not article 11) |
| 9 | 9. Score calibration | PASS | Poor article scores calibrated |
| 10 | 10. Message length | PASS | All messages ≥ 15 words |
| 11 | 11. Good minimum | PASS | ≥ 2 good annotations present |
| 12 | 12. Score stability | PASS | All traits within ±10 across runs |

### 03-avg-essay.md (average) — 11/12

| # | Criterion | Result | Detail |
|---|-----------|--------|--------|
| 1 | 1. Honesty | PASS | N/A (not a poor article) |
| 2 | 2. Rewrite | PASS | All suggestion/issue annotations have rewrite |
| 3 | 3. No empty praise | PASS | No empty praise found |
| 4 | 4. Good specificity | PASS | All good annotations cite specific text |
| 5 | 5. Question quality | PASS | All questions end with ? |
| 6 | 6. Liz Lerman order | PASS | Liz Lerman order correct |
| 7 | 7. Suppression fires | PASS | Suppression fired correctly |
| 8 | 8. Suppression no false-fire | PASS | N/A (not article 11) |
| 9 | 9. Score calibration | PASS | N/A (not poor or excellent) |
| 10 | 10. Message length | PASS | All messages ≥ 15 words |
| 11 | 11. Good minimum | PASS | ≥ 2 good annotations present |
| 12 | 12. Score stability | FAIL | ideas: spread=15 (45,55,60); organization: spread=15 (40,45,55) |

### 04-avg-article.md (average) — 12/12

| # | Criterion | Result | Detail |
|---|-----------|--------|--------|
| 1 | 1. Honesty | PASS | N/A (not a poor article) |
| 2 | 2. Rewrite | PASS | All suggestion/issue annotations have rewrite |
| 3 | 3. No empty praise | PASS | No empty praise found |
| 4 | 4. Good specificity | PASS | All good annotations cite specific text |
| 5 | 5. Question quality | PASS | All questions end with ? |
| 6 | 6. Liz Lerman order | PASS | Liz Lerman order correct |
| 7 | 7. Suppression fires | PASS | Suppression fired correctly |
| 8 | 8. Suppression no false-fire | PASS | N/A (not article 11) |
| 9 | 9. Score calibration | PASS | N/A (not poor or excellent) |
| 10 | 10. Message length | PASS | All messages ≥ 15 words |
| 11 | 11. Good minimum | PASS | ≥ 2 good annotations present |
| 12 | 12. Score stability | PASS | All traits within ±10 across runs |

### 05-avg-academic.md (average) — 10/12

| # | Criterion | Result | Detail |
|---|-----------|--------|--------|
| 1 | 1. Honesty | PASS | N/A (not a poor article) |
| 2 | 2. Rewrite | PASS | All suggestion/issue annotations have rewrite |
| 3 | 3. No empty praise | PASS | No empty praise found |
| 4 | 4. Good specificity | PASS | All good annotations cite specific text |
| 5 | 5. Question quality | PASS | All questions end with ? |
| 6 | 6. Liz Lerman order | PASS | Liz Lerman order correct |
| 7 | 7. Suppression fires | FAIL | Run 2: conventionsSuppressed should be true |
| 8 | 8. Suppression no false-fire | PASS | N/A (not article 11) |
| 9 | 9. Score calibration | PASS | N/A (not poor or excellent) |
| 10 | 10. Message length | PASS | All messages ≥ 15 words |
| 11 | 11. Good minimum | PASS | ≥ 2 good annotations present |
| 12 | 12. Score stability | FAIL | ideas: spread=15 (45,60,60); organization: spread=25 (40,65,55) |

### 06-good-essay.md (good) — 11/12

| # | Criterion | Result | Detail |
|---|-----------|--------|--------|
| 1 | 1. Honesty | PASS | N/A (not a poor article) |
| 2 | 2. Rewrite | PASS | All suggestion/issue annotations have rewrite |
| 3 | 3. No empty praise | PASS | No empty praise found |
| 4 | 4. Good specificity | FAIL | Run 2: good annotation quotes not found in text: "The opening paragraph effectively establishes the essay's ar..."; Run 2: good annotation quotes not found in text: "The paragraph provides strong evidence with 'When the Seattl..."; Run 3: good annotation quotes not found in text: "The opening paragraph effectively establishes the essay's ar..." |
| 5 | 5. Question quality | PASS | All questions end with ? |
| 6 | 6. Liz Lerman order | PASS | Liz Lerman order correct |
| 7 | 7. Suppression fires | PASS | N/A (no ideas/org issues detected) |
| 8 | 8. Suppression no false-fire | PASS | N/A (not article 11) |
| 9 | 9. Score calibration | PASS | N/A (not poor or excellent) |
| 10 | 10. Message length | PASS | All messages ≥ 15 words |
| 11 | 11. Good minimum | PASS | ≥ 2 good annotations present |
| 12 | 12. Score stability | PASS | All traits within ±10 across runs |

### 07-good-business.md (good) — 11/12

| # | Criterion | Result | Detail |
|---|-----------|--------|--------|
| 1 | 1. Honesty | PASS | N/A (not a poor article) |
| 2 | 2. Rewrite | PASS | All suggestion/issue annotations have rewrite |
| 3 | 3. No empty praise | PASS | No empty praise found |
| 4 | 4. Good specificity | PASS | All good annotations cite specific text |
| 5 | 5. Question quality | PASS | All questions end with ? |
| 6 | 6. Liz Lerman order | PASS | Liz Lerman order correct |
| 7 | 7. Suppression fires | PASS | Suppression fired correctly |
| 8 | 8. Suppression no false-fire | PASS | N/A (not article 11) |
| 9 | 9. Score calibration | PASS | N/A (not poor or excellent) |
| 10 | 10. Message length | PASS | All messages ≥ 15 words |
| 11 | 11. Good minimum | PASS | ≥ 2 good annotations present |
| 12 | 12. Score stability | FAIL | ideas: spread=15 (70,85,75); voice: spread=20 (60,80,65) |

### 08-good-creative.md (good) — 11/12

| # | Criterion | Result | Detail |
|---|-----------|--------|--------|
| 1 | 1. Honesty | PASS | N/A (not a poor article) |
| 2 | 2. Rewrite | PASS | All suggestion/issue annotations have rewrite |
| 3 | 3. No empty praise | PASS | No empty praise found |
| 4 | 4. Good specificity | PASS | All good annotations cite specific text |
| 5 | 5. Question quality | PASS | All questions end with ? |
| 6 | 6. Liz Lerman order | PASS | Liz Lerman order correct |
| 7 | 7. Suppression fires | FAIL | Run 1: conventionsSuppressed should be true |
| 8 | 8. Suppression no false-fire | PASS | N/A (not article 11) |
| 9 | 9. Score calibration | PASS | N/A (not poor or excellent) |
| 10 | 10. Message length | PASS | All messages ≥ 15 words |
| 11 | 11. Good minimum | PASS | ≥ 2 good annotations present |
| 12 | 12. Score stability | PASS | All traits within ±10 across runs |

### 09-excellent-essay.md (excellent) — 11/12

| # | Criterion | Result | Detail |
|---|-----------|--------|--------|
| 1 | 1. Honesty | PASS | N/A (not a poor article) |
| 2 | 2. Rewrite | PASS | All suggestion/issue annotations have rewrite |
| 3 | 3. No empty praise | PASS | No empty praise found |
| 4 | 4. Good specificity | PASS | All good annotations cite specific text |
| 5 | 5. Question quality | PASS | All questions end with ? |
| 6 | 6. Liz Lerman order | PASS | Liz Lerman order correct |
| 7 | 7. Suppression fires | PASS | N/A (no ideas/org issues detected) |
| 8 | 8. Suppression no false-fire | PASS | N/A (not article 11) |
| 9 | 9. Score calibration | FAIL | Run 1: ideas=75, org=70 — need both > 75; Run 3: ideas=80, org=70 — need both > 75 |
| 10 | 10. Message length | PASS | All messages ≥ 15 words |
| 11 | 11. Good minimum | PASS | ≥ 2 good annotations present |
| 12 | 12. Score stability | PASS | All traits within ±10 across runs |

### 10-excellent-academic.md (excellent) — 9/12

| # | Criterion | Result | Detail |
|---|-----------|--------|--------|
| 1 | 1. Honesty | PASS | N/A (not a poor article) |
| 2 | 2. Rewrite | PASS | All suggestion/issue annotations have rewrite |
| 3 | 3. No empty praise | PASS | No empty praise found |
| 4 | 4. Good specificity | PASS | All good annotations cite specific text |
| 5 | 5. Question quality | PASS | All questions end with ? |
| 6 | 6. Liz Lerman order | PASS | Liz Lerman order correct |
| 7 | 7. Suppression fires | FAIL | Run 2: conventionsSuppressed should be true |
| 8 | 8. Suppression no false-fire | PASS | N/A (not article 11) |
| 9 | 9. Score calibration | FAIL | Run 1: ideas=75, org=70 — need both > 75; Run 2: ideas=75, org=70 — need both > 75 |
| 10 | 10. Message length | PASS | All messages ≥ 15 words |
| 11 | 11. Good minimum | PASS | ≥ 2 good annotations present |
| 12 | 12. Score stability | FAIL | voice: spread=20 (80,65,85); wordChoice: spread=15 (85,70,80) |

### 11-edge-good-ideas-bad-grammar.md (edge-case) — 9/12

| # | Criterion | Result | Detail |
|---|-----------|--------|--------|
| 1 | 1. Honesty | PASS | N/A (not a poor article) |
| 2 | 2. Rewrite | PASS | All suggestion/issue annotations have rewrite |
| 3 | 3. No empty praise | PASS | No empty praise found |
| 4 | 4. Good specificity | PASS | All good annotations cite specific text |
| 5 | 5. Question quality | PASS | All questions end with ? |
| 6 | 6. Liz Lerman order | PASS | Liz Lerman order correct |
| 7 | 7. Suppression fires | FAIL | Run 2: 5 conventions annotations should be suppressed |
| 8 | 8. Suppression no false-fire | FAIL | Run 1: no conventions annotations — false suppression; Run 1: conventionsSuppressed=true is a false fire; Run 2: conventionsSuppressed=true is a false fire |
| 9 | 9. Score calibration | PASS | N/A (not poor or excellent) |
| 10 | 10. Message length | PASS | All messages ≥ 15 words |
| 11 | 11. Good minimum | PASS | ≥ 2 good annotations present |
| 12 | 12. Score stability | FAIL | ideas: spread=15 (45,45,60) |

### 12-edge-perfect-grammar-no-thesis.md (edge-case) — 12/12

| # | Criterion | Result | Detail |
|---|-----------|--------|--------|
| 1 | 1. Honesty | PASS | N/A (not a poor article) |
| 2 | 2. Rewrite | PASS | All suggestion/issue annotations have rewrite |
| 3 | 3. No empty praise | PASS | No empty praise found |
| 4 | 4. Good specificity | PASS | All good annotations cite specific text |
| 5 | 5. Question quality | PASS | All questions end with ? |
| 6 | 6. Liz Lerman order | PASS | Liz Lerman order correct |
| 7 | 7. Suppression fires | PASS | Suppression fired correctly |
| 8 | 8. Suppression no false-fire | PASS | N/A (not article 11) |
| 9 | 9. Score calibration | PASS | N/A (not poor or excellent) |
| 10 | 10. Message length | PASS | All messages ≥ 15 words |
| 11 | 11. Good minimum | PASS | ≥ 2 good annotations present |
| 12 | 12. Score stability | PASS | All traits within ±10 across runs |

## Trait Scores by Article

| Article | Run | Ideas | Org | Voice | Word | Fluency | Conv | Pres |
|---------|-----|-------|-----|-------|------|---------|------|------|
| 01-poor-essay.md | 1 | 30 | 25 | 60 | 45 | 40 | 35 | 50 |
| 01-poor-essay.md | 2 | 35 | 30 | 60 | 45 | 40 | 50 | 55 |
| 01-poor-essay.md | 3 | 35 | 30 | 50 | 45 | 40 | 40 | 50 |
| 02-poor-creative.md | 1 | 35 | 30 | 45 | 35 | 40 | 50 | 45 |
| 02-poor-creative.md | 2 | 35 | 30 | 45 | 40 | 35 | 60 | 50 |
| 02-poor-creative.md | 3 | 35 | 30 | 40 | 40 | 35 | 60 | 50 |
| 03-avg-essay.md | 1 | 45 | 40 | 50 | 50 | 45 | 75 | 60 |
| 03-avg-essay.md | 2 | 55 | 45 | 50 | 55 | 50 | 75 | 70 |
| 03-avg-essay.md | 3 | 60 | 55 | 50 | 55 | 55 | 75 | 70 |
| 04-avg-article.md | 1 | 55 | 45 | 50 | 60 | 55 | 85 | 70 |
| 04-avg-article.md | 2 | 60 | 35 | 50 | 60 | 55 | 75 | 65 |
| 04-avg-article.md | 3 | 55 | 45 | 50 | 60 | 55 | 75 | 65 |
| 05-avg-academic.md | 1 | 45 | 40 | 50 | 60 | 55 | 75 | 70 |
| 05-avg-academic.md | 2 | 60 | 65 | 55 | 60 | 60 | 75 | 70 |
| 05-avg-academic.md | 3 | 60 | 55 | 50 | 55 | 60 | 75 | 70 |
| 06-good-essay.md | 1 | 85 | 75 | 80 | 75 | 80 | 95 | 90 |
| 06-good-essay.md | 2 | 80 | 75 | 70 | 70 | 75 | 85 | 80 |
| 06-good-essay.md | 3 | 85 | 80 | 75 | 75 | 80 | 90 | 85 |
| 07-good-business.md | 1 | 70 | 65 | 60 | 75 | 70 | 85 | 80 |
| 07-good-business.md | 2 | 85 | 75 | 80 | 75 | 80 | 95 | 90 |
| 07-good-business.md | 3 | 75 | 70 | 65 | 70 | 75 | 85 | 80 |
| 08-good-creative.md | 1 | 70 | 65 | 80 | 75 | 70 | 85 | 80 |
| 08-good-creative.md | 2 | 70 | 65 | 80 | 75 | 70 | 85 | 80 |
| 08-good-creative.md | 3 | 70 | 65 | 80 | 75 | 70 | 85 | 80 |
| 09-excellent-essay.m | 1 | 75 | 70 | 80 | 75 | 75 | 85 | 80 |
| 09-excellent-essay.m | 2 | 85 | 80 | 85 | 85 | 80 | 95 | 90 |
| 09-excellent-essay.m | 3 | 80 | 70 | 85 | 80 | 75 | 95 | 90 |
| 10-excellent-academi | 1 | 75 | 70 | 80 | 85 | 75 | 95 | 90 |
| 10-excellent-academi | 2 | 75 | 70 | 65 | 70 | 65 | 85 | 80 |
| 10-excellent-academi | 3 | 85 | 80 | 85 | 80 | 75 | 95 | 90 |
| 11-edge-good-ideas-b | 1 | 45 | 40 | 60 | 55 | 50 | 35 | 60 |
| 11-edge-good-ideas-b | 2 | 45 | 40 | 55 | 50 | 45 | 35 | 50 |
| 11-edge-good-ideas-b | 3 | 60 | 40 | 65 | 55 | 50 | 35 | 60 |
| 12-edge-perfect-gram | 1 | 35 | 30 | 40 | 50 | 45 | 75 | 60 |
| 12-edge-perfect-gram | 2 | 30 | 25 | 40 | 60 | 50 | 75 | 70 |
| 12-edge-perfect-gram | 3 | 30 | 25 | 40 | 50 | 45 | 75 | 60 |
