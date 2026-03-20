# Writing Center — Design Spec

> Date: 2026-03-20
> Status: **Final (v6)** — ready for implementation planning
> Module: AI Text X-Ray / Writing Center

---

## 1. Positioning

| Module | Purpose | User |
|--------|---------|------|
| AI Detector | Detect whether text is AI-generated | Has text, wants to check |
| Humanizer | Rewrite AI text to pass detection | Has AI text, wants to bypass |
| **Writing Center** | **Teach users to write well** | **Wants to learn/improve writing** |

Writing Center is independent from Detector/Humanizer. Own editor state, own profile, own pedagogy.

**Writing Lab** bridges the gap: users come for AI detection → Lab shows *why* AI text lacks "temperature" → Writing Center teaches intentional voice.

---

## 2. Core User Flow

```
Open Writing Center
  │
  ├→ Daily Tip (LLM-personalized if ≥3 analyses; static fallback otherwise)
  │   "Try it" / "Skip" / "Don't show tips"
  │
  ├→ Choose genre + enter topic
  │
  ├→ STEP MODE (card count matches strategy steps, depth by experience)
  │   genreExperience increments once per draft session on first analyze
  │
  ├→ DIALOGUE MODE (Socratic, Liz Lerman four-phase)
  │   60s idle → proactive prompt (reset on any content change)
  │
  ├→ ANALYZE (requires ≥100 words)
  │   Liz Lerman order: good → question → suggestion → issue
  │   Conventions suppressed if Ideas/Organization has issue
  │   Recovery message when suppression lifts on re-analyze
  │   traitScores saved to profile
  │   ↕ Bidirectional annotation-editor linking
  │
  ├→ Revise → Re-analyze → See improvement
  │
  └→ Done → Copy as plain text
```

---

## 3. Layout

```
┌──────────────────────────────────────────────────────────┐
│ Activity Bar │ Writing Center                [layout ▫▪▫]│
│              ├──────────────────┬─────────────────────────┤
│  [Detector]  │                  │ [Chat] [Dashboard] [Lab]│
│  [Humanizer] │   EDITOR         │                         │
│  [Writing]←  │   + inline       │  (tab content)          │
│              │   highlights     │                         │
│              │                  │                         │
│              │   [Analyze ▶]    │  [input box ________]   │
│              ├──────────────────┴─────────────────────────┤
│     [X]      │ 352 words · Essay · Draft · 🔥 3-day streak│
└──────────────────────────────────────────────────────────┘
```

Presets: side-by-side (default) / top-bottom / editor fullscreen. Draggable divider.

### Collaboration Panel Tabs

| Tab | Content | MVP |
|-----|---------|-----|
| **Chat** | Daily Tip → Step cards → Chat → Annotation details | MVP-1 |
| **Dashboard** | Radar, trends, heatmap, stats, report | MVP-2 |
| **Lab** | Temperature comparison experiments | MVP-1 |

### Status Bar

| State | Display |
|-------|---------|
| Ready | `0 words · Essay · 🔥 3-day streak` |
| Draft | `352 words · Essay · Draft · 🔥 3-day streak` |
| Below threshold | `52 words · Essay · (100 words to analyze)` |
| Analyzing | `Analyzing...` |
| Analyzed | `352 words · Essay · 7 annotations · 🔥 3` |
| Streak broken | `352 words · Essay · Draft` (no flame) |

### Annotation ↔ Editor Bidirectional Linking

- Click annotation card in panel → editor `scrollIntoView` + 0.5s yellow pulse on Decoration
- Click highlighted text in editor → panel scrolls to matching annotation card + pulse
- Link key: `annotation.id` ↔ `data-annotation-id` on Decoration
- **Multiple annotations on same paragraph**: clicking the paragraph highlight scrolls to the *first* annotation in Liz Lerman order (i.e., the `good` one if present, otherwise `question`, etc.). A small badge shows "3 annotations" — clicking cycles through them.

### Integration with AppShell

Own Tiptap state, independent from shared `text`. Status bar reads conditionally. Existing `WritingPanel` and `PROMPT_TEMPLATES` fully replaced.

---

## 4. Three AI Modes

| Stage | Mode | Trigger | UI |
|-------|------|---------|-----|
| Before writing | **Step** | Editor empty, genre selected | Strategy cards |
| During writing | **Dialogue** | Editor has content | Socratic chat |
| After analysis | **Annotation** | Clicked "Analyze" | Highlights + details |

Input box always available → typing forces Dialogue.

**Idle detection**: `setTimeout(60000)` reset on every Tiptap `update` event (content change, not cursor movement). On fire, sends a proactive dialogue prompt.

---

## 5. Pedagogy

### 5.1 SRSD — Genre Strategy Mapping with Card Counts

| Genre | Mnemonic | Steps | Experience 0 cards | Exp 1-2 cards | Exp 3+ |
|-------|----------|-------|-------------------|---------------|--------|
| Essay | **TIDE** | T→I→D→E | 4 strategy + 2 meta (background, independence) = **6** | 2 (strategy summary + "jump in") | 0 |
| Article | **5W+H** | W→W→W→W→W→H | 6 strategy + 2 meta = **8** | 2 | 0 |
| Academic | **PLAN+WRITE** | P→L→A→N→W→R→I→T→E | 5 grouped + 2 meta = **7** | 2 | 0 |
| Creative | **POW+WWW** | P→O→W + W→W→W→W→W→H | 4 grouped + 2 meta = **6** | 2 | 0 |
| Business | **AIDA** | A→I→D→A | 4 strategy + 2 meta = **6** | 2 | 0 |

"Meta cards" are: card 1 = Background Knowledge ("What do you know about this topic?"), last card = Independence ("You're ready — start writing, ask me anything").

Experience 1-2 always returns exactly 2 cards: strategy summary + jump-in. Experience 3+ returns empty array.

### 5.2 SRSD Scaffolding Decay

`genreExperience[genre]` increments **once per draft session** on first successful analyze (≥100 words). Tracked by `hasIncrementedThisSession` in component state. New draft (clear editor + new topic) resets session flag.

### 5.3 6+1 Traits

| Trait | Color |
|-------|-------|
| Ideas | Blue |
| Organization | Purple |
| Voice | Orange |
| Word Choice | Teal |
| Sentence Fluency | Green |
| Conventions | Gray |
| Presentation | Pink |

### 5.4 Conventions Suppression

Ideas or Organization has `issue` → suppress all Conventions annotations. `traitScores` for Conventions still returned.

**Recovery**: when previous analyze had suppression but current does not, show transition message: *"Structure looks solid now — let's fine-tune the grammar."*

### 5.5 Liz Lerman

**Dialogue**: four phases via conversation context.
**Annotations**: strict order `good → question → suggestion → issue`.

### 5.6 Annotation Calibration Rules

1. Every `message` ≥ 15 words
2. Every `suggestion`/`issue` MUST have `rewrite` field — a string showing the suggested replacement text (the "before" is the annotated span itself, so only the replacement is needed)
3. No generic praise without evidence
4. Every `good` cites specific word/sentence/technique
5. Every `question` is open-ended and actionable
6. Good Ideas + good Organization + bad grammar → Conventions NOT suppressed
7. ≥200 words → minimum 2 `good` annotations

---

## 6. Habit Mechanism

### 6.1 Daily Tip

On session open, if `profile.preferences.showDailyTips`:
- If `profile.stats.totalAnalyses >= 3` → call `action: "daily-tip"` (LLM-personalized)
- Else → static fallback from `src/lib/daily-tips.ts` (35 entries, 5/trait)

Static selection: filter by lowest-scoring trait (ties broken alphabetically) → `dateHash % filtered.length`.

Tip IDs: static tips have `"static-{index}"`. LLM tips get `"llm-{date}"`. Both tracked in `completedExercises`.

User prefs: "Skip" (session) / "Don't show tips" (persistent, `showDailyTips = false`).

### 6.2 Streak

- ≥50 words written OR 1 analysis → day counts as active
- Yesterday → `current++` / Today → noop / 2+ days → `current = 0`
- `🔥 {n}-day streak` when `current > 0`

---

## 7. Writing Lab

### 7.1 Concept

"Two Kinds of Temperature" — AI temperature = sampling randomness; human temperature = voice, intention, lived experience. Lab makes this visceral through comparison.

### 7.2 Flow

```
Open Lab tab → Select example (10-15 topics)
  │
  ├→ Displays "cold text" (bland, correct, lifeless)
  │
  ├→ "See how AI adds temperature"
  │   Backend rewrites at temperatures [0, 0.7, 1.3]
  │   Three-column comparison with explanation cards
  │
  ├→ Human-warm version (pre-written) always visible in 4th column
  │   Teaching point: "AI randomness ≠ human voice"
  │
  └→ "Your turn" → fills cold text into editor
      Sets genre to example's focusTrait-appropriate genre (default: Essay)
      Sets topic to example's topic
      Creates new draft session (resets hasIncrementedThisSession)
      Switches to Chat tab
      Does NOT auto-set any other state
```

### 7.3 Examples

10-15 sets in `src/lib/lab-examples.ts`:

```typescript
interface LabExample {
  id: string;
  topic: string;
  coldText: string;           // ~100 words, bland
  humanWarmText: string;      // handcrafted
  humanExplanation: string;   // what makes it warm
  teachingPoint: string;      // overarching lesson
  focusTrait: Trait;
}
```

Content creation can run in parallel with prompt calibration.

### 7.4 Lab Rate Limiting

Max 5 `lab-rewrite` calls per session (client-side counter). Each call = 3 LLM invocations. Beyond 5: "You've explored enough for today — try writing your own version!"

---

## 8. Data Model

### 8.1 Types

```typescript
type Genre = "essay" | "article" | "academic" | "creative" | "business";
type Trait = "ideas" | "organization" | "voice" | "wordChoice" | "fluency" | "conventions" | "presentation";
type Severity = "good" | "question" | "suggestion" | "issue";
```

### 8.2 WriterProfile

```typescript
interface WriterProfile {
  userId: string;
  genreExperience: Record<Genre, number>;
  analysisHistory: AnalysisSnapshot[];
  traitScores: Record<Trait, { date: string; score: number }[]>;
  streak: { current: number; longest: number; lastActiveDate: string };
  completedExercises: string[];
  stats: { totalWords: number; totalSessions: number; totalAnalyses: number };
  preferences: { showDailyTips: boolean };
}

interface AnalysisSnapshot {
  date: string;
  genre: Genre;
  wordCount: number;
  traitScores: Record<Trait, number>;
  annotationCounts: Record<Severity, number>;
}
```

### 8.3 localStorage

```typescript
interface WritingCenterState {
  draft: {
    genre: Genre;
    topic: string;
    document: string;
    messages: ChatMessage[];
    annotations: Annotation[];
    lastSaved: number;
  };
  profile: WriterProfile;
}
// Key: "writing-center". Auto-save debounced 2s. Single draft MVP-1.
```

### 8.4 Daily Tip

```typescript
interface DailyTip {
  id: string;       // "static-3" or "llm-2026-03-20"
  trait: Trait;
  tip: string;
  example?: { before: string; after: string };
  exercisePrompt?: string;
}
```

### 8.5 MVP-2 Migration

Supabase Auth → real `userId`. Merge: higher streak/experience wins, history concatenated, dedup by date+genre+wordCount.

---

## 9. Technical Architecture

### 9.1 Frontend

- **Editor**: Tiptap (`@tiptap/react`, `@tiptap/starter-kit`, `@tiptap/extension-highlight`). Inline Decorations with `data-annotation-id`. Bidirectional scroll + pulse.
- **Collaboration Panel**: Chat (MVP-1) + Lab (MVP-1) + Dashboard (MVP-2).
- **Dashboard** (MVP-2): Recharts (already in `package.json`).
- **Lab** (MVP-1): Three-column diff view + "Your turn" button.
- **State**: React state + localStorage. Supabase MVP-2.

### 9.2 API

`POST /api/writing-assist`

```typescript
// ── Messages ──
interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}

// ── Step Cards ──
interface StepCard {
  id: string;
  stepIndex: number;
  totalSteps: number;
  title: string;
  mnemonic?: string;
  instructions: string;
  checklist?: string[];
  example?: string;
  completed: boolean;
}

// ── Annotations ──
interface Annotation {
  id: string;
  paragraph: number;       // 0-indexed
  startOffset: number;     // -1 = whole paragraph
  endOffset: number;       // -1 = whole paragraph
  trait: Trait;
  severity: Severity;
  message: string;         // ≥15 words
  rewrite?: string;        // replacement text. REQUIRED for suggestion/issue.
}

// ── Request ──
interface WritingAssistRequest {
  action: "guide" | "analyze" | "expand" | "daily-tip" | "lab-rewrite" | "report";
  mode?: "step" | "dialogue";
  genre?: Genre;
  topic?: string;
  document?: string;
  messages?: ChatMessage[];
  annotationId?: string;
  annotationContext?: Annotation;    // the full annotation being expanded (for expand action)
  experienceLevel?: number;
  traitScores?: Record<Trait, number>;
  analysisHistory?: AnalysisSnapshot[];
  text?: string;                     // for lab-rewrite
  temperatures?: number[];           // for lab-rewrite, validated: 1-3 values, each 0-2.0
  profile?: ReportProfileData;
}

// ── Responses ──

// guide (step)
interface GuideStepResponse {
  type: "step";
  cards: StepCard[];     // count varies by genre + experience (see §5.1)
}

// guide (dialogue) — complete JSON in MVP-1, SSE in MVP-2
interface GuideDialogueResponse {
  type: "dialogue";
  message: string;
}

// analyze — requires ≥100 words, else 400
interface AnalyzeResponse {
  annotations: Annotation[];            // Liz Lerman order
  traitScores: Record<Trait, number>;   // 0-100, all 7 always present
  summary: string;
  conventionsSuppressed: boolean;
}

// expand — sends annotationContext so backend has full context
interface ExpandResponse {
  detail: string;
  suggestion?: string;   // only for suggestion/issue
  question: string;      // Socratic follow-up, always present
}

// daily-tip
interface DailyTipResponse {
  tip: DailyTip;
}

// lab-rewrite — temperatures validated server-side: array of 1-3 floats, each 0.0-2.0
interface LabRewriteResponse {
  rewrites: { temperature: number; text: string; explanation: string }[];
}

// report (501 in MVP-1)
interface ReportProfileData {
  recentAnalyses: AnalysisSnapshot[];
  traitTrends: Record<Trait, number[]>;
  streak: number;
  totalWordsThisWeek: number;
  genresThisWeek: string[];
}

interface ReportResponse {
  summary: string;
  improvements: string;
  weakPoints: string;
  nextWeekFocus: string;
  encouragement: string;
}
```

### 9.3 Streaming

| Action | MVP-1 | MVP-2 |
|--------|-------|-------|
| guide (step) | JSON | JSON |
| guide (dialogue) | JSON | SSE |
| analyze | JSON | JSON |
| expand | JSON | JSON |
| daily-tip | JSON | JSON |
| lab-rewrite | JSON | JSON |
| report | 501 | JSON |

### 9.4 Paragraph Mapping

0-indexed Tiptap paragraph nodes. Empty counted. Invalid index → drop. Invalid offsets → whole-paragraph (`-1, -1`).

### 9.5 Error Handling

| Error | Response |
|-------|----------|
| Malformed JSON (analyze) | Toast: "Analysis failed" |
| Timeout >30s | Retry button |
| Network error | Error banner |
| Rate limit | "Daily limit reached" |
| Document >15,000 chars | Status bar message |
| Document <100 words (analyze) | Button disabled + tooltip |
| `report` in MVP-1 | 501 |
| `daily-tip` LLM fails | Static fallback |
| `lab-rewrite` LLM fails | Show error, keep cold + human visible |
| `lab-rewrite` invalid temps | 400: "temperatures must be 1-3 values between 0-2.0" |
| Lab session limit (>5 rewrites) | Client-side block: "Try writing your own version!" |

### 9.6 Role Prompts

Seven files in `src/lib/prompts/writing/`:

| File | Role |
|------|------|
| `guide-step.ts` | SRSD coach. Receives experienceLevel + genre + topic. Returns cards[]. |
| `guide-dialogue.ts` | Socratic mentor. Liz Lerman four-phase. 60s idle → proactive. |
| `analyze.ts` | 6+1 reviewer. Liz Lerman order. Suppression. Calibration rules §5.6. Returns traitScores. |
| `expand.ts` | Feedback coach. Receives full annotation via annotationContext. |
| `daily-tip.ts` | Personalized tip. Receives traitScores + recent history. |
| `lab-rewrite.ts` | "Rewrite to make more engaging. Do not change meaning." Used at variable temperature. |
| `report.ts` | Weekly narrator (MVP-2). Stats only, no raw text. |

### 9.7 LLM Configuration

| Action | Temperature | Rationale |
|--------|-------------|-----------|
| analyze | **0** | Deterministic scoring |
| report | **0** | Consistent stats summary |
| guide (step) | **0.3** | Consistent but slightly varied cards |
| guide (dialogue) | **0.7** | Natural conversation |
| expand | **0.3** | Consistent feedback |
| daily-tip | **0.7** | Fresh-feeling tips |
| lab-rewrite | **per request** | 0, 0.7, 1.3 — the point of Lab |

Model: `claude-sonnet-4-6`. Key: server-side `ANTHROPIC_API_KEY`. Max doc: 15,000 chars. Min analyze: 100 words.

---

## 10. MVP Phasing

### MVP-1: Editor + AI + Habits + Lab

- Tiptap editor + split-pane layout (3 presets, draggable)
- Chat tab: Daily Tip (LLM + static fallback) → Step cards → Dialogue → Annotation details
- Lab tab: 10-15 examples, 3-temperature comparison, "Your turn" handoff
- API: guide / analyze / expand / daily-tip / lab-rewrite (report → 501)
- 7 prompt files
- 5 genres with SRSD strategies (variable card counts per genre)
- SRSD scaffolding decay (per-session increment)
- Liz Lerman annotation ordering
- Conventions suppression + recovery message
- Calibration hard rules
- Bidirectional annotation-editor linking
- Streak counter
- localStorage (draft + profile)
- Copy as plain text
- Error handling + fallbacks

### MVP-2: Progress System

- Dashboard tab (Recharts: radar, trends, heatmap, stats, weak-spot, report)
- `action: "report"` + report.ts
- Supabase Auth + persistence + migration
- SSE streaming for dialogue
- Rate limiting (10 analyses/day free, unlimited premium)
- Expand tips to 100+

### Forward Compatibility

MVP-1 localStorage matches full WriterProfile. AnalyzeResponse includes traitScores + conventionsSuppressed. Dashboard has data on MVP-2 launch.

---

## 11. Prompt Calibration Plan

**Phase 1 of implementation. Must pass before any frontend code.**

### Test Corpus (12 articles, multi-genre)

| # | Quality | Genre | Purpose |
|---|---------|-------|---------|
| 1 | Poor | Essay | Weak thesis, no structure |
| 2 | Poor | Creative | No arc, flat characters |
| 3 | Average | Essay | Vague thesis, some structure |
| 4 | Average | Article | OK 5W+H, weak lead |
| 5 | Average | Academic | Decent argument, citation issues |
| 6 | Good | Essay | Clear thesis, logical |
| 7 | Good | Business | Clear AIDA, tone issues |
| 8 | Good | Creative | Engaging, pacing issues |
| 9 | Excellent | Essay | Strong voice, tight |
| 10 | Excellent | Academic | Original, well-cited |
| 11 | Edge | Essay | Good Ideas + Organization, terrible grammar |
| 12 | Edge | Essay | Perfect grammar, zero thesis |

### 12 Pass Criteria

| # | Criterion | Condition |
|---|-----------|-----------|
| 1 | Honesty | Poor articles: ≥3 issue annotations |
| 2 | Rewrite | Every suggestion/issue has rewrite string |
| 3 | No empty praise | No generic phrases without evidence |
| 4 | Good specificity | Every good cites word/sentence/technique |
| 5 | Question quality | Open-ended, actionable |
| 6 | Liz Lerman order | Strict: good → question → suggestion → issue |
| 7 | Suppression fires | Articles 1, 2, 12: zero Conventions annotations |
| 8 | Suppression doesn't false-fire | Article 11: Conventions present |
| 9 | Score calibration | Poor <40 on weak traits; Excellent >75 |
| 10 | Message length | Every message ≥15 words |
| 11 | Good minimum | ≥200 words → ≥2 good annotations |
| 12 | Score stability | Same article ×3: traitScores ±5 (temp=0) |

### Process

1. Write analyze.ts prompt
2. Run 12 articles × 3 runs = 36 evaluations
3. Score against 12 criteria
4. Iterate until all pass
5. Document in `docs/prompt-eval-results.md`
6. Lab examples + daily tips can be written in parallel
7. **Then** begin frontend

---

## 12. Module Relationships

```
┌─────────────┐  ┌─────────────┐  ┌───────────────────────┐
│ AI Detector │  │  Humanizer  │  │    Writing Center      │
│ Detect AI   │  │ Rewrite AI  │  │ Chat: Teach writing   │
│             │  │             │  │ Lab:  Show why AI is  │
│ Shared text ◄──► Shared text │  │       "cold"          │
│ state       │  │ state       │  │ Own Tiptap + Profile  │
└─────────────┘  └─────────────┘  └───────────────────────┘
                                    Lab bridges Detector →
                                    Writing Center
```

---

## 13. Risks

| # | Risk | Mitigation |
|---|------|------------|
| 1 | AI feedback quality | §11 calibration gates frontend. temp=0 for analyze. |
| 2 | API cost | ~$0.01/analyze + $0.002/tip + $0.03/lab. 100 DAU ≈ $20/day. |
| 3 | Tiptap offset misalignment | Validate → fallback whole-paragraph. |
| 4 | SRSD decay abrupt | Transition messages with experience count. |
| 5 | Lab temps produce similar output | Calibrate lab-rewrite prompt. Test 3-5 examples. |
| 6 | Daily tip generic for new users | <3 analyses → static fallback. |
| 7 | genreExperience inflation | Per-session-only increment. |
| 8 | Lab cost spam | 5 calls/session client-side limit. |
| 9 | Brand mismatch | Lab bridges narrative. Long-term rebrand if needed. |
| 10 | 10-15 handcrafted Lab examples | Content creation parallel with prompt calibration. Can use 50M corpus for inspiration. |
