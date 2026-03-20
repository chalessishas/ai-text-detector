# Writing Center Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Writing Center module — an AI-powered writing education tool with Tiptap editor, multi-mode AI collaboration (Step/Dialogue/Annotation), Writing Lab experiments, daily tips, and streak tracking.

**Architecture:** Next.js API route (`/api/writing-assist`) dispatches to 7 role-specific system prompts with action-specific temperature settings. Frontend uses Tiptap for the editor, React state + localStorage for persistence, and a split-pane layout matching the existing VSCode-style AppShell.

**Tech Stack:** Next.js 16, React 19, Tiptap (ProseMirror), Tailwind CSS 4, Recharts, Claude API (claude-sonnet-4-6), localStorage.

**Spec:** `docs/superpowers/specs/2026-03-20-writing-center-design.md` (v6 final)

---

## File Map

### New files

| Path | Responsibility |
|------|---------------|
| `src/lib/writing/types.ts` | All shared TypeScript types (Genre, Trait, Severity, WriterProfile, Annotation, ChatMessage, StepCard, DailyTip, LabExample, all Request/Response interfaces) |
| `src/lib/writing/storage.ts` | localStorage read/write with debounced auto-save, profile updates, streak logic |
| `src/lib/writing/daily-tips.ts` | 35 static DailyTip entries (5 per trait) |
| `src/lib/writing/lab-examples.ts` | 10 LabExample entries with cold text + handcrafted human warm text |
| `src/lib/prompts/writing/analyze.ts` | System prompt for 6+1 Traits analysis (Liz Lerman order, calibration rules, Conventions suppression) |
| `src/lib/prompts/writing/guide-step.ts` | System prompt for SRSD step cards (experience-aware, genre-specific) |
| `src/lib/prompts/writing/guide-dialogue.ts` | System prompt for Socratic dialogue (Liz Lerman four-phase) |
| `src/lib/prompts/writing/expand.ts` | System prompt for annotation expansion |
| `src/lib/prompts/writing/daily-tip.ts` | System prompt for personalized tip generation |
| `src/lib/prompts/writing/lab-rewrite.ts` | System prompt for Lab temperature rewriting |
| `src/lib/prompts/writing/report.ts` | System prompt for weekly report narrator (MVP-2 content, stub file for MVP-1) |
| `src/app/api/writing-assist/route.ts` | API route — dispatches by action, selects prompt + temperature, calls Claude API |
| `src/components/writing/WritingCenter.tsx` | Top-level Writing Center component — layout, split pane, tab switching, state orchestration |
| `src/components/writing/Editor.tsx` | Tiptap editor wrapper with inline annotation Decorations |
| `src/components/writing/ChatPanel.tsx` | Chat tab — Daily Tip card, Step cards, chat messages, annotation detail expansion |
| `src/components/writing/LabPanel.tsx` | Lab tab — example selector, temperature comparison columns, "Your turn" handoff |
| `src/components/writing/StepCard.tsx` | Single SRSD step card component |
| `src/components/writing/AnnotationCard.tsx` | Single annotation detail card (for Chat panel) |
| `src/components/writing/DailyTipCard.tsx` | Daily tip card with "Try it" / "Skip" / "Don't show" |
| `docs/prompt-eval-results.md` | Prompt calibration evaluation results |
| `docs/test-articles/` | 12 test articles for prompt calibration (01-poor-essay.md through 12-edge-grammar.md) |

### Modified files

| Path | Change |
|------|--------|
| `src/components/AppShell.tsx` | Replace `WritingPanel` with `WritingCenter`. Remove old Writing Center state. Conditional status bar for writing mode (reads from WritingCenter state). |
| `package.json` | Add `@tiptap/react`, `@tiptap/starter-kit`, `@tiptap/extension-highlight`, `@tiptap/pm`, `@anthropic-ai/sdk` |

---

## Phase 0: Prompt Calibration (before any frontend code)

> Spec §11 gates frontend work. This phase produces calibrated prompts and test documentation.

### Task 0.1: Write Test Articles

**Files:**
- Create: `docs/test-articles/01-poor-essay.md`
- Create: `docs/test-articles/02-poor-creative.md`
- Create: `docs/test-articles/03-avg-essay.md`
- Create: `docs/test-articles/04-avg-article.md`
- Create: `docs/test-articles/05-avg-academic.md`
- Create: `docs/test-articles/06-good-essay.md`
- Create: `docs/test-articles/07-good-business.md`
- Create: `docs/test-articles/08-good-creative.md`
- Create: `docs/test-articles/09-excellent-essay.md`
- Create: `docs/test-articles/10-excellent-academic.md`
- Create: `docs/test-articles/11-edge-good-ideas-bad-grammar.md`
- Create: `docs/test-articles/12-edge-perfect-grammar-no-thesis.md`

- [ ] **Step 1:** Write articles 1-4 (poor + average). Each 200-400 words. Poor articles: deliberately weak thesis, monotone voice, no transitions. Average: thesis exists but vague, some structure.

- [ ] **Step 2:** Write articles 5-8 (average + good). Average academic: decent argument but citation issues. Good: clear thesis, logical structure, varied sentences, minor issues.

- [ ] **Step 3:** Write articles 9-10 (excellent). Strong voice, tight structure, precise vocabulary. These are the "gold standard" — should score >75 on all traits.

- [ ] **Step 4:** Write articles 11-12 (edge cases). #11: Good ideas and organization but terrible grammar (missing commas, subject-verb disagreement, run-ons). #12: Perfect grammar, perfect punctuation, but zero thesis — just a collection of bland factual statements.

- [ ] **Step 5:** Commit.
```bash
git add docs/test-articles/
git commit -m "docs: add 12 test articles for prompt calibration"
```

### Task 0.2: Write Types

**Files:**
- Create: `src/lib/writing/types.ts`

- [ ] **Step 1:** Create the shared types file with all TypeScript interfaces from spec §8.1, §8.2, §8.3, §8.4, §9.2.

```typescript
// src/lib/writing/types.ts

export type Genre = "essay" | "article" | "academic" | "creative" | "business";
export type Trait = "ideas" | "organization" | "voice" | "wordChoice" | "fluency" | "conventions" | "presentation";
export type Severity = "good" | "question" | "suggestion" | "issue";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}

export interface StepCard {
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

export interface Annotation {
  id: string;
  paragraph: number;
  startOffset: number;
  endOffset: number;
  trait: Trait;
  severity: Severity;
  message: string;
  rewrite?: string;
}

export interface DailyTip {
  id: string;
  trait: Trait;
  tip: string;
  example?: { before: string; after: string };
  exercisePrompt?: string;
}

export interface LabExample {
  id: string;
  topic: string;
  coldText: string;
  humanWarmText: string;
  humanExplanation: string;
  teachingPoint: string;
  focusTrait: Trait;
}

export interface AnalysisSnapshot {
  date: string;
  genre: Genre;
  wordCount: number;
  traitScores: Record<Trait, number>;
  annotationCounts: Record<Severity, number>;
}

export interface WriterProfile {
  userId: string;
  genreExperience: Record<Genre, number>;
  analysisHistory: AnalysisSnapshot[];
  traitScores: Record<Trait, { date: string; score: number }[]>;
  streak: { current: number; longest: number; lastActiveDate: string };
  completedExercises: string[];
  stats: { totalWords: number; totalSessions: number; totalAnalyses: number };
  preferences: { showDailyTips: boolean };
}

export interface WritingCenterState {
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

// ── API Request/Response ──

export interface WritingAssistRequest {
  action: "guide" | "analyze" | "expand" | "daily-tip" | "lab-rewrite" | "report";
  mode?: "step" | "dialogue";
  genre?: Genre;
  topic?: string;
  document?: string;
  messages?: ChatMessage[];
  annotationId?: string;
  annotationContext?: Annotation;
  experienceLevel?: number;
  traitScores?: Record<Trait, number>;
  analysisHistory?: AnalysisSnapshot[];
  text?: string;
  temperatures?: number[];
  profile?: ReportProfileData;
}

export interface GuideStepResponse {
  type: "step";
  cards: StepCard[];
}

export interface GuideDialogueResponse {
  type: "dialogue";
  message: string;
}

export interface AnalyzeResponse {
  annotations: Annotation[];
  traitScores: Record<Trait, number>;
  summary: string;
  conventionsSuppressed: boolean;
}

export interface ExpandResponse {
  detail: string;
  suggestion?: string;
  question: string;
}

export interface DailyTipResponse {
  tip: DailyTip;
}

export interface LabRewriteResponse {
  rewrites: { temperature: number; text: string; explanation: string }[];
}

export interface ReportProfileData {
  recentAnalyses: AnalysisSnapshot[];
  traitTrends: Record<Trait, number[]>;
  streak: number;
  totalWordsThisWeek: number;
  genresThisWeek: string[];
}

export interface ReportResponse {
  summary: string;
  improvements: string;
  weakPoints: string;
  nextWeekFocus: string;
  encouragement: string;
}
```

- [ ] **Step 2:** Verify TypeScript compiles: `npx tsc --noEmit`

- [ ] **Step 3:** Commit.
```bash
git add src/lib/writing/types.ts
git commit -m "feat(writing): add shared TypeScript types"
```

### Task 0.3: Write analyze.ts Prompt + Calibration Script

**Files:**
- Create: `src/lib/prompts/writing/analyze.ts`
- Create: `scripts/calibrate-analyze.ts`

- [ ] **Step 1:** Write the analyze system prompt. Must encode: Liz Lerman output order, Conventions suppression, all 7 calibration rules from §5.6, 6+1 Traits framework, JSON output format matching `AnalyzeResponse`.

- [ ] **Step 2:** Write a calibration script that:
  1. Reads all 12 test articles from `docs/test-articles/`
  2. Calls the analyze prompt via Claude API (temperature=0) 3 times per article
  3. Checks each of the 12 pass criteria from spec §11
  4. Outputs a pass/fail report per criterion per article
  5. Saves results to `docs/prompt-eval-results.md`

- [ ] **Step 3:** Install Anthropic SDK: `npm install @anthropic-ai/sdk`

- [ ] **Step 4:** Run calibration: `ANTHROPIC_API_KEY=... npx tsx scripts/calibrate-analyze.ts`

- [ ] **Step 5:** Review results. If any criterion fails, iterate on the prompt and re-run.

- [ ] **Step 6:** Once all 12 criteria pass for all 12 articles, commit.
```bash
git add src/lib/prompts/writing/analyze.ts scripts/calibrate-analyze.ts docs/prompt-eval-results.md
git commit -m "feat(writing): calibrated analyze prompt — all 12 criteria pass"
```

### Task 0.4: Write Remaining Prompts

**Files:**
- Create: `src/lib/prompts/writing/guide-step.ts`
- Create: `src/lib/prompts/writing/guide-dialogue.ts`
- Create: `src/lib/prompts/writing/expand.ts`
- Create: `src/lib/prompts/writing/daily-tip.ts`
- Create: `src/lib/prompts/writing/lab-rewrite.ts`

- [ ] **Step 1:** Write `guide-step.ts` — SRSD coach. Receives experienceLevel + genre + topic. Returns StepCard[] JSON. Experience 0: full cards (count per genre from §5.1). Experience 1-2: 2 cards. Experience 3+: empty array. Temperature: 0.3.

- [ ] **Step 2:** Write `guide-dialogue.ts` — Socratic mentor. Liz Lerman four-phase via conversation. Never gives direct answers. References specific text from user's draft. Temperature: 0.7.

- [ ] **Step 3:** Write `expand.ts` — Feedback coach. Receives full annotation context. Returns detail + suggestion (for suggestion/issue severity only) + Socratic follow-up question. Temperature: 0.3.

- [ ] **Step 4:** Write `daily-tip.ts` — Personalized tip generator. Receives traitScores + recent analysisHistory. Returns DailyTip JSON targeting weakest trait. Temperature: 0.7.

- [ ] **Step 5:** Write `lab-rewrite.ts` — Simple rewrite prompt: "Rewrite this text to make it more engaging. Do not change the meaning." Temperature: per request (the whole point).

- [ ] **Step 6:** Write `report.ts` — Stub for MVP-2. Export the system prompt string and a comment: "// MVP-2: Weekly narrator. Receives stats, not raw text." This ensures the 7-file structure matches the spec.

- [ ] **Step 7:** Manually test each prompt (except report) with 2-3 sample inputs to verify JSON output is well-formed.

- [ ] **Step 8:** Commit.
```bash
git add src/lib/prompts/writing/
git commit -m "feat(writing): all 7 role prompts (guide-step, guide-dialogue, expand, daily-tip, lab-rewrite, report stub)"
```

### Task 0.5: Write Content (parallel with 0.3-0.4)

**Files:**
- Create: `src/lib/writing/daily-tips.ts`
- Create: `src/lib/writing/lab-examples.ts`

- [ ] **Step 1:** Write 35 static daily tips (5 per trait). Each tip: id (`"static-0"` through `"static-34"`), trait, tip text, optional before/after example, optional exercisePrompt. Import `DailyTip` and `Trait` from `types.ts`. Also export a `selectStaticTip(traitScores: Record<Trait, number> | null, date: Date): DailyTip` function that: filters by lowest-scoring trait (ties broken alphabetically), then selects by `dateHash % filtered.length`. If no traitScores, uses `dateHash % allTips.length`.

- [ ] **Step 2:** Write 10 Lab examples. Each: id, topic, coldText (~100 words bland prose), humanWarmText (handcrafted — NOT AI-generated), humanExplanation, teachingPoint, focusTrait. Prioritize Voice (4 examples), Ideas (2), Word Choice (2), Fluency (2).

- [ ] **Step 3:** Commit.
```bash
git add src/lib/writing/daily-tips.ts src/lib/writing/lab-examples.ts
git commit -m "content(writing): 35 daily tips + 10 lab examples"
```

---

## Phase 1: API Route

### Task 1.1: API Route — Skeleton + Analyze Action

**Files:**
- Create: `src/app/api/writing-assist/route.ts`

- [ ] **Step 1:** Create the route with action dispatch. Start with just `analyze` action. Read the existing `/api/analyze/route.ts` and `/api/humanize/route.ts` for the project's API patterns (NextRequest/NextResponse, error handling, input validation).

- [ ] **Step 2:** Implement `analyze` handler:
  - Validate: `document` required, ≥100 words (split on whitespace), ≤15,000 chars
  - Call Claude API with `analyze.ts` prompt, temperature=0
  - Parse JSON response, validate it matches `AnalyzeResponse`
  - Return response

- [ ] **Step 3:** Test manually with curl:
```bash
curl -X POST http://localhost:3000/api/writing-assist \
  -H 'Content-Type: application/json' \
  -d '{"action":"analyze","genre":"essay","document":"<paste 200+ word test text>"}'
```

- [ ] **Step 4:** Commit.
```bash
git add src/app/api/writing-assist/route.ts
git commit -m "feat(writing): API route with analyze action"
```

### Task 1.2: API Route — Guide, Expand, Daily-Tip, Lab-Rewrite

**Files:**
- Modify: `src/app/api/writing-assist/route.ts`

- [ ] **Step 1:** Add `guide` handler (step + dialogue modes). Step: temperature=0.3, returns GuideStepResponse. Dialogue: temperature=0.7, returns GuideDialogueResponse.

- [ ] **Step 2:** Add `expand` handler. Temperature=0.3. Receives annotationContext in request body.

- [ ] **Step 3:** Add `daily-tip` handler. Temperature=0.7. Falls through to static fallback if Claude call fails (import from daily-tips.ts, select by lowest trait score).

- [ ] **Step 4:** Add `lab-rewrite` handler. Validate temperatures array: 1-3 values, each 0.0-2.0. Calls Claude 3 times in parallel with `Promise.all`, same prompt, different temperature. Returns `LabRewriteResponse`.

- [ ] **Step 5:** Add `report` handler — returns 501: `{ error: "Not available yet" }`.

- [ ] **Step 6:** Test each action with curl.

- [ ] **Step 7:** Commit.
```bash
git add src/app/api/writing-assist/route.ts
git commit -m "feat(writing): all API actions (guide, expand, daily-tip, lab-rewrite)"
```

---

## Phase 2: Frontend — Core Editor + Chat

### Task 2.1: Install Tiptap + Storage Utility

**Files:**
- Modify: `package.json`
- Create: `src/lib/writing/storage.ts`

- [ ] **Step 1:** Install Tiptap:
```bash
cd ai-text-detector && npm install @tiptap/react @tiptap/starter-kit @tiptap/extension-highlight @tiptap/pm
```

- [ ] **Step 2:** Write `storage.ts` — localStorage wrapper:
  - `loadState(): WritingCenterState | null`
  - `saveState(state: WritingCenterState): void` (debounced, called by auto-save)
  - `createDefaultProfile(): WriterProfile` (all zeros, empty arrays, showDailyTips: true)
  - `updateStreak(profile: WriterProfile): WriterProfile` (compare lastActiveDate vs today: yesterday → current++, today → noop, 2+ days → current=0)
  - `markDayActive(profile: WriterProfile): WriterProfile` (set lastActiveDate to today, called when ≥50 words written OR 1 analysis completed — caller is responsible for the threshold check)
  - `addAnalysisToProfile(profile, snapshot): WriterProfile`
  - `incrementGenreExperience(profile, genre): WriterProfile`

- [ ] **Step 3:** Commit.
```bash
git add package.json package-lock.json src/lib/writing/storage.ts
git commit -m "feat(writing): tiptap deps + localStorage storage utility"
```

### Task 2.2: WritingCenter Shell Component

**Files:**
- Create: `src/components/writing/WritingCenter.tsx`
- Modify: `src/components/AppShell.tsx`

- [ ] **Step 1:** Create `WritingCenter.tsx` — the top-level component:
  - State: `genre`, `topic`, `document` (Tiptap HTML), `messages`, `annotations`, `activeTab` ("chat" | "lab" | "dashboard"), `layoutPreset` ("side" | "top" | "full"), `loading`, `error`, `profile` (from localStorage), `hasIncrementedThisSession`, `previousConventionsSuppressed`
  - Layout: CSS Grid split-pane with **draggable divider** (mouse/touch drag handler on a 4px bar between editor and panel, updates CSS Grid column/row template). Three preset buttons in toolbar reset to fixed ratios.
  - Tabs: Chat | Dashboard (disabled, shows "Coming soon — track your progress here" placeholder) | Lab
  - Toolbar: genre selector + topic input + layout toggle + **"Copy" button** (calls `navigator.clipboard.writeText(editor.getText())`)
  - Status bar: word count (dynamic: `{n} words · {genre} · (100 words to analyze)` when <100 words), genre, draft status, streak
  - Auto-save: `useEffect` with 2s debounce writing to localStorage

- [ ] **Step 2:** Modify `AppShell.tsx`:
  - Import `WritingCenter` with `dynamic()` (SSR: false)
  - Replace the old `WritingPanel` render with `<WritingCenter />`
  - Remove old Writing Center state variables (`promptStyle`, `topic`, `generatedPrompt`)
  - Status bar: when `activePanel === "writing"`, render WritingCenter's own status bar content

- [ ] **Step 3:** Verify the app builds: `npx next build`

- [ ] **Step 4:** Commit.
```bash
git add src/components/writing/WritingCenter.tsx src/components/AppShell.tsx
git commit -m "feat(writing): WritingCenter shell + AppShell integration"
```

### Task 2.3: Tiptap Editor Component

**Files:**
- Create: `src/components/writing/Editor.tsx`

- [ ] **Step 1:** Create `Editor.tsx`:
  - Tiptap `useEditor` with StarterKit + Highlight extension
  - Props: `content: string`, `onUpdate: (html: string) => void`, `annotations: Annotation[]`
  - Render annotations as Tiptap Decorations (colored underlines per trait color from §5.3)
  - `data-annotation-id` attribute on each Decoration
  - Click handler: when user clicks a decorated span, call `onAnnotationClick(annotationId)`
  - Paragraph node counter (for annotation mapping validation)
  - Fallback: if annotation paragraph/offset is invalid, highlight whole paragraph

- [ ] **Step 2:** Add the genre selector and topic input above the editor (simple select + text input).

- [ ] **Step 3:** Add the "Analyze" button below the editor. Disabled when word count < 100. Tooltip: "Need 100+ words to analyze".

- [ ] **Step 4:** Commit.
```bash
git add src/components/writing/Editor.tsx
git commit -m "feat(writing): Tiptap editor with annotation decorations"
```

### Task 2.4: Chat Panel — Step Cards + Chat Messages

**Files:**
- Create: `src/components/writing/ChatPanel.tsx`
- Create: `src/components/writing/StepCard.tsx`
- Create: `src/components/writing/AnnotationCard.tsx`
- Create: `src/components/writing/DailyTipCard.tsx`

- [ ] **Step 1:** Create `DailyTipCard.tsx` — displays tip with trait badge, before/after example, "Try it" / "Skip" / "Don't show" buttons.

- [ ] **Step 2:** Create `StepCard.tsx` — single SRSD step card with stepIndex/totalSteps progress, title, mnemonic, instructions, optional checklist (checkable items), optional example. Completed state with checkmark.

- [ ] **Step 3:** Create `AnnotationCard.tsx` — annotation detail card. Shows trait badge (colored), severity icon, message, rewrite (if present). Click to trigger expand action. Pulse animation class for bidirectional linking.

- [ ] **Step 4:** Create `ChatPanel.tsx` — assembles all pieces:
  - Top: DailyTipCard (if showDailyTips and tip loaded)
  - Below: StepCards (if mode is "step" and cards exist)
  - Below: Chat messages (alternating user/assistant bubbles)
  - Below: AnnotationCards (when annotations exist, in Liz Lerman order)
  - Bottom: input box for user messages (sends as dialogue)
  - Auto-scroll to bottom on new messages

- [ ] **Step 5:** Commit.
```bash
git add src/components/writing/ChatPanel.tsx src/components/writing/StepCard.tsx src/components/writing/AnnotationCard.tsx src/components/writing/DailyTipCard.tsx
git commit -m "feat(writing): Chat panel with DailyTip, StepCards, AnnotationCards"
```

### Task 2.5: Wire Up AI Actions

**Files:**
- Modify: `src/components/writing/WritingCenter.tsx`

- [ ] **Step 1:** Wire Daily Tip — on mount, check profile.preferences.showDailyTips + totalAnalyses >= 3 → call `/api/writing-assist` with `action: "daily-tip"`. Fallback to static tip if API fails.

- [ ] **Step 2:** Wire Guide (Step) — when genre + topic are set and editor is empty, call `action: "guide"`, `mode: "step"` with experienceLevel. Render returned cards.

- [ ] **Step 3:** Wire Guide (Dialogue) — when user types in Chat input, call `action: "guide"`, `mode: "dialogue"` with messages + document.

- [ ] **Step 4:** Wire Analyze — when user clicks Analyze button, call `action: "analyze"` with document + genre. Parse response → update annotations state → update profile (addAnalysisToProfile, incrementGenreExperience if !hasIncrementedThisSession).

- [ ] **Step 5:** Wire Expand — when user clicks AnnotationCard's "expand" button, call `action: "expand"` with annotationId + annotationContext.

- [ ] **Step 6:** Wire Conventions suppression recovery — track `previousConventionsSuppressed` state. If was true and now false, prepend recovery message to Chat: "Structure looks solid now — let's fine-tune the grammar."

- [ ] **Step 7:** Wire 60s idle detection — `setTimeout(60000)` reset on Tiptap `update` event. On fire, auto-send dialogue prompt.

- [ ] **Step 8:** Commit.
```bash
git add src/components/writing/WritingCenter.tsx
git commit -m "feat(writing): wire all AI actions (tip, guide, analyze, expand)"
```

### Task 2.6: Bidirectional Annotation Linking

**Files:**
- Modify: `src/components/writing/Editor.tsx`
- Modify: `src/components/writing/ChatPanel.tsx`
- Modify: `src/components/writing/WritingCenter.tsx`

- [ ] **Step 1:** Editor → Panel: when user clicks a decorated span, `WritingCenter` receives annotationId, sets `focusedAnnotationId` state, ChatPanel scrolls to matching AnnotationCard and applies pulse class.

- [ ] **Step 2:** Panel → Editor: when user clicks an AnnotationCard, `WritingCenter` calls editor ref's `scrollToAnnotation(id)` which finds the Decoration, calls ProseMirror `scrollIntoView`, and applies yellow pulse CSS (0.5s animation).

- [ ] **Step 3:** Multiple annotations on same paragraph: clicking paragraph highlight cycles through annotations in Liz Lerman order. Small badge "3 annotations" visible on highlight.

- [ ] **Step 4:** Commit.
```bash
git add src/components/writing/Editor.tsx src/components/writing/ChatPanel.tsx src/components/writing/WritingCenter.tsx
git commit -m "feat(writing): bidirectional annotation-editor linking"
```

---

## Phase 3: Lab Tab

### Task 3.1: Lab Panel Component

**Files:**
- Create: `src/components/writing/LabPanel.tsx`

- [ ] **Step 1:** Create `LabPanel.tsx`:
  - Example selector: dropdown or card grid showing all 10 lab examples (title + focusTrait badge)
  - Selected example: displays coldText in a card
  - "See how AI adds temperature" button → calls `action: "lab-rewrite"` with temperatures [0, 0.7, 1.3]
  - Loading state while waiting for 3 LLM calls
  - Three-column comparison: AI-Cold (t=0) | AI-Hot (t=1.3) | displayed side-by-side
  - Each column: text + explanation card below
  - Human-Warm column: always visible from `humanWarmText`, `humanExplanation`
  - Teaching point card at bottom
  - "Your turn" button: fills coldText into editor, sets genre to Essay, sets topic to example.topic, creates new session (resets hasIncrementedThisSession), switches to Chat tab

- [ ] **Step 2:** Add session rate limiting: `labRewriteCount` state, increment per call, disable button after 5 with message "Try writing your own version!"

- [ ] **Step 3:** Commit.
```bash
git add src/components/writing/LabPanel.tsx
git commit -m "feat(writing): Lab panel with temperature comparison"
```

### Task 3.2: Integrate Lab into WritingCenter

**Files:**
- Modify: `src/components/writing/WritingCenter.tsx`

- [ ] **Step 1:** Add "Lab" tab to the collaboration panel tab bar.

- [ ] **Step 2:** Wire Lab's "Your turn" to WritingCenter state: update genre, topic, clear document, switch activeTab to "chat", reset session flag.

- [ ] **Step 3:** Wire lab-rewrite API call from LabPanel through WritingCenter.

- [ ] **Step 4:** Commit.
```bash
git add src/components/writing/WritingCenter.tsx
git commit -m "feat(writing): integrate Lab tab into WritingCenter"
```

---

## Phase 4: Polish + Final Integration

### Task 4.1: Streak + Profile Integration

**Files:**
- Modify: `src/components/writing/WritingCenter.tsx`

- [ ] **Step 1:** On mount, call `updateStreak(profile)` from storage.ts. If streak changed, update localStorage.

- [ ] **Step 2:** Status bar: show `🔥 {n}-day streak` when streak.current > 0. Show nothing when 0.

- [ ] **Step 3:** After each analyze: save AnalysisSnapshot to profile, update traitScores history, increment stats.totalAnalyses, call `markDayActive(profile)`. On auto-save: update stats.totalWords; if totalWords for this session crossed 50-word threshold, call `markDayActive(profile)`.

- [ ] **Step 4:** Commit.
```bash
git add src/components/writing/WritingCenter.tsx
git commit -m "feat(writing): streak tracking + profile stats"
```

### Task 4.2: Error Handling

**Files:**
- Modify: `src/components/writing/WritingCenter.tsx`
- Modify: `src/app/api/writing-assist/route.ts`

- [ ] **Step 1:** Frontend: Add toast/banner for each error case from spec §9.5. Use the existing error banner pattern from AppShell.

- [ ] **Step 2:** Backend: Add input validation for all actions (document length, word count, temperatures array). Return proper 400 status codes.

- [ ] **Step 3:** Daily-tip fallback: if API returns error, load static tip from daily-tips.ts.

- [ ] **Step 4:** Lab-rewrite error: if API fails, show error but keep cold text + human warm text visible.

- [ ] **Step 5:** Commit.
```bash
git add src/components/writing/WritingCenter.tsx src/app/api/writing-assist/route.ts
git commit -m "feat(writing): error handling for all actions"
```

### Task 4.3: Full Build + Smoke Test

- [ ] **Step 1:** Run `npx next build` — fix any TypeScript or build errors.

- [ ] **Step 2:** Start dev server: `npm run dev`. Open Writing Center panel.

- [ ] **Step 3:** Smoke test checklist:
  - [ ] Daily tip appears on first open (static fallback)
  - [ ] Genre selector + topic input works
  - [ ] Step cards appear for first-time genre
  - [ ] Typing in editor works, word count updates
  - [ ] Analyze button disabled under 100 words, enabled above
  - [ ] Analyze returns annotations, highlights appear on text
  - [ ] Click highlight → panel scrolls to annotation
  - [ ] Click annotation → editor scrolls to text
  - [ ] Chat dialogue works (type question, get response)
  - [ ] Lab tab: select example, see cold text, "See AI temperature" works
  - [ ] Lab "Your turn" fills editor, switches to Chat
  - [ ] Streak shows in status bar
  - [ ] Layout presets toggle (side-by-side / top-bottom / fullscreen)
  - [ ] Auto-save works (reload page, content persists)

- [ ] **Step 4:** Commit.
```bash
git add -A
git commit -m "feat(writing): Writing Center MVP-1 complete"
```

---

## Implementation Order Summary

```
Phase 0: Prompt Calibration (~1 week)
  0.1 Write 12 test articles        (content, can start day 1)
  0.2 Write types.ts                (code, can start day 1)
  0.3 Write + calibrate analyze.ts  (core, blocks Phase 1)
  0.4 Write 5 other prompts         (after 0.3 pattern established)
  0.5 Write daily tips + lab content (content, parallel with 0.3-0.4)

Phase 1: API Route (~2-3 days)
  1.1 Route skeleton + analyze      (blocked by 0.3)
  1.2 All other actions             (blocked by 0.4)

Phase 2: Frontend Core (~1 week)
  2.1 Tiptap + storage              (blocked by 1.1 for testing)
  2.2 WritingCenter shell           (blocked by 2.1)
  2.3 Editor component              (blocked by 2.2)
  2.4 Chat panel components         (parallel with 2.3)
  2.5 Wire AI actions               (blocked by 2.3 + 2.4)
  2.6 Bidirectional linking         (blocked by 2.5)

Phase 3: Lab Tab (~2-3 days)
  3.1 Lab panel                     (blocked by 0.5 content + 1.2 lab-rewrite API)
  3.2 Lab integration               (blocked by 3.1)

Phase 4: Polish (~2 days)
  4.1 Streak + profile              (blocked by 2.5)
  4.2 Error handling                (blocked by 2.5)
  4.3 Build + smoke test            (blocked by everything)
```

**Total estimated time: 3-4 weeks for a single developer.**
