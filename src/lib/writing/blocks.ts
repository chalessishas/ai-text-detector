// Block definitions for the Writing Center "building blocks" system.
// Each block is an independent writing activity with clear input/output.
// Users pick blocks, arrange them, and work through them at their own pace.
//
// 21 blocks across 8 categories, based on composition research:
// Sommers (revising ≠ editing ≠ proofreading), Flower & Hayes (cognitive process),
// PAIRR (separate drafting from AI feedback), Yancey (metacognitive reflection).

export type BlockType =
  // Pre-writing
  | "brainstorm"
  | "audience"
  | "research"
  // Planning
  | "thesis"
  | "outline"
  | "evidence-eval"
  // Drafting
  | "hook"
  | "draft"
  | "evidence"
  | "counterargument"
  | "conclusion"
  // Reviewing
  | "peer-review"
  | "reader-sim"
  // Revising
  | "analyze"
  | "logic-check"
  | "voice-lab"
  // Editing
  | "transitions"
  | "style-edit"
  // Proofreading
  | "grammar"
  // Publishing
  | "submit-ready"
  // Reflecting
  | "self-review";

export type BlockCategory =
  | "pre-writing"
  | "planning"
  | "drafting"
  | "reviewing"
  | "revising"
  | "editing"
  | "proofreading"
  | "publishing"
  | "reflecting";

// How the block renders its UI
export type BlockMode = "chat" | "editor" | "lab" | "checklist";

export interface BlockAutoConfig {
  needs: string[];       // Required inputs from user or previous block
  produces: string;      // Key name for this block's output in PipelineOutputs
  checkpoint: boolean;   // Whether to pause for user input
  promptKey: string;     // Key into AUTO_PROMPTS (lives in auto-prompts.ts)
}

export interface BlockDef {
  type: BlockType;
  name: string;
  nameZh: string;
  description: string;
  category: BlockCategory;
  color: string;
  mode: BlockMode;
  aiRole: string;
  auto?: BlockAutoConfig | null;
}

// Runtime state for a block on the user's board
export interface BlockInstance {
  id: string;
  type: BlockType;
  status: "todo" | "active" | "done";
  output: string;
}

export interface BlockPreset {
  id: string;
  name: string;
  nameZh: string;
  description: string;
  blocks: BlockType[];
}

// ── Block catalog (21 blocks) ──

export const BLOCK_CATALOG: BlockDef[] = [
  // ─── Pre-writing ───
  {
    type: "brainstorm",
    name: "Brainstorm",
    nameZh: "头脑风暴",
    description: "Explore ideas freely with AI asking probing questions",
    category: "pre-writing",
    color: "#ee6c4d",
    mode: "chat",
    aiRole: "Asks open-ended questions to help you discover what you really want to say",
  },
  {
    type: "audience",
    name: "Audience",
    nameZh: "受众画像",
    description: "Define who you're writing for and what they care about",
    category: "pre-writing",
    color: "#e07a5f",
    mode: "chat",
    aiRole: "Helps you think through your reader's perspective, knowledge, and expectations",
  },
  {
    type: "research",
    name: "Research",
    nameZh: "调研收集",
    description: "Gather evidence, sources, and supporting material",
    category: "pre-writing",
    color: "#d4775c",
    mode: "chat",
    aiRole: "Suggests research directions and helps evaluate source quality — never provides content",
  },

  // ─── Planning ───
  {
    type: "thesis",
    name: "Thesis",
    nameZh: "论点锤炼",
    description: "Craft a clear, arguable thesis statement",
    category: "planning",
    color: "#e0a458",
    mode: "chat",
    aiRole: "Challenges your thesis through Socratic questioning until it's sharp and defensible",
    auto: {
      needs: ["topic"],
      produces: "thesis",
      checkpoint: true,
      promptKey: "thesis-checkpoint",
    },
  },
  {
    type: "outline",
    name: "Outline",
    nameZh: "大纲搭建",
    description: "Build the skeleton structure of your piece",
    category: "planning",
    color: "#d4983e",
    mode: "chat",
    aiRole: "Helps organize your ideas into a logical structure without writing content for you",
    auto: {
      needs: ["thesis"],
      produces: "outline",
      checkpoint: true,
      promptKey: "outline-checkpoint",
    },
  },
  {
    type: "evidence-eval",
    name: "Evidence Eval",
    nameZh: "论据评估",
    description: "Evaluate the strength and relevance of your evidence before drafting",
    category: "planning",
    color: "#c4892e",
    mode: "chat",
    aiRole: "Helps you assess whether your evidence is strong, relevant, and sufficient — before you start writing",
  },

  // ─── Drafting ───
  {
    type: "hook",
    name: "Hook",
    nameZh: "开篇设计",
    description: "Craft a compelling opening that pulls readers in",
    category: "drafting",
    color: "#2d4a6f",
    mode: "chat",
    aiRole: "Guides you to write an opening that grabs attention — asks what you want the reader to feel in the first 10 seconds",
    auto: {
      needs: ["thesis", "outline"],
      produces: "hook",
      checkpoint: true,
      promptKey: "hook-checkpoint",
    },
  },
  {
    type: "draft",
    name: "Draft",
    nameZh: "自由写作",
    description: "Write freely — AI steps back, this is your space",
    category: "drafting",
    color: "#3d5a80",
    mode: "editor",
    aiRole: "Minimal presence. Only offers encouragement and word count goals. Your words, your voice.",
    auto: {
      needs: ["thesis", "outline", "hook"],
      produces: "draft",
      checkpoint: false,
      promptKey: "draft-auto",
    },
  },
  {
    type: "evidence",
    name: "Evidence Integrate",
    nameZh: "论据编织",
    description: "Weave quotes, data, and sources into your argument effectively",
    category: "drafting",
    color: "#4d6a90",
    mode: "chat",
    aiRole: "Teaches how to introduce, integrate, and analyze evidence — not just drop quotes in",
  },
  {
    type: "counterargument",
    name: "Counterargument",
    nameZh: "反驳预演",
    description: "AI plays devil's advocate against your argument",
    category: "drafting",
    color: "#5b7ea8",
    mode: "chat",
    aiRole: "Presents the strongest possible objections to your thesis so you can prepare rebuttals",
  },
  {
    type: "conclusion",
    name: "Conclusion",
    nameZh: "收束总结",
    description: "Write an ending that ties back to your thesis and leaves an impression",
    category: "drafting",
    color: "#6b8eb8",
    mode: "chat",
    aiRole: "Guides you to write a conclusion that synthesizes — not just repeats — your argument, and opens a wider perspective",
  },

  // ─── Reviewing (human + AI feedback, per PAIRR model) ───
  {
    type: "peer-review",
    name: "Peer Review",
    nameZh: "同伴互评",
    description: "Get structured feedback from a real reader using a guided rubric",
    category: "reviewing",
    color: "#6a8e6a",
    mode: "checklist",
    aiRole: "Provides a structured rubric for your peer reviewer and helps you process their feedback",
  },
  {
    type: "reader-sim",
    name: "Reader Sim",
    nameZh: "模拟读者",
    description: "AI simulates a first-time reader's honest reaction to your draft",
    category: "reviewing",
    color: "#5a9e6a",
    mode: "chat",
    aiRole: "Reads your draft as a real reader would — reports confusion, engagement, and questions, not corrections",
  },

  // ─── Revising (global concerns: argument, structure, coherence) ───
  {
    type: "analyze",
    name: "Analyze",
    nameZh: "七维分析",
    description: "Get trait-based feedback on ideas, structure, voice, and more",
    category: "revising",
    color: "#5b8a72",
    mode: "editor",
    aiRole: "Scores 7 writing traits and provides specific, actionable annotations",
    auto: {
      needs: ["draft"],
      produces: "analyze",
      checkpoint: false,
      promptKey: "analyze-auto",
    },
  },
  {
    type: "logic-check",
    name: "Logic Check",
    nameZh: "逻辑审查",
    description: "Find logical fallacies and argument gaps",
    category: "revising",
    color: "#4a7c62",
    mode: "chat",
    aiRole: "Examines your reasoning step by step, flags logical fallacies and unsupported claims",
  },
  {
    type: "voice-lab",
    name: "Voice Lab",
    nameZh: "风格实验室",
    description: "Explore why AI text feels 'cold' and learn to write with warmth",
    category: "revising",
    color: "#7b6d8d",
    mode: "lab",
    aiRole: "Shows temperature-based rewrites side by side so you can see what makes writing feel human",
  },

  // ─── Editing (sentence-level: clarity, style, transitions) ───
  {
    type: "transitions",
    name: "Transitions",
    nameZh: "衔接打磨",
    description: "Strengthen the logical flow between paragraphs and sections",
    category: "editing",
    color: "#9d7e63",
    mode: "chat",
    aiRole: "Reviews paragraph-to-paragraph connections, teaches transition techniques, flags jumpy logic",
  },
  {
    type: "style-edit",
    name: "Style Polish",
    nameZh: "风格打磨",
    description: "Improve clarity, word choice, and sentence variety",
    category: "editing",
    color: "#8d6e63",
    mode: "chat",
    aiRole: "Suggests sentence-level improvements with explanations — teaches 'why', not just 'what'",
  },

  // ─── Proofreading (surface mechanics only — Sommers distinction) ───
  {
    type: "grammar",
    name: "Proofread",
    nameZh: "终校",
    description: "Fix grammar, spelling, and punctuation — surface errors only",
    category: "proofreading",
    color: "#7d6558",
    mode: "editor",
    aiRole: "Flags mechanical errors with direct corrections. The one block where AI gives you the fix, not just a question.",
    auto: {
      needs: ["analyze"],
      produces: "grammar",
      checkpoint: false,
      promptKey: "grammar-auto",
    },
  },

  // ─── Publishing ───
  {
    type: "submit-ready",
    name: "Submit Ready",
    nameZh: "提交检查",
    description: "Final checks before submission: format, citations, word count, completeness",
    category: "publishing",
    color: "#4a6896",
    mode: "checklist",
    aiRole: "Runs a submission readiness checklist — format compliance, citation completeness, word count, and final review",
  },

  // ─── Reflecting ───
  {
    type: "self-review",
    name: "Self-Review",
    nameZh: "自我检查",
    description: "Checklist-based self-assessment based on your writing and past weak points",
    category: "reflecting",
    color: "#4a7c96",
    mode: "checklist",
    aiRole: "Generates a personalized checklist from your writing history and trait scores — you check, not AI",
  },
];

export function getBlockDef(type: BlockType): BlockDef {
  return BLOCK_CATALOG.find((b) => b.type === type)!;
}

// ── Preset kits (6 templates) ──

export const BLOCK_PRESETS: BlockPreset[] = [
  {
    id: "academic",
    name: "Academic Essay",
    nameZh: "学术论文",
    description: "Full process for argumentative essays",
    blocks: [
      "brainstorm", "thesis", "outline", "research",
      "hook", "draft", "evidence", "counterargument", "conclusion",
      "reader-sim", "analyze", "transitions", "grammar", "self-review",
    ],
  },
  {
    id: "research-paper",
    name: "Research Paper",
    nameZh: "研究论文",
    description: "Evidence-heavy paper with rigorous structure",
    blocks: [
      "research", "thesis", "outline", "evidence-eval",
      "hook", "draft", "evidence", "counterargument", "conclusion",
      "peer-review", "analyze", "logic-check", "transitions",
      "grammar", "submit-ready", "self-review",
    ],
  },
  {
    id: "quick",
    name: "Quick Write",
    nameZh: "快速写作",
    description: "Draft, get feedback, polish",
    blocks: ["draft", "analyze", "grammar"],
  },
  {
    id: "creative",
    name: "Creative",
    nameZh: "创意写作",
    description: "Explore voice and style",
    blocks: ["brainstorm", "hook", "draft", "voice-lab", "style-edit", "conclusion"],
  },
  {
    id: "toefl",
    name: "TOEFL",
    nameZh: "托福写作",
    description: "Timed essay with structure and polish",
    blocks: ["thesis", "outline", "hook", "draft", "analyze", "grammar"],
  },
  {
    id: "revision-only",
    name: "Revision Focus",
    nameZh: "专项修改",
    description: "Already have a draft? Focus on improving it",
    blocks: ["reader-sim", "analyze", "logic-check", "transitions", "style-edit", "grammar"],
  },
];

// ── Category metadata ──

export const CATEGORY_LABELS: Record<BlockCategory, { name: string; nameZh: string }> = {
  "pre-writing": { name: "Pre-Writing", nameZh: "构思" },
  planning: { name: "Planning", nameZh: "规划" },
  drafting: { name: "Drafting", nameZh: "起草" },
  reviewing: { name: "Reviewing", nameZh: "评审" },
  revising: { name: "Revising", nameZh: "修改" },
  editing: { name: "Editing", nameZh: "编辑" },
  proofreading: { name: "Proofreading", nameZh: "终校" },
  publishing: { name: "Publishing", nameZh: "发布" },
  reflecting: { name: "Reflecting", nameZh: "反思" },
};

export const CATEGORY_ORDER: BlockCategory[] = [
  "pre-writing",
  "planning",
  "drafting",
  "reviewing",
  "revising",
  "editing",
  "proofreading",
  "publishing",
  "reflecting",
];

// ── Helpers ──

export function createBlockInstance(type: BlockType): BlockInstance {
  return {
    id: crypto.randomUUID(),
    type,
    status: "todo",
    output: "",
  };
}

// ── Chat-based system prompts per block type ──

export const BLOCK_SYSTEM_PROMPTS: Partial<Record<BlockType, string>> = {
  brainstorm: `You are a brainstorming coach. Your ONLY job is to help the writer discover what they want to say.
Rules:
- Ask ONE open-ended question at a time
- Never suggest content, topics, or arguments
- Reflect back what you hear: "So you're saying X — is that the core of it?"
- When the writer has a clear direction, say: "Sounds like you know what you want to say. Ready to move on?"
- Keep responses under 3 sentences`,

  audience: `You help writers think about their readers. Ask questions like:
- Who will read this? What do they already know?
- What do they care about? What might they disagree with?
- What tone would resonate with them?
Never write content. Just help the writer see through their reader's eyes. One question at a time.`,

  research: `You are a research advisor. Help the writer think about what evidence they need.
- What claims need supporting evidence?
- What types of sources would be strongest? (data, expert opinion, case study)
- How will they evaluate source credibility?
Suggest search strategies and source types. Never provide facts, quotes, or citations yourself.`,

  thesis: `You are a thesis coach. Help the writer craft a strong thesis statement.
A good thesis: has a clear position (not just a fact), is arguable (reasonable people could disagree), and is specific enough to be supported in the essay.
Method: Ask what they believe → challenge it ("But couldn't someone say X?") → refine until it's sharp.
Never write the thesis for them. Ask questions that lead them to write it themselves.`,

  outline: `You help writers organize their ideas into a structure. Ask:
- What are your 2-4 main supporting points?
- In what order should they appear? Why?
- What evidence do you have for each point?
Help them see the logical flow. Suggest structural patterns (chronological, compare/contrast, problem/solution) but let them choose. Never fill in content.`,

  "evidence-eval": `You help writers evaluate evidence BEFORE they start drafting.
For each piece of evidence the writer mentions, ask:
- Is it relevant to your thesis? How directly?
- Is it credible? (peer-reviewed? primary source? expert opinion?)
- Is it sufficient? Or do you need more to convince a skeptic?
- Could someone use this same evidence to argue the opposite?
Help them identify gaps in their evidence and prioritize the strongest pieces. Never provide evidence yourself.`,

  hook: `You help writers craft compelling openings. NEVER write the hook for them.
Instead, ask:
- What emotion or reaction do you want from your reader in the first sentence?
- What's the most surprising or counterintuitive aspect of your topic?
- Could you start with a vivid scene, a striking question, or a bold claim?
Offer 3-4 opening STRATEGIES (not actual sentences) and let them choose and write. If they share a draft opening, give honest feedback: does it make you want to keep reading? Why or why not?`,

  evidence: `You help writers integrate evidence into their arguments effectively. The writer already has their evidence — your job is to help them USE it well.
Teach the ICE method:
- Introduce: Who said this? Why should the reader care?
- Cite: The actual quote, data, or paraphrase
- Explain: What does this prove? How does it support YOUR argument?
Review their paragraphs one at a time. Flag "dropped quotes" (evidence with no introduction or analysis). Never provide evidence or rewrite their paragraphs.`,

  counterargument: `You are a skilled devil's advocate. Your job:
1. Read the writer's argument carefully
2. Present the STRONGEST possible objection (not strawmen)
3. Ask: "How would you respond to this?"
4. If their response is weak, push harder
5. When they've strengthened their argument, acknowledge it
Be intellectually honest. Good counterarguments make the final essay stronger.`,

  conclusion: `You help writers craft strong conclusions. NEVER write the conclusion for them.
A good conclusion does NOT just repeat the thesis. It should:
- Synthesize (not summarize) — show how the argument BUILT to something larger
- Answer "so what?" — why does this matter beyond the essay?
- Leave the reader thinking — a question, a call to action, or a wider implication
Ask: "If your reader remembers ONE thing from your essay tomorrow, what should it be?" Work from there.`,

  "reader-sim": `You are simulating a first-time reader. You have NOT read this text before. Read the writer's draft and report your HONEST reader experience:

1. First impression: What did you expect from the title/opening? Were you pulled in or confused?
2. Flow: Where did you follow easily? Where did you get lost or have to re-read?
3. Questions: What questions popped into your head that the text didn't answer?
4. Engagement: Where were you most interested? Where did your attention drop?
5. Takeaway: After reading, what's the ONE main point you took away? (If it's not what the writer intended, that's important feedback.)

Do NOT suggest fixes. Just report your experience as a reader. Be honest — a polite reader is a useless reader.`,

  "logic-check": `You are a logic reviewer. Examine the writer's reasoning:
- Is each claim supported by evidence?
- Are there logical fallacies? (ad hominem, slippery slope, false dichotomy, etc.)
- Does the conclusion follow from the premises?
- Are there gaps in the argument?
Point out ONE issue at a time with a clear explanation. Ask: "How might you address this?"`,

  transitions: `You help writers improve the flow between paragraphs and sections.
Read the writer's draft and focus on:
- Does each paragraph connect logically to the next? Or does the reader have to "jump"?
- Are there transition sentences that show the relationship? (contrast, cause-effect, addition, example)
- Does the overall sequence make sense, or should sections be reordered?
Point out ONE weak transition at a time. Explain WHY it feels disconnected and suggest a transition STRATEGY (not an actual sentence). Let the writer do the rewriting.`,

  "style-edit": `You are a style coach. Focus on sentence-level improvements:
- Identify unclear or wordy passages
- Suggest more precise word choices (explain WHY the alternative is better)
- Point out monotonous sentence patterns
- Flag jargon or overly complex phrasing
Show the original and explain what could improve, but let the writer do the rewriting. Teach the principle, not just the fix.`,
};

// ── Phase transition micro-reflection prompts (Yancey model) ──
// Shown when user moves between category boundaries

export const PHASE_TRANSITION_PROMPTS: Partial<Record<`${BlockCategory}->${BlockCategory}`, string>> = {
  "pre-writing->planning": "Before you start organizing: what surprised you during brainstorming? Did your direction change?",
  "planning->drafting": "You have a plan. Before you write: what part are you most excited to draft? What part feels hardest?",
  "drafting->reviewing": "Your draft is down. Before getting feedback: what do YOU think is the strongest part? The weakest?",
  "drafting->revising": "Before revising: re-read your thesis. Does your draft actually argue what you planned to argue?",
  "reviewing->revising": "You've received feedback. What surprised you? What feedback do you disagree with, and why?",
  "revising->editing": "Big-picture issues are addressed. Before sentence-level polishing: read one paragraph aloud. How does it sound?",
  "editing->proofreading": "Style is polished. Now just surface errors. Read backwards, sentence by sentence — it helps catch typos.",
  "proofreading->publishing": "Almost done. Before submitting: does the final version match the assignment requirements?",
  "proofreading->reflecting": "Before you reflect: are you proud of this piece? What would you do differently next time?",
  "publishing->reflecting": "It's submitted. Take a moment: what did you learn about your writing process this time?",
};
