// Block definitions for the Writing Center "building blocks" system.
// Each block is an independent writing activity with clear input/output.
// Users pick blocks, arrange them, and work through them at their own pace.

export type BlockType =
  | "brainstorm"
  | "audience"
  | "thesis"
  | "outline"
  | "research"
  | "draft"
  | "counterargument"
  | "analyze"
  | "logic-check"
  | "voice-lab"
  | "style-edit"
  | "grammar"
  | "self-review";

export type BlockCategory =
  | "pre-writing"
  | "planning"
  | "drafting"
  | "revising"
  | "editing"
  | "reflecting";

// How the block renders its UI
export type BlockMode = "chat" | "editor" | "lab" | "checklist";

export interface BlockDef {
  type: BlockType;
  name: string;
  nameZh: string;
  description: string;
  category: BlockCategory;
  color: string;
  mode: BlockMode;
  // What AI does in this block (shown to user as tooltip)
  aiRole: string;
}

// Runtime state for a block on the user's board
export interface BlockInstance {
  id: string;
  type: BlockType;
  status: "todo" | "active" | "done";
  // Block-specific output (persisted between visits)
  output: string;
}

export interface BlockPreset {
  id: string;
  name: string;
  nameZh: string;
  description: string;
  blocks: BlockType[];
}

// ── Block catalog ──

export const BLOCK_CATALOG: BlockDef[] = [
  // Pre-writing
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

  // Planning
  {
    type: "thesis",
    name: "Thesis",
    nameZh: "论点锤炼",
    description: "Craft a clear, arguable thesis statement",
    category: "planning",
    color: "#e0a458",
    mode: "chat",
    aiRole: "Challenges your thesis through Socratic questioning until it's sharp and defensible",
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
  },

  // Drafting
  {
    type: "draft",
    name: "Draft",
    nameZh: "自由写作",
    description: "Write freely — AI steps back, this is your space",
    category: "drafting",
    color: "#3d5a80",
    mode: "editor",
    aiRole: "Minimal presence. Only offers encouragement and word count goals. Your words, your voice.",
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

  // Revising
  {
    type: "analyze",
    name: "Analyze",
    nameZh: "七维分析",
    description: "Get trait-based feedback on ideas, structure, voice, and more",
    category: "revising",
    color: "#5b8a72",
    mode: "editor",
    aiRole: "Scores 7 writing traits and provides specific, actionable annotations",
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

  // Editing
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
  {
    type: "grammar",
    name: "Grammar",
    nameZh: "语法校对",
    description: "Fix grammar, spelling, and punctuation",
    category: "editing",
    color: "#7d6558",
    mode: "editor",
    aiRole: "Flags surface errors with corrections — the one block where AI gives direct fixes",
  },

  // Reflecting
  {
    type: "self-review",
    name: "Self-Review",
    nameZh: "自我检查",
    description: "Checklist-based self-assessment before you submit",
    category: "reflecting",
    color: "#4a7c96",
    mode: "checklist",
    aiRole: "Generates a personalized checklist based on your writing and past weak points",
  },
];

export function getBlockDef(type: BlockType): BlockDef {
  return BLOCK_CATALOG.find((b) => b.type === type)!;
}

// ── Preset kits ──

export const BLOCK_PRESETS: BlockPreset[] = [
  {
    id: "academic",
    name: "Academic Essay",
    nameZh: "学术论文",
    description: "Full process for argumentative essays",
    blocks: ["brainstorm", "thesis", "outline", "draft", "counterargument", "analyze", "grammar", "self-review"],
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
    blocks: ["brainstorm", "draft", "voice-lab", "style-edit"],
  },
  {
    id: "toefl",
    name: "TOEFL",
    nameZh: "托福写作",
    description: "Timed essay with structure",
    blocks: ["thesis", "outline", "draft", "analyze", "grammar"],
  },
];

// ── Category metadata ──

export const CATEGORY_LABELS: Record<BlockCategory, { name: string; nameZh: string }> = {
  "pre-writing": { name: "Pre-Writing", nameZh: "构思" },
  planning: { name: "Planning", nameZh: "规划" },
  drafting: { name: "Drafting", nameZh: "起草" },
  revising: { name: "Revising", nameZh: "修改" },
  editing: { name: "Editing", nameZh: "编辑" },
  reflecting: { name: "Reflecting", nameZh: "反思" },
};

// Ordered categories for display
export const CATEGORY_ORDER: BlockCategory[] = [
  "pre-writing",
  "planning",
  "drafting",
  "revising",
  "editing",
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

// Chat-based system prompts per block type
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

  thesis: `You are a thesis coach. Help the writer craft a strong thesis statement.
A good thesis: has a clear position (not just a fact), is arguable (reasonable people could disagree), and is specific enough to be supported in the essay.
Method: Ask what they believe → challenge it ("But couldn't someone say X?") → refine until it's sharp.
Never write the thesis for them. Ask questions that lead them to write it themselves.`,

  outline: `You help writers organize their ideas into a structure. Ask:
- What are your 2-4 main supporting points?
- In what order should they appear? Why?
- What evidence do you have for each point?
Help them see the logical flow. Suggest structural patterns (chronological, compare/contrast, problem/solution) but let them choose. Never fill in content.`,

  research: `You are a research advisor. Help the writer think about what evidence they need.
- What claims need supporting evidence?
- What types of sources would be strongest? (data, expert opinion, case study)
- How will they evaluate source credibility?
Suggest search strategies and source types. Never provide facts, quotes, or citations yourself.`,

  counterargument: `You are a skilled devil's advocate. Your job:
1. Read the writer's argument carefully
2. Present the STRONGEST possible objection (not strawmen)
3. Ask: "How would you respond to this?"
4. If their response is weak, push harder
5. When they've strengthened their argument, acknowledge it
Be intellectually honest. Good counterarguments make the final essay stronger.`,

  "logic-check": `You are a logic reviewer. Examine the writer's reasoning:
- Is each claim supported by evidence?
- Are there logical fallacies? (ad hominem, slippery slope, false dichotomy, etc.)
- Does the conclusion follow from the premises?
- Are there gaps in the argument?
Point out ONE issue at a time with a clear explanation. Ask: "How might you address this?"`,

  "style-edit": `You are a style coach. Focus on sentence-level improvements:
- Identify unclear or wordy passages
- Suggest more precise word choices (explain WHY the alternative is better)
- Point out monotonous sentence patterns
- Flag jargon or overly complex phrasing
Show the original and explain what could improve, but let the writer do the rewriting. Teach the principle, not just the fix.`,
};
