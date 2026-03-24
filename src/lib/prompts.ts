/**
 * Prompt templates that guide AI to generate text matching our corpus style.
 *
 * These prompts encode the statistical fingerprint of the humanizer corpus:
 * - Avg 23 words/sentence, 1.2 commas/sentence
 * - Heavy use of "However,", "Although", "Despite", "While"
 * - Mix of active/passive voice (~25% passive)
 * - Sentence starters: "It was", "In the", "According to", "This was"
 * - Word-level complexity: ~4.5 chars/word (not too simple, not too academic)
 */

export const CORPUS_STYLE_GUIDE = `Writing style requirements (follow these exactly):

SENTENCE STRUCTURE:
- Each sentence should be 18-28 words long
- Use 1-2 commas per sentence for moderate complexity
- Vary sentence openings: start some with "However,", "Although", "Despite", "In the", "According to", "While"
- Mix active and passive voice — roughly 1 in 4 sentences should use passive voice (e.g., "was established", "were discovered", "is considered")

VOCABULARY:
- Use common but precise words (average 4-5 characters per word)
- Avoid overly academic jargon or overly casual slang
- Do NOT use these AI-typical phrases: "delve into", "it's important to note", "in conclusion", "furthermore", "leveraging", "holistic", "paradigm", "multifaceted"

FLOW:
- Use transition words naturally: "however", "although", "while", "despite", "meanwhile"
- Do NOT start every sentence the same way — vary between declarative, "In the..." prepositional, and "However,..." transitional openings
- Avoid numbered lists or bullet points — write in flowing paragraphs

WHAT TO AVOID:
- No sentences shorter than 12 words or longer than 35 words
- No rhetorical questions
- No exclamation marks
- No first-person ("I think", "I believe") unless specifically asked
- No hedging phrases ("it could be argued", "one might suggest")`;

export const PROMPT_TEMPLATES: Record<string, string> = {
  essay: `Write an essay about the following topic. ${CORPUS_STYLE_GUIDE}

Topic: {topic}

Write 3-5 paragraphs. Each paragraph should have 3-5 sentences.`,

  article: `Write a news-style article about the following topic. ${CORPUS_STYLE_GUIDE}

Additional rules for news style:
- Use third person perspective throughout
- Include specific but fictional details (names, places, numbers) to feel concrete
- Start with the most important information

Topic: {topic}`,

  academic: `Write an academic analysis of the following topic. ${CORPUS_STYLE_GUIDE}

Additional rules for academic style:
- Use more passive voice (about 1 in 3 sentences)
- Reference general research findings ("Studies have shown...", "Research suggests...")
- Maintain an objective, analytical tone throughout

Topic: {topic}`,

  opinion: `Write an opinion piece about the following topic. ${CORPUS_STYLE_GUIDE}

Additional rules for opinion style:
- Take a clear position but present counterarguments
- Use "While..." and "Although..." to acknowledge opposing views
- Support claims with general reasoning, not personal anecdotes

Topic: {topic}`,
};

export type PromptStyle = keyof typeof PROMPT_TEMPLATES;

export function buildPrompt(style: PromptStyle, topic: string): string {
  return PROMPT_TEMPLATES[style].replace('{topic}', topic);
}
