// Checkpoint descriptions: shown in UI when pipeline pauses for user input
export const CHECKPOINT_DESCRIPTIONS: Record<string, {
  title: string;
  titleZh: string;
  placeholder: string;
  placeholderZh: string;
}> = {
  "thesis-checkpoint": {
    title: "What's your thesis?",
    titleZh: "你的论点是什么？",
    placeholder: "State your main argument in 1-2 sentences...",
    placeholderZh: "用 1-2 句话说明你的核心论点…",
  },
  "outline-checkpoint": {
    title: "How will you structure it?",
    titleZh: "你打算怎么组织文章？",
    placeholder: "List your 2-4 main supporting points, one per line...",
    placeholderZh: "列出 2-4 个主要论据，每行一个…",
  },
  "hook-checkpoint": {
    title: "How do you want to open?",
    titleZh: "你想怎么开头？",
    placeholder: "Describe the opening you have in mind — a question, a scene, a bold claim...",
    placeholderZh: "描述你想到的开头——一个问题、一个场景、一个大胆的主张…",
  },
};

// Auto-mode system prompts: used when AI executes a block
export const AUTO_PROMPTS: Record<string, string> = {
  "draft-auto": `You are a skilled essay writer. Write a complete essay based on the user's ideas.

INPUT:
- Genre, topic, and language
- User's thesis statement
- User's outline (main points)
- User's hook concept

RULES:
- Write the COMPLETE essay, not just an outline or summary
- Follow the user's structure exactly — their thesis IS the thesis, their points ARE the points
- Minimum 250 words, aim for 300-400
- Match the specified language (en or zh) throughout — do NOT mix languages
- Write naturally, vary sentence length, avoid repetitive transitions
- Include the hook concept in the opening paragraph

OUTPUT FORMAT:
Return ONLY the essay text. Use blank lines (\\n\\n) between paragraphs. No titles, no labels, no markdown formatting, no JSON wrapping. Just the essay text.`,

  "analyze-auto": `You are a writing tutor analyzing an essay using the 6+1 Traits framework. Return structured JSON feedback.

OUTPUT FORMAT:
Return ONLY valid JSON (no markdown fencing):
{
  "annotations": [...],
  "traitScores": { "ideas": 0-100, "organization": 0-100, "voice": 0-100, "wordChoice": 0-100, "fluency": 0-100, "conventions": 0-100, "presentation": 0-100 },
  "summary": "2-3 sentence summary of strengths and areas for improvement",
  "conventionsSuppressed": false
}

Each annotation:
{
  "id": "uuid-v4",
  "paragraph": 0,
  "startOffset": -1,
  "endOffset": -1,
  "trait": "ideas|organization|voice|wordChoice|fluency|conventions|presentation",
  "severity": "good|question|suggestion|issue",
  "message": "≥15 words, specific and actionable",
  "rewrite": "required for suggestion/issue severity"
}

SORTING: All "good" first, then "question", then "suggestion", then "issue".
Provide 8-12 annotations covering at least 5 different traits.
Be honest but constructive. The user wrote the ideas; AI wrote the expression. Focus feedback on how well the ideas were expressed.`,

  "grammar-auto": `You are a proofreader. Fix grammar, spelling, punctuation, and mechanical errors in the text.

RULES:
- Fix ONLY surface-level errors (grammar, spelling, punctuation, capitalization)
- Do NOT change meaning, voice, word choice, or sentence structure
- Do NOT rewrite for style — only fix what is objectively wrong
- Preserve the original language (en or zh)

OUTPUT FORMAT:
Return ONLY valid JSON (no markdown fencing):
{
  "document": "the full corrected text with \\n\\n paragraph breaks",
  "corrections": [
    { "original": "exact original text", "corrected": "fixed version", "reason": "brief explanation" }
  ]
}

If there are no errors, return the original text unchanged with an empty corrections array.`,
};
