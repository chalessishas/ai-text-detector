export const LAB_REWRITE_SYSTEM_PROMPT = `Rewrite this text to make it more engaging and vivid. Do not change the core meaning or add new factual claims. Make stylistic improvements only.

Keep the output approximately the same length as the input.

Return a JSON object with two fields:
- "text": the rewritten text
- "explanation": 1-2 sentences explaining what you changed and why (e.g. "Replaced passive constructions with active voice to create urgency")`;
