"""Configuration for multi-model dataset generation.

27 models across 7 providers, 3 tiers.
21 prompt styles × 60 topics × 4 temperatures × 3 lengths.
"""

# --- API Provider Configs ---
# Each provider has: base_url, env_key (for API key), models list

PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "models": {
            "gpt-5.4": {"tier": "flagship", "output_price": 15.00, "max_temp": 0.7},
            "gpt-5": {"tier": "flagship", "output_price": 10.00, "fixed_temp": 1.0},
            "gpt-4o": {"tier": "mid", "output_price": 10.00},
            "gpt-4o-mini": {"tier": "mid", "output_price": 0.60},
            "gpt-5.4-mini": {"tier": "mid", "output_price": 0.60, "max_temp": 0.7},
            "gpt-5.4-nano": {"tier": "budget", "output_price": 0.40, "max_temp": 0.7},
        },
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "env_key": "ANTHROPIC_API_KEY",
        "models": {
            "claude-opus-4-6": {"tier": "flagship", "output_price": 25.00},
            "claude-sonnet-4-6": {"tier": "mid", "output_price": 15.00},
            "claude-haiku-4-5": {"tier": "mid", "output_price": 5.00},
        },
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "env_key": "GEMINI_API_KEY",
        "models": {
            "gemini-3.1-pro-preview": {"tier": "flagship", "output_price": 10.00, "thinking": True},
            "gemini-3-flash-preview": {"tier": "mid", "output_price": 2.50, "thinking": True},
            "gemini-3.1-flash-lite-preview": {"tier": "budget", "output_price": 0.40},
            "gemini-2.5-pro": {"tier": "flagship", "output_price": 10.00, "thinking": True},
            "gemini-2.5-flash": {"tier": "mid", "output_price": 2.50, "thinking": True},
            "gemini-2.5-flash-lite": {"tier": "budget", "output_price": 0.40},
        },
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "QWEN_API_KEY",
        "models": {
            "qwen3-max": {"tier": "flagship", "output_price": 3.90},
            "qwen3.5-plus": {"tier": "mid", "output_price": 1.56},
            "qwen-turbo": {"tier": "budget", "output_price": 0.06},
        },
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "env_key": "GLM_API_KEY",
        "models": {
            "GLM-5": {"tier": "flagship", "output_price": 3.20},
            "GLM-4.7": {"tier": "mid", "output_price": 2.20, "max_temp": 0.7},
            "glm-4.5-flash": {"tier": "budget", "output_price": 0.00},
        },
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
        "models": {
            "deepseek-chat": {"tier": "mid", "output_price": 0.42},
        },
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "env_key": "MISTRAL_API_KEY",
        "models": {
            "mistral-large-2512": {"tier": "mid", "output_price": 1.50},
            "mistral-medium-3.1": {"tier": "mid", "output_price": 2.00},
            "mistral-small-3.2-24b": {"tier": "budget", "output_price": 0.18},
            "mistral-nemo": {"tier": "budget", "output_price": 0.04},
            "devstral-small-2505": {"tier": "budget", "output_price": 0.00},
        },
    },
}

# Default max temperature for models without explicit max_temp
DEFAULT_MAX_TEMP = 2.0

# --- Prompt Styles ---

PROMPT_STYLES = {
    "direct": "Write a {length} essay about: {topic}",

    "academic": (
        "Write a scholarly analysis of the following topic in about {length} words. "
        "Use formal academic tone, cite general research findings, "
        "and maintain an objective perspective throughout. "
        "Topic: {topic}"
    ),

    "casual": (
        "Write about {topic} like you're explaining it to a friend "
        "over coffee. Keep it conversational but informative. "
        "About {length} words."
    ),

    "persona": (
        "You are a {persona}. Write about {topic} from your perspective. "
        "Use natural language that reflects your background and experience. "
        "About {length} words."
    ),

    "anti_detect": (
        "Write about {topic} in about {length} words. Make it indistinguishable "
        "from human writing by following ALL of these rules:\n"
        "- Vary sentence length dramatically: some under 8 words, some over 30\n"
        "- Use unexpected word combinations — avoid common collocations like "
        "'play a role', 'it is important to note', 'in today's world'\n"
        "- Transitions between paragraphs should NOT be smooth — jump between "
        "subtopics the way a real person's mind wanders, leave some connections implicit\n"
        "- Include 2-3 minor imperfections: a repeated word, a slightly awkward "
        "phrasing, a sentence that starts one way and finishes differently\n"
        "- Occasionally use an uncommon or slightly surprising word where a common one would do\n"
        "- Do NOT use any of these AI-typical phrases: 'delve into', 'it's worth noting', "
        "'furthermore', 'in conclusion', 'plays a crucial role', 'it is essential', "
        "'navigating', 'landscape', 'multifaceted', 'holistic'\n"
        "- Include a personal aside, opinion, or tangent that doesn't perfectly fit the structure"
    ),

    "anti_detect_v2": (
        "Write about {topic} in about {length} words, mimicking a real college student's "
        "writing process. Real student writing has these properties:\n"
        "- Starts strong but quality drops toward the end (fatigue pattern)\n"
        "- Some paragraphs are much better developed than others\n"
        "- Uses a mix of simple and complex vocabulary inconsistently\n"
        "- Has 1-2 sentences that are unnecessarily wordy or could be cut\n"
        "- Occasionally restates a point slightly differently without realizing\n"
        "- Transitions are sometimes abrupt ('Anyway,', 'Moving on,', 'Another thing is')\n"
        "- May include a half-formed thought that trails off\n"
        "Do NOT write perfectly polished prose. Write like a real person drafting."
    ),

    "rewrite": (
        "Rewrite the following text in your own words. Keep the same meaning "
        "but change the structure, word choice, and flow completely.\n\n"
        "Original text: {source_text}"
    ),

    # --- New styles to fix coverage gaps ---

    "literature_review": (
        "Write a comprehensive literature review on {topic} in about {length} words. "
        "Cite specific researchers by name (you may invent plausible names and dates), "
        "include quantitative findings (percentages, sample sizes, effect sizes), "
        "and compare methodologies across studies. Use academic citation style."
    ),

    "tech_report": (
        "Write a technical report analyzing {topic} in about {length} words. "
        "Include specific metrics, benchmark comparisons, implementation details, "
        "and concrete recommendations. Use a structured format with clear sections."
    ),

    "news_article": (
        "Write a news article about {topic} in about {length} words. "
        "Use the inverted pyramid structure. Include quotes from experts "
        "(you may invent plausible names and affiliations), specific dates, "
        "and factual details. Write in journalistic style."
    ),

    "business_memo": (
        "Write an internal business memo about {topic} in about {length} words. "
        "Address it to 'Senior Leadership Team'. Include an executive summary, "
        "key findings, cost implications, and recommended next steps. "
        "Use professional but direct language."
    ),

    "how_to_guide": (
        "Write a step-by-step guide on {topic} in about {length} words. "
        "Include numbered steps, practical tips, common mistakes to avoid, "
        "and expected outcomes. Write for a beginner audience."
    ),

    "product_review": (
        "Write a detailed review of {topic} in about {length} words. "
        "Include pros and cons, specific use cases, comparisons to alternatives, "
        "and a clear recommendation. Write as an experienced user, not a marketer."
    ),

    "opinion_editorial": (
        "Write an opinion piece arguing a specific position on {topic} in about "
        "{length} words. Take a strong stance, acknowledge counterarguments, "
        "and use rhetorical techniques. Write with personality and conviction."
    ),

    "qa_explanation": (
        "Answer this question in about {length} words: What is {topic} and why "
        "does it matter? Explain clearly for a general audience. Include concrete "
        "examples, avoid jargon, and anticipate follow-up questions."
    ),

    "creative_nonfiction": (
        "Write a creative nonfiction piece about {topic} in about {length} words. "
        "Open with a vivid scene or anecdote. Weave personal observation with "
        "factual information. Use sensory details and narrative techniques."
    ),

    "email_professional": (
        "Write a professional email about {topic} in about {length} words. "
        "Include a clear subject line, greeting, structured body with bullet points "
        "where appropriate, and a specific call to action. Be concise but thorough."
    ),

    "debate_argument": (
        "Present a structured argument about {topic} in about {length} words. "
        "State your thesis clearly, provide three distinct supporting points with "
        "evidence, address the strongest counterargument, and conclude with a "
        "compelling restatement. Use logical connectives."
    ),

    "abstract_summary": (
        "Write a research abstract about {topic} in about {length} words. "
        "Follow the IMRAD structure: background/motivation (2 sentences), "
        "methods (2 sentences), results with specific numbers, and implications. "
        "Be precise and information-dense."
    ),

    "deep_research": (
        "Write a comprehensive analysis of {topic} in about {length} words. "
        "Structure it as a research briefing: cite specific papers with author names "
        "and years, include exact statistics and metrics, compare multiple approaches, "
        "and synthesize findings into actionable conclusions. Use parenthetical "
        "citations and reference specific datasets or benchmarks."
    ),

    "code_documentation": (
        "Write technical documentation about {topic} in about {length} words. "
        "Include an overview section, key concepts explained with examples, "
        "common patterns and anti-patterns, and troubleshooting tips. "
        "Write for developers with intermediate experience."
    ),
}

PERSONAS = [
    "20-year-old college student majoring in psychology",
    "retired high school teacher with 30 years of experience",
    "immigrant small business owner in their 40s",
    "journalist at a local newspaper",
    "graduate student writing their thesis",
    "working mother balancing career and family",
    "software engineer who blogs on the side",
    "first-generation college student from a rural town",
]

# --- Topics ---

TOPICS = [
    # --- Original 20 (social issues / essays) ---
    "the impact of social media on teenage mental health",
    "whether standardized testing should be abolished",
    "how artificial intelligence is changing the job market",
    "the ethics of gene editing in humans",
    "why college tuition has become unaffordable",
    "the role of government in addressing climate change",
    "how remote work has changed workplace culture",
    "the growing wealth gap and its consequences",
    "whether social media companies should be regulated",
    "the mental health crisis among college students",
    "how technology is reshaping public education",
    "the future of renewable energy adoption",
    "immigration policy and its economic effects",
    "the decline of local journalism and its impact on democracy",
    "whether universal basic income is feasible",
    "the opioid crisis and pharmaceutical accountability",
    "how streaming services have changed the entertainment industry",
    "the pros and cons of electric vehicles",
    "food insecurity in developed nations",
    "the ethics of surveillance technology in public spaces",
    # --- STEM / Technology ---
    "transformer architecture and self-attention mechanisms in deep learning",
    "comparing SQL and NoSQL database performance for web applications",
    "the evolution of container orchestration from Docker to Kubernetes",
    "quantum computing's potential impact on cryptography",
    "CRISPR-Cas9 gene editing techniques and off-target effects",
    "machine learning model evaluation metrics and their tradeoffs",
    "the chemistry of lithium-ion battery degradation",
    "advances in protein structure prediction using AlphaFold",
    "TCP congestion control algorithms and their real-world performance",
    "the physics of semiconductor manufacturing at 3nm nodes",
    # --- Business / Finance ---
    "supply chain disruptions and inventory management strategies",
    "the rise of decentralized finance and its regulatory challenges",
    "customer retention strategies for SaaS companies",
    "venture capital funding trends in the AI startup ecosystem",
    "the economics of subscription-based business models",
    # --- History / Humanities ---
    "the fall of the Roman Republic and lessons for modern democracies",
    "the Harlem Renaissance and its lasting cultural impact",
    "how the printing press transformed European society",
    "the partition of India in 1947 and its long-term consequences",
    "the role of propaganda in World War II",
    # --- Medicine / Health ---
    "antibiotic resistance as a global public health crisis",
    "the gut-brain axis and its role in mental health disorders",
    "mRNA vaccine technology beyond COVID-19 applications",
    "sleep deprivation effects on cognitive performance and decision-making",
    "the ethics of clinical trial design in developing countries",
    # --- Daily Life / Practical ---
    "how to build a personal budgeting system that actually works",
    "the science behind effective study habits and memory retention",
    "meal prep strategies for busy professionals on a tight budget",
    "choosing between renting and buying a home in 2026",
    "how to evaluate the credibility of online health information",
    # --- Arts / Culture ---
    "the influence of Japanese animation on Western storytelling",
    "how hip-hop evolved from protest music to mainstream culture",
    "the impact of AI-generated art on professional illustrators",
    "why independent bookstores are making a comeback",
    "the relationship between architecture and community well-being",
    # --- Law / Policy ---
    "data privacy regulations across different jurisdictions",
    "the legal implications of autonomous vehicle accidents",
    "intellectual property challenges in the age of generative AI",
    "criminal justice reform and recidivism reduction programs",
    "the debate over Section 230 and platform liability",
]

# --- Academic Topics (518 topics from MIT/Stanford/UNC course catalogs) ---
# Auto-generated 2026-03-24 from scripts/academic_taxonomy.json
from academic_topics import ACADEMIC_TOPICS
TOPICS = TOPICS + ACADEMIC_TOPICS

# --- Temperatures ---

TEMPERATURES = [0.3, 0.7, 1.0, 1.3]  # Expanded range — higher temps are harder to detect

# --- Lengths ---

LENGTHS = {
    "short": {"words": "300-400", "target_tokens": 500},
    "medium": {"words": "500-800", "target_tokens": 900},
    "long": {"words": "1000-1500", "target_tokens": 1700},
}

# --- Labels ---

LABELS = {
    0: "human",
    1: "ai",
    2: "ai_polished",
    3: "human_polished",
    4: "ai_humanized",
}
