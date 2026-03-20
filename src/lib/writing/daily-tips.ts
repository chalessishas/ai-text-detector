import type { DailyTip, Trait } from "./types";

export const staticTips: DailyTip[] = [
  // ── Ideas (5) ──────────────────────────────────────────────
  {
    id: "static-0",
    trait: "ideas",
    tip: "Before writing, list 3 specific examples that support your argument. If you can't find 3, your thesis might be too broad.",
    exercisePrompt:
      "Pick any opinion you hold. Set a timer for 3 minutes and list as many concrete examples as you can. Circle the 3 strongest.",
  },
  {
    id: "static-1",
    trait: "ideas",
    tip: "Turn abstract claims into 'show, don't tell' moments. Replace 'Pollution is harmful' with a specific scene: who is harmed, where, and how?",
    example: {
      before: "Pollution is harmful to communities.",
      after:
        "In Flint, Michigan, families boiled tap water for months because lead had leached into the city's aging pipes.",
    },
  },
  {
    id: "static-2",
    trait: "ideas",
    tip: "Ask 'So what?' after every paragraph. If you can't answer it, the paragraph isn't pulling its weight.",
    exercisePrompt:
      "Take a paragraph from your latest draft. Write the 'So what?' answer below it. If the answer repeats the paragraph, cut or rewrite.",
  },
  {
    id: "static-3",
    trait: "ideas",
    tip: "Strengthen your thesis by including a counter-argument. Acknowledging the opposition and refuting it shows depth of thought.",
    example: {
      before: "Social media is bad for teenagers.",
      after:
        "While social media gives teenagers a platform for creative expression, the algorithmic push toward comparison content measurably increases anxiety in 13-to-17-year-olds.",
    },
  },
  {
    id: "static-4",
    trait: "ideas",
    tip: "Replace second-hand generalizations with first-hand evidence. One personal observation is more persuasive than three borrowed statistics.",
    exercisePrompt:
      "Rewrite your weakest paragraph using only things you have personally seen, heard, or experienced. No Google allowed.",
  },

  // ── Organization (5) ───────────────────────────────────────
  {
    id: "static-5",
    trait: "organization",
    tip: "Write your conclusion first. It forces you to know your destination before you start the journey.",
    exercisePrompt:
      "Draft the final paragraph of your next essay before anything else. Then outline the steps a reader needs to arrive there.",
  },
  {
    id: "static-6",
    trait: "organization",
    tip: "Each paragraph should answer exactly one question. If it answers two, split it. If it answers none, cut it.",
    example: {
      before:
        "Dogs are loyal. They also need regular exercise. Many breeds are prone to hip problems. Owning a dog teaches responsibility.",
      after:
        "Paragraph 1: Why dogs build loyalty. Paragraph 2: Exercise needs by breed. Paragraph 3: What dog ownership teaches children.",
    },
  },
  {
    id: "static-7",
    trait: "organization",
    tip: "Use the 'last sentence → first sentence' test: does the last sentence of each paragraph connect logically to the first sentence of the next?",
    exercisePrompt:
      "Copy only the last and first sentences of consecutive paragraphs and read them as pairs. Fix any pair that feels like a jump cut.",
  },
  {
    id: "static-8",
    trait: "organization",
    tip: "Put your strongest argument second-to-last, not first. Readers remember endings. Open with something interesting, build, then land your best punch before the conclusion.",
  },
  {
    id: "static-9",
    trait: "organization",
    tip: "If you're stuck outlining, try the 'sticky note' method: write one idea per note, spread them on a table, and physically rearrange until the order feels inevitable.",
    exercisePrompt:
      "Take 5 index cards. Write one key point per card. Shuffle them and try 3 different orderings. Pick the one that tells the clearest story.",
  },

  // ── Voice (5) ──────────────────────────────────────────────
  {
    id: "static-10",
    trait: "voice",
    tip: "Delete every 'I think that...' and 'I believe that...' and state your opinion directly. Hedging weakens your voice.",
    example: {
      before: "I think that school uniforms limit self-expression.",
      after: "School uniforms strip students of a basic form of self-expression.",
    },
  },
  {
    id: "static-11",
    trait: "voice",
    tip: "Read your essay aloud in the voice of someone giving a TED talk. If any sentence would make the audience zone out, rewrite it with energy.",
    exercisePrompt:
      "Stand up, read your introduction aloud as if presenting to 200 people. Mark every line where your energy drops.",
  },
  {
    id: "static-12",
    trait: "voice",
    tip: "Inject one honest, unexpected opinion into an otherwise standard essay. A single moment of candor can make the whole piece feel alive.",
    example: {
      before: "Volunteering is a rewarding experience.",
      after:
        "Most of my volunteering was resume padding — until the afternoon I watched a seven-year-old read her own name for the first time.",
    },
  },
  {
    id: "static-13",
    trait: "voice",
    tip: "Match your sentence rhythm to your meaning. Short sentences create urgency. Longer, more flowing sentences suggest reflection or complexity.",
    example: {
      before:
        "The fire spread quickly. It consumed the building in a matter of minutes, which was surprising to everyone watching.",
      after:
        "The fire spread. In four minutes the entire east wing was gone.",
    },
  },
  {
    id: "static-14",
    trait: "voice",
    tip: "Cut filler phrases that add no meaning: 'It is important to note that', 'In today's society', 'At the end of the day'. They make you sound like everyone else.",
    exercisePrompt:
      "Search your draft for 'It is', 'There are', and 'In today's'. Delete or rewrite every instance you find.",
  },

  // ── Word Choice (5) ────────────────────────────────────────
  {
    id: "static-15",
    trait: "wordChoice",
    tip: "Replace 'very' + adjective with a single precise word. 'Very tired' becomes 'exhausted'. 'Very happy' becomes 'elated'. 'Very scared' becomes 'terrified'.",
    example: {
      before: "She was very sad about the very bad news.",
      after: "She was devastated by the grim news.",
    },
  },
  {
    id: "static-16",
    trait: "wordChoice",
    tip: "Ban the word 'things' from your writing. Every time you type it, force yourself to name the actual object, concept, or detail.",
    example: {
      before: "There are many things that contribute to climate change.",
      after:
        "Deforestation, methane emissions, and fossil-fuel combustion each accelerate climate change.",
    },
  },
  {
    id: "static-17",
    trait: "wordChoice",
    tip: "Use concrete nouns instead of abstract ones. 'Transportation' is forgettable; 'a rusted 1998 Honda Civic' is vivid.",
    example: {
      before: "He had an old vehicle.",
      after:
        "He drove a dented Civic with a cracked windshield and 230,000 miles on it.",
    },
  },
  {
    id: "static-18",
    trait: "wordChoice",
    tip: "Prefer active verbs over 'to be' + adjective. 'The sunset was beautiful' tells; 'The sunset bled orange across the lake' shows.",
    example: {
      before: "The city was busy and the streets were noisy.",
      after:
        "Taxi horns punched through the chatter of sidewalk vendors and the hiss of bus brakes.",
    },
  },
  {
    id: "static-19",
    trait: "wordChoice",
    tip: "When you catch yourself using a cliche ('tip of the iceberg', 'at the end of the day'), ask: can I say this in a way no one has said before?",
    exercisePrompt:
      "Write down 5 cliches you use often. For each one, invent a fresh image that conveys the same meaning.",
  },

  // ── Fluency (5) ────────────────────────────────────────────
  {
    id: "static-20",
    trait: "fluency",
    tip: "Read your paragraph aloud. If you run out of breath, the sentence is too long. If it sounds choppy, combine short sentences with a conjunction or semicolon.",
    exercisePrompt:
      "Read your latest paragraph aloud without pausing except at punctuation. Mark every place you gasped or stumbled.",
  },
  {
    id: "static-21",
    trait: "fluency",
    tip: "Vary your sentence openers. If three sentences in a row start with 'The', rewrite at least one to begin with a verb, prepositional phrase, or subordinate clause.",
    example: {
      before:
        "The students gathered in the hall. The principal gave a speech. The ceremony lasted two hours.",
      after:
        "Students packed the hall shoulder to shoulder. From the podium, the principal spoke about resilience. Two hours later, the last name was called.",
    },
  },
  {
    id: "static-22",
    trait: "fluency",
    tip: "Use a short sentence after a long one for emphasis. The contrast creates a natural rhythm that keeps readers engaged.",
    example: {
      before:
        "The project required months of planning, coordination across departments, and a complete redesign of the workflow. It was a significant undertaking that demanded considerable resources.",
      after:
        "The project required months of planning, coordination across departments, and a complete redesign of the workflow. It nearly broke us.",
    },
  },
  {
    id: "static-23",
    trait: "fluency",
    tip: "Transition words are bridges, not crutches. Use 'however', 'therefore', and 'moreover' only when the logical relationship isn't already clear from context.",
    exercisePrompt:
      "Highlight every transition word in your draft. Delete half of them and re-read. If the logic still flows, leave them out.",
  },
  {
    id: "static-24",
    trait: "fluency",
    tip: "Break the 'subject-verb-object' pattern occasionally. Fronting a phrase ('Exhausted and sunburned, we reached camp') adds variety without sacrificing clarity.",
    example: {
      before: "We reached camp. We were exhausted and sunburned.",
      after: "Exhausted and sunburned, we reached camp at dusk.",
    },
  },

  // ── Conventions (5) ────────────────────────────────────────
  {
    id: "static-25",
    trait: "conventions",
    tip: "After finishing your draft, read it backwards sentence by sentence. This breaks the flow and helps you catch grammar and spelling errors your brain auto-corrects when reading forward.",
    exercisePrompt:
      "Take your latest draft and read it from the last sentence to the first. Write down every error you find.",
  },
  {
    id: "static-26",
    trait: "conventions",
    tip: "Use a comma before a coordinating conjunction (and, but, or, so) only when it joins two independent clauses. 'I went to the store and bought milk' needs no comma; 'I went to the store, and she stayed home' does.",
    example: {
      before: "I went to the store, and bought milk.",
      after: "I went to the store and bought milk.",
    },
  },
  {
    id: "static-27",
    trait: "conventions",
    tip: "Semicolons connect two related independent clauses without a conjunction. If you can replace the semicolon with a period and both sentences work, you used it correctly.",
    example: {
      before:
        "She loved the city, however she missed the quiet of the countryside.",
      after:
        "She loved the city; however, she missed the quiet of the countryside.",
    },
  },
  {
    id: "static-28",
    trait: "conventions",
    tip: "Its vs. it's: 'it's' always means 'it is' or 'it has'. If you can't expand it to 'it is', use 'its'. This one mistake appears in 40% of student essays.",
    example: {
      before: "The company lost it's competitive edge.",
      after: "The company lost its competitive edge.",
    },
  },
  {
    id: "static-29",
    trait: "conventions",
    tip: "Keep verb tenses consistent within a paragraph. If you start in past tense, stay in past tense unless you have a reason to shift.",
    example: {
      before:
        "She walked into the room and sees the broken vase. She picked up the pieces.",
      after:
        "She walked into the room and saw the broken vase. She picked up the pieces.",
    },
  },

  // ── Presentation (5) ───────────────────────────────────────
  {
    id: "static-30",
    trait: "presentation",
    tip: "If a paragraph exceeds 6 sentences, it probably contains two ideas. Split it. White space is your reader's breathing room.",
    exercisePrompt:
      "Count the sentences in every paragraph of your draft. Split any paragraph with more than 6 sentences into two focused paragraphs.",
  },
  {
    id: "static-31",
    trait: "presentation",
    tip: "Use headings and subheadings in longer pieces. They act as signposts — a reader scanning your essay should understand your argument from headings alone.",
    exercisePrompt:
      "Write a subheading for each section of your essay. Read only the subheadings in order. Do they tell a coherent story?",
  },
  {
    id: "static-32",
    trait: "presentation",
    tip: "Indent or block-quote any direct quotation longer than 3 lines. Wall-of-text quotations buried in a paragraph make readers skip them entirely.",
  },
  {
    id: "static-33",
    trait: "presentation",
    tip: "Front-load your paragraphs: put the main point in the first sentence, then support it. Readers who skim will still absorb your argument.",
    example: {
      before:
        "Many studies have been conducted over the years, and researchers have found varying results, but the consensus is that sleep deprivation impairs memory.",
      after:
        "Sleep deprivation impairs memory. Decades of research confirm that even one night of poor sleep reduces recall accuracy by up to 40%.",
    },
  },
  {
    id: "static-34",
    trait: "presentation",
    tip: "End your essay with a single, punchy sentence — not a summary paragraph that restates everything. Trust your reader to remember what you wrote.",
    example: {
      before:
        "In conclusion, as this essay has shown, there are many reasons why public libraries remain important to communities everywhere.",
      after: "Close a library, and you close a door that some people have no other way to open.",
    },
  },
];

const TRAIT_ORDER: Trait[] = [
  "conventions",
  "fluency",
  "ideas",
  "organization",
  "presentation",
  "voice",
  "wordChoice",
];

function dateHash(date: Date): number {
  return (
    date.getFullYear() * 10000 +
    (date.getMonth() + 1) * 100 +
    date.getDate()
  );
}

export function selectStaticTip(
  traitScores: Record<Trait, number> | null,
  date: Date,
): DailyTip {
  const hash = dateHash(date);

  if (!traitScores) {
    return staticTips[hash % staticTips.length];
  }

  // Find lowest-scoring trait, ties broken alphabetically
  let lowestTrait: Trait = TRAIT_ORDER[0];
  let lowestScore = Infinity;
  for (const trait of TRAIT_ORDER) {
    const score = traitScores[trait] ?? Infinity;
    if (score < lowestScore) {
      lowestScore = score;
      lowestTrait = trait;
    }
  }

  const filtered = staticTips.filter((t) => t.trait === lowestTrait);
  return filtered[hash % filtered.length];
}
