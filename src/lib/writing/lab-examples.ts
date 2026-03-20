import type { LabExample } from "./types";

export const labExamples: LabExample[] = [
  // ── Voice (4) ──────────────────────────────────────────────
  {
    id: "lab-0",
    topic: "Social Media & Mental Health",
    focusTrait: "voice",
    coldText:
      "Social media platforms have been shown to affect the mental health of young people. Studies indicate that excessive use correlates with increased rates of anxiety and depression among teenagers. Many experts recommend limiting screen time and encouraging face-to-face interactions. Parents and educators should be aware of these risks and take appropriate measures to protect young users from potential harm.",
    humanWarmText:
      "My sister stopped making eye contact at dinner last March. She was fourteen, glued to her phone under the table, scrolling through highlight reels of people she barely knew. Her therapist later told my parents that she'd been comparing herself to influencers for months and had started skipping meals. I don't think social media is evil — she uses it to talk to her best friend who moved to Portland — but the algorithm doesn't care that she's fourteen. It just keeps feeding her content that makes her feel less than. That's not a design flaw. That's the design.",
    humanExplanation:
      "Opens with a specific, observed moment ('stopped making eye contact'). Uses a named relationship and timeline. States a strong opinion in the final line. The personal stake makes the argument credible without citing a single study.",
    teachingPoint:
      "A personal story with precise details can carry more persuasive weight than a stack of citations — because readers feel it before they evaluate it.",
  },
  {
    id: "lab-1",
    topic: "Remote Work",
    focusTrait: "voice",
    coldText:
      "Remote work has become increasingly common since the pandemic. Many employees report higher productivity when working from home due to fewer distractions and eliminated commute times. However, some challenges exist, including difficulty separating work and personal life, reduced collaboration, and feelings of isolation. Companies must develop policies that balance flexibility with the need for team cohesion.",
    humanWarmText:
      "I wrote this paragraph in my kitchen at 6 AM, still in the T-shirt I slept in, coffee going cold because I forgot about it twenty minutes ago. That's remote work. I'm more productive than I ever was in our open-plan office — nobody taps my shoulder to ask about the Jenkins build — but I also haven't spoken a word out loud since yesterday afternoon. Last Tuesday I realized I'd gone three days without leaving my apartment. I put on real shoes, walked to the bodega on Myrtle Avenue, and bought a banana I didn't need, just to hear someone say 'have a nice day.' Remote work solved my commute problem and gave me a loneliness problem. I'm not sure that's a trade I'd undo, but I wish someone had warned me.",
    humanExplanation:
      "Written in present-tense first person with mundane, hyper-specific details (cold coffee, Jenkins build, bodega on Myrtle Avenue). Admits ambivalence rather than arguing one side. The banana anecdote is the kind of detail AI doesn't generate because it's too small and too real.",
    teachingPoint:
      "Authentic voice comes from admitting complexity. Real people rarely have clean opinions — writing that reflects genuine ambivalence feels more human than a tidy argument.",
  },
  {
    id: "lab-2",
    topic: "Street Food",
    focusTrait: "voice",
    coldText:
      "Street food is an important part of culinary culture in many countries around the world. Vendors offer affordable meals that reflect local ingredients and traditions. Street food provides economic opportunities for small business owners and contributes to the vibrancy of urban areas. Health and safety regulations vary by region, and some consumers express concerns about hygiene standards.",
    humanWarmText:
      "The best meal I had last year cost four dollars. It was a lamb shawarma from a cart on West 34th Street — the one wedged between the pretzel guy and the handbag table. The man running it never told me his name, but he remembered that I like extra white sauce and no tomato. The lamb had been turning on that spit since before sunrise, and you could smell the garlic and cumin from half a block away, cutting right through the diesel and the wet garbage smell that is just Manhattan in August. People in suits lined up next to bike messengers. Nobody looked at their phones. We were all just standing on a sidewalk, eating with our hands, and for three minutes, the city felt like a small town. I'll take that over a Michelin star any day.",
    humanExplanation:
      "Rich sensory language (smell of garlic and cumin, diesel, wet garbage). Specific location and price. The vendor remembering the order creates a relationship. 'Nobody looked at their phones' captures a shared human moment. Ends with a strong, confident opinion.",
    teachingPoint:
      "Sensory details and specificity are the fastest way to create voice. Name the street, state the price, describe the smell — readers trust writers who notice things.",
  },
  {
    id: "lab-3",
    topic: "Music Education",
    focusTrait: "voice",
    coldText:
      "Music education provides numerous benefits for students. Research shows that learning an instrument improves cognitive development, including memory, attention, and mathematical reasoning. Participation in school music programs also promotes teamwork and discipline. Despite these benefits, many school districts have reduced or eliminated music programs due to budget constraints, which has been a subject of ongoing debate among educators and policymakers.",
    humanWarmText:
      "I was a terrible trumpet player. Three years of lessons and I never made it past third chair. But Mr. Patterson kept me in the band because — his words — 'the section needs your lungs, even if your fingers are confused.' That man taught me more about showing up than any honors class ever did. When they cut the music program my junior year to pay for new Chromebooks, we lost the one room in the school where a kid who couldn't sit still was told that his energy was useful. I don't have studies to cite. I just know that my nephew, who has ADHD and hates every class, comes home from drumline practice grinning. If you need a study to tell you that matters, we're measuring the wrong things.",
    humanExplanation:
      "Uses self-deprecation ('terrible trumpet player') to establish authenticity. Mr. Patterson's direct quote adds a real character. The connection to the nephew grounds the argument in continuing personal experience. The final line is a strong, direct challenge to the reader.",
    teachingPoint:
      "Naming real people (even by role: 'Mr. Patterson', 'my nephew') makes writing feel lived-in. Characters turn arguments into stories, and stories are harder to dismiss.",
  },

  // ── Ideas (2) ──────────────────────────────────────────────
  {
    id: "lab-4",
    topic: "Public Libraries",
    focusTrait: "ideas",
    coldText:
      "Public libraries serve many important functions in communities. They provide free access to books, computers, and internet services. Libraries also offer programs for children, adults, and seniors. In recent years, some have questioned whether libraries remain relevant in the digital age, but supporters argue that they continue to serve vital community needs, especially for underserved populations.",
    humanWarmText:
      "The branch library on Prospect Avenue doesn't look like much — beige carpet, fluorescent lights, a water fountain that's been broken since Obama's first term. But on any given Tuesday afternoon you'll find a Haitian grandmother learning to file taxes online, two teenagers doing homework because their apartment has no desk, and a man in a worn coat sitting in the warmest chair he can find because the shelter kicks everyone out at 8 AM. None of them need a book. All of them need the building. That's what people miss when they ask why we still fund libraries in the age of Google: a library is the last public space in America where you are allowed to exist without buying something.",
    humanExplanation:
      "Replaces the abstract claim 'libraries serve community needs' with three specific people doing three specific things. The broken water fountain signals firsthand observation. The closing line reframes the entire debate with a concrete insight ('exist without buying something') that is specific enough to be memorable.",
    teachingPoint:
      "Strong ideas aren't abstract — they're specific observations elevated to insight. Start with what you see, then tell the reader what it means.",
  },
  {
    id: "lab-5",
    topic: "Climate Change",
    focusTrait: "ideas",
    coldText:
      "Climate change is one of the most pressing issues facing the world today. Rising global temperatures are causing melting ice caps, rising sea levels, and more frequent extreme weather events. Scientists agree that human activity, particularly the burning of fossil fuels, is the primary driver. Addressing climate change requires international cooperation and significant changes to energy production, transportation, and industrial practices.",
    humanWarmText:
      "My grandfather's farm in central Nebraska used to get its first frost in mid-September. Now it comes in October, sometimes not until Halloween. He didn't need a climate report to know something had shifted — his soybeans told him. The growing season is longer, which sounds like good news until you realize the corn beetles also get an extra month, and the aquifer he irrigates from dropped fourteen feet in the last decade. Climate change isn't an ice-cap story for him. It's a well-pump story. It's the reason he sold forty acres last spring to cover the cost of drilling deeper. When we talk about 'international cooperation and significant changes,' what we mean is: my grandfather needs water, and the Ogallala Aquifer is not refilling. Scale that to ten thousand farms across six states and you stop needing abstractions.",
    humanExplanation:
      "Translates a global issue into one family's experience. Uses specific data points (fourteen feet, forty acres) embedded in narrative rather than listed as statistics. The pivot from 'ice-cap story' to 'well-pump story' reframes the issue in a way the reader hasn't heard before.",
    teachingPoint:
      "The best ideas connect the universal to the specific. Find one person, one place, one concrete change — and let that carry the weight of the larger argument.",
  },

  // ── Word Choice (2) ────────────────────────────────────────
  {
    id: "lab-6",
    topic: "Urban Parks",
    focusTrait: "wordChoice",
    coldText:
      "Urban parks provide important benefits to city residents. They offer green spaces where people can exercise, relax, and socialize. Parks improve air quality, reduce urban heat island effects, and increase property values in surrounding neighborhoods. City planners recognize that well-maintained parks contribute to the overall quality of life and should be prioritized in urban development strategies.",
    humanWarmText:
      "On the first warm Saturday in April, Prospect Park turns feral. Drum circles thump near the bandshell. Somebody's uncle is grilling jerk chicken on a portable Weber, and the smoke drifts across the soccer fields where teenagers play in mismatched cleats. Toddlers eat grass. Dogs ignore their owners. A woman in a folding chair is reading a romance novel with a shirtless cowboy on the cover and she does not care who sees it. This is what a thirty-five-million-dollar annual parks budget actually buys: permission to be a body in the sun, surrounded by other bodies in the sun, doing absolutely nothing productive. Every city spreadsheet calls this 'quality of life.' I call it the only reason I haven't moved to Vermont.",
    humanExplanation:
      "Every noun is specific: 'jerk chicken on a portable Weber', 'mismatched cleats', 'romance novel with a shirtless cowboy'. Active verbs drive the scene: 'thump', 'drifts', 'eat'. The word 'feral' in the opening sentence does more work than the entire cold paragraph.",
    teachingPoint:
      "Precise nouns and active verbs create vivid writing. One well-chosen word ('feral') can replace an entire paragraph of explanation.",
  },
  {
    id: "lab-7",
    topic: "Childhood Education",
    focusTrait: "wordChoice",
    coldText:
      "Early childhood education plays a critical role in child development. Research indicates that children who attend quality preschool programs demonstrate better academic performance, social skills, and emotional regulation. Investment in early education yields significant returns, as studies show that every dollar spent on preschool generates several dollars in long-term economic benefits through reduced crime rates and increased earning potential.",
    humanWarmText:
      "Watch a four-year-old in a good preschool classroom and you'll see learning that doesn't look like learning. She's at the water table, pouring from a tall skinny cup into a short fat one, frowning because the water won't fit even though the short cup looks bigger. She just stumbled into conservation of volume — Piaget's bread and butter — and nobody told her to. Across the room, two boys are negotiating who gets the red fire truck. One offers a trade: 'You can have it after lunch if I get the blocks now.' That's contract law at age four, hammered out in socks on a carpet square. The returns-on-investment crowd likes to talk about reduced crime rates and earning potential, and they're not wrong. But they're measuring the shadow, not the thing. The thing is a kid discovering that the world has rules she can figure out herself.",
    humanExplanation:
      "Replaces generic terms ('academic performance', 'social skills') with observed moments: a girl at a water table, boys trading toys. 'Conservation of volume' and 'contract law' reframe mundane moments in surprising language. 'Measuring the shadow, not the thing' is a fresh metaphor that replaces the cliche of ROI framing.",
    teachingPoint:
      "Strong word choice means replacing categories with instances. Don't say 'social skills' — show two four-year-olds negotiating over a fire truck.",
  },

  // ── Fluency (2) ────────────────────────────────────────────
  {
    id: "lab-8",
    topic: "Space Exploration",
    focusTrait: "fluency",
    coldText:
      "Space exploration has led to many important scientific discoveries. Missions to the moon and Mars have expanded our understanding of the solar system. Space technology has also produced practical benefits, including satellite communications, GPS, and advances in materials science. However, space programs require enormous financial investment, leading some to question whether the resources could be better spent addressing problems on Earth.",
    humanWarmText:
      "In 1977, NASA bolted a golden record to the side of Voyager 1 and launched it toward interstellar space. The record carried greetings in 55 languages, the sound of a mother kissing her newborn, and Chuck Berry's 'Johnny B. Goode.' Forty-seven years later, that spacecraft is still transmitting — fourteen billion miles from Earth, powered by a plutonium battery the size of a coffee can, sending signals so faint that the receiving dish in Canberra has to filter out the noise of its own electronics to hear them. That's space exploration at its most honest: a whisper across an unimaginable distance, carried by hardware older than most of the engineers monitoring it. Meanwhile, the satellites that Voyager's generation made possible now guide your Uber to within three feet of the curb. We're living in the practical aftermath of audacious, impractical dreams. The money question — could we spend those billions on schools, on hospitals, on clean water — is fair. But it assumes we know what practical means. We didn't launch Voyager to improve GPS. We launched it because we wanted to know what was out there. GPS was an accident. Most of the best things are.",
    humanExplanation:
      "Sentence lengths vary dramatically: the Voyager facts build in long, layered sentences, then 'That's space exploration at its most honest' lands as a shorter pivot. 'Most of the best things are' ends the paragraph with a five-word sentence after several complex ones. The rhythm mirrors the argument — grand ambition punctuated by simple truths.",
    teachingPoint:
      "Fluency isn't about making every sentence smooth — it's about controlling rhythm. Long sentences build momentum; short sentences stop the reader and make them think. Alternate deliberately.",
  },
  {
    id: "lab-9",
    topic: "Immigration",
    focusTrait: "fluency",
    coldText:
      "Immigration is a complex and multifaceted issue. Immigrants contribute to the economy through labor, entrepreneurship, and cultural diversity. However, immigration also raises concerns about job competition, public services, and national security. Effective immigration policy must balance economic needs with social considerations, ensuring that both immigrants and native-born citizens benefit from a fair and well-managed system.",
    humanWarmText:
      "My mother came to this country with two suitcases and an English vocabulary of maybe two hundred words. She had a degree in chemical engineering from a university in Taipei that nobody here had heard of. So she cleaned houses. She cleaned houses for three years while studying for her American certification exams at the kitchen table after I went to bed. She passed on the first try. Within a decade she was managing a water treatment facility in northern New Jersey, and the county gave her an award for reducing chemical runoff into the Passaic River by thirty percent. That's one immigration story. It's mine, so I tell it. But I'm not naive enough to think it's everyone's. The man who does drywall in my building works twelve-hour days and sends half his paycheck to Guatemala. He's been here eleven years and his English is still rough. His story doesn't end with an award. It's a story about endurance, not triumph, and those stories matter too — maybe more, because there are more of them. Immigration policy that only values the success stories is policy built on a flattering lie.",
    humanExplanation:
      "The opening sentences are short and declarative, building a timeline. Mid-paragraph, sentences lengthen as the reflection deepens. The transition ('That's one immigration story') is a single short sentence that pivots the entire piece. The final sentence is long and complex, matching the complexity of the idea it carries. No transition words are needed — the rhythm itself signals the shifts.",
    teachingPoint:
      "When your sentence structure matches your meaning, you don't need transition words to hold the paragraph together. The rhythm does the work. Short sentences for facts; longer sentences for nuance.",
  },
];
