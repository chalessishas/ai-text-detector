"""False Positive Rate (FPR) test suite — 50+ diverse human texts.

Tests that authentic human writing is NOT falsely flagged as AI.
This is the most critical metric for a production detector.

Run: python3.11 -m pytest tests/test_fpr.py -v
Requires: detection server on port 5001
"""
import json
import urllib.request
import pytest

SERVER = "http://127.0.0.1:5001/analyze"


def get_prediction(text: str) -> tuple:
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request(SERVER, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    fused = result.get("fused", {})
    return fused.get("ai_score", 50), fused.get("prediction", "unknown")


# ═══════════════════════════════════════════════════════════════
# HUMAN TEXT SAMPLES — diverse genres, styles, domains
# Every single one must NOT be classified as "ai"
# ═══════════════════════════════════════════════════════════════

HUMAN_SAMPLES = {
    # --- Casual / Social Media ---
    "casual_baking": "I honestly had no idea what I was doing when I first tried to bake bread. The dough stuck to everything, my kitchen looked like a flour bomb went off, and the end result was basically a brick. But you know what? I kept trying, and eventually something clicked. Now I make a decent loaf every Sunday.",
    "casual_dentist": "Just got back from the dentist and honestly it wasnt as bad as I thought. They said I need a crown on one of my molars which sucks but at least no root canal. The hygienist was super nice and even gave me extra numbing when I asked. Pro tip: always ask for more novocaine.",
    "casual_dog": "My dog ate an entire bag of chocolate chips last night and I freaked out. Called the emergency vet at like 2am, they said to watch for vomiting. He threw up twice and then went back to sleep like nothing happened. $200 vet bill for him to literally sleep it off.",
    "casual_neighbor": "So my neighbor has been playing drums at midnight for the past three weeks. I've asked nicely, left notes, even brought cookies. Nothing works. Last night I snapped and started playing the tuba at 6am. He hasn't played since. Problem solved I guess.",
    "casual_pizza": "Burned my tongue on hot pizza for the third time this week because I have zero patience. The roof of my mouth is basically sandpaper at this point. My wife says just wait five minutes but who has that kind of self control when theres fresh pizza sitting right there. Nobody thats who.",
    "casual_cat": "My cat has decided that 4am is the perfect time to practice his opera singing. Every. Single. Morning. I've tried everything — feeding him before bed, playing with him until he's tired, even those calming treats. Nothing works. He just really loves 4am apparently.",
    "casual_grocery": "Anyone else feel like grocery prices are completely out of control? I spent 87 dollars today and came home with like two bags. Eggs are 6 bucks. SIX. I remember when they were like a dollar fifty. What is happening.",
    "casual_wallet": "Lost my wallet at target yesterday and some random dude found it and turned it in with all my cash still inside. Theres like 200 bucks in there too. Faith in humanity restored honestly. Whoever you are, thank you.",

    # --- Personal Essay / Memoir ---
    "essay_sixteen": "The summer I turned sixteen, my dad lost his job and everything changed. We went from eating out twice a week to counting pennies for groceries. I had to drop out of soccer because we couldnt afford the cleats. Looking back, it taught me more about money than any class ever could.",
    "essay_smalltown": "I grew up in a small town in Iowa where everyone knew everyone. The kind of place where you couldn't speed because someone would call your mom before you got home. I moved to Chicago at 18 and the anonymity was both terrifying and freeing.",
    "essay_failure": "The first time I failed was in seventh grade when I bombed my science fair project. Everyone else had these elaborate displays with blinking lights and I showed up with a poster board about mold growing on bread. The judges didnt even stop at my table. I cried in the bathroom for twenty minutes.",

    # --- Professional / Business ---
    "business_q3": "Q3 revenue came in at 14.2M, up 8% YoY but below our 15M target. The miss was primarily driven by delayed enterprise deals in EMEA — two contracts worth a combined 1.8M slipped to Q4. Pipeline for Q4 looks strong with 23M in qualified opportunities.",
    "business_merger": "The merger between the two companies was finalized on March 15th after eighteen months of negotiations. Regulatory concerns delayed the process by approximately six months, primarily due to antitrust considerations in the European market.",
    "business_audit": "The audit identified 47 instances of non-compliance with SOC 2 Type II controls, primarily concentrated in access management and change control processes. Of these, 12 were classified as high severity requiring immediate remediation.",
    "business_freelance": "Honestly the hardest part of being a freelance designer isn't the work itself, it's chasing invoices. I have one client who is currently 90 days past due on a $4,500 invoice and just keeps saying we'll get to it. Meanwhile I still have rent to pay.",

    # --- Technical / Dev ---
    "tech_api": "After three iterations we finally got the API latency down to under 50ms p99. The bottleneck was the N+1 query in the user permissions check — switched it to a single JOIN with a materialized view and it dropped from 200ms to 12ms. Deployed to staging, monitoring overnight.",
    "tech_docker": "yo did anyone elses docker build just completely die? im getting some weird segfault in the node_modules step that wasnt there yesterday. already tried rm -rf node_modules and a fresh install, same thing. mike said he saw something similar last week but his fix was to just restart his mac which feels wrong lol.",
    "tech_pipeline": "We need to talk about the deployment pipeline. Three times this month we've had hotfixes that bypassed staging entirely because it was urgent. I get it, stuff breaks, but we can't keep doing cowboy deploys to prod. I'm proposing a 30-minute fast-track staging gate.",
    "tech_bios": "UPDATE: Fixed it!! For anyone having the same problem — turns out the issue was my BIOS was set to UEFI-only mode but the drive was formatted as MBR. Once I switched to Legacy+UEFI boot mode it found the drive immediately. Took me three days and about 47 forum posts to figure this out.",

    # --- Academic / Scientific ---
    "academic_sleep": "While the study had several limitations including a small sample size and the lack of a control group, the findings suggest a correlation between sleep duration and cognitive performance in elderly adults. More research is needed, particularly longitudinal studies that track participants over multiple years.",
    "academic_medical": "The patient presented with acute onset of chest pain radiating to the left arm, accompanied by diaphoresis and nausea. Initial troponin levels were elevated at 0.45 ng/mL. An ECG revealed ST-segment elevation in leads II, III, and aVF consistent with an inferior STEMI.",
    "academic_spectrum": "The absorption spectrum showed a characteristic peak at 420nm consistent with the formation of the gold nanoparticle conjugate. TEM imaging confirmed an average particle diameter of 15.3 plus or minus 2.1nm with a narrow size distribution.",
    "academic_data": "The dataset contains approximately 2.3 million records spanning fiscal years 2018 through 2023. After removing duplicates and entries with missing key fields, the clean dataset comprises 1.87 million unique records suitable for analysis.",

    # --- Creative Writing ---
    "creative_rain": "The rain came down in sheets, turning the parking lot into a shallow lake. Sarah sat in her car, watching the windshield wipers fight a losing battle. She was twenty minutes early for the interview, which meant she had twenty minutes to talk herself out of going in.",
    "creative_letter": "He found the letter in a shoebox under his mother's bed, three weeks after the funeral. The handwriting was shaky, the ink faded to a pale blue, but he recognized it immediately. It was from his father. A man who, as far as he'd known his whole life, had died before he was born.",

    # --- Reviews ---
    "review_restaurant": "Went here for my anniversary and wow. The steak was cooked perfectly medium rare, the mashed potatoes were creamy and had just the right amount of garlic. Service was a bit slow at first but our waiter Carlos made up for it — super attentive once he got to us and even comped our dessert.",
    "review_kindle": "Picked up a used Kindle for 20 bucks at a garage sale and honestly it might be the best purchase ive made all year. Already read three books this month which is more than I read all of last year. Something about not having notifications pop up every 30 seconds makes a huge difference.",

    # --- Informal / Text Message Style ---
    "text_flights": "hey so i was thinking about what u said yesterday and ur totally right we should just go for it. worst case scenario we lose like 200 bucks which whatever thats like two dinners out. ive been looking at flights and if we book by friday theres some decent deals to denver.",
    "text_ISP": "Can someone PLEASE explain to me why my ISP thinks its acceptable to charge me 80 bucks a month for speeds that barely hit 10mbps?? I ran a speed test at 3pm on a TUESDAY and got 8.2 download. Eight. Point. Two. I swear these companies just dont care.",

    # --- Parenting ---
    "parent_sky": "My 4 year old asked me why the sky is blue and I said something about light scattering and he just stared at me and said no daddy its because God painted it and honestly his answer was better than mine.",
    "parent_dino": "My 3 year old just told me she wants to be a dinosaur when she grows up and I dont have the heart to tell her thats not a career option. She also insists that broccoli is tiny trees and honestly shes not wrong.",
    "parent_jellyfish": "Took the kids to the aquarium yesterday and my 5yo spent literally 45 minutes just staring at the jellyfish. Wouldnt move. We missed the dolphin show and she did not care one bit. Then on the way home she announced that when she grows up she wants to be a jellyfish scientist.",

    # --- Legal ---
    "legal_motion": "The defendant's motion to dismiss is hereby denied. The plaintiff has demonstrated sufficient facts to state a claim under Section 1983. The case will proceed to discovery. Counsel for both parties shall submit a proposed scheduling order within 14 days.",

    # --- Wedding / Events ---
    "wedding_toast": "When Jake first told me he was dating someone new, I was like oh here we go again. But then I met Sarah and within five minutes I knew this was different. She laughed at his terrible puns, she didnt try to change his obsession with fantasy football, and most importantly she calls me every Sunday just to chat.",

    # --- Home / DIY ---
    "diy_plumbing": "Tried to fix my own plumbing yesterday. YouTube made it look so easy. Four hours later I'm standing in two inches of water calling a real plumber. The guy charged me double because of the mess I made. Lesson learned.",
    "diy_painting": "Finally finished painting the bedroom and I gotta say it looks amazing. Went with this sage green color my wife picked and I was skeptical at first but she was totally right. Only took 6 hours, two trips to Lowes for more tape, and one near-disaster when I knocked a full can off the ladder.",
    "diy_garden": "Started a garden last spring on a total whim. Bought some tomato plants and basil from Home Depot, stuck them in the ground, and kinda forgot about them for two weeks. When I came back out, the tomatoes were going crazy but something had eaten all my basil. Turns out rabbits love basil. Who knew?",
    "diy_garage": "Finally finished reorganizing my garage after putting it off for like three years. Found my old skateboard from high school, a box of VHS tapes including all three Lord of the Rings extended editions, and somehow four mismatched garden gloves. Not a single pair just four random lefts.",

    # --- Food / Cooking ---
    "cooking_marinade": "Okay so for the marinade youll need about a third cup of soy sauce, two tablespoons of sesame oil, a big chunk of ginger grated, and like four cloves of garlic minced. Mix it all up and dump your chicken thighs in there. Let it sit in the fridge for at least an hour but honestly overnight is way better.",
    "cooking_grandma": "Made my grandmas pasta recipe last night — the one she refuses to write down. I watched her make it once and took secret notes on my phone. Pretty sure she saw me but pretended not to. The sauce came out almost as good as hers. Almost.",
    "cooking_thanksgiving": "Thanksgiving dinner was a disaster this year. Uncle Jim brought politics to the table again, mom burned the turkey because she was on the phone, and the dog ate an entire pie off the counter. We ended up ordering pizza. Best Thanksgiving ever honestly.",

    # --- Frustration / Complaints ---
    "frustration_email": "Hi Mark, This is the third time I am asking about the refund for order #45892. It has been over 6 weeks since I returned the item and I still see nothing in my account. I have the tracking number showing it was delivered back to your warehouse on Feb 3rd. I have been a customer for over 10 years.",
    "frustration_rent": "So my landlord just told me he is raising rent by 400 dollars next month and I literally cannot afford it. I have been here 3 years and never missed a payment. Looked into tenants rights in my state and apparently he has to give 60 days notice which he didnt so I might have some leverage.",
    "frustration_elevator": "Got stuck in an elevator for 45 minutes today at work and honestly the worst part wasnt the waiting it was being trapped with Dave from accounting who used the entire time to tell me about his fantasy baseball draft strategy. I now know more about relief pitchers than any human should.",
    "frustration_interview": "The interview went terribly. I showed up 10 minutes late because Google Maps sent me to the wrong building, my shirt had a coffee stain I didn't notice until I sat down, and I blanked on the most basic question about my own resume.",

    # --- Focus Group / Market Research ---
    "research_focus": "The focus group results were mixed. Participants aged 25-34 responded positively to the new branding (NPS +22) but the 45-54 cohort found it too modern and hard to read. We're recommending a hybrid approach that retains the new color palette but uses the original serif typeface for body copy.",

    # --- Sober / Health ---
    "health_sober": "Been sober for 6 months now and honestly the hardest part isn't the cravings. It's explaining to people why you're not drinking at a party. No, I don't want a mocktail. No, I'm not pregnant. I just don't drink anymore, Karen.",
    "health_sleep": "Couldn't sleep again last night. Brain just won't shut off. Kept replaying that conversation with Mike over and over. He didn't mean it the way I took it, I know that now, but in the moment it felt like a punch to the gut. Gonna try the meditation app again tonight.",

    # --- Bakery / Small Business ---
    "business_bakery": "I run a small bakery in Portland and let me tell you, the gluten-free trend has been both a blessing and a curse. Revenue is up 30% but my ingredient costs have tripled. Almond flour ain't cheap people.",
}


class TestFalsePositiveRate:
    """Every human text MUST NOT be classified as 'ai'.
    Classification as 'uncertain' is acceptable (cautious), but 'ai' is a failure."""

    @pytest.mark.parametrize("name,text", list(HUMAN_SAMPLES.items()))
    def test_human_not_flagged_ai(self, name, text):
        score, pred = get_prediction(text)
        assert pred != "ai", (
            f"FALSE POSITIVE [{name}]: human text flagged as AI "
            f"(score={score}, pred={pred})\n"
            f"Text: {text[:80]}..."
        )


def test_fpr_summary():
    """Print FPR summary at the end."""
    total = len(HUMAN_SAMPLES)
    ai_count = 0
    uncertain_count = 0
    human_count = 0
    for name, text in HUMAN_SAMPLES.items():
        _, pred = get_prediction(text)
        if pred == "ai":
            ai_count += 1
        elif pred == "uncertain":
            uncertain_count += 1
        else:
            human_count += 1
    fpr = ai_count / total * 100
    print(f"\n{'='*50}")
    print(f"FPR TEST SUITE: {total} human texts")
    print(f"  Human: {human_count} ({human_count/total*100:.1f}%)")
    print(f"  Uncertain: {uncertain_count} ({uncertain_count/total*100:.1f}%)")
    print(f"  AI (false positive): {ai_count} ({fpr:.1f}%)")
    print(f"  FALSE POSITIVE RATE: {fpr:.1f}%")
    print(f"{'='*50}")
