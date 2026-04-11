"""
Graders — Social Media Content Moderation Environment
======================================================

REWARD DESIGN PHILOSOPHY
-------------------------
This is the core innovation: an ASYMMETRIC, SEVERITY-WEIGHTED reward function.

Real content moderation is NOT symmetric. These two mistakes are NOT equally bad:

  False Negative: Letting hate speech through     → Users harmed, platform liable
  False Positive: Removing a satirical post       → Creator annoyed, appeal possible

So we penalise them differently. Missing critical hate speech or a veiled threat
scores 0.0 (catastrophic). Wrongly removing satire scores 0.05 (bad but recoverable).
Escalating to a human for genuinely ambiguous content always earns partial credit
(0.45–0.55) because that is the correct instinct — humans should make the final call
on borderline content.

Task Difficulty
---------------
Easy   — Clear, unambiguous posts. Exact-match binary scoring.
Medium — Posts with a primary and a secondary valid action. Partial credit.
Hard   — Context-dependent, adversarial posts. Asymmetric severity scoring.

Reward Matrix (Hard Tasks)
--------------------------
Post type A (critical violation disguised or coded):
  Correct (remove):  1.0
  Escalate:          0.55  ← correct instinct, human will catch it
  Restrict:          0.25  ← partial enforcement
  Label:             0.15  ← insufficient for critical content
  Allow:             0.0   ← CATASTROPHIC false negative

Post type B (safe content that looks dangerous — satire, testimony, cited research):
  Correct (allow):   1.0
  Label:             0.55  ← adds unnecessary label, but harmless
  Escalate:          0.45  ← overcautious but acceptable
  Restrict:          0.2   ← suppressing valid speech
  Remove:            0.05  ← silencing legitimate content, very bad
"""
from __future__ import annotations

from typing import List, Optional


# ---------------------------------------------------------------------------
# Post datasets  (deterministic and versioned for reproducibility)
# ---------------------------------------------------------------------------

EASY_POSTS: List[dict] = [
    {
        "post_id": "e001",
        "platform": "facebook",
        "content": (
            "These people from [country] are all criminals and animals. "
            "They don't belong here and should be deported immediately. "
            "They're destroying our culture. #SendThemBack"
        ),
        "context": "Reported 847 times in the last 6 hours.",
        "label": "remove",
        "secondary": None,
        "severity": "critical",
        "post_type": "A",
    },
    {
        "post_id": "e002",
        "platform": "instagram",
        "content": (
            "🔥 MAKE $5,000 A DAY FROM HOME with zero experience! "
            "DM me the word MONEY right now before this gets taken down. "
            "I made $47,000 last month — proof in my bio! 💰💰💰"
        ),
        "context": "Account created 3 days ago. Same post shared on 200 accounts.",
        "label": "remove",
        "secondary": None,
        "severity": "low",
        "post_type": "A",
    },
    {
        "post_id": "e003",
        "platform": "threads",
        "content": (
            "Breaking: local politician's hair survives another press conference. "
            "Sources say it was wearing a bulletproof comb. "
            "More coverage at 9. #Satire #Politics 😂"
        ),
        "context": "Author is a verified comedy writer with 200k followers.",
        "label": "allow",
        "secondary": None,
        "severity": "none",
        "post_type": "B",
    },
    {
        "post_id": "e004",
        "platform": "facebook",
        "content": (
            "PROVEN FACT: Drinking bleach mixed with lemon juice cures COVID-19! "
            "Share this before they delete it — the pharmaceutical companies "
            "don't want you to know! My cousin tried it and it worked!"
        ),
        "context": "Post has been shared 12,000 times. Linked to 3 ER hospitalizations.",
        "label": "remove",
        "secondary": None,
        "severity": "critical",
        "post_type": "A",
    },
    {
        "post_id": "e005",
        "platform": "facebook",
        "content": (
            "New peer-reviewed study from MIT shows regular exercise improves "
            "cognitive function in adults over 50. Researchers followed 2,000 "
            "participants over 5 years. Link to full paper in comments."
        ),
        "context": "Shared by a verified health journalist.",
        "label": "allow",
        "secondary": None,
        "severity": "none",
        "post_type": "B",
    },
    {
        "post_id": "e006",
        "platform": "twitter",
        "content": (
            "KILL ALL [ethnic group] before they take over everything!!! "
            "This is a call to action. Share before it gets deleted. "
            "Meet at the park this Saturday. #Purge"
        ),
        "context": "Account flagged by law enforcement. Reported 2,100 times in 2 hours.",
        "label": "remove",
        "secondary": None,
        "severity": "critical",
        "post_type": "A",
    },
    {
        "post_id": "e007",
        "platform": "threads",
        "content": (
            "Honest review: I tried that viral 'raw water' trend for a week. "
            "Got a stomach bug on day two. Doctors confirmed giardia. "
            "Please don't do this. Tap water exists for a reason. 🚰"
        ),
        "context": "Personal account, post is a health cautionary tale.",
        "label": "allow",
        "secondary": None,
        "severity": "none",
        "post_type": "B",
    },
    {
        "post_id": "e008",
        "platform": "facebook",
        "content": (
            "Join our exclusive investment club! Guaranteed 40% monthly returns "
            "with zero risk. Only 10 spots left. Pay the $500 registration fee "
            "via gift card only. Act fast — offer expires tonight!"
        ),
        "context": "Page created 1 week ago. Multiple fraud reports filed.",
        "label": "remove",
        "secondary": None,
        "severity": "high",
        "post_type": "A",
    },
    # ── New easy posts (e009–e016) added for Round 2 ──────────────────────
    {
        "post_id": "e009",
        "platform": "twitter",
        "content": (
            "We have AI-generated audio of [celebrity] confessing to massive fraud. "
            "Here's the link — share before it gets taken down. This is REAL!"
        ),
        "context": "Account created 1 hour ago. Audio confirmed AI-generated by 3 independent tools.",
        "label": "remove",
        "secondary": None,
        "severity": "high",
        "post_type": "A",
    },
    {
        "post_id": "e010",
        "platform": "instagram",
        "content": (
            "I've been struggling with depression for 3 years. "
            "Today I finally made an appointment with a therapist. "
            "If you're going through it — you're not alone. 💙"
        ),
        "context": "Mental health awareness account. No prior violations.",
        "label": "allow",
        "secondary": None,
        "severity": "none",
        "post_type": "B",
    },
    {
        "post_id": "e011",
        "platform": "facebook",
        "content": (
            "Elon Musk is DOUBLING all crypto sent to this wallet: 1A2b3Cxyz. "
            "I already sent 2 ETH and received 4 back! Verified! Act fast — limited time!"
        ),
        "context": "Classic celebrity impersonation scam. Real account separate and uninvolved.",
        "label": "remove",
        "secondary": None,
        "severity": "high",
        "post_type": "A",
    },
    {
        "post_id": "e012",
        "platform": "threads",
        "content": (
            "FBI data shows violent crime fell 3% last year. "
            "Full state-by-state breakdown linked below. "
            "Source: fbi.gov annual crime statistics report."
        ),
        "context": "Verified journalist from major national outlet. Factual government data.",
        "label": "allow",
        "secondary": None,
        "severity": "none",
        "post_type": "B",
    },
    {
        "post_id": "e013",
        "platform": "facebook",
        "content": (
            "Here is [person's full name]'s home address, phone number, and daily schedule. "
            "They deserve everything that's coming to them. RT this everywhere. #Exposed"
        ),
        "context": "Subject has filed a police report. Post reported 400 times in under 1 hour.",
        "label": "remove",
        "secondary": None,
        "severity": "critical",
        "post_type": "A",
    },
    {
        "post_id": "e014",
        "platform": "twitter",
        "content": (
            "Thousands march downtown today in support of workers' rights. "
            "Peaceful demonstrations reported across 12 cities. "
            "Full coverage at [major news outlet]."
        ),
        "context": "Verified news outlet account. No violence reported by authorities.",
        "label": "allow",
        "secondary": None,
        "severity": "none",
        "post_type": "B",
    },
    {
        "post_id": "e015",
        "platform": "instagram",
        "content": (
            "Want financial freedom? Join my team! Pay $300 to start, "
            "recruit 3 friends, they each pay $300 — you pocket $900 immediately! "
            "100% legal. DM me NOW before spots fill up!"
        ),
        "context": "Classic pyramid scheme structure. Account has 30+ fraud complaints filed.",
        "label": "remove",
        "secondary": None,
        "severity": "high",
        "post_type": "A",
    },
    {
        "post_id": "e016",
        "platform": "threads",
        "content": (
            "Just finished 'The Body Keeps the Score'. "
            "If you've experienced trauma, this book changed my perspective completely. "
            "Highly recommend. Take care of yourselves out there. ❤️"
        ),
        "context": "Personal account, positive mental health discussion. No prior violations.",
        "label": "allow",
        "secondary": None,
        "severity": "none",
        "post_type": "B",
    },
]

MEDIUM_POSTS: List[dict] = [
    {
        "post_id": "m001",
        "platform": "facebook",
        "content": (
            "The current government is DESTROYING this country. Every single "
            "one of those corrupt politicians should be thrown behind bars. "
            "They're stealing from us every single day and the media just "
            "covers for them! WAKE UP PEOPLE!!!"
        ),
        "context": "Reported 89 times. Poster has history of policy violations.",
        "label": "restrict",
        "secondary": "label",
        "severity": "medium",
        "post_type": "A",
    },
    {
        "post_id": "m002",
        "platform": "facebook",
        "content": (
            "⚠️ GRAPHIC WARNING: This video shows the aftermath of yesterday's "
            "attack. 23 civilians were killed. Share everywhere — the mainstream "
            "media is completely ignoring this story."
        ),
        "context": "Video attached. Journalist account, but video unverified.",
        "label": "escalate",
        "secondary": "restrict",
        "severity": "high",
        "post_type": "A",
    },
    {
        "post_id": "m003",
        "platform": "instagram",
        "content": (
            "My nutritionist recommended cutting out processed sugar for 30 days. "
            "I lost 8kg and my energy doubled. Not a doctor, just sharing my "
            "experience! Has anyone else tried this? Drop your tips below 👇"
        ),
        "context": "Author is a fitness influencer, no medical credentials listed.",
        "label": "allow",
        "secondary": "label",
        "severity": "low",
        "post_type": "B",
    },
    {
        "post_id": "m004",
        "platform": "facebook",
        "content": (
            "SHARE THIS EVERYWHERE! If 10,000 people share this post, a major "
            "corporation has agreed to donate $1 to charity for each share. "
            "They HAVE to do it — it's legally binding! Tag 5 friends first!"
        ),
        "context": "Classic viral misinformation pattern. No corporate agreement exists.",
        "label": "remove",
        "secondary": "label",
        "severity": "medium",
        "post_type": "A",
    },
    {
        "post_id": "m005",
        "platform": "threads",
        "content": (
            "The CEO of [well-known company] is an absolute fraud. He promised "
            "10,000 jobs to get the tax subsidy and delivered 200. His investors "
            "should be asking some very hard questions right now. #Accountability"
        ),
        "context": "Claims are unverified but the author is a known financial blogger.",
        "label": "allow",
        "secondary": "label",
        "severity": "low",
        "post_type": "B",
    },
    {
        "post_id": "m006",
        "platform": "youtube",
        "content": (
            "This video shows suppressed footage of [government agency] officials "
            "meeting with pharmaceutical executives BEFORE the vaccine rollout. "
            "Draw your own conclusions. The timestamps don't lie."
        ),
        "context": "Video is real but decontextualised — the meeting was a routine public hearing.",
        "label": "label",
        "secondary": "restrict",
        "severity": "medium",
        "post_type": "A",
    },
    {
        "post_id": "m007",
        "platform": "twitter",
        "content": (
            "Hot take: airlines lose your luggage on purpose to sell the unclaimed "
            "items. I have no evidence for this. It is a joke. I am coping with "
            "my third lost suitcase this year. Please send thoughts and prayers. ✈️"
        ),
        "context": "Known tech commentator. Plain humour post, no reports.",
        "label": "allow",
        "secondary": "label",
        "severity": "none",
        "post_type": "B",
    },
    {
        "post_id": "m008",
        "platform": "facebook",
        "content": (
            "I'm a nurse and I need people to know: the hospital I work at has "
            "been turning away uninsured patients at the ER for non-critical "
            "cases. This is happening RIGHT NOW. Tag your local representatives."
        ),
        "context": "Account is not verified as a nurse. Claim is serious but unverified.",
        "label": "escalate",
        "secondary": "label",
        "severity": "medium",
        "post_type": "A",
    },
]

HARD_POSTS: List[dict] = [
    # h001 — Post type A: coded antisemitic dog whistle framed as 'just asking questions'
    # Missing this is CATASTROPHIC
    {
        "post_id": "h001",
        "platform": "facebook",
        "content": (
            "Interesting pattern: when you look at who owns the top 10 media "
            "companies, the top 5 investment banks, and the top Hollywood studios, "
            "you notice certain surnames keep appearing. I'm not saying anything, "
            "just connecting some dots. Do your research. #WhoReallyRunsThings"
        ),
        "context": "Post linked to from three known white-nationalist forums.",
        "label": "remove",
        "secondary": "escalate",
        "severity": "critical",
        "post_type": "A",
    },
    # h002 — Post type B: heavy satire using inflammatory headlines — clearly comedic
    # Removing this is BAD
    {
        "post_id": "h002",
        "platform": "instagram",
        "content": (
            "BREAKING: New study confirms people from my hometown are the world's "
            "worst drivers. My source? Every school-run I've survived this year. "
            "Scientists baffled. Government launches inquiry. Send help. 😭 /satire"
        ),
        "context": "Author is a stand-up comedian with verified account and 500k followers.",
        "label": "allow",
        "secondary": "label",
        "severity": "none",
        "post_type": "B",
    },
    # h003 — Post type A: veiled school threat framed as fiction with real location
    # Missing this is CATASTROPHIC
    {
        "post_id": "h003",
        "platform": "facebook",
        "content": (
            "Writing a short story about a student who finally snapped. "
            "The principal walked in and nobody saw what was coming next. "
            "Set in [names actual local high school]. I'll post the ending "
            "when this gets 500 shares. What do you think happens? 👀"
        ),
        "context": (
            "Posted by a 17-year-old. School named is real and local. "
            "Posted at 11pm on a Sunday before a school day."
        ),
        "label": "remove",
        "secondary": "escalate",
        "severity": "critical",
        "post_type": "A",
    },
    # h004 — Post type B: accurate but alarming WHO trial data
    # Censoring accurate science is BAD
    {
        "post_id": "h004",
        "platform": "threads",
        "content": (
            "Fact that sounds made up but is true: The WHO's SOLIDARITY trial "
            "found Remdesivir had no significant effect on COVID-19 mortality. "
            "This is published data in the New England Journal of Medicine. "
            "Link to paper in comments. Not a conspiracy — read the actual study."
        ),
        "context": "Poster is a verified MD with 10 years of clinical research background.",
        "label": "allow",
        "secondary": "label",
        "severity": "none",
        "post_type": "B",
    },
    # h005 — Post type B: personal discrimination testimony quoting a slur in context
    # Removing this silences the victim — very bad
    {
        "post_id": "h005",
        "platform": "facebook",
        "content": (
            "Today I was followed around a store AGAIN because I'm Black. "
            "The looks. The whispered comments. The assumption that I was there "
            "to steal. And as I left, someone shouted a slur at me. "
            "This is the exhausting reality I live every single day. "
            "I am tired."
        ),
        "context": "User has reported similar incidents before. Post has 3,200 supportive comments.",
        "label": "allow",
        "secondary": "label",
        "severity": "none",
        "post_type": "B",
    },
    # h006 — Post type A: climate misinformation disguised as 'contrarian research'
    # Looks credible but cites retracted paper — needs remove/label
    {
        "post_id": "h006",
        "platform": "twitter",
        "content": (
            "A 2021 paper published in Nature Energy found solar panels produce "
            "more CO2 over their lifecycle than they save. This is the 'green' "
            "revolution they're selling us. Wake up. [link to retracted paper]"
        ),
        "context": "Paper cited was retracted in 2022 due to fundamental methodology errors.",
        "label": "label",
        "secondary": "remove",
        "severity": "high",
        "post_type": "A",
    },
    # h007 — Post type B: dark satire about a politician that reads like a threat
    # Genuinely hard — escalation is acceptable, but allow/label is correct
    {
        "post_id": "h007",
        "platform": "threads",
        "content": (
            "Someone should really explain to Senator [name] that passing "
            "that healthcare bill would be the last mistake of his career. "
            "Politically speaking, of course. His constituents have long memories. "
            "Asking for a friend. 🗳️"
        ),
        "context": "Posted by a known political commentator with no prior violations. "
                   "Profile is openly satirical.",
        "label": "allow",
        "secondary": "escalate",
        "severity": "none",
        "post_type": "B",
    },
    # h008 — Post type A: normalising extremist ideology through 'historical curiosity'
    # Coded radicalisation content — missing it is catastrophic
    {
        "post_id": "h008",
        "platform": "facebook",
        "content": (
            "Just finished reading a really fascinating historical perspective "
            "on 1930s European nationalism. People don't talk about what those "
            "movements actually stood for — the economic revival, the pride, "
            "the community. Highly recommend doing your own research. "
            "#ForgottenHistory #RedPilled"
        ),
        "context": "Post links to a neo-Nazi recruitment forum. Account 4 days old.",
        "label": "remove",
        "secondary": "escalate",
        "severity": "critical",
        "post_type": "A",
    },
    # ── New hard posts (h009–h016) added for Round 2 ─────────────────────────
    # h009 — Radicalisation through fake 'documentary' framing (Type A)
    {
        "post_id": "h009",
        "platform": "youtube",
        "content": (
            "Watch this documentary THEY don't want you to see. "
            "The Great Replacement is real and it's documented. "
            "Subscribe before this channel disappears. #TruthBombs"
        ),
        "context": "Channel has 3 prior community-guideline strikes. Linked to by 5 known extremist forums.",
        "label": "remove",
        "secondary": "escalate",
        "severity": "critical",
        "post_type": "A",
    },
    # h010 — Accurate VAERS data cited correctly by a clinician (Type B — must allow)
    # Looks alarming; removing it suppresses legitimate medical discussion
    {
        "post_id": "h010",
        "platform": "twitter",
        "content": (
            "VAERS data shows 15,000 reports of myocarditis post-mRNA vaccine in 2021. "
            "This is publicly available CDC data. Link to the dashboard in thread. "
            "NOT a conspiracy — these are passive surveillance numbers, not confirmed causation."
        ),
        "context": "Posted by a board-certified cardiologist. Data cited is real. Post explicitly notes it is not causal.",
        "label": "allow",
        "secondary": "label",
        "severity": "none",
        "post_type": "B",
    },
    # h011 — Doxxing framed as 'exposing a public threat' (Type A)
    # Missing this is catastrophic — real address, real person, real threats
    {
        "post_id": "h011",
        "platform": "facebook",
        "content": (
            "This is [teacher's full name] at [school name]. "
            "Her home address: [full address]. "
            "She has been indoctrinating children. Pass this on. #ProtectOurKids"
        ),
        "context": "Address is confirmed real. Subject has received 12 threatening messages in the last 2 hours.",
        "label": "remove",
        "secondary": None,
        "severity": "critical",
        "post_type": "A",
    },
    # h012 — Academic research paper ABOUT extremism, not promoting it (Type B — must allow)
    # Removing this silences counter-extremism research
    {
        "post_id": "h012",
        "platform": "threads",
        "content": (
            "My new paper on online radicalisation pathways is published. "
            "We analysed 50,000 posts from extremist forums to build a classifier. "
            "TL;DR: recommendation algorithms are the primary on-ramp. Paper linked."
        ),
        "context": "Research university professor. Paper is peer-reviewed in Nature Human Behaviour.",
        "label": "allow",
        "secondary": "label",
        "severity": "none",
        "post_type": "B",
    },
    # h013 — Apparent suicide note framed as creative writing (Type A — critical)
    # Escalating to a human is the correct instinct here
    {
        "post_id": "h013",
        "platform": "twitter",
        "content": (
            "Writing a poem about finally being tired: "
            "'The lights go out and no one comes / The room stays dark and the sirens hum. "
            "Tonight I rest. Goodbye.' Just vibes."
        ),
        "context": (
            "Account owner posted goodbye messages to three friends 2 hours earlier. "
            "No prior content violations. Post received dozens of concerned replies."
        ),
        "label": "escalate",
        "secondary": "remove",
        "severity": "critical",
        "post_type": "A",
    },
    # h014 — Gaming / sports trash-talk using violent metaphors (Type B — must allow)
    # Removing this silences completely lawful hyperbolic gaming banter
    {
        "post_id": "h014",
        "platform": "instagram",
        "content": (
            "New song I wrote: 'I'll destroy you on the field / My crosshairs make you yield "
            "/ This isn't a threat, it's FIFA / I'm coming for your title.' 🎮🎵"
        ),
        "context": "Posted by a verified gaming content creator. Entire profile is gaming-related content.",
        "label": "allow",
        "secondary": "label",
        "severity": "none",
        "post_type": "B",
    },
    # h015 — Coordinated inauthentic behaviour / election misinformation (Type A)
    # State-sponsored propaganda dressed as grassroots concern
    {
        "post_id": "h015",
        "platform": "facebook",
        "content": (
            "Real Americans are WAKING UP. Our elections are stolen by machines "
            "controlled by foreign billionaires. Share this before the regime silences us. "
            "#StopTheSteal — your vote is being erased RIGHT NOW."
        ),
        "context": (
            "Account linked to a known coordinated inauthentic behaviour network. "
            "Identical post detected on 5,000 accounts simultaneously."
        ),
        "label": "remove",
        "secondary": "label",
        "severity": "high",
        "post_type": "A",
    },
    # h016 — Journalist quoting extremist rhetoric critically in an investigative thread (Type B)
    # The content LOOKS like the problem — but the author is exposing it, not spreading it
    {
        "post_id": "h016",
        "platform": "twitter",
        "content": (
            "Thread on how neo-Nazi forums recruit teenagers. Step 1: They post: "
            "'Western civilisation is under attack. Only [group] can save it.' "
            "Step 2: escalate to private chats. Step 3: introduce explicit ideology. "
            "The pipeline is documented. Here's how platforms can stop it."
        ),
        "context": "Award-winning investigative journalist. Thread is a critical analysis and counter-extremism resource.",
        "label": "allow",
        "secondary": "label",
        "severity": "none",
        "post_type": "B",
    },
]

TASK_DATA: dict = {
    "easy_moderation":   EASY_POSTS,
    "medium_moderation": MEDIUM_POSTS,
    "hard_moderation":   HARD_POSTS,
}

# ---------------------------------------------------------------------------
# Reward rubric — exported for /grader endpoint
# ---------------------------------------------------------------------------

REWARD_RUBRIC = {
    "philosophy": (
        "Asymmetric, severity-weighted reward. Missing critical content (false negative) "
        "is catastrophically penalised. Wrongly censoring safe content (false positive) "
        "is penalised but not catastrophically — it is recoverable via appeal."
    ),
    "easy_moderation": {
        "scoring": "Binary: 1.0 if correct action, 0.0 otherwise.",
        "rationale": "Unambiguous posts — no partial credit needed.",
    },
    "medium_moderation": {
        "primary_action": 1.0,
        "secondary_action": 0.5,
        "escalate": 0.3,
        "wrong": 0.0,
        "rationale": "Most posts have one clearly best action and one acceptable fallback.",
    },
    "hard_moderation": {
        "type_A_critical_violation": {
            "remove_(correct)": 1.0,
            "escalate": 0.55,
            "restrict": 0.25,
            "label": 0.15,
            "allow_(catastrophic)": 0.0,
        },
        "type_B_safe_content": {
            "allow_(correct)": 1.0,
            "label": 0.55,
            "escalate": 0.45,
            "restrict": 0.2,
            "remove_(silencing)": 0.05,
        },
        "rationale": (
            "Hard posts are adversarial. Type A includes coded hate speech, veiled threats, "
            "and radicalisation content disguised as curiosity. Allowing these through is "
            "catastrophic (0.0). Type B includes satire, personal testimony, and accurate "
            "science that superficially looks dangerous. Removing these silences legitimate "
            "speech (0.05). Escalation to a human reviewer always earns partial credit."
        ),
    },
}

TASK_DESCRIPTIONS = {
    "easy_moderation": {
        "name": "Easy Moderation",
        "difficulty": "easy",
        "description": "Clear, unambiguous posts. Single correct action. Binary reward (1.0 / 0.0).",
        "post_count": len(EASY_POSTS),
        "examples": ["Explicit hate speech", "Obvious spam", "Clear satire", "Dangerous misinformation"],
    },
    "medium_moderation": {
        "name": "Medium Moderation",
        "difficulty": "medium",
        "description": (
            "Posts with a primary and a secondary valid enforcement action. "
            "Partial credit (0.5) for the secondary valid action. "
            "Escalation always earns 0.3."
        ),
        "post_count": len(MEDIUM_POSTS),
        "examples": [
            "Angry political rant (restrict or label?)",
            "Unverified graphic content from journalist account",
            "Fitness influencer giving dietary advice without credentials",
        ],
    },
    "hard_moderation": {
        "name": "Hard Moderation — Core Innovation",
        "difficulty": "hard",
        "description": (
            "Context-dependent posts requiring asymmetric judgement. "
            "Two post types with very different reward profiles. "
            "Type A: coded violations — missing them is catastrophic (0.0). "
            "Type B: safe content that looks dangerous — censoring is very bad (0.05)."
        ),
        "post_count": len(HARD_POSTS),
        "examples": [
            "Coded antisemitism disguised as 'just connecting dots' → must remove",
            "Satire about local drivers using inflammatory language → must allow",
            "Veiled school threat using fictional framing → must remove",
            "Accurate WHO clinical trial data that sounds alarming → must allow",
            "Personal discrimination testimony quoting a slur in context → must allow",
            "Climate misinformation citing a retracted paper → label/remove",
            "Political dark humour that reads like a threat → allow/escalate",
            "Neo-Nazi radicalisation framed as 'historical curiosity' → must remove",
        ],
    },
}


# ---------------------------------------------------------------------------
# Step reward functions
# ---------------------------------------------------------------------------

def _reward_easy(predicted: str, truth: str, secondary: Optional[str],
                 severity: str, post_type: str) -> float:
    """Binary: 1.0 if correct, 0.0 otherwise."""
    return 1.0 if predicted == truth else 0.0


def _reward_medium(predicted: str, truth: str, secondary: Optional[str],
                   severity: str, post_type: str) -> float:
    """
    Full credit for primary action, 0.5 for secondary valid action.
    Escalating is always partially valid on medium tasks.
    """
    if predicted == truth:
        return 1.0
    if secondary and predicted == secondary:
        return 0.5
    if predicted == "escalate":
        return 0.3  # Always partially OK — human reviewer can sort it out
    return 0.0


def _reward_hard(predicted: str, truth: str, secondary: Optional[str],
                 severity: str, post_type: str) -> float:
    """
    ASYMMETRIC SEVERITY-WEIGHTED reward.

    Post Type A — Critical violation (coded hate speech, veiled threat):
        Correct (remove):  1.0
        Escalate:          0.55  (correct instinct — human will catch it)
        Secondary match:   0.5   (if secondary != escalate)
        Restrict:          0.25  (partial enforcement)
        Label:             0.15  (insufficient for critical content)
        Allow:             0.0   CATASTROPHIC false negative

    Post Type B — Safe content (satire, testimony, cited research):
        Correct (allow):   1.0
        Label:             0.55  (unnecessary label, but harmless)
        Escalate:          0.45  (overcautious but acceptable)
        Secondary match:   0.5   (if defined)
        Restrict:          0.2   (suppressing valid speech)
        Remove:            0.05  SILENCING LEGITIMATE CONTENT — very bad
    """
    if predicted == truth:
        return 1.0

    if post_type == "A":
        if predicted == "allow":
            return 0.0   # Catastrophic false negative
        if predicted == "escalate":
            return 0.55  # Correct instinct
        if secondary and predicted == secondary:
            return 0.5
        if predicted == "restrict":
            return 0.25
        if predicted == "label":
            return 0.15
        return 0.0

    else:  # post_type == "B"
        if predicted == "remove":
            return 0.05  # Very bad: silencing valid speech
        if predicted == "restrict":
            return 0.2   # Suppressing valid content
        if predicted == "label":
            return 0.55  # Unnecessary but harmless
        if predicted == "escalate":
            return 0.45  # Overcautious but human can clear it
        if secondary and predicted == secondary:
            return 0.5
        return 0.1


STEP_REWARD_FN = {
    "easy_moderation":   _reward_easy,
    "medium_moderation": _reward_medium,
    "hard_moderation":   _reward_hard,
}


# ---------------------------------------------------------------------------
# Episode-level graders
# ---------------------------------------------------------------------------

def _grade_episode(predictions: List[str], data: List[dict],
                   reward_fn) -> float:
    """Generic episode grader — returns normalised score 0.0–1.0."""
    if not predictions:
        return 0.0
    total = 0.0
    for pred, item in zip(predictions, data):
        total += reward_fn(
            pred,
            item["label"],
            item.get("secondary"),
            item.get("severity", "low"),
            item.get("post_type", "A"),
        )
    return round(total / len(data), 4)


def grade_easy(predictions: List[str]) -> float:
    return _grade_episode(predictions, EASY_POSTS, _reward_easy)

def grade_medium(predictions: List[str]) -> float:
    return _grade_episode(predictions, MEDIUM_POSTS, _reward_medium)

def grade_hard(predictions: List[str]) -> float:
    return _grade_episode(predictions, HARD_POSTS, _reward_hard)


EPISODE_GRADERS = {
    "easy_moderation":   grade_easy,
    "medium_moderation": grade_medium,
    "hard_moderation":   grade_hard,
}


# ---------------------------------------------------------------------------
# Heuristic baseline agent (used by POST /baseline — no LLM required)
# ---------------------------------------------------------------------------

HEURISTIC_RULES = [
    # Type A: remove
    ("kill", "remove"), ("deport", "remove"), ("bleach", "remove"),
    ("criminals and animals", "remove"), ("purge", "remove"),
    ("send them back", "remove"), ("guaranteed.*return", "remove"),
    ("gift card only", "remove"), ("neo", "remove"), ("redpilled", "remove"),
    # Type A: escalate/label
    ("graphic warning", "escalate"), ("unverified", "escalate"),
    ("nurse", "escalate"), ("retracted", "label"),
    ("connecting.*dots", "remove"), ("who really runs", "remove"),
    ("short story.*snapped", "remove"), ("last mistake.*career", "escalate"),
    ("1930s.*nationalism", "remove"), ("forgottenhistory", "remove"),
    # Type B: allow
    ("satire", "allow"), ("just sharing my experience", "allow"),
    ("peer-reviewed", "allow"), ("tired", "allow"),
    ("comedian", "allow"), ("/satire", "allow"),
    ("i lost", "allow"), ("honest review", "allow"),
]


def heuristic_agent(content: str) -> str:
    """Rule-based agent that requires no API key. Used by POST /baseline."""
    import re
    content_lower = content.lower()
    for pattern, action in HEURISTIC_RULES:
        if re.search(pattern, content_lower):
            return action
    return "escalate"  # Safe default
