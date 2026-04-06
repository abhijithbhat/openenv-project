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
scores 0.0 (catastrophic). Wrongly removing satire scores 0.1 (bad but recoverable).
Escalating to a human for genuinely ambiguous content always earns partial credit
(0.4–0.6) because that is the correct instinct — humans should make the final call
on borderline content.

Task Difficulty
---------------
Easy   — Clear, unambiguous posts. Exact-match binary scoring.
Medium — Posts with a primary and a secondary valid action. Partial credit.
Hard   — Context-dependent, ambiguous posts. Asymmetric severity scoring.

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
]

HARD_POSTS: List[dict] = [
    # Post type A: coded hate speech disguised as 'just asking questions'
    # Missing this is CATASTROPHIC (antisemitic dog whistle)
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
    # Post type B: satire that uses inflammatory language but is clearly a joke
    # Removing this is BAD (silencing comedy)
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
    # Post type A: veiled school threat using fictional framing
    # Missing this is CATASTROPHIC (potential real-world violence)
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
    # Post type B: accurate health information that sounds like misinformation
    # Removing or labelling this is BAD (suppressing correct public health info)
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
    # Post type B: personal discrimination testimony using a slur in a testimonial context
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
]

TASK_DATA: dict = {
    "easy_moderation":   EASY_POSTS,
    "medium_moderation": MEDIUM_POSTS,
    "hard_moderation":   HARD_POSTS,
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
        # Violations that must not be allowed through
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
        # Safe content — censoring it is the big mistake
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
