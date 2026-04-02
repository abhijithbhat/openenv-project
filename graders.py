"""
Graders — Email Triage Environment
====================================
Programmatic graders for each difficulty level.
All graders return a float in [0.0, 1.0].

Design principles
-----------------
- Easy   : exact-match only. Binary scoring.
- Medium : full credit for primary label; 0.5 for the secondary (alternative
           valid) label. Simulates real support queues where some tickets
           legitimately belong to two departments.
- Hard   : full credit for primary; 0.4 for secondary; 0.2 if the agent at
           least detected escalation intent (complaint) for high-frustration
           tickets. Penalises completely wrong answers with 0.0.
"""
from __future__ import annotations

from typing import List, Optional


# ---------------------------------------------------------------------------
# Email datasets  (deterministic and versioned for reproducibility)
# ---------------------------------------------------------------------------

EASY_EMAILS: List[dict] = [
    {
        "email_id": "e001",
        "subject": "my payment has failed",
        "body": "hello, my credit card payment was declined when I tried to renew it. "
                "please fix this help me",
        "label": "billing",
        "secondary": None,
    },
    {
        "email_id": "e002",
        "subject": "App Keeps Crashing every time",
        "body": "The mobile application crashes every time I open the dashboard. "
                "I have tried reinstalling, but the problem occurs each and every time",
        "label": "technical",
        "secondary": None,
    },
    {
        "email_id": "e003",
        "subject": "How does your service work?",
        "body": "I'm considering signing up. Could you explain how the service "
                "works and what plans are available?",
        "label": "general",
        "secondary": None,
    },
    {
        "email_id": "e004",
        "subject": "Wrong Invoice Amount",
        "body": "I was charged $200, but I chose the plan of $99. "
                "Please correct my invoice.",
        "label": "billing",
        "secondary": None,
    },
    {
        "email_id": "e005",
        "subject": "Login Not Working",
        "body": "I cannot login into my account. It always keeps saying incorrect "
                "password even though I just reset it.",
        "label": "technical",
        "secondary": None,
    },
]

MEDIUM_EMAILS: List[dict] = [
    {
        "email_id": "m001",
        "subject": "Payment failing again — is it a bug?",
        "body": "Hi,\n\nMy payment has failed again. My card is 100% valid so I think "
                "there might be some bug in your payment system.\n\nCan you check "
                "both the billing side and whether there is a technical issue causing this?",
        "label": "billing",
        "secondary": "technical",
    },
    {
        "email_id": "m002",
        "subject": "App crashes when updating card details",
        "body": "Hello,\n\nEvery time I try to update my credit card in the app, it "
                "crashes immediately.\n\nI am not sure if my payment went through or "
                "not since the card was not saved. Please check this for me.",
        "label": "technical",
        "secondary": "billing",
    },
    {
        "email_id": "m003",
        "subject": "I want to cancel and get a refund",
        "body": "Hi,\n\nI would like to cancel my subscription right away and get a "
                "full refund. The product was not what was promised during the sales call. "
                "I am really not happy with the experience overall.",
        "label": "refund",
        "secondary": "complaint",
    },
    {
        "email_id": "m004",
        "subject": "Dashboard is very slow",
        "body": "The dashboard keeps timing out whenever I try to check my account "
                "balance or see my billing history. Is this something you are aware of?",
        "label": "technical",
        "secondary": "general",
    },
    {
        "email_id": "m005",
        "subject": "Invoice export not working + one suggestion",
        "body": "Hi team,\n\nThe invoice CSV export is broken — it keeps downloading "
                "an empty file.\n\nAlso, small suggestion: it would be great if you "
                "could add a dark mode to the dashboard. Would really help.",
        "label": "technical",
        "secondary": "general",
    },
]

HARD_EMAILS: List[dict] = [
    {
        "email_id": "h001",
        "subject": "Charged twice & no response",
        "body": "Hi,\n\nI just noticed I've been charged twice this month. I tried "
                "raising a dispute in the app but it crashed midway. I've also "
                "reached out to support earlier but haven't heard back for almost "
                "2 weeks now.\n\nThis is really frustrating. I need a refund ASAP, "
                "otherwise I'll have to contact my bank and take this further.\n\nThanks.",
        "label": "complaint",
        "secondary": "billing",
    },
    {
        "email_id": "h002",
        "subject": "Still facing balance mismatch",
        "body": "Hello,\n\nThis is actually the third time I'm writing about this. "
                "My account balance just doesn't match what's shown in my bank "
                "statement.\n\nWhenever I try to check it in the app, I keep getting "
                "a 500 error which makes it impossible to verify anything.\n\nPlease "
                "look into this urgently. I really need this sorted today.",
        "label": "billing",
        "secondary": "technical",
    },
    {
        "email_id": "h003",
        "subject": "API issues + billing concern",
        "body": "Hey team,\n\nOur team has been dealing with constant API timeouts "
                "for the past few weeks now. It's affecting our work quite badly.\n\n"
                "On top of that, we've been billed for Enterprise features that we "
                "can't even use because of these issues.\n\nAt this point we're "
                "considering cancelling the contract altogether. Please advise on "
                "refund options.",
        "label": "complaint",
        "secondary": "technical",
    },
    {
        "email_id": "h004",
        "subject": "Subscription confusion",
        "body": "Hi,\n\nI'm a bit confused about my subscription. I got an email "
                "saying it will auto-renew next week, but I was pretty sure I "
                "cancelled it last month.\n\nAlso, the pricing on the website looks "
                "different from what I was originally told. And weirdly, the app "
                "shows my plan as expired while the website says it's active.\n\n"
                "Can you please clarify? I don't want to be charged unexpectedly.",
        "label": "billing",
        "secondary": "general",
    },
    {
        "email_id": "h005",
        "subject": "Data export issue (urgent)",
        "body": "Hi,\n\nWe recently exported data from your platform and noticed "
                "some serious issues. A number of records are missing, and a few "
                "are duplicated.\n\nWe're in healthcare, so this is quite critical "
                "for us.\n\nWe need a clean re-export of the data, along with some "
                "explanation of what went wrong. Also, it would help if billing "
                "could be paused until this is resolved.\n\nPlease treat this as urgent.",
        "label": "technical",
        "secondary": "billing",
    },
]

TASK_DATA: dict = {
    "easy_triage": EASY_EMAILS,
    "medium_triage": MEDIUM_EMAILS,
    "hard_triage": HARD_EMAILS,
}


# ---------------------------------------------------------------------------
# Individual step reward functions
# ---------------------------------------------------------------------------

def _reward_easy(predicted: str, truth: str, secondary: Optional[str]) -> float:
    """Binary: 1.0 if correct, 0.0 otherwise."""
    return 1.0 if predicted == truth else 0.0


def _reward_medium(predicted: str, truth: str, secondary: Optional[str]) -> float:
    """Full credit for primary label, 0.5 for secondary."""
    if predicted == truth:
        return 1.0
    if secondary and predicted == secondary:
        return 0.5
    return 0.0


def _reward_hard(predicted: str, truth: str, secondary: Optional[str]) -> float:
    """
    Full credit for primary; 0.4 for secondary; 0.2 for detecting
    escalation intent (complaint) when frustration is high.
    """
    if predicted == truth:
        return 1.0
    if secondary and predicted == secondary:
        return 0.4
    # Partial credit: agent identified that customer is frustrated / escalating
    if predicted == "complaint" and truth in ("billing", "technical", "refund"):
        return 0.2
    return 0.0


STEP_REWARD_FN = {
    "easy_triage": _reward_easy,
    "medium_triage": _reward_medium,
    "hard_triage": _reward_hard,
}


# ---------------------------------------------------------------------------
# Episode-level graders (used by scoring scripts)
# ---------------------------------------------------------------------------

def grade_easy(predictions: List[str]) -> float:
    """Grade a full easy episode. Returns episode score 0.0–1.0."""
    data = EASY_EMAILS
    if not predictions:
        return 0.0
    correct = sum(
        1.0 for pred, item in zip(predictions, data) if pred == item["label"]
    )
    return round(correct / len(data), 4)


def grade_medium(predictions: List[str]) -> float:
    """Grade a full medium episode. Returns episode score 0.0–1.0."""
    data = MEDIUM_EMAILS
    if not predictions:
        return 0.0
    total = sum(
        _reward_medium(pred, item["label"], item["secondary"])
        for pred, item in zip(predictions, data)
    )
    return round(total / len(data), 4)


def grade_hard(predictions: List[str]) -> float:
    """Grade a full hard episode. Returns episode score 0.0–1.0."""
    data = HARD_EMAILS
    if not predictions:
        return 0.0
    total = sum(
        _reward_hard(pred, item["label"], item["secondary"])
        for pred, item in zip(predictions, data)
    )
    return round(total / len(data), 4)


EPISODE_GRADERS = {
    "easy_triage": grade_easy,
    "medium_triage": grade_medium,
    "hard_triage": grade_hard,
}
