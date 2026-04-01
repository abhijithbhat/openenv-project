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
        "subject": "Payment Failed",
        "body": "Hi, my credit card payment was declined when I tried to renew. "
                "Please help me fix this.",
        "label": "billing",
        "secondary": None,
    },
    {
        "email_id": "e002",
        "subject": "App Keeps Crashing",
        "body": "The mobile application crashes every time I open the dashboard. "
                "I've tried reinstalling but the problem persists.",
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
        "body": "I was charged $200 but I selected the $99 plan. "
                "Please correct my invoice.",
        "label": "billing",
        "secondary": None,
    },
    {
        "email_id": "e005",
        "subject": "Login Not Working",
        "body": "I cannot log into my account. It keeps saying incorrect "
                "password even though I just reset it.",
        "label": "technical",
        "secondary": None,
    },
]

MEDIUM_EMAILS: List[dict] = [
    {
        "email_id": "m001",
        "subject": "Payment Failed – Possible Bug?",
        "body": "My payment failed again. I suspect it's a bug in your checkout "
                "system because my card is definitely valid. Can you look into both "
                "the charge issue and the technical bug?",
        "label": "billing",
        "secondary": "technical",
    },
    {
        "email_id": "m002",
        "subject": "App Crash During Billing Update",
        "body": "Whenever I try to update my credit card details, the app crashes "
                "immediately. I'm worried the payment went through but the card "
                "wasn't saved. Please investigate.",
        "label": "technical",
        "secondary": "billing",
    },
    {
        "email_id": "m003",
        "subject": "Cancel and Refund Request",
        "body": "I want to cancel my subscription immediately and receive a full "
                "refund. The product didn't meet the promises made during the sales "
                "call. I'm very unhappy with the experience.",
        "label": "refund",
        "secondary": "complaint",
    },
    {
        "email_id": "m004",
        "subject": "Slow Dashboard",
        "body": "The web dashboard is extremely slow and times out when I try to "
                "view my account balance and billing history. Is this a known issue?",
        "label": "technical",
        "secondary": "general",
    },
    {
        "email_id": "m005",
        "subject": "Broken Invoicing Feature + Feature Request",
        "body": "The CSV export of invoices is broken — it produces empty files. "
                "Also, could you add a dark mode to the dashboard? Both would "
                "be very helpful.",
        "label": "technical",
        "secondary": "general",
    },
]

HARD_EMAILS: List[dict] = [
    {
        "email_id": "h001",
        "subject": "Multiple Failures – Escalation",
        "body": "I have been charged twice this month, the app crashed when I tried "
                "to raise a dispute, and your support team hasn't responded in two "
                "weeks. I want an immediate refund and I am considering contacting "
                "my bank and filing a formal complaint.",
        "label": "complaint",
        "secondary": "billing",
    },
    {
        "email_id": "h002",
        "subject": "Balance Mismatch – Third Time Writing",
        "body": "This is the third time I'm writing about the same issue. My account "
                "balance doesn't match my bank statement. The mobile app throws a "
                "500 error when I try to reconcile. I need this resolved today or "
                "I will file a chargeback.",
        "label": "billing",
        "secondary": "technical",
    },
    {
        "email_id": "h003",
        "subject": "API Timeouts + Billing for Inaccessible Features",
        "body": "Your API keeps timing out. My engineering team has wasted three "
                "weeks on this. Worse, you charged us for Enterprise features we "
                "cannot even access because the API is broken. We want out of this "
                "contract and a full refund for the Enterprise tier.",
        "label": "complaint",
        "secondary": "technical",
    },
    {
        "email_id": "h004",
        "subject": "Confused About Subscription Status",
        "body": "Hi — I got an email saying my subscription auto-renews next week, "
                "but I thought I cancelled it last month. The pricing on your website "
                "also differs from what your sales rep quoted. The mobile app shows "
                "my plan as 'expired' but the website shows 'active'. I'm confused "
                "and don't want to be charged unexpectedly.",
        "label": "billing",
        "secondary": "general",
    },
    {
        "email_id": "h005",
        "subject": "Critical: Corrupted Data Export",
        "body": "We discovered that data exported from your platform is corrupted — "
                "records are missing and some are duplicated. We are a healthcare "
                "provider and this is critical. We need the full dataset re-exported, "
                "a root-cause analysis, and we want billing paused until this is "
                "fully resolved.",
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
