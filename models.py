"""
Typed Pydantic models for the Content Moderation OpenEnv Environment.

Action space  : remove | restrict | label | escalate | allow
Observation   : a social media post with platform context
StepResult    : reward (asymmetric, severity-weighted) + next observation
EnvState      : episode metadata
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

ActionType = Literal["remove", "restrict", "label", "escalate", "allow"]

class ContentModerationAction(BaseModel):
    """
    The action an agent takes after reviewing a post.

    Fields
    ------
    action     : One of the five enforcement decisions.
    confidence : Optional float [0.0–1.0] — how confident the agent is.
    reasoning  : Optional short explanation (useful for debugging agents).

    Action semantics
    ----------------
    remove    — Take the post down completely. Highest enforcement.
    restrict  — Reduce distribution / age-gate. Softer than remove.
    label     — Add a warning or context label. Lightest intervention.
    escalate  — Send to a human reviewer. Use when unsure.
    allow     — No action. Content is within policy.
    """
    action: ActionType
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reasoning: Optional[str] = Field(default=None, max_length=500)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class ContentObservation(BaseModel):
    """
    A single social media post for the agent to review.

    Fields
    ------
    post_id                  : Unique identifier for the post.
    content                  : The full text of the post.
    platform                 : Platform the post appeared on.
    context                  : Additional signal (e.g. report count, account type).
    author_account_age_days  : Days since the posting account was created.
                               Very new accounts (<7 days) warrant extra scrutiny.
    author_previous_strikes  : Number of prior policy violations on this account.
                               High-strike accounts should shift your thresholds.
    user_reports_count       : How many users have reported this post.
                               High report counts are a strong signal — but not conclusive.
    task_name                : Which task is active.
    step                     : Current step (1-based).
    max_steps                : Total posts in this episode.
    available_actions        : The valid action values for this environment.
    """
    episode_id: str = Field(
        description="Session identifier — pass this back to /step as session_id for concurrency-safe routing."
    )
    post_id: str
    content: str
    platform: str
    context: Optional[str] = None
    author_account_age_days: int = Field(
        default=365,
        ge=0,
        description="Days since account creation. Very new accounts (<7 days) are higher risk.",
    )
    author_previous_strikes: int = Field(
        default=0,
        ge=0,
        description="Prior policy violations on this account. High values should raise your suspicion threshold.",
    )
    user_reports_count: int = Field(
        default=0,
        ge=0,
        description="Number of user reports on this post. High volumes are a signal but not proof of a violation.",
    )
    task_name: str
    step: int
    max_steps: int
    available_actions: List[ActionType]


# ---------------------------------------------------------------------------
# State (episode metadata)
# ---------------------------------------------------------------------------

class EnvState(BaseModel):
    """
    Current episode metadata — safe to call at any time via GET /state.

    Fields
    ------
    episode_id   : UUID for reproducibility and logging.
    step_count   : How many posts have been reviewed so far.
    total_reward : Cumulative reward so far (0.0–max_steps).
    task_name    : Active task.
    difficulty   : 'easy' | 'medium' | 'hard'.
    """
    episode_id: str
    step_count: int
    total_reward: float
    task_name: str
    difficulty: str


# ---------------------------------------------------------------------------
# StepResult (returned by POST /step)
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """
    Result of one agent action.

    Fields
    ------
    observation : Next post to review, or None if episode is complete.
    reward      : Step reward [0.0–1.0]. Asymmetric on hard tasks.
    done        : True when all posts have been reviewed.
    info        : Dict with correct action, severity, and reward explanation.
    """
    observation: Optional[ContentObservation]
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict
