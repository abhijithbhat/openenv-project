"""
OpenEnv Typed Models — Email Triage Environment
================================================
Defines Action, Observation, State, and StepResult as Pydantic models,
as required by the OpenEnv specification.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class EmailAction(BaseModel):
    """
    Action space: classify the current email into a support category.

    Fields
    ------
    category : one of the five support routing labels.
    confidence : agent's self-reported confidence (0.0–1.0). Optional but
                 used by the hard-task grader for partial-credit rubrics.
    """

    category: Literal["billing", "technical", "general", "refund", "complaint"] = Field(
        description="Support category assigned to the email."
    )
    confidence: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in the classification (0.0-1.0).",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class EmailObservation(BaseModel):
    """
    Observation space: a single customer-support email awaiting classification.

    Fields
    ------
    email_id        : stable ID for reproducibility / debugging.
    subject         : email subject line.
    body            : email body text.
    task_name       : which task the episode is running.
    step            : current step index (1-based).
    max_steps       : total emails in this episode.
    available_categories : valid action values the agent may use.
    """

    email_id: str = Field(description="Unique email identifier.")
    subject: str = Field(description="Email subject line.")
    body: str = Field(description="Full email body text.")
    task_name: str = Field(description="Active task name.")
    step: int = Field(description="Current step number (1-based).")
    max_steps: int = Field(description="Total steps in this episode.")
    available_categories: List[str] = Field(
        default=["billing", "technical", "general", "refund", "complaint"],
        description="Valid category labels the agent may choose from.",
    )


# ---------------------------------------------------------------------------
# State (episode metadata)
# ---------------------------------------------------------------------------

class EnvState(BaseModel):
    """
    Episode state / metadata returned by GET /state.

    Fields
    ------
    episode_id   : unique UUID for the current episode.
    step_count   : number of steps taken so far.
    total_reward : cumulative reward earned in this episode.
    task_name    : active task name.
    difficulty   : 'easy' | 'medium' | 'hard'.
    """

    episode_id: str = Field(description="Unique UUID for this episode.")
    step_count: int = Field(description="Steps completed so far.")
    total_reward: float = Field(description="Cumulative reward this episode.")
    task_name: str = Field(description="Active task name.")
    difficulty: str = Field(description="Task difficulty level.")


# ---------------------------------------------------------------------------
# StepResult (returned by POST /step)
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """
    Result returned after calling POST /step.

    Fields
    ------
    observation : next observation (None when done=True).
    reward      : reward earned for this step (0.0–1.0).
    done        : whether the episode has ended.
    info        : auxiliary diagnostics (predicted label, correct label, etc.).
    """

    observation: Optional[EmailObservation] = Field(
        default=None,
        description="Next observation. None when the episode is done.",
    )
    reward: float = Field(ge=0.0, le=1.0, description="Step reward (0.0–1.0).")
    done: bool = Field(description="True when the episode has finished.")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary info: predicted, correct, partial_credit, etc.",
    )
