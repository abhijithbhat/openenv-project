"""
Email Triage Environment — Core Logic
======================================
Implements the OpenEnv Environment interface:
    reset()  → EmailObservation
    step()   → StepResult
    state()  → EnvState

This module is imported by server.py (FastAPI) and by inference.py.
"""
from __future__ import annotations

import uuid
from typing import List, Optional

from graders import STEP_REWARD_FN, TASK_DATA
from models import EmailAction, EmailObservation, EnvState, StepResult


_VALID_TASKS = ("easy_triage", "medium_triage", "hard_triage")


class EmailTriageEnvironment:
    """
    Real-world customer-support email triage environment.

    An AI agent is shown a sequence of customer emails and must classify
    each one into one of five routing categories:
        billing | technical | general | refund | complaint

    Episode flow
    ------------
    1. Call reset(task_name) → get the first EmailObservation.
    2. For each step, call step(EmailAction) → get StepResult.
    3. Continue until StepResult.done is True.
    4. Call state() at any time for episode metadata.

    Tasks
    -----
    - easy_triage   : 5 single-intent emails, binary scoring.
    - medium_triage : 5 emails with primary + secondary labels, partial credit.
    - hard_triage   : 5 complex multi-intent emails, escalation-aware scoring.
    """

    def __init__(self) -> None:
        self._task_name: str = "easy_triage"
        self._emails: List[dict] = []
        self._step_idx: int = 0
        self._episode_id: str = str(uuid.uuid4())
        self._total_reward: float = 0.0
        self._predictions: List[str] = []

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task_name: str = "easy_triage") -> EmailObservation:
        """
        Start a new episode for the given task.

        Parameters
        ----------
        task_name : one of 'easy_triage', 'medium_triage', 'hard_triage'.

        Returns
        -------
        EmailObservation — the first email in the episode.
        """
        if task_name not in _VALID_TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Valid tasks: {_VALID_TASKS}"
            )

        self._task_name = task_name
        self._emails = TASK_DATA[task_name]
        self._step_idx = 0
        self._episode_id = str(uuid.uuid4())
        self._total_reward = 0.0
        self._predictions = []

        return self._make_observation()

    def step(self, action: EmailAction) -> StepResult:
        """
        Execute one classification action.

        Parameters
        ----------
        action : EmailAction with category and optional confidence.

        Returns
        -------
        StepResult with next observation (or None if done), reward, done flag,
        and info dict containing the correct label for transparency.
        """
        if self._step_idx >= len(self._emails):
            # Environment already done — return terminal result
            return StepResult(
                observation=None,
                reward=0.0,
                done=True,
                info={"error": "Episode already finished. Call reset() to start a new one."},
            )

        current_email = self._emails[self._step_idx]
        predicted = action.category
        truth = current_email["label"]
        secondary = current_email.get("secondary")

        # Calculate reward using the task-specific function
        reward_fn = STEP_REWARD_FN[self._task_name]
        reward = reward_fn(predicted, truth, secondary)

        self._total_reward += reward
        self._predictions.append(predicted)
        self._step_idx += 1

        done = self._step_idx >= len(self._emails)

        info: dict = {
            "email_id": current_email["email_id"],
            "predicted": predicted,
            "correct": truth,
            "secondary_valid": secondary,
            "reward": reward,
            "step_reward_explanation": _explain_reward(
                predicted, truth, secondary, reward, self._task_name
            ),
        }

        next_obs: Optional[EmailObservation] = (
            None if done else self._make_observation()
        )

        return StepResult(
            observation=next_obs,
            reward=reward,
            done=done,
            info=info,
        )

    def state(self) -> EnvState:
        """Return current episode metadata."""
        difficulty = self._task_name.split("_")[0]  # 'easy' | 'medium' | 'hard'
        return EnvState(
            episode_id=self._episode_id,
            step_count=self._step_idx,
            total_reward=round(self._total_reward, 4),
            task_name=self._task_name,
            difficulty=difficulty,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_observation(self) -> EmailObservation:
        email = self._emails[self._step_idx]
        return EmailObservation(
            email_id=email["email_id"],
            subject=email["subject"],
            body=email["body"],
            task_name=self._task_name,
            step=self._step_idx + 1,
            max_steps=len(self._emails),
            available_categories=["billing", "technical", "general", "refund", "complaint"],
        )


# ---------------------------------------------------------------------------
# Private utility
# ---------------------------------------------------------------------------

def _explain_reward(
    predicted: str,
    truth: str,
    secondary: Optional[str],
    reward: float,
    task: str,
) -> str:
    if reward == 1.0:
        return f"Correct! '{predicted}' matches the primary label."
    if reward == 0.5:
        return (
            f"Partial credit: '{predicted}' matches the secondary label "
            f"'{secondary}'. Primary was '{truth}'."
        )
    if reward == 0.4:
        return (
            f"Partial credit (hard task): '{predicted}' matches secondary "
            f"label '{secondary}'. Primary was '{truth}'."
        )
    if reward == 0.2:
        return (
            f"Minimal credit: agent detected escalation intent ('complaint') "
            f"but primary label was '{truth}'."
        )
    return (
        f"Incorrect. Predicted '{predicted}', correct was '{truth}'"
        + (f" (secondary: '{secondary}')." if secondary else ".")
    )