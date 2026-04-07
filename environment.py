"""
Content Moderation Environment — OpenEnv Implementation
========================================================
Simulates a real-world social media content review queue.
An agent reviews posts and decides: remove | restrict | label | escalate | allow

Reward is ASYMMETRIC — missing critical hate speech earns 0.0,
wrongly removing satire earns 0.05. See graders.py for full rationale.
"""
from __future__ import annotations

import random
import uuid
from typing import Optional

from models import (
    ActionType,
    ContentModerationAction,
    ContentObservation,
    EnvState,
    StepResult,
)
from graders import TASK_DATA, STEP_REWARD_FN

VALID_TASKS = list(TASK_DATA.keys())
AVAILABLE_ACTIONS: list[ActionType] = ["remove", "restrict", "label", "escalate", "allow"]


class ContentModerationEnvironment:
    """
    OpenEnv-compliant environment for social media content moderation.

    Public API
    ----------
    reset(task_name)  →  ContentObservation   (first post in episode)
    step(action)      →  StepResult           (reward + next post)
    state()           →  EnvState             (current episode metadata)
    """

    def __init__(self) -> None:
        self._episode_id: str = ""
        self._task_name: str = ""
        self._posts: list[dict] = []
        self._step_idx: int = 0
        self._total_reward: float = 0.0
        self._started: bool = False

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task_name: str = "easy_moderation") -> ContentObservation:
        """
        Start a new episode.

        Parameters
        ----------
        task_name : One of 'easy_moderation', 'medium_moderation', 'hard_moderation'.

        Returns
        -------
        ContentObservation with the first post in the queue.
        """
        if task_name not in VALID_TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Valid tasks: {VALID_TASKS}"
            )

        self._episode_id = str(uuid.uuid4())
        self._task_name = task_name
        # Shuffle posts so each episode feels different
        self._posts = list(TASK_DATA[task_name])
        random.shuffle(self._posts)
        self._step_idx = 0
        self._total_reward = 0.0
        self._started = True

        return self._make_observation()

    def step(self, action: ContentModerationAction) -> StepResult:
        """
        Apply an enforcement decision to the current post.

        Parameters
        ----------
        action : ContentModerationAction with .action field.

        Returns
        -------
        StepResult with reward, done flag, and next observation.
        """
        if not self._started:
            raise RuntimeError("Call reset() before step().")

        if self._step_idx >= len(self._posts):
            # Episode already finished — return terminal result
            return StepResult(
                observation=None,
                reward=0.0,
                done=True,
                info={"message": "Episode already complete."},
            )

        current_post = self._posts[self._step_idx]

        # Calculate asymmetric reward using the task-specific function
        reward_fn = STEP_REWARD_FN[self._task_name]
        reward = reward_fn(
            action.action,
            current_post["label"],
            current_post.get("secondary"),
            current_post.get("severity", "low"),
            current_post.get("post_type", "A"),
        )
        reward = round(float(reward), 4)
        self._total_reward += reward
        self._step_idx += 1

        done = self._step_idx >= len(self._posts)
        next_obs = None if done else self._make_observation()

        return StepResult(
            observation=next_obs,
            reward=reward,
            done=done,
            info={
                "post_id": current_post["post_id"],
                "correct_action": current_post["label"],
                "secondary_action": current_post.get("secondary"),
                "severity": current_post.get("severity"),
                "post_type": current_post.get("post_type"),
                "agent_action": action.action,
                "reward_explanation": _explain_reward(
                    action.action,
                    current_post["label"],
                    current_post.get("post_type", "A"),
                    reward,
                ),
            },
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

    def _make_observation(self) -> ContentObservation:
        post = self._posts[self._step_idx]
        return ContentObservation(
            episode_id=self._episode_id,
            post_id=post["post_id"],
            content=post["content"],
            platform=post["platform"],
            context=post.get("context"),
            task_name=self._task_name,
            step=self._step_idx + 1,
            max_steps=len(self._posts),
            available_actions=AVAILABLE_ACTIONS,
        )


# ---------------------------------------------------------------------------
# Private utility
# ---------------------------------------------------------------------------

def _explain_reward(predicted: str, truth: str, post_type: str,
                    reward: float) -> str:
    """Human-readable explanation of why this reward was awarded."""
    if reward == 1.0:
        return "Correct decision."
    if predicted == truth:
        return "Correct decision."
    if post_type == "A" and predicted == "allow":
        return (
            "CATASTROPHIC false negative: harmful content was allowed through. "
            "Reward: 0.0."
        )
    if post_type == "B" and predicted == "remove":
        return (
            "Legitimate content was silenced. "
            f"Correct action was '{truth}'. Reward: {reward}."
        )
    if predicted == "escalate":
        return (
            "Escalation to human reviewer is always partially valid "
            f"for ambiguous content. Reward: {reward}."
        )
    return f"Suboptimal decision. Correct action was '{truth}'. Reward: {reward}."