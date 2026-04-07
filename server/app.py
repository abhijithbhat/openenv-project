"""
server/app.py — Content Moderation OpenEnv (self-contained entry point)
========================================================================
Uvicorn target: server.app:app
This file is fully self-contained so there is no circular import between
server.py (root) and the server/ package.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional

# Ensure project root is on sys.path so graders/environment/models are importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import ContentModerationEnvironment, VALID_TASKS
from models import ContentModerationAction, ContentObservation, EnvState, StepResult
from graders import (
    REWARD_RUBRIC,
    TASK_DESCRIPTIONS,
    TASK_DATA,
    STEP_REWARD_FN,
    heuristic_agent,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concurrency-safe session store
# ---------------------------------------------------------------------------

active_sessions: Dict[str, ContentModerationEnvironment] = {}
_session_timestamps: Dict[str, float] = {}
_default_env = ContentModerationEnvironment()
SESSION_TTL_SECONDS = 3600


def _get_env(session_id: Optional[str]) -> ContentModerationEnvironment:
    if session_id and session_id in active_sessions:
        return active_sessions[session_id]
    return _default_env


def _prune_old_sessions() -> None:
    now = time.time()
    expired = [sid for sid, ts in _session_timestamps.items()
               if now - ts > SESSION_TTL_SECONDS]
    for sid in expired:
        active_sessions.pop(sid, None)
        _session_timestamps.pop(sid, None)
    if expired:
        logger.info("Pruned %d expired sessions.", len(expired))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ContentGuard Moderation Environment starting up.")
    yield
    logger.info("ContentGuard shutting down.")
    active_sessions.clear()

app = FastAPI(
    title="ContentGuard — Content Moderation OpenEnv",
    description=(
        "Social media content moderation environment for training AI agents. "
        "Features an asymmetric, severity-weighted reward function. "
        "Each /reset returns an `episode_id` — use it as `session_id` in /step "
        "for concurrency-safe routing.\n\n"
        "**New endpoints:** GET /tasks · GET /grader · POST /baseline"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class StepRequest(BaseModel):
    action: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    session_id: Optional[str] = None


class BaselineResult(BaseModel):
    agent: str
    scores: dict
    average_score: float
    note: str


# ---------------------------------------------------------------------------
# Core OpenEnv routes
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=ContentObservation, summary="Start a new episode")
async def reset(
    task_name: str = Query(
        default="easy_moderation",
        description="easy_moderation | medium_moderation | hard_moderation",
    )
) -> ContentObservation:
    if task_name not in VALID_TASKS:
        raise HTTPException(400, detail=f"Invalid task. Valid: {VALID_TASKS}")
    _prune_old_sessions()
    try:
        env = ContentModerationEnvironment()
        obs = env.reset(task_name=task_name)
        sid = obs.episode_id
        active_sessions[sid] = env
        _session_timestamps[sid] = time.time()
        global _default_env
        _default_env = env
        logger.info("Reset | task=%s | session=%s | total_sessions=%d",
                    task_name, sid, len(active_sessions))
        return obs
    except Exception as exc:
        logger.exception("Error in /reset")
        raise HTTPException(500, detail=str(exc)) from exc


@app.post("/step", response_model=StepResult, summary="Submit enforcement decision")
async def step(request: StepRequest) -> StepResult:
    valid = ["remove", "restrict", "label", "escalate", "allow"]
    if request.action not in valid:
        raise HTTPException(400, detail=f"Invalid action. Valid: {valid}")
    env = _get_env(request.session_id)
    if request.session_id:
        _session_timestamps[request.session_id] = time.time()
    try:
        action = ContentModerationAction(
            action=request.action,  # type: ignore[arg-type]
            confidence=request.confidence,
            reasoning=request.reasoning,
        )
        result = env.step(action)
        logger.info("Step | session=%s | action=%s | reward=%.2f | done=%s",
                    request.session_id or "default", request.action,
                    result.reward, result.done)
        return result
    except RuntimeError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error in /step")
        raise HTTPException(500, detail=str(exc)) from exc


@app.get("/state", response_model=EnvState, summary="Current episode state")
async def state(session_id: Optional[str] = Query(default=None)) -> EnvState:
    env = _get_env(session_id)
    try:
        return env.state()
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc


@app.get("/health", summary="Liveness probe")
async def health():
    return {
        "status": "healthy",
        "environment": "content_moderation",
        "version": "2.0.0",
        "active_sessions": len(active_sessions),
    }


# ---------------------------------------------------------------------------
# Quality-of-life / documentation endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks", summary="List all available tasks")
async def list_tasks():
    return {
        "tasks": TASK_DESCRIPTIONS,
        "valid_task_names": VALID_TASKS,
        "action_space": ["remove", "restrict", "label", "escalate", "allow"],
    }


@app.get("/grader", summary="Asymmetric reward rubric")
async def grader_docs():
    return REWARD_RUBRIC


@app.post("/baseline", response_model=BaselineResult,
          summary="Run heuristic baseline agent (no API key needed)")
async def baseline() -> BaselineResult:
    task_scores: dict = {}
    for task_name in VALID_TASKS:
        posts = TASK_DATA[task_name]
        reward_fn = STEP_REWARD_FN[task_name]
        total = 0.0
        for post in posts:
            predicted = heuristic_agent(post["content"])
            reward = reward_fn(
                predicted,
                post["label"],
                post.get("secondary"),
                post.get("severity", "low"),
                post.get("post_type", "A"),
            )
            total += reward
        task_scores[task_name] = round(total / len(posts), 4)

    avg = round(sum(task_scores.values()) / len(task_scores), 4)
    logger.info("Baseline | scores=%s | avg=%.4f", task_scores, avg)
    return BaselineResult(
        agent="heuristic_keyword_rules",
        scores=task_scores,
        average_score=avg,
        note=(
            "Rule-based heuristic baseline. No LLM required. "
            "A trained agent should significantly exceed these scores."
        ),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for [project.scripts] — required by openenv validate."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()
