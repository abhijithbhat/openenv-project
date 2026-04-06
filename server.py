"""
FastAPI Server — Content Moderation OpenEnv Environment
=========================================================
Exposes the standard OpenEnv API:
  POST /reset   → Start new episode, returns first post
  POST /step    → Submit enforcement decision, returns reward + next post
  GET  /state   → Current episode metadata
  GET  /health  → Liveness probe (required for HuggingFace Spaces)
  GET  /docs    → Auto-generated OpenAPI docs (FastAPI built-in)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import ContentModerationEnvironment, VALID_TASKS
from models import ContentModerationAction, ContentObservation, EnvState, StepResult

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Content Moderation Environment starting up.")
    yield
    logger.info("Content Moderation Environment shutting down.")

app = FastAPI(
    title="Content Moderation OpenEnv",
    description=(
        "A social media content moderation environment for training and evaluating "
        "AI agents. Features an asymmetric, severity-weighted reward function — "
        "missing hate speech is far worse than flagging satire."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow cross-origin requests (required for HuggingFace Spaces)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global environment instance
# ---------------------------------------------------------------------------

env = ContentModerationEnvironment()

# ---------------------------------------------------------------------------
# Request / Response schemas (thin wrappers for OpenAPI docs)
# ---------------------------------------------------------------------------

class ResetResponse(BaseModel):
    observation: ContentObservation

class StepRequest(BaseModel):
    action: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post(
    "/reset",
    response_model=ContentObservation,
    summary="Start a new episode",
    description=(
        "Resets the environment and returns the first post to review. "
        f"Valid task names: {VALID_TASKS}"
    ),
)
async def reset(
    task_name: str = Query(
        default="easy_moderation",
        description="Task difficulty: easy_moderation | medium_moderation | hard_moderation",
    )
) -> ContentObservation:
    if task_name not in VALID_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{task_name}'. Valid tasks: {VALID_TASKS}",
        )
    try:
        obs = env.reset(task_name=task_name)
        logger.info("Episode started | task=%s | episode_id=%s", task_name, env.state().episode_id)
        return obs
    except Exception as exc:
        logger.exception("Error in /reset")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/step",
    response_model=StepResult,
    summary="Submit an enforcement decision",
    description=(
        "Submit your action for the current post. "
        "Valid actions: remove | restrict | label | escalate | allow. "
        "Returns the step reward (asymmetric on hard tasks), done flag, "
        "and the next post to review."
    ),
)
async def step(request: StepRequest) -> StepResult:
    valid_actions = ["remove", "restrict", "label", "escalate", "allow"]
    if request.action not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{request.action}'. Valid actions: {valid_actions}",
        )
    try:
        action = ContentModerationAction(
            action=request.action,  # type: ignore[arg-type]
            confidence=request.confidence,
            reasoning=request.reasoning,
        )
        result = env.step(action)
        logger.info(
            "Step | action=%s | reward=%.2f | done=%s",
            request.action, result.reward, result.done,
        )
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error in /step")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/state",
    response_model=EnvState,
    summary="Get current episode state",
    description="Returns episode metadata: step count, cumulative reward, task name.",
)
async def state() -> EnvState:
    try:
        return env.state()
    except Exception as exc:
        logger.exception("Error in /state")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/health",
    summary="Health check",
    description="Liveness probe. Returns 200 if the server is running.",
)
async def health():
    return {"status": "healthy", "environment": "content_moderation"}
