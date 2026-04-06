"""
server/app.py — Content Moderation OpenEnv
===========================================
Entry point for the FastAPI server. Required by openenv validate.
Imports and re-exports the app from the root server module.
"""
from __future__ import annotations

import sys
import os

# Ensure the project root is on the path so we can import environment, models, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
# Request / Response schemas
# ---------------------------------------------------------------------------

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
        logger.info("Episode started | task=%s", task_name)
        return obs
    except Exception as exc:
        logger.exception("Error in /reset")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step", response_model=StepResult, summary="Submit an enforcement decision")
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
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error in /step")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state", response_model=EnvState, summary="Get current episode state")
async def state() -> EnvState:
    try:
        return env.state()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health", summary="Health check")
async def health():
    return {"status": "healthy", "environment": "content_moderation"}


def start():
    """Entry point for project.scripts."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    start()
