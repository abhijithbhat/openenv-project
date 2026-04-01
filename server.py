"""
FastAPI Server — Email Triage OpenEnv Environment
==================================================
Exposes the full OpenEnv HTTP interface:

    POST /reset   → EmailObservation   (start a new episode)
    POST /step    → StepResult         (take one classification action)
    GET  /state   → EnvState           (episode metadata)
    GET  /tasks   → list of available tasks
    GET  /health  → health check
    GET  /        → environment info

The pre-submission validator (validate-submission.sh) pings POST /reset
and expects HTTP 200. This server must be running for validation to pass.

Start locally:
    uvicorn server:app --host 0.0.0.0 --port 7860 --reload

In Docker / HuggingFace Spaces:
    CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from environment import EmailTriageEnvironment
from models import EmailAction, EmailObservation, EnvState, StepResult

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Email Triage Environment",
    description=(
        "Real-world customer support email triage environment built on the "
        "OpenEnv specification. Supports step() / reset() / state() API."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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
# (single-session for hackathon evaluation; one episode at a time)
# ---------------------------------------------------------------------------

_env = EmailTriageEnvironment()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["info"])
def root() -> dict:
    """Environment info and available endpoints."""
    return {
        "environment": "email-triage-env",
        "version": "1.0.0",
        "description": (
            "Real-world customer support email triage environment. "
            "An AI agent classifies customer emails into billing / technical / "
            "general / refund / complaint categories."
        ),
        "endpoints": {
            "POST /reset":  "Start a new episode",
            "POST /step":   "Take a classification action",
            "GET  /state":  "Get current episode metadata",
            "GET  /tasks":  "List available tasks",
            "GET  /health": "Health check",
        },
        "tasks": ["easy_triage", "medium_triage", "hard_triage"],
        "action_space": ["billing", "technical", "general", "refund", "complaint"],
    }


@app.get("/health", tags=["info"])
def health() -> dict:
    """Simple health check — returns 200 OK when the server is live."""
    return {"status": "healthy"}


@app.get("/tasks", tags=["info"])
def list_tasks() -> dict:
    """List all available tasks with descriptions."""
    return {
        "tasks": [
            {
                "name": "easy_triage",
                "difficulty": "easy",
                "description": "5 simple, single-intent customer support emails. Binary scoring.",
                "n_emails": 5,
                "scoring": "1.0 for correct category, 0.0 otherwise.",
            },
            {
                "name": "medium_triage",
                "difficulty": "medium",
                "description": "5 emails with primary + secondary valid labels. Partial credit.",
                "n_emails": 5,
                "scoring": "1.0 for primary, 0.5 for secondary, 0.0 otherwise.",
            },
            {
                "name": "hard_triage",
                "difficulty": "hard",
                "description": "5 complex, multi-intent emails. Escalation-aware scoring.",
                "n_emails": 5,
                "scoring": "1.0 primary, 0.4 secondary, 0.2 escalation-detected, 0.0 wrong.",
            },
        ]
    }


@app.post(
    "/reset",
    response_model=EmailObservation,
    tags=["openenv"],
    summary="Reset the environment and start a new episode.",
)
def reset(
    task_name: Optional[str] = Query(
        default="easy_triage",
        description="Task to run: easy_triage | medium_triage | hard_triage",
    )
) -> EmailObservation:
    """
    Reset the environment.

    - Initialises a new episode for the given task.
    - Returns the first EmailObservation (the first email to classify).
    - Must be called before any step().

    This endpoint is pinged by the OpenEnv validator — it MUST return HTTP 200.
    """
    valid = ("easy_triage", "medium_triage", "hard_triage")
    if task_name not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_name '{task_name}'. Must be one of {valid}.",
        )

    try:
        obs = _env.reset(task_name=task_name)
        logger.info("Episode reset. task=%s episode_id=%s", task_name, _env.state().episode_id)
        return obs
    except Exception as exc:
        logger.exception("reset() error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/step",
    response_model=StepResult,
    tags=["openenv"],
    summary="Execute one classification action.",
)
def step(action: EmailAction) -> StepResult:
    """
    Take one step in the current episode.

    - Accepts an EmailAction (category + optional confidence).
    - Returns a StepResult with the next observation, reward, and done flag.
    - `info` contains the correct label and reward explanation for debugging.
    - When `done=True`, call /reset to start a new episode.
    """
    try:
        result = _env.step(action)
        logger.info(
            "Step taken. category=%s reward=%.2f done=%s",
            action.category,
            result.reward,
            result.done,
        )
        return result
    except Exception as exc:
        logger.exception("step() error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/state",
    response_model=EnvState,
    tags=["openenv"],
    summary="Get current episode state / metadata.",
)
def state() -> EnvState:
    """
    Return current episode metadata.

    Includes episode_id, step_count, total_reward, task_name, and difficulty.
    Safe to call at any time.
    """
    try:
        return _env.state()
    except Exception as exc:
        logger.exception("state() error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
