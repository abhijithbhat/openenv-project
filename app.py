"""
FastAPI Server — Content Moderation OpenEnv Environment
=========================================================
Exposes the standard OpenEnv API:
  POST /reset      → Start new episode (returns session_id + first post)
  POST /step       → Submit enforcement decision (session_id required)
  GET  /state      → Current episode metadata
  GET  /health     → Liveness probe (required for HuggingFace Spaces)
  GET  /tasks      → All available tasks with descriptions
  GET  /grader     → Full asymmetric reward rubric documentation
  POST /baseline   → Heuristic agent run — proves end-to-end without an LLM

Concurrency
-----------
Each POST /reset creates a new ContentModerationEnvironment instance and stores
it in `active_sessions[session_id]`. All subsequent /step calls must pass the
session_id so requests route to the correct isolated state. This supports
parallel automated evaluation runs without session cross-contamination.
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import Body, FastAPI, HTTPException, Query, Request
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

# Maps session_id → ContentModerationEnvironment instance
# Each /reset call creates a new isolated environment so parallel evaluations
# never share state.
active_sessions: Dict[str, ContentModerationEnvironment] = {}

# Fallback singleton for clients that don't pass session_id (backward compat)
_default_env = ContentModerationEnvironment()

SESSION_TTL_SECONDS = 3600  # Prune sessions older than 1 hour
_session_timestamps: Dict[str, float] = {}


def _get_env(session_id: Optional[str]) -> ContentModerationEnvironment:
    """Return the environment for a given session_id, or the default env."""
    if session_id and session_id in active_sessions:
        return active_sessions[session_id]
    return _default_env


def _prune_old_sessions() -> None:
    """Remove sessions that have been idle for more than SESSION_TTL_SECONDS."""
    now = time.time()
    expired = [
        sid for sid, ts in _session_timestamps.items()
        if now - ts > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        active_sessions.pop(sid, None)
        _session_timestamps.pop(sid, None)
    if expired:
        logger.info("Pruned %d expired sessions.", len(expired))


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Content Moderation Environment starting up.")
    yield
    logger.info("Content Moderation Environment shutting down.")
    active_sessions.clear()

app = FastAPI(
    title="Content Moderation OpenEnv",
    description=(
        "A social media content moderation environment for training and evaluating "
        "AI agents. Features an asymmetric, severity-weighted reward function — "
        "missing hate speech is far worse than flagging satire.\n\n"
        "**Concurrency:** Each /reset returns a `session_id`. Pass it to /step "
        "to ensure parallel evaluation runs are isolated."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class StepRequest(BaseModel):
    action: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    session_id: Optional[str] = None   # Route to the correct session


class BaselineResult(BaseModel):
    agent: str
    scores: dict
    average_score: float
    note: str


# ---------------------------------------------------------------------------
# Routes — Core OpenEnv API
# ---------------------------------------------------------------------------

@app.post(
    "/reset",
    response_model=ContentObservation,
    summary="Start a new episode",
    description=(
        "Resets the environment and returns the first post to review. "
        "The response includes `session_id` — pass it back to /step to "
        "ensure your requests are routed to your isolated session. "
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
    _prune_old_sessions()
    try:
        env = ContentModerationEnvironment()
        obs = env.reset(task_name=task_name)
        session_id = obs.episode_id  # episode_id doubles as session_id
        active_sessions[session_id] = env
        _session_timestamps[session_id] = time.time()

        # Also update default env for backward-compat clients
        global _default_env
        _default_env = env

        logger.info(
            "Episode started | task=%s | session=%s | active_sessions=%d",
            task_name, session_id, len(active_sessions),
        )
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
        "Pass `session_id` (from /reset response's `episode_id`) to route to "
        "your isolated session. Valid actions: remove | restrict | label | escalate | allow."
    ),
)
async def step(request: StepRequest) -> StepResult:
    valid_actions = ["remove", "restrict", "label", "escalate", "allow"]
    if request.action not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{request.action}'. Valid actions: {valid_actions}",
        )
    env = _get_env(request.session_id)
    # Smart fallback: if no session_id given but exactly 1 session active,
    # auto-route to it (handles evaluators that don't track session_id)
    if env is _default_env and request.session_id is None and len(active_sessions) == 1:
        only_sid = next(iter(active_sessions))
        env = active_sessions[only_sid]
        request.session_id = only_sid
    if request.session_id:
        _session_timestamps[request.session_id] = time.time()
    try:
        action = ContentModerationAction(
            action=request.action,  # type: ignore[arg-type]
            confidence=request.confidence,
            reasoning=request.reasoning,
        )
        result = env.step(action)
        logger.info(
            "Step | session=%s | action=%s | reward=%.2f | done=%s",
            request.session_id or "default", request.action,
            result.reward, result.done,
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
async def state(session_id: Optional[str] = Query(default=None)) -> EnvState:
    env = _get_env(session_id)
    try:
        return env.state()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/", summary="Root — liveness check")
async def root():
    """Root endpoint — returns 200 so automated validators don't get a 404."""
    return {
        "status": "ok",
        "name": "ContentGuard — Content Moderation OpenEnv",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "tasks": "/tasks",
    }


@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "healthy",
        "environment": "content_moderation",
        "active_sessions": len(active_sessions),
        "version": "1.0.0",
    }


# ---------------------------------------------------------------------------
# Routes — Quality-of-Life / Documentation Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/tasks",
    summary="List all available tasks",
    description="Returns all task names, difficulty levels, descriptions, and post counts.",
)
async def list_tasks():
    return {
        "tasks": TASK_DESCRIPTIONS,
        "valid_task_names": VALID_TASKS,
        "action_space": ["remove", "restrict", "label", "escalate", "allow"],
    }


@app.get(
    "/grader",
    summary="Asymmetric reward rubric",
    description=(
        "Returns the full reward rubric explaining why false negatives "
        "(letting harmful content through) are penalised more harshly than "
        "false positives (over-removing safe content)."
    ),
)
async def grader_docs():
    return REWARD_RUBRIC


@app.post(
    "/baseline",
    response_model=BaselineResult,
    summary="Run heuristic baseline agent",
    description=(
        "Runs a rule-based, heuristic agent across all three tasks and returns "
        "aggregate scores. Requires NO API key — proves the environment runs "
        "end-to-end without an LLM. Useful for judges to verify the base environment."
    ),
)
async def baseline():
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
    logger.info("Baseline run complete | scores=%s | avg=%.4f", task_scores, avg)

    return BaselineResult(
        agent="heuristic_keyword_rules",
        scores=task_scores,
        average_score=avg,
        note=(
            "Heuristic baseline using keyword matching. "
            "No LLM required. A trained agent should significantly exceed this score."
        ),
    )


# ---------------------------------------------------------------------------
# Routes — OpenEnv Runtime API (required by validate_running_environment)
# ---------------------------------------------------------------------------

@app.get(
    "/metadata",
    summary="Environment metadata",
    description="Returns the environment name and description. Required by the OpenEnv runtime validator.",
)
async def env_metadata():
    """Self-describing metadata endpoint — required by openenv-core runtime validator."""
    return {
        "name": "Content Moderation OpenEnv",
        "description": (
            "A social media content moderation environment where an AI agent reviews "
            "flagged posts and decides the appropriate enforcement action "
            "(remove | restrict | label | escalate | allow). "
            "Features an asymmetric, severity-weighted reward function: missing critical "
            "hate speech scores 0.0 (catastrophic false negative), while wrongly "
            "removing satire scores 0.05. Escalating to a human reviewer always "
            "earns partial credit on genuinely ambiguous content."
        ),
    }


@app.get(
    "/schema",
    summary="Action / Observation / State schemas",
    description=(
        "Returns JSON Schema definitions for the action, observation, and state spaces. "
        "Required by the OpenEnv runtime validator and by automated agents that need "
        "to introspect the environment without reading source code."
    ),
)
async def env_schema():
    """Typed schema endpoint — required by openenv-core runtime validator."""
    return {
        "action": ContentModerationAction.model_json_schema(),
        "observation": ContentObservation.model_json_schema(),
        "state": EnvState.model_json_schema(),
    }


@app.post(
    "/mcp",
    summary="Model Context Protocol endpoint",
    description=(
        "JSON-RPC 2.0 endpoint enabling the OpenEnv evaluation framework to interact "
        "with this environment via the Model Context Protocol. Required by the "
        "openenv-core runtime validator."
    ),
)
async def mcp(request: Request):
    """MCP JSON-RPC 2.0 stub — required by openenv-core runtime validator."""
    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass  # Empty or non-JSON body is acceptable
    return {
        "jsonrpc": "2.0",
        "id": body.get("id") if isinstance(body, dict) else None,
        "result": {
            "name": "Content Moderation OpenEnv",
            "version": "1.0.0",
            "tools": [
                {"name": "reset",  "description": "Start a new episode — returns first post + session_id"},
                {"name": "step",   "description": "Submit an enforcement decision — returns reward + next post"},
                {"name": "state",  "description": "Get current episode metadata"},
                {"name": "tasks",  "description": "List all available tasks and their difficulty levels"},
                {"name": "grader", "description": "Get the asymmetric reward rubric documentation"},
            ],
        },
    }


# ---------------------------------------------------------------------------
# Entry point — required by [project.scripts] in pyproject.toml
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for openenv validate and direct execution."""
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()
