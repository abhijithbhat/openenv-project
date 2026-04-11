"""
Baseline Inference Script — Content Moderation OpenEnv
=======================================================
Runs a baseline LLM agent against all three tasks over HTTP and reports
reproducible scores. Communicates with the FastAPI server running at
http://localhost:7860. Passes session_id on every /step call to ensure
correct routing in concurrent evaluation environments.

Usage:
    # 1. Start the server first (in a separate terminal):
    #    uvicorn server:app --host 0.0.0.0 --port 7860
    #
    # 2. Then run this script:
    export HF_TOKEN=hf_xxxx
    export MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
    export API_BASE_URL=https://router.huggingface.co/v1
    python inference.py
"""
from __future__ import annotations

import os
import sys
import time

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — all credentials read from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN: str = os.getenv("HF_TOKEN")           # No default — must be set explicitly
LOCAL_IMAGE_NAME: str = os.getenv("LOCAL_IMAGE_NAME")  # Optional — only needed for from_docker_image()
ENV_SERVER_URL: str = os.getenv("ENV_SERVER_URL", "http://localhost:7860")
TEMPERATURE: float = 0.0   # deterministic for reproducibility
MAX_TOKENS: int = 20       # we only need one word

TASK_NAMES = ["easy_moderation", "medium_moderation", "hard_moderation"]
VALID_ACTIONS = ["remove", "restrict", "label", "escalate", "allow"]

SYSTEM_PROMPT = """You are a senior content moderation expert at a major social media platform.

You must reason through each post using three steps:

STEP 1 — CLASSIFY: What type of content is this?
  (hate speech / coded hate / doxxing / threat / misinformation / scam / satire / testimony / credible data / neutral)

STEP 2 — ASSESS ASYMMETRY: Which error is worse in this specific case?
  - If Type A (critical violation): letting it through (false negative) = catastrophic. Remove or escalate.
  - If Type B (legitimate speech): removing it (false positive) = silencing a real voice. Allow or label.

STEP 3 — DECIDE: Choose the single best enforcement action.

Actions:
- remove   : Clear violation — hate speech, threats, dangerous misinformation, doxxing, scams.
- restrict : Borderline graphic content — reduce distribution or age-gate.
- label    : Add context warning — misleading but not outright harmful.
- escalate : Genuinely ambiguous — send to a human reviewer.
- allow    : Within policy — satire, personal opinion, testimony, credible sources.

CRITICAL RULES:
- Coded / disguised hate speech ('just asking questions', dog whistles) → remove
- Personal testimony of discrimination or hardship → almost always allow
- Accurate data from credible sources, even if alarming → allow
- Veiled threats referencing real locations or real named people → remove immediately
- Satire from verified comedy accounts → allow
- Research ABOUT extremism is not the same as promoting it → allow
- Suicide risk signals masked as poetry or fiction → escalate immediately
- Coordinated inauthentic behaviour (same post, 1000s of accounts) → remove

Format your response EXACTLY as two lines:
REASONING: [one concise sentence explaining your decision]
DECISION: [one word: remove/restrict/label/escalate/allow]"""


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def extract_action(text: str) -> tuple[str, str]:
    """
    Parse a CoT response and return (action, reasoning).

    Expected format:
        REASONING: <one sentence>
        DECISION: <one word>
    Falls back to scanning the full text for any valid action word.
    """
    reasoning = ""
    decision = ""
    for line in text.strip().split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("REASONING:"):
            reasoning = stripped[len("REASONING:"):].strip()
        elif stripped.upper().startswith("DECISION:"):
            decision = stripped[len("DECISION:"):].strip().lower().rstrip(".,!?;:")
    if decision in VALID_ACTIONS:
        return decision, reasoning
    # Fallback: scan full text for any valid action word
    for action in VALID_ACTIONS:
        if action in text.lower():
            return action, reasoning
    return "escalate", reasoning  # Safe default


def get_agent_action(client: OpenAI, post_content: str,
                     platform: str, context: str | None) -> tuple[str, str]:
    """Ask the LLM to moderate a post and return (action, reasoning)."""
    context_str = f"\nAdditional context: {context}" if context else ""
    user_message = (
        f"Platform: {platform}\n"
        f"Post content:\n\"\"\"\n{post_content}\n\"\"\""
        f"{context_str}\n\n"
        "Apply your three-step analysis and provide your REASONING and DECISION."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            max_tokens=80,   # enough for REASONING + DECISION lines
        )
        raw = response.choices[0].message.content or ""
        return extract_action(raw)
    except Exception as exc:
        print(f"    [LLM error] {exc} — defaulting to escalate")
        return "escalate", ""


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_post(endpoint: str, payload: dict) -> dict:
    url = f"{ENV_SERVER_URL}{endpoint}"
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def http_get(endpoint: str) -> dict:
    url = f"{ENV_SERVER_URL}{endpoint}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Episode runner — passes session_id on every /step call
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, task_name: str) -> float:
    """Run one full episode over HTTP and return the normalised episode score."""
    # Reset — creates a new isolated session on the server
    obs = http_post(f"/reset?task_name={task_name}", {})
    session_id = obs.get("episode_id", None)
    max_steps = obs.get("max_steps", 8)
    total_reward = 0.0
    step = 0

    # Required structured log: START (flush=True prevents Docker stdout buffering)
    print(f"[START] task={task_name} session={session_id} max_steps={max_steps}", flush=True)

    while True:
        step += 1
        post_content = obs.get("content", "")
        platform = obs.get("platform", "unknown")
        context = obs.get("context")

        action_str, reasoning = get_agent_action(client, post_content, platform, context)

        # Pass session_id so the server routes this step to our session
        # Also pass reasoning so the server can log it
        result = http_post("/step", {
            "action": action_str,
            "confidence": None,
            "reasoning": reasoning[:500] if reasoning else None,
            "session_id": session_id,
        })

        reward = result.get("reward", 0.0)
        done = result.get("done", True)
        info = result.get("info", {})
        total_reward += reward

        # Required structured log: STEP (includes agent reasoning for transparency)
        reasoning_short = reasoning[:100].replace('"', "'") if reasoning else "(no reasoning)"
        print(
            f"[STEP] step={step} action={action_str} reward={reward:.4f} "
            f"correct={info.get('correct_action', 'unknown')} "
            f'reasoning="{reasoning_short}"',
            flush=True,
        )

        if done:
            break

        obs = result.get("observation", {})

    episode_score = round(total_reward / max_steps, 4)
    # Phase 2 Task Validation requires scores strictly in (0, 1), not 0.0 or 1.0
    episode_score = max(0.0001, min(0.9999, episode_score))
    # Required structured log: END
    print(f"[END] task={task_name} score={episode_score:.4f} total_reward={total_reward:.4f} steps={step}", flush=True)
    return episode_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable not set.")
        print("Run: export HF_TOKEN=hf_xxxx")
        sys.exit(1)

    try:
        health = http_get("/health")
        assert health.get("status") == "healthy"
        print(f"[OK] Server is live at {ENV_SERVER_URL}")
        print(f"     Active sessions: {health.get('active_sessions', 0)}")
    except Exception:
        print(f"\nError: Could not reach the environment server at {ENV_SERVER_URL}")
        print("Start the server first with:")
        print("  uvicorn server:app --host 0.0.0.0 --port 7860")
        sys.exit(1)

    print(f"Model : {MODEL_NAME}")
    print(f"API   : {API_BASE_URL}")
    print(f"Tasks : {TASK_NAMES}")

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    scores: dict[str, float] = {}
    for task in TASK_NAMES:
        scores[task] = run_episode(client, task)

    print("\n" + "=" * 52)
    print("BASELINE SCORES — ContentGuard")
    print("=" * 52)
    for task, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task:<24} {score:.4f}  {bar}")
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average score:           {avg:.4f}")
    print("=" * 52)


if __name__ == "__main__":
    main()