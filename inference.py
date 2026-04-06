"""
Baseline Inference Script — Content Moderation OpenEnv
=======================================================
Runs a baseline LLM agent (via HuggingFace Router) against all three tasks
and reports reproducible scores.

Usage:
    export HF_TOKEN=hf_xxxx
    export MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct   # or any OpenAI-compat model
    export API_BASE_URL=https://router.huggingface.co/v1     # default
    python inference.py

The script calls a local server, so start the server first:
    uvicorn server:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.error

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — all credentials read from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
ENV_SERVER_URL: str = os.getenv("ENV_SERVER_URL", "http://localhost:7860")
TEMPERATURE: float = 0.0   # deterministic for reproducibility
MAX_TOKENS: int = 20       # we only need one word

TASK_NAMES = ["easy_moderation", "medium_moderation", "hard_moderation"]
VALID_ACTIONS = ["remove", "restrict", "label", "escalate", "allow"]

SYSTEM_PROMPT = """You are a content moderation expert for a major social media platform.
Your job is to review posts and decide on the appropriate enforcement action.

Actions:
- remove   : Take the post down completely. Use for clear violations (hate speech, threats, dangerous misinformation, spam).
- restrict : Reduce the post's distribution or add an age gate. Use for borderline or graphic content.
- label    : Add a warning or context label. Use for content that is misleading but not outright harmful.
- escalate : Send to a human reviewer. Use when you are genuinely unsure.
- allow    : No action needed. Content is within policy.

IMPORTANT GUIDELINES:
- Satire, comedy, and opinion — even if harsh — are generally allowed.
- Personal testimony of discrimination or hardship should almost always be allowed.
- Coded or disguised hate speech (using euphemisms, 'just asking questions') should be removed.
- Accurate information from credible sources should be allowed, even if it sounds alarming.
- Veiled threats — especially involving real locations — should be removed immediately.

Respond with ONLY one word: remove, restrict, label, escalate, or allow."""


# ---------------------------------------------------------------------------
# HTTP helpers — calling the local environment server
# ---------------------------------------------------------------------------

def _http_post(url: str, data: dict) -> dict:
    """POST JSON to a URL, return parsed response."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _http_get(url: str) -> dict:
    """GET a URL, return parsed response."""
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def extract_action(text: str) -> str:
    """Extract the action word from the model's response."""
    clean = text.strip().lower().split()[0] if text.strip() else ""
    # Remove punctuation
    clean = clean.rstrip(".,!?;:")
    if clean in VALID_ACTIONS:
        return clean
    # Fuzzy fallback
    for action in VALID_ACTIONS:
        if action in text.lower():
            return action
    return "escalate"  # Safe fallback — send to human when unsure


def get_agent_action(client: OpenAI, post_content: str,
                     platform: str, context: str | None) -> str:
    """Ask the LLM to moderate a post and return a valid action."""
    context_str = f"\nAdditional context: {context}" if context else ""
    user_message = (
        f"Platform: {platform}\n"
        f"Post content:\n\"\"\"\n{post_content}\n\"\"\""
        f"{context_str}\n\n"
        "What is your enforcement decision?"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = response.choices[0].message.content or ""
        return extract_action(raw)
    except Exception as exc:
        print(f"    [LLM error] {exc} — defaulting to escalate")
        return "escalate"


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, task_name: str) -> float:
    """Run one full episode and return the normalised episode score."""
    print(f"\n  Running task: {task_name}")

    # Reset environment
    obs_data = _http_post(f"{ENV_SERVER_URL}/reset?task_name={task_name}", {})

    total_reward = 0.0
    step = 0

    while True:
        step += 1
        post_content = obs_data.get("content", "")
        platform = obs_data.get("platform", "unknown")
        context = obs_data.get("context")
        max_steps = obs_data.get("max_steps", 5)

        action = get_agent_action(client, post_content, platform, context)
        print(f"    Step {step}/{max_steps} | action={action:<10}", end="")

        result = _http_post(f"{ENV_SERVER_URL}/step", {"action": action})
        reward = result.get("reward", 0.0)
        done = result.get("done", True)
        info = result.get("info", {})

        total_reward += reward
        print(f"| reward={reward:.2f} | correct={info.get('correct_action', '?')}")

        if done:
            break
        obs_data = result["observation"]
        time.sleep(0.3)  # avoid rate limiting

    episode_score = round(total_reward / max_steps, 4)
    print(f"  Episode score: {episode_score:.4f} ({total_reward:.1f}/{max_steps})")
    return episode_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Validate credentials
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not set.")
        print("Export your HuggingFace token: export HF_TOKEN=hf_xxxx")
        sys.exit(1)

    # Check server is up
    try:
        health = _http_get(f"{ENV_SERVER_URL}/health")
        assert health.get("status") == "healthy"
        print(f"Server is live at {ENV_SERVER_URL}")
    except Exception:
        print(f"Error: Could not reach environment server at {ENV_SERVER_URL}")
        print("Start the server with: uvicorn server:app --host 0.0.0.0 --port 7860")
        sys.exit(1)

    print(f"\nModel: {MODEL_NAME}")
    print(f"API:   {API_BASE_URL}")
    print(f"Tasks: {TASK_NAMES}")

    # Initialise OpenAI client pointing at HuggingFace Router
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    # Run all tasks
    scores: dict[str, float] = {}
    for task in TASK_NAMES:
        scores[task] = run_episode(client, task)

    # Summary
    print("\n" + "=" * 50)
    print("BASELINE SCORES")
    print("=" * 50)
    for task, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task:<22} {score:.4f}  {bar}")
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average score:         {avg:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()