"""
Baseline Inference Script — Content Moderation OpenEnv
=======================================================
Runs a baseline LLM agent against all three tasks and reports
reproducible scores.

The environment is imported DIRECTLY — no running server required.
This guarantees the script works in any sterile execution context.

Usage:
    export HF_TOKEN=hf_xxxx
    export MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct   # or any OpenAI-compat model
    export API_BASE_URL=https://router.huggingface.co/v1     # default
    python inference.py
"""
from __future__ import annotations

import os
import sys

from openai import OpenAI

# Import environment directly — no HTTP server required
from environment import ContentModerationEnvironment
from graders import TASK_DATA
from models import ContentModerationAction

# ---------------------------------------------------------------------------
# Configuration — all credentials read from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
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
# Agent helpers
# ---------------------------------------------------------------------------

def extract_action(text: str) -> str:
    """Extract a valid action word from the model's response."""
    clean = text.strip().lower().split()[0] if text.strip() else ""
    clean = clean.rstrip(".,!?;:")
    if clean in VALID_ACTIONS:
        return clean
    # Fuzzy fallback — check if any action appears anywhere in the response
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
# Episode runner — uses environment directly (no HTTP server needed)
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, env: ContentModerationEnvironment,
                task_name: str) -> float:
    """Run one full episode and return the normalised episode score."""
    print(f"\n  Running task: {task_name}")

    obs = env.reset(task_name=task_name)
    max_steps = obs.max_steps
    total_reward = 0.0
    step = 0

    while True:
        step += 1
        action_str = get_agent_action(
            client, obs.content, obs.platform, obs.context
        )
        print(f"    Step {step}/{max_steps} | action={action_str:<10}", end="")

        result = env.step(ContentModerationAction(action=action_str))  # type: ignore[arg-type]
        total_reward += result.reward
        correct = result.info.get("correct_action", "?")
        print(f"| reward={result.reward:.2f} | correct={correct}")

        if result.done:
            break
        obs = result.observation  # type: ignore[assignment]

    episode_score = round(total_reward / max_steps, 4)
    print(f"  Episode score: {episode_score:.4f}  ({total_reward:.1f}/{max_steps})")
    return episode_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Validate credentials
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not set.")
        print("Export your HuggingFace token: export HF_TOKEN=hf_xxxx")
        sys.exit(1)

    print(f"Model : {MODEL_NAME}")
    print(f"API   : {API_BASE_URL}")
    print(f"Tasks : {TASK_NAMES}")

    # Initialise OpenAI client (pointing at HuggingFace Router)
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    # Single environment instance reused across all tasks
    env = ContentModerationEnvironment()

    # Run all tasks and collect scores
    scores: dict[str, float] = {}
    for task in TASK_NAMES:
        scores[task] = run_episode(client, env, task)

    # Print summary
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