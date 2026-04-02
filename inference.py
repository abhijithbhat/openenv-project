"""
Inference Script — Email Triage Environment
============================================
MANDATORY requirements (from hackathon rules):
  - Named `inference.py` in the root directory.
  - Uses OpenAI Client for all LLM calls.
  - Reads API credentials from environment variables:
        API_BASE_URL   The LLM API endpoint.
        MODEL_NAME     The model identifier.
        HF_TOKEN       Your Hugging Face / API key.
  - Runs all 3 tasks and produces reproducible baseline scores.
  - Runtime < 20 min on vcpu=2, memory=8GB.

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
    export HF_TOKEN="hf_your_token_here"
    python inference.py

The script runs against the environment DIRECTLY (no HTTP required)
so it works both locally and inside the Docker container.
"""
from __future__ import annotations

import os
import sys
import textwrap
from typing import List, Literal

from openai import OpenAI

from environment import EmailTriageEnvironment
from models import EmailAction

# ---------------------------------------------------------------------------
# Configuration — all credentials read from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

MAX_RETRIES: int = 3
TEMPERATURE: float = 0.0   # deterministic for reproducibility
MAX_TOKENS: int = 20        # we only need one word

CATEGORIES = ("billing", "technical", "general", "refund", "complaint")

FALLBACK_CATEGORY: Literal["general"] = "general"

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a customer-support routing assistant.
    Your task is to classify each customer email into EXACTLY ONE of these categories:
      billing   — payment issues, invoices, charges, subscriptions, refund requests involving money
      technical — bugs, crashes, errors, API issues, performance problems, login failures
      general   — product questions, feature inquiries, general information requests
      refund    — explicit refund request without strong technical or billing context
      complaint — escalated frustration, threats to leave, legal action, repeated failures

    Rules:
    - Reply with ONLY the single category word. No punctuation. No explanation.
    - If unsure between two categories, pick the one that best describes what the customer needs FIRST.
    - 'complaint' is for highly escalated, threatening, or multi-failure emails.
    """
).strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_user_prompt(subject: str, body: str, step: int, max_steps: int) -> str:
    return textwrap.dedent(
        f"""
        Email {step}/{max_steps}

        Subject: {subject}

        Body:
        {body}

        Classify this email. Reply with exactly one word from:
        billing | technical | general | refund | complaint
        """
    ).strip()


def parse_category(text: str) -> str:
    """Extract the first valid category word from model output."""
    cleaned = text.strip().lower().split()[0] if text.strip() else ""
    # Strip trailing punctuation
    cleaned = cleaned.rstrip(".,!?:")
    if cleaned in CATEGORIES:
        return cleaned
    # Fuzzy fallback: check if any category appears in the full text
    for cat in CATEGORIES:
        if cat in text.lower():
            return cat
    return FALLBACK_CATEGORY


def call_llm(client: OpenAI, subject: str, body: str, step: int, max_steps: int) -> str:
    """Call the LLM and return a validated category string."""
    user_prompt = build_user_prompt(subject, body, step, max_steps)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            raw = completion.choices[0].message.content or ""
            category = parse_category(raw)
            return category
        except Exception as exc:  # noqa: BLE001
            print(f"  [attempt {attempt}/{MAX_RETRIES}] LLM call failed: {exc}")
            if attempt == MAX_RETRIES:
                print(f"  Falling back to '{FALLBACK_CATEGORY}'.")
                return FALLBACK_CATEGORY

    return FALLBACK_CATEGORY


def run_task(client: OpenAI, env: EmailTriageEnvironment, task_name: str) -> float:
    """Run one full task episode and return the normalised score (0.0–1.0)."""
    print(f"\n{'='*55}")
    print(f"  TASK: {task_name.upper()}")
    print(f"{'='*55}")

    obs = env.reset(task_name=task_name)
    total_reward = 0.0
    step_rewards: List[float] = []

    while True:
        print(f"\n  Step {obs.step}/{obs.max_steps}")
        print(f"  Subject : {obs.subject}")
        print(f"  Body    : {obs.body[:120]}{'...' if len(obs.body) > 120 else ''}")

        category = call_llm(client, obs.subject, obs.body, obs.step, obs.max_steps)
        action = EmailAction(category=category)  # type: ignore[arg-type]

        result = env.step(action)
        reward = result.reward
        total_reward += reward
        step_rewards.append(reward)

        info = result.info
        print(f"  Predicted : {info.get('predicted', category)}")
        print(f"  Correct   : {info.get('correct', '?')}")
        print(f"  Reward    : {reward:.2f}  — {info.get('step_reward_explanation', '')}")

        if result.done:
            break

        obs = result.observation  # type: ignore[assignment]

    # Normalised episode score
    n = len(step_rewards)
    episode_score = round(total_reward / n, 4) if n else 0.0

    print(f"\n  ── Task score: {episode_score:.4f} "
          f"({total_reward:.1f}/{n}) ──")
    return episode_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Validate credentials
    if not API_KEY:
        print(
            "ERROR: HF_TOKEN (or API_KEY) environment variable is not set.\n"
            "Export it before running:\n"
            "  export HF_TOKEN=hf_your_token_here"
        )
        sys.exit(1)

    print("\n" + "="*55)
    print("  Email Triage Environment — Baseline Inference")
    print("="*55)
    print(f"  API_BASE_URL : {API_BASE_URL}")
    print(f"  MODEL_NAME   : {MODEL_NAME}")
    print(f"  HF_TOKEN     : {API_KEY[:8]}{'*' * (len(API_KEY) - 8) if len(API_KEY) > 8 else '***'}")

    # Initialise OpenAI client (pointing to HuggingFace or any OpenAI-compatible endpoint)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EmailTriageEnvironment()

    scores: dict = {}
    tasks = ["easy_triage", "medium_triage", "hard_triage"]

    for task in tasks:
        scores[task] = run_task(client, env, task)

    # Summary
    print("\n" + "="*55)
    print("  BASELINE SCORES")
    print("="*55)
    for task, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task:<16} │ {bar} │ {score:.4f}")

    overall = round(sum(scores.values()) / len(scores), 4)
    print(f"\n  Overall average : {overall:.4f}")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()