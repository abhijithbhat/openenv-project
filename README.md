# Email Triage Environment

> **Meta × Hugging Face OpenEnv Hackathon — Round 1 Submission**

A **complete, real-world OpenEnv environment** for customer-support email triage. An AI agent classifies incoming customer emails into one of five routing categories, with three difficulty levels and meaningful partial-credit reward signals.

---

## 🎯 Environment Description

Real companies receive thousands of customer emails per day. Mis-routing a ticket wastes agent time and frustrates customers. This environment trains AI agents to accurately classify emails so they can be routed to the right team automatically.

**Domain:** Customer Support Operations  
**Task type:** Multi-class text classification with escalation detection  
**Real-world utility:** Directly applicable to any SaaS, e-commerce, or FinTech support queue

---

## 🔑 Key Features

| Feature | Details |
|---|---|
| Real-world task | Customer support email triage (not a game/toy) |
| Full OpenEnv spec | `step()` / `reset()` / `state()` + typed Pydantic models |
| 3 difficulty levels | easy → medium → hard with increasing ambiguity |
| Partial-credit rewards | Signals guide agents on multi-intent emails |
| Reproducible baseline | LLM inference script with per-task scores |
| Docker ready | FastAPI + uvicorn on port 7860 |

---

## 📐 Action Space

The agent must output **exactly one** of:

| Category | When to use |
|---|---|
| `billing` | Payment failures, invoice errors, wrong charges |
| `technical` | App crashes, API errors, login issues, bugs |
| `general` | Product questions, feature inquiries, general info |
| `refund` | Explicit refund or cancellation request |
| `complaint` | Escalated frustration, legal threats, repeated failures |

```json
{
  "category": "billing",
  "confidence": 0.95
}
```

---

## 👁️ Observation Space

Each step returns an email to classify:

```json
{
  "email_id": "m003",
  "subject": "Cancel and Refund Request",
  "body": "I want to cancel my subscription and receive a full refund...",
  "task_name": "medium_triage",
  "step": 3,
  "max_steps": 5,
  "available_categories": ["billing", "technical", "general", "refund", "complaint"]
}
```

---

## 🏆 Tasks & Reward Function

### Task 1: `easy_triage` (Easy)
- 5 single-intent emails with clear keywords
- **Scoring:** 1.0 correct / 0.0 wrong (binary)
- Expected baseline score: ~0.80–1.00

### Task 2: `medium_triage` (Medium)
- 5 emails with a primary label and a valid secondary label
- **Scoring:** 1.0 primary / 0.5 secondary / 0.0 wrong
- Expected baseline score: ~0.60–0.85

### Task 3: `hard_triage` (Hard)
- 5 complex, multi-intent, or escalated emails
- **Scoring:** 1.0 primary / 0.4 secondary / 0.2 escalation-detected / 0.0 wrong
- Expected baseline score: ~0.40–0.70

The reward function provides **partial progress signals** throughout the episode rather than only at the end. This enables RL algorithms to learn incrementally.

---

## 🔌 API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode. Param: `task_name` |
| `POST` | `/step` | Take a classification action |
| `GET` | `/state` | Get current episode metadata |
| `GET` | `/tasks` | List all available tasks |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive API documentation |

---

## 🚀 Setup & Usage

### Prerequisites
- Python 3.10+
- Docker (for containerised deployment)
- Hugging Face account + token

### Local Development (without Docker)

```bash
# 1. Clone the repo
git clone https://github.com/your-team/openenv-project.git
cd openenv-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn server:app --host 0.0.0.0 --port 7860 --reload

# 4. Test in another terminal
curl -X POST http://localhost:7860/reset?task_name=easy_triage
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"category": "billing", "confidence": 0.9}'
curl http://localhost:7860/state
```

### Run Inference Script

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
export HF_TOKEN="hf_your_token_here"

python inference.py
```

### Docker Build & Run

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 \
  -e HF_TOKEN="${HF_TOKEN}" \
  email-triage-env
```

---

## 📊 Baseline Scores

Run with `meta-llama/Meta-Llama-3-8B-Instruct` via HuggingFace Router:

| Task | Score |
|---|---|
| easy_triage | ~0.90 |
| medium_triage | ~0.70 |
| hard_triage | ~0.52 |
| **Overall** | **~0.71** |

*(Exact scores vary by model. Run `python inference.py` to reproduce.)*

---

## 📁 Project Structure

```
openenv-project/
├── server.py          # FastAPI server (POST /reset, /step, GET /state)
├── environment.py     # Core environment logic (reset/step/state)
├── models.py          # Typed Pydantic models (Action, Observation, State)
├── graders.py         # Task graders + email datasets
├── inference.py       # Baseline inference script (OpenAI client)
├── openenv.yaml       # OpenEnv manifest
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container definition
└── README.md          # This file
```

---

## 🧠 Design Decisions

1. **Why email triage?** It's a concrete, measurable task every company faces — not contrived.
2. **Why 5 emails per task?** Balances evaluation speed (< 20 min inference budget) with statistical validity.
3. **Why partial credit on hard tasks?** Multi-intent tickets genuinely belong to multiple queues. Binary scoring would punish reasonable predictions.
4. **Why the `complaint` category?** Escalation detection is a real operational need — it triggers SLA escalation workflows in production systems.
5. **Why FastAPI?** It's the standard for OpenEnv deployments and generates automatic `/docs` for transparency.

---

## ⚙️ Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | LLM API endpoint |
| `MODEL_NAME` | Yes | Model identifier |
| `HF_TOKEN` | Yes | HuggingFace / API key |

---

## 📜 License

MIT License