---
title: AI Email Triage Environment
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - email-triage
  - nlp
  - agent-environment
---

# 📧 AI Email Triage & Response Environment

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://openenv.dev)
[![HuggingFace Spaces](https://img.shields.io/badge/HuggingFace-Spaces-yellow)](https://huggingface.co/spaces)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)

A complete, real-world **OpenEnv-compatible** environment for training and evaluating AI agents on realistic email triage tasks. The environment simulates a corporate email inbox where agents must classify priority, route emails to departments, detect spam/phishing, compose draft responses, and select appropriate follow-up actions.

---

## 🌍 Environment Description & Motivation

Email is one of the most universally demanding cognitive tasks in the modern workplace. A professional knowledge worker receives **120+ emails per day** and must rapidly make triage decisions: What is urgent? Who should handle this? Is this a phishing attempt? What should I reply? What follow-up is needed?

This environment captures the full complexity of this task with:
- **Realistic email scenarios** drawn from common workplace patterns
- **Adversarial examples** (phishing, spam) requiring careful detection
- **Multi-dimensional decisions** that interact in non-trivial ways
- **Partial-credit rewards** that provide dense learning signals throughout an episode

Unlike toy environments, every email in this dataset reflects a scenario a real knowledge worker encounters daily.

---

## 📊 Observation Space

Each observation is a structured representation of an incoming email:

| Field | Type | Description |
|-------|------|-------------|
| `email_id` | `string` | Unique identifier |
| `from_address` | `string` | Sender email address |
| `from_name` | `string` | Sender display name |
| `to_address` | `string` | Recipient email address |
| `subject` | `string` | Email subject line |
| `body` | `string` | Full email body |
| `timestamp` | `string` | ISO 8601 timestamp |
| `attachments` | `list[Attachment]` | File attachments (name, type, size) |
| `thread_history` | `list[Message]` | Prior thread messages |
| `labels` | `list[string]` | Existing labels (e.g., `vip`, `automated`) |
| `is_reply` | `bool` | Whether this is a reply |
| `urgency_indicators` | `list[string]` | Detected urgency signals |
| `task_id` | `string` | Current task (`task1`/`task2`/`task3`) |
| `step_number` | `int` | Current position in episode |
| `total_steps` | `int` | Total emails in episode |

---

## 🎯 Action Space

Each action is a triage decision:

| Field | Type | Options | Description |
|-------|------|---------|-------------|
| `priority` | `enum` | `high`, `medium`, `low` | Urgency classification |
| `department` | `enum` | `sales`, `support`, `engineering`, `hr`, `marketing`, `legal`, `executive`, `spam` | Routing target |
| `response` | `string\|null` | Free text | Draft reply (required for task3) |
| `follow_up_actions` | `list[enum]` | `reply`, `escalate`, `schedule_meeting`, `create_ticket`, `archive`, `forward`, `flag_review`, `no_action` | Actions to take |
| `is_spam` | `bool` | — | True if spam/phishing |
| `confidence` | `float` | 0.0–1.0 | Agent confidence |
| `reasoning` | `string\|null` | Free text | Explanation for the decision |

---

## 🏆 Tasks

### Task 1 — Priority Classification `[EASY]`
**5 emails | Score weight: 20%**

Classify each email as `high`, `medium`, or `low` priority. No routing or response required. Focuses on understanding urgency signals.

**Scoring:**
- Priority label: **70%** of step reward
- Department hint: **20%**
- Spam detection: **10%**

**Expected difficulty:** An LLM zero-shot baseline scores ~0.82

---

### Task 2 — Priority + Department Routing `[MEDIUM]`
**7 emails | Score weight: 35%**

Triage emails by assigning priority and routing to the correct department. Must detect spam and phishing. Includes edge cases: VIP escalations, automated alerts, phishing attempts, regulatory notices.

**Scoring:**
- Priority label: **35%** of step reward
- Department routing: **35%**
- Spam/phishing detection: **20%**
- Follow-up action selection: **10%**

**Expected difficulty:** An LLM zero-shot baseline scores ~0.71

---

### Task 3 — Full Triage & Response `[HARD]`
**10 emails | Score weight: 45%**

Complete email triage: priority, department, spam detection, **compose a draft response**, and select follow-up actions. Emails include security breaches, investor communications, legal compliance notices, VIP escalations, and sophisticated phishing attempts.

**Scoring:**
- Priority label: **20%** of step reward
- Department routing: **20%**
- Spam/phishing detection: **15%**
- Response quality (keyword coverage + length): **25%**
- Follow-up action F1 score: **20%**

**Expected difficulty:** An LLM zero-shot baseline scores ~0.63

---

## 🎁 Reward Function

The reward function provides **dense, partial-credit signals** throughout the episode:

- **Priority score** (0.0–1.0): Exact match = 1.0; adjacent priority = 0.5; opposite = 0.0
- **Department score** (0.0–1.0): Exact = 1.0; same domain group (e.g., support↔engineering) = 0.5
- **Spam score** (0.0–1.0): Correct detection = 1.0; missed spam = 0.0; false positive = 0.3
- **Response score** (0.0–1.0): Scored on keyword coverage (60%) + response length (40%)
- **Follow-up score** (0.0–1.0): F1 score vs. ground truth action set

**Penalties** (applied before total is computed):
- Missing spam/phishing: −0.30
- High-priority demoted to low: −0.20
- Missing required escalation (task3 only): −0.10

Final reward = `weighted_sum(scores) − penalties`, clipped to [0.0, 1.0].

---

## 🚀 Setup & Usage

### Option 1: Docker (Recommended)

```bash
# Build
docker build -t email-triage-env .

# Run
docker run -p 7860:7860 -e OPENAI_API_KEY=$OPENAI_API_KEY email-triage-env

# API is available at: http://localhost:7860
```

### Option 2: Local Python

```bash
pip install -r requirements.txt
python server.py
# API available at: http://localhost:7860
```

### Interacting with the Environment

```python
import requests

BASE = "http://localhost:7860"

# 1. Reset Task 2
obs = requests.post(f"{BASE}/env/task2/reset").json()
print(f"Email: {obs['subject']}")

# 2. Submit a triage action
action = {
    "priority": "high",
    "department": "support",
    "response": "Thank you for reaching out. I'm escalating this immediately.",
    "follow_up_actions": ["escalate", "create_ticket"],
    "is_spam": False,
    "confidence": 0.95,
    "reasoning": "Critical production outage with financial impact."
}
result = requests.post(f"{BASE}/env/task2/step", json=action).json()
print(f"Reward: {result['reward']['total']}")
print(f"Feedback: {result['reward']['feedback']}")

# 3. Check state
state = requests.get(f"{BASE}/env/task2/state").json()
print(f"Progress: {state['emails_processed']}/{state['total_emails']}")
```

---

## 🤖 Baseline Inference Script

Run the baseline LLM agent against all tasks:

```bash
export OPENAI_API_KEY=sk-...

# Run all tasks
python inference.py

# Run single task
python inference.py --task task3

# Use custom model or base URL (OpenRouter, local vLLM, etc.)
OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
OPENAI_MODEL=openai/gpt-4o \
python inference.py

# Results saved to baseline_results.json
```

### Baseline Scores (gpt-4o-mini, temperature=0)

| Task | Difficulty | Score |
|------|-----------|-------|
| Task 1: Priority Classification | Easy | 0.82 |
| Task 2: Priority + Routing | Medium | 0.71 |
| Task 3: Full Triage & Response | Hard | 0.63 |
| **Weighted Overall** | — | **0.70** |

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info and task overview |
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List all task metadata |
| `GET` | `/tasks/{task_id}` | Single task info |
| `POST` | `/env/{task_id}/reset` | Start a new episode |
| `POST` | `/env/{task_id}/step` | Submit triage action |
| `GET` | `/env/{task_id}/state` | Current state |
| `POST` | `/env/{task_id}/evaluate` | Batch offline evaluation |
| `GET` | `/openenv.yaml` | OpenEnv spec file |

Interactive API docs: **http://localhost:7860/docs**

---

## 🗂️ Project Structure

```
email-triage-env/
├── Dockerfile           # Container definition (HF Spaces compatible)
├── README.md            # This file
├── openenv.yaml         # OpenEnv specification
├── requirements.txt     # Python dependencies
├── models.py            # Pydantic models (Observation, Action, Reward, EnvState)
├── rewards.py           # Multi-dimensional reward functions
├── tasks.py             # Task definitions + graders
├── environment.py       # Core OpenEnv interface (reset/step/state)
├── server.py            # FastAPI REST server
├── inference.py         # Baseline LLM inference script
└── data/
    └── emails.py        # 22 synthetic email scenarios with ground truth
```

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | For inference only | — | OpenAI API key |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | API base URL (OpenRouter, etc.) |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model name |
| `PORT` | No | `7860` | Server port |

---

## 📄 License

MIT License — See LICENSE for details.
