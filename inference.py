"""
Baseline inference script for the AI Email Triage Environment.

Uses the OpenAI-compatible API to run a model against all three tasks.

Usage:
    export HF_TOKEN=hf_...
    export API_BASE_URL=https://your-active-endpoint.com/v1
    export MODEL_NAME=gpt-4o-mini
    python inference.py

    # Run a single task:
    python inference.py --task task1

    # Optional: use a local Docker image
    LOCAL_IMAGE_NAME=email-triage-env python inference.py
"""
import argparse
import json
import os
import sys
import time
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

from environment import EmailTriageEnv
from models import Action, Priority, Department, FollowUpAction
from tasks import TASKS


# ─── Configuration (hackathon-required variable names) ───────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default — must be set

# Optional: used when running from a local Docker image
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

MAX_RETRIES = 3
RETRY_DELAY = 2.0


# ─── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert email triage assistant for a professional organization.
Your job is to analyze each incoming email and make a structured triage decision.

For each email you must respond with a valid JSON object with the following fields:

{
  "priority": "high" | "medium" | "low",
  "department": "sales" | "support" | "engineering" | "hr" | "marketing" | "legal" | "executive" | "spam",
  "response": "<draft response text, or null if not appropriate>",
  "follow_up_actions": ["reply", "escalate", "schedule_meeting", "create_ticket", "archive", "forward", "flag_review", "no_action"],
  "is_spam": true | false,
  "confidence": 0.0-1.0,
  "reasoning": "<one-sentence explanation>"
}

Priority guidelines:
- high: immediate action required (security incidents, system outages, legal deadlines, VIP escalations, C-level)
- medium: important but not urgent (feature bugs, partner requests, HR matters with upcoming deadlines)
- low: informational, routine, or can wait (newsletters, surveys, low-priority feature requests)

Department routing:
- executive: C-level matters, investor relations, M&A, board communications
- legal: compliance, audits, contracts, IP, regulatory
- engineering: technical issues, API bugs, infrastructure, security incidents
- support: customer issues, account problems, product questions
- sales: deals, RFPs, renewals, leads
- hr: hiring, benefits, employee relations, resignations
- marketing: brand, campaigns, partnerships, newsletters
- spam: phishing, scams, unsolicited bulk email

Spam/phishing detection signals:
- Mismatched sender domain (e.g. "Microsoft" but from @microsofft.com)
- Urgency + credential harvesting links
- Excessive punctuation, ALL CAPS, guaranteed results
- Requests to click suspicious URLs

Respond ONLY with valid JSON. No markdown, no explanation outside the JSON."""


# ─── Prompt builder ──────────────────────────────────────────────────────────

def build_user_prompt(obs) -> str:
    parts = [
        f"Task: {obs.task_id} | Step {obs.step_number}/{obs.total_steps}",
        f"From: {obs.from_name} <{obs.from_address}>",
        f"To: {obs.to_address}",
        f"Subject: {obs.subject}",
        f"Timestamp: {obs.timestamp}",
    ]
    if obs.labels:
        parts.append(f"Labels: {', '.join(obs.labels)}")
    if obs.attachments:
        att_list = ", ".join(f"{a.filename} ({a.file_type}, {a.size_kb}KB)" for a in obs.attachments)
        parts.append(f"Attachments: {att_list}")
    if obs.urgency_indicators:
        parts.append(f"Urgency signals: {', '.join(obs.urgency_indicators)}")
    if obs.thread_history:
        parts.append("\n--- Thread History ---")
        for msg in obs.thread_history:
            parts.append(f"[{msg.timestamp}] {msg.sender}: {msg.body[:200]}")
        parts.append("--- End Thread ---")
    parts.append(f"\n{obs.body}")
    return "\n".join(parts)


# ─── LLM call ────────────────────────────────────────────────────────────────

def call_llm(client: OpenAI, obs, task_id: str) -> Optional[Action]:
    user_prompt = build_user_prompt(obs)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)

            valid_actions = {a.value for a in FollowUpAction}
            follow_up = [
                FollowUpAction(a) for a in data.get("follow_up_actions", [])
                if a in valid_actions
            ]

            action = Action(
                priority=Priority(data.get("priority", "medium")),
                department=Department(data.get("department", "support")),
                response=data.get("response"),
                follow_up_actions=follow_up,
                is_spam=bool(data.get("is_spam", False)),
                confidence=float(data.get("confidence", 1.0)),
                reasoning=data.get("reasoning"),
            )
            return action

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"  [attempt {attempt + 1}] Parse error: {e}. Retrying...")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            if "rate" in str(e).lower():
                wait = RETRY_DELAY * (attempt + 1) * 2
                print(f"  [attempt {attempt + 1}] Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [attempt {attempt + 1}] API error: {e}")
                time.sleep(RETRY_DELAY)

    print("  All retries exhausted — using fallback action.")
    return Action(
        priority=Priority.MEDIUM,
        department=Department.SUPPORT,
        follow_up_actions=[FollowUpAction.NO_ACTION],
        is_spam=False,
        confidence=0.0,
        reasoning="Fallback: LLM call failed.",
    )


# ─── Run task with structured START/STEP/END logging ─────────────────────────

def run_task(client: OpenAI, task_id: str) -> dict:
    env = EmailTriageEnv(task_id=task_id)
    task_info = TASKS[task_id]

    # START log (required structured format)
    print(json.dumps({
        "type": "START",
        "task_id": task_id,
        "task_name": task_info.name,
        "difficulty": task_info.difficulty,
        "num_emails": task_info.num_emails,
        "model": MODEL_NAME,
    }))

    obs = env.reset()
    step_results = []

    while obs is not None:
        action = call_llm(client, obs, task_id)
        result = env.step(action)
        reward = result.reward

        # STEP log (required structured format)
        print(json.dumps({
            "type": "STEP",
            "task_id": task_id,
            "step": obs.step_number,
            "total_steps": obs.total_steps,
            "email_id": obs.email_id,
            "subject": obs.subject[:80],
            "action": {
                "priority": action.priority.value,
                "department": action.department.value,
                "is_spam": action.is_spam,
                "confidence": action.confidence,
                "reasoning": action.reasoning,
            },
            "reward": {
                "total": reward.total,
                "priority_score": reward.priority_score,
                "department_score": reward.department_score,
                "spam_score": reward.spam_score,
                "response_score": reward.response_score,
                "follow_up_score": reward.follow_up_score,
                "penalty": reward.penalty,
            },
        }))

        step_results.append({
            "email_id": obs.email_id,
            "subject": obs.subject,
            "action": {
                "priority": action.priority.value,
                "department": action.department.value,
                "is_spam": action.is_spam,
            },
            "reward": reward.total,
            "reasoning": action.reasoning,
        })

        obs = result.observation

    state = env.state()
    episode_score = state.cumulative_reward

    task_result = {
        "task_id": task_id,
        "task_name": task_info.name,
        "difficulty": task_info.difficulty,
        "model": MODEL_NAME,
        "episode_score": episode_score,
        "num_emails": task_info.num_emails,
        "steps": step_results,
    }

    # END log (required structured format)
    print(json.dumps({
        "type": "END",
        "task_id": task_id,
        "episode_score": episode_score,
        "num_emails": task_info.num_emails,
        "model": MODEL_NAME,
    }))

    return task_result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline inference for AI Email Triage Env")
    parser.add_argument("--task", choices=list(TASKS.keys()) + ["all"], default="all",
                        help="Which task to run (default: all)")
    parser.add_argument("--output", default="baseline_results.json",
                        help="Output JSON file (default: baseline_results.json)")
    args = parser.parse_args()

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.")
        print("  export HF_TOKEN=hf_...")
        sys.exit(1)

    client = OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
    )

    task_ids = list(TASKS.keys()) if args.task == "all" else [args.task]
    all_results = []

    for task_id in task_ids:
        result = run_task(client, task_id)
        all_results.append(result)

    weights = {"task1": 0.2, "task2": 0.35, "task3": 0.45}
    total_weighted = sum(
        r["episode_score"] * weights.get(r["task_id"], 1.0 / len(all_results))
        for r in all_results
    )

    output = {
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": all_results,
        "summary": {
            "scores": {r["task_id"]: r["episode_score"] for r in all_results},
            "weighted_overall": round(total_weighted, 4) if len(all_results) == 3 else None,
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
