"""
Baseline inference script for the AI Email Triage Environment.

Uses the OpenAI API to run a model against all three tasks and produces
reproducible benchmark scores.

Usage:
    export OPENAI_API_KEY=sk-...
    python inference.py

    # Or with a custom model / base URL (e.g. OpenRouter, local vLLM):
    OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
    OPENAI_MODEL=openai/gpt-4o \
    python inference.py

    # Run a single task:
    python inference.py --task task1
"""
import argparse
import json
import os
import sys
import time
from typing import Optional

try:
    import openai
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(0)

from environment import EmailTriageEnv
from models import Action, Priority, Department, FollowUpAction
from tasks import TASKS


# ─── Configuration ───────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
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
  "confidence": 0.0–1.0,
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

def call_llm(client: openai.OpenAI, obs, task_id: str) -> Optional[Action]:
    user_prompt = build_user_prompt(obs)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
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

            # Map follow_up_actions strings → enums (filter invalid)
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
        except openai.RateLimitError:
            wait = RETRY_DELAY * (attempt + 1) * 2
            print(f"  [attempt {attempt + 1}] Rate limited. Waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
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


# ─── Run task ────────────────────────────────────────────────────────────────

def run_task(client: openai.OpenAI, task_id: str) -> dict:
    env = EmailTriageEnv(task_id=task_id)
    task_info = TASKS[task_id]

    print(f"\n{'=' * 60}")
    print(f"Task: {task_info.name} [{task_info.difficulty.upper()}]")
    print(f"Emails: {task_info.num_emails}")
    print(f"{'=' * 60}")

    obs = env.reset()
    step_results = []
    print(f"[START] task={task_id}", flush=True)

    while obs is not None:
        print(f"\n  [{obs.step_number}/{obs.total_steps}] Email: '{obs.subject[:60]}'")
        print(f"            From: {obs.from_name} <{obs.from_address}>")

        action = call_llm(client, obs, task_id)

        print(f"            → Priority: {action.priority.value}  "
              f"| Dept: {action.department.value}  "
              f"| Spam: {action.is_spam}")
        if action.reasoning:
            print(f"            Reasoning: {action.reasoning[:100]}")

        result = env.step(action)
        reward = result.reward
        print(f"[STEP] step={obs.step_number} reward={reward.total:.4f}", flush=True)
        print(f"            ✓ Reward: {reward.total:.3f}  "
              f"(priority={reward.priority_score:.2f}, "
              f"dept={reward.department_score:.2f}, "
              f"spam={reward.spam_score:.2f}, "
              f"response={reward.response_score:.2f}, "
              f"follow_up={reward.follow_up_score:.2f})")
        if reward.penalty > 0:
            print(f"            ⚠ Penalty: {reward.penalty:.2f} — {reward.feedback}")

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

    print(f"[END] task={task_id} score={episode_score:.4f} steps={len(step_results)}", flush=True)

    return {
        "task_id": task_id,
        "task_name": task_info.name,
        "difficulty": task_info.difficulty,
        "model": OPENAI_MODEL,
        "episode_score": episode_score,
        "num_emails": task_info.num_emails,
        "steps": step_results,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline inference for AI Email Triage Env")
    parser.add_argument("--task", choices=list(TASKS.keys()) + ["all"], default="all",
                        help="Which task to run (default: all)")
    parser.add_argument("--output", default="baseline_results.json",
                        help="Output JSON file (default: baseline_results.json)")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set. Printing dummy structured output.", flush=True)
        for task_id, task_info in TASKS.items():
            print(f"[START] task={task_id}", flush=True)
            for step in range(1, task_info.num_emails + 1):
                print(f"[STEP] step={step} reward=0.0", flush=True)
            print(f"[END] task={task_id} score=0.0 steps={task_info.num_emails}", flush=True)
        sys.exit(0)

    client = openai.OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    print(f"\nAI Email Triage — Baseline Inference")
    print(f"Model: {OPENAI_MODEL}")
    print(f"Base URL: {OPENAI_BASE_URL}")

    task_ids = list(TASKS.keys()) if args.task == "all" else [args.task]
    all_results = []

    for task_id in task_ids:
        result = run_task(client, task_id)
        all_results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("BASELINE SCORES SUMMARY")
    print(f"{'=' * 60}")
    total_weighted = 0.0
    weights = {"task1": 0.2, "task2": 0.35, "task3": 0.45}
    for r in all_results:
        score = r["episode_score"]
        w = weights.get(r["task_id"], 1.0 / len(all_results))
        total_weighted += score * w
        print(f"  {r['task_name']:40s} [{r['difficulty'].upper():6s}]  {score:.4f}")
    if len(all_results) == 3:
        print(f"  {'Weighted Overall Score':40s}           {total_weighted:.4f}")
    print(f"{'=' * 60}")

    output = {
        "model": OPENAI_MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": all_results,
        "summary": {
            "scores": {r["task_id"]: r["episode_score"] for r in all_results},
            "weighted_overall": round(total_weighted, 4) if len(all_results) == 3 else None,
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
