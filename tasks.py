"""
Task definitions and graders for the AI Email Triage environment.

Task 1 (Easy)   — Priority Classification only (5 emails)
Task 2 (Medium) — Priority + Department Routing + Spam Detection (7 emails)
Task 3 (Hard)   — Full Triage: priority, department, spam, response, follow-up (10 emails)
"""
from models import Action, TaskInfo
from data.emails import get_emails_for_task
from rewards import compute_reward


TASKS: dict[str, TaskInfo] = {
    "task1": TaskInfo(
        task_id="task1",
        name="Priority Classification",
        description=(
            "Classify each incoming email as high, medium, or low priority. "
            "No response or routing is required — focus solely on understanding "
            "the urgency signals in the email content."
        ),
        difficulty="easy",
        num_emails=5,
        scoring_criteria=[
            "Correct priority label (high/medium/low) — 70% of score",
            "Correct department hint — 20% of score",
            "Spam detection — 10% of score",
        ],
    ),
    "task2": TaskInfo(
        task_id="task2",
        name="Priority + Department Routing",
        description=(
            "Triage each email by assigning a priority and routing it to the correct department. "
            "You must also identify spam and phishing emails. "
            "Tasks include realistic edge cases: VIP escalations, automated alerts, "
            "phishing attempts, and regulatory notices."
        ),
        difficulty="medium",
        num_emails=7,
        scoring_criteria=[
            "Correct priority label — 35% of score",
            "Correct department routing — 35% of score",
            "Spam/phishing detection — 20% of score",
            "Appropriate follow-up action selection — 10% of score",
        ],
    ),
    "task3": TaskInfo(
        task_id="task3",
        name="Full Email Triage & Response",
        description=(
            "Perform complete email triage: assign priority, route to department, "
            "detect spam/phishing, compose a contextually appropriate draft response, "
            "and select follow-up actions. Emails include security incidents, "
            "investor communications, legal compliance notices, and VIP escalations."
        ),
        difficulty="hard",
        num_emails=10,
        scoring_criteria=[
            "Correct priority label — 20% of score",
            "Correct department routing — 20% of score",
            "Spam/phishing detection — 15% of score",
            "Response quality (keyword coverage + length) — 25% of score",
            "Follow-up action F1 score — 20% of score",
        ],
    ),
}


def get_task_emails(task_id: str) -> list[dict]:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}. Valid tasks: {list(TASKS.keys())}")
    return get_emails_for_task(task_id)


def grade_action(action: Action, email: dict) -> dict:
    """Grade a single action against the email's ground truth. Returns reward dict + details."""
    reward = compute_reward(action, email)
    gt = email["ground_truth"]

    return {
        "reward": reward,
        "breakdown": {
            "email_id": email["email_id"],
            "ground_truth": gt,
            "predicted": {
                "priority": action.priority.value,
                "department": action.department.value,
                "is_spam": action.is_spam,
                "follow_up_actions": [a.value for a in (action.follow_up_actions or [])],
                "has_response": bool(action.response and len(action.response) > 20),
            },
            "scores": {
                "priority": reward.priority_score,
                "department": reward.department_score,
                "spam": reward.spam_score,
                "response": reward.response_score,
                "follow_up": reward.follow_up_score,
                "penalty": reward.penalty,
                "total": reward.total,
            },
        },
    }


def run_episode_grader(task_id: str, actions: list[Action]) -> dict:
    """
    Batch grader — runs the full episode offline and returns aggregate scores.
    Used by the inference script for reproducible evaluation.
    """
    emails = get_task_emails(task_id)
    if len(actions) != len(emails):
        raise ValueError(
            f"Expected {len(emails)} actions for {task_id}, got {len(actions)}"
        )

    results = [grade_action(action, email) for action, email in zip(actions, emails)]
    rewards = [r["reward"].total for r in results]

    return {
        "task_id": task_id,
        "task_name": TASKS[task_id].name,
        "difficulty": TASKS[task_id].difficulty,
        "num_emails": len(emails),
        "episode_score": round(sum(rewards) / len(rewards), 4),
        "min_score": round(min(rewards), 4),
        "max_score": round(max(rewards), 4),
        "per_email": results,
        "dimension_averages": {
            "priority": round(sum(r["breakdown"]["scores"]["priority"] for r in results) / len(results), 4),
            "department": round(sum(r["breakdown"]["scores"]["department"] for r in results) / len(results), 4),
            "spam": round(sum(r["breakdown"]["scores"]["spam"] for r in results) / len(results), 4),
            "response": round(sum(r["breakdown"]["scores"]["response"] for r in results) / len(results), 4),
            "follow_up": round(sum(r["breakdown"]["scores"]["follow_up"] for r in results) / len(results), 4),
        },
    }
