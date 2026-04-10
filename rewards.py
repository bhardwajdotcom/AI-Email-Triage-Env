"""
Reward functions for the AI Email Triage environment.
All rewards are in [0.0, 1.0]. Partial credit is given at each step.
"""
from models import Action, Reward, Priority, Department


# Priority adjacency — partial credit for "close" guesses
PRIORITY_ADJACENCY = {
    ("high", "medium"): 0.5,
    ("medium", "high"): 0.5,
    ("medium", "low"): 0.5,
    ("low", "medium"): 0.5,
    ("high", "low"): 0.0,
    ("low", "high"): 0.0,
}

# Department partial-credit groups (same-group routing gets 0.5)
DEPARTMENT_GROUPS = {
    "support": ["support", "engineering"],
    "engineering": ["engineering", "support"],
    "executive": ["executive", "legal"],
    "legal": ["legal", "executive"],
    "sales": ["sales", "marketing"],
    "marketing": ["marketing", "sales"],
    "hr": ["hr"],
    "spam": ["spam"],
}


def score_priority(predicted: str, actual: str) -> float:
    if predicted == actual:
        return 1.0
    return PRIORITY_ADJACENCY.get((predicted, actual), 0.0)


def score_department(predicted: str, actual: str, is_spam_email: bool) -> float:
    if predicted == actual:
        return 1.0
    if is_spam_email:
        # Spam must go to spam — no partial credit for other departments
        return 0.0
    group = DEPARTMENT_GROUPS.get(actual, [actual])
    if predicted in group:
        return 0.5
    return 0.0


def score_spam_detection(predicted_spam: bool, actual_spam: bool) -> float:
    if predicted_spam == actual_spam:
        return 1.0
    # False negative (missed spam) is worse than false positive
    if actual_spam and not predicted_spam:
        return 0.0   # missed phishing / spam — no credit
    return 0.3       # false positive — minor penalty only


def score_response(response: str | None, keyword_hints: list[str], task_id: str) -> float:
    """
    Score the response quality.
    - Task 1: responses not required (always 1.0)
    - Task 2: responses not required (always 1.0)
    - Task 3: required; scored on keyword coverage + length heuristics
    """
    if task_id in ("task1", "task2"):
        return 1.0

    if not response or len(response.strip()) < 20:
        return 0.0

    # Length bonus — we want substantive responses
    length_score = min(1.0, len(response) / 300)

    # Keyword coverage
    if not keyword_hints:
        keyword_score = 1.0
    else:
        resp_lower = response.lower()
        matched = sum(1 for kw in keyword_hints if kw.lower() in resp_lower)
        keyword_score = matched / len(keyword_hints)

    # Avoid canned/empty responses
    canned_phrases = ["please let me know", "i will get back to you", "thank you for your email"]
    canned_penalty = 0.2 if any(p in response.lower() for p in canned_phrases) and len(response) < 100 else 0.0

    return max(0.0, min(1.0, (0.4 * length_score + 0.6 * keyword_score) - canned_penalty))


def score_follow_up_actions(
    predicted_actions: list[str],
    ground_truth_actions: list[str],
    task_id: str,
) -> float:
    """
    Score follow-up action selection.
    - Tasks 1 & 2: not strictly evaluated (1.0 if reasonable, 0.5 otherwise)
    - Task 3: precision/recall F1 against ground truth
    """
    if task_id == "task1":
        return 1.0

    if not ground_truth_actions:
        # Penalise non-empty predictions against emails requiring no action
        return 1.0 if not predicted_actions else 0.7

    predicted_set = set(predicted_actions)
    truth_set = set(ground_truth_actions)

    if not predicted_set:
        return 0.0

    precision = len(predicted_set & truth_set) / len(predicted_set)
    recall = len(predicted_set & truth_set) / len(truth_set)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)

    # Task 3 demands higher accuracy
    if task_id == "task3":
        return f1
    else:
        # Task 2: partial credit — recall matters more than precision
        return 0.6 * recall + 0.4 * precision


def compute_penalty(action: Action, email: dict) -> float:
    """
    Penalties for clearly bad decisions.
    - Routing spam to non-spam department (and missing is_spam) = 0.3
    - Routing a critical/Sev1 email to low priority = 0.2
    - No follow-up on escalation-required emails = 0.1
    """
    penalty = 0.0
    is_spam_email = email["ground_truth"].get("is_spam", False)
    true_priority = email["ground_truth"]["priority"]

    if is_spam_email and not action.is_spam:
        penalty += 0.3  # missed phishing/spam

    if true_priority == "high" and action.priority == Priority.LOW:
        penalty += 0.2  # demoted a critical issue

    escalation_required = "escalate" in email["ground_truth"].get("follow_up_actions", [])
    if escalation_required and "escalate" not in [a.value for a in (action.follow_up_actions or [])]:
        if email["task"] == "task3":
            penalty += 0.1

    return min(0.4, penalty)


def compute_reward(action: Action, email: dict) -> Reward:
    task_id = email["task"]
    gt = email["ground_truth"]

    priority_score = score_priority(action.priority.value, gt["priority"])
    department_score = score_department(action.department.value, gt["department"], gt.get("is_spam", False))
    spam_score = score_spam_detection(action.is_spam, gt.get("is_spam", False))
    response_score = score_response(action.response, gt.get("response_keywords", []), task_id)
    follow_up_score = score_follow_up_actions(
        [a.value for a in (action.follow_up_actions or [])],
        gt.get("follow_up_actions", []),
        task_id,
    )
    penalty = compute_penalty(action, email)

    # Weighted total by task complexity
    if task_id == "task1":
        weights = {"priority": 0.7, "department": 0.2, "spam": 0.1, "response": 0.0, "follow_up": 0.0}
    elif task_id == "task2":
        weights = {"priority": 0.35, "department": 0.35, "spam": 0.2, "response": 0.0, "follow_up": 0.1}
    else:  # task3
        weights = {"priority": 0.2, "department": 0.2, "spam": 0.15, "response": 0.25, "follow_up": 0.2}

    weighted = (
        weights["priority"] * priority_score
        + weights["department"] * department_score
        + weights["spam"] * spam_score
        + weights["response"] * response_score
        + weights["follow_up"] * follow_up_score
    )

    total = max(0.0, min(1.0, weighted - penalty))

    # Build human-readable feedback
    feedback_parts = []
    if priority_score == 1.0:
        feedback_parts.append(f"Priority '{action.priority.value}' is correct.")
    elif priority_score > 0:
        feedback_parts.append(f"Priority '{action.priority.value}' is close (expected '{gt['priority']}') — partial credit.")
    else:
        feedback_parts.append(f"Priority '{action.priority.value}' is wrong (expected '{gt['priority']}').")

    if department_score == 1.0:
        feedback_parts.append(f"Department '{action.department.value}' is correct.")
    elif department_score > 0:
        feedback_parts.append(f"Department '{action.department.value}' is partially correct (expected '{gt['department']}').")
    else:
        feedback_parts.append(f"Department '{action.department.value}' is wrong (expected '{gt['department']}').")

    if gt.get("is_spam") and not action.is_spam:
        feedback_parts.append("Spam/phishing detection failed — this email should be flagged.")
    elif spam_score == 1.0 and gt.get("is_spam"):
        feedback_parts.append("Spam correctly detected.")

    if penalty > 0:
        feedback_parts.append(f"Penalty applied: {penalty:.2f}.")

    return Reward(
        total=round(total, 4),
        priority_score=round(priority_score, 4),
        department_score=round(department_score, 4),
        response_score=round(response_score, 4),
        follow_up_score=round(follow_up_score, 4),
        spam_score=round(spam_score, 4),
        penalty=round(penalty, 4),
        feedback=" ".join(feedback_parts),
    )
