"""
AI Email Triage OpenEnv Environment.

Implements the full OpenEnv interface:
  - reset()   → initial Observation
  - step()    → StepResult (observation, reward, done, info)
  - state()   → EnvState

The environment simulates a realistic corporate email inbox. An agent must
read each email and produce an Action (triage decision). It is scored per
action with partial-credit reward signals.
"""
import uuid
from datetime import datetime, timezone
from models import (
    Action,
    Observation,
    EnvState,
    StepResult,
    EmailAttachment,
    ThreadMessage,
)
from tasks import get_task_emails, grade_action, TASKS
from rewards import compute_reward


def _build_observation(email: dict, task_id: str, step: int, total: int) -> Observation:
    attachments = [
        EmailAttachment(**a) for a in email.get("attachments", [])
    ]
    thread_history = [
        ThreadMessage(**m) for m in email.get("thread_history", [])
    ]
    return Observation(
        email_id=email["email_id"],
        from_address=email["from_address"],
        from_name=email["from_name"],
        to_address=email["to_address"],
        subject=email["subject"],
        body=email["body"],
        timestamp=email["timestamp"],
        attachments=attachments,
        thread_history=thread_history,
        labels=email.get("labels", []),
        is_reply=email.get("is_reply", False),
        urgency_indicators=email.get("urgency_indicators", []),
        task_id=task_id,
        step_number=step,
        total_steps=total,
    )


class EmailTriageEnv:
    """
    OpenEnv-compliant Email Triage Environment.

    Supports three tasks of increasing difficulty:
      task1 — Priority Classification (easy, 5 emails)
      task2 — Priority + Department Routing (medium, 7 emails)
      task3 — Full Triage with Response (hard, 10 emails)
    """

    SUPPORTED_TASKS = list(TASKS.keys())

    def __init__(self, task_id: str = "task1"):
        if task_id not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"Unknown task '{task_id}'. Choose from: {self.SUPPORTED_TASKS}"
            )
        self.task_id = task_id
        self._emails: list[dict] = []
        self._step_idx: int = 0
        self._done: bool = True
        self._episode_rewards: list[float] = []
        self._session_id: str = ""

    # ─── OpenEnv Interface ───────────────────────────────────────────────────

    def reset(self) -> Observation:
        """
        Reset the environment to the beginning of a new episode.
        Returns the first email as an Observation.
        """
        self._emails = get_task_emails(self.task_id)
        self._step_idx = 0
        self._done = False
        self._episode_rewards = []
        self._session_id = str(uuid.uuid4())

        return _build_observation(
            self._emails[0],
            self.task_id,
            step=1,
            total=len(self._emails),
        )

    def step(self, action: Action) -> StepResult:
        """
        Apply an action (triage decision) to the current email.

        Returns:
          observation — next email (or None if episode is done)
          reward      — scored Reward with breakdown
          done        — True when all emails have been triaged
          info        — grading details and cumulative progress
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._step_idx >= len(self._emails):
            raise RuntimeError("No emails remaining. Call reset().")

        current_email = self._emails[self._step_idx]
        grade = grade_action(action, current_email)
        reward = grade["reward"]
        self._episode_rewards.append(reward.total)
        self._step_idx += 1

        done = self._step_idx >= len(self._emails)
        self._done = done

        next_obs = None
        if not done:
            next_obs = _build_observation(
                self._emails[self._step_idx],
                self.task_id,
                step=self._step_idx + 1,
                total=len(self._emails),
            )

        info = {
            "email_id": current_email["email_id"],
            "step": self._step_idx,
            "total_steps": len(self._emails),
            "grading_breakdown": grade["breakdown"],
            "cumulative_reward": round(sum(self._episode_rewards) / len(self._episode_rewards), 4),
            "emails_remaining": len(self._emails) - self._step_idx,
            "session_id": self._session_id,
        }
        if done:
            info["episode_summary"] = {
                "total_emails": len(self._emails),
                "episode_score": round(sum(self._episode_rewards) / len(self._episode_rewards), 4),
                "min_step_reward": round(min(self._episode_rewards), 4),
                "max_step_reward": round(max(self._episode_rewards), 4),
                "rewards_per_step": self._episode_rewards,
            }

        return StepResult(
            observation=next_obs,
            reward=reward,
            done=done,
            info=info,
        )

    def state(self) -> EnvState:
        """Return the current environment state."""
        current_obs = None
        if not self._done and self._step_idx < len(self._emails):
            current_obs = _build_observation(
                self._emails[self._step_idx],
                self.task_id,
                step=self._step_idx + 1,
                total=len(self._emails),
            )

        return EnvState(
            task_id=self.task_id,
            current_step=self._step_idx,
            total_steps=len(self._emails),
            emails_processed=self._step_idx,
            total_emails=len(self._emails),
            cumulative_reward=(
                round(sum(self._episode_rewards) / len(self._episode_rewards), 4)
                if self._episode_rewards else 0.0
            ),
            done=self._done,
            current_observation=current_obs,
            episode_rewards=self._episode_rewards,
        )

    # ─── Utility ─────────────────────────────────────────────────────────────

    def get_task_info(self) -> dict:
        task = TASKS[self.task_id]
        return task.model_dump()
