from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Department(str, Enum):
    SALES = "sales"
    SUPPORT = "support"
    ENGINEERING = "engineering"
    HR = "hr"
    MARKETING = "marketing"
    LEGAL = "legal"
    EXECUTIVE = "executive"
    SPAM = "spam"


class FollowUpAction(str, Enum):
    REPLY = "reply"
    ESCALATE = "escalate"
    SCHEDULE_MEETING = "schedule_meeting"
    CREATE_TICKET = "create_ticket"
    ARCHIVE = "archive"
    FORWARD = "forward"
    FLAG_REVIEW = "flag_review"
    NO_ACTION = "no_action"


class EmailAttachment(BaseModel):
    filename: str
    file_type: str
    size_kb: int


class ThreadMessage(BaseModel):
    sender: str
    timestamp: str
    body: str


class Observation(BaseModel):
    email_id: str = Field(description="Unique identifier for the email")
    from_address: str = Field(description="Sender email address")
    from_name: str = Field(description="Sender display name")
    to_address: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")
    timestamp: str = Field(description="ISO 8601 timestamp")
    attachments: list[EmailAttachment] = Field(default_factory=list)
    thread_history: list[ThreadMessage] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list, description="Existing labels/tags")
    is_reply: bool = Field(default=False)
    urgency_indicators: list[str] = Field(
        default_factory=list,
        description="Detected urgency signals in the email"
    )
    task_id: str = Field(description="Current task identifier")
    step_number: int = Field(description="Current step in the episode")
    total_steps: int = Field(description="Total steps in the episode")


class Action(BaseModel):
    priority: Priority = Field(description="Assigned priority level")
    department: Department = Field(description="Target department for routing")
    response: Optional[str] = Field(
        default=None,
        description="Draft response text (required for medium/hard tasks)"
    )
    follow_up_actions: list[FollowUpAction] = Field(
        default_factory=list,
        description="Follow-up actions to take (required for hard task)"
    )
    is_spam: bool = Field(default=False, description="Whether email is spam/phishing")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in the classification"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's reasoning for the triage decision"
    )


class Reward(BaseModel):
    total: float = Field(ge=0.0, le=1.0, description="Total reward score")
    priority_score: float = Field(ge=0.0, le=1.0, description="Priority classification score")
    department_score: float = Field(ge=0.0, le=1.0, description="Department routing score")
    response_score: float = Field(ge=0.0, le=1.0, description="Response quality score")
    follow_up_score: float = Field(ge=0.0, le=1.0, description="Follow-up actions score")
    spam_score: float = Field(ge=0.0, le=1.0, description="Spam detection score")
    penalty: float = Field(ge=0.0, le=1.0, description="Penalty deductions")
    feedback: str = Field(description="Human-readable feedback")


class EnvState(BaseModel):
    task_id: str
    current_step: int
    total_steps: int
    emails_processed: int
    total_emails: int
    cumulative_reward: float
    done: bool
    current_observation: Optional[Observation] = None
    episode_rewards: list[float] = Field(default_factory=list)


class StepResult(BaseModel):
    observation: Optional[Observation]
    reward: Reward
    done: bool
    info: dict


class TaskInfo(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: str
    num_emails: int
    scoring_criteria: list[str]
