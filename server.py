"""
FastAPI server for the AI Email Triage OpenEnv Environment.

Endpoints:
  GET  /                         → HTML landing page / dashboard
  GET  /health                   → Health check
  GET  /tasks                    → List all task metadata
  GET  /tasks/{task_id}          → Single task info
  POST /env/{task_id}/reset      → Start a new episode
  POST /env/{task_id}/step       → Submit a triage action
  GET  /env/{task_id}/state      → Current environment state
  POST /env/{task_id}/evaluate   → Batch evaluate (offline grader)
  GET  /openenv.yaml             → OpenEnv spec file
"""
import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, HTMLResponse

from models import Action, StepResult, EnvState, Observation
from environment import EmailTriageEnv
from tasks import TASKS, run_episode_grader

# ─── Session store (in-memory per-session environments) ──────────────────────
# In production use Redis or a proper session store
_sessions: dict[str, EmailTriageEnv] = {}


def _get_env(task_id: str) -> EmailTriageEnv:
    """Get or create an environment for the given task."""
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task: '{task_id}'. Valid tasks: {list(TASKS.keys())}")
    if task_id not in _sessions:
        _sessions[task_id] = EmailTriageEnv(task_id=task_id)
    return _sessions[task_id]


# ─── App setup ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm environments for all tasks
    for task_id in TASKS:
        _sessions[task_id] = EmailTriageEnv(task_id=task_id)
    yield


app = FastAPI(
    title="AI Email Triage Environment",
    description=(
        "A complete OpenEnv-compatible environment for training and evaluating "
        "AI agents on realistic email triage tasks. Agents must classify priority, "
        "route emails to departments, detect spam/phishing, compose responses, "
        "and select follow-up actions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ──────────────────────────────────────────────────────────────────

LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>AI Email Triage Environment</title>
<style>
  :root{--bg:#0f1117;--card:#1a1d27;--border:#2a2d3e;--accent:#6c63ff;--accent2:#00d4aa;--warn:#ff6b6b;--text:#e2e8f0;--muted:#8892a4;--green:#22d3a3;--yellow:#fbbf24;--red:#f87171}
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;min-height:100vh;padding:0}
  a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
  .top-bar{background:linear-gradient(135deg,#1a1d27 0%,#0f1117 100%);border-bottom:1px solid var(--border);padding:20px 40px;display:flex;align-items:center;justify-content:space-between}
  .logo{display:flex;align-items:center;gap:12px}
  .logo-icon{width:40px;height:40px;background:linear-gradient(135deg,var(--accent),var(--accent2));border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px}
  .logo-text h1{font-size:18px;font-weight:700;color:var(--text)}
  .logo-text p{font-size:12px;color:var(--muted)}
  .badge{background:var(--accent);color:#fff;font-size:11px;font-weight:600;padding:3px 10px;border-radius:20px;letter-spacing:.5px}
  .nav-links{display:flex;gap:16px;align-items:center}
  .nav-links a{color:var(--muted);font-size:13px;font-weight:500;padding:6px 14px;border-radius:6px;transition:all .2s}
  .nav-links a:hover{color:var(--text);background:var(--border)}
  .nav-links .btn-docs{background:var(--accent);color:#fff!important;padding:7px 18px}
  .nav-links .btn-docs:hover{background:#5a52e0;text-decoration:none}
  .hero{padding:60px 40px 40px;max-width:1100px;margin:0 auto}
  .hero-label{font-size:12px;font-weight:600;color:var(--accent2);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px}
  .hero h2{font-size:36px;font-weight:800;line-height:1.2;margin-bottom:16px;background:linear-gradient(135deg,#fff 0%,var(--muted) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .hero p{font-size:16px;color:var(--muted);max-width:600px;line-height:1.7}
  .stats-row{display:flex;gap:20px;margin:30px 0}
  .stat{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px 24px;flex:1}
  .stat-val{font-size:28px;font-weight:800;color:var(--accent2)}
  .stat-label{font-size:12px;color:var(--muted);margin-top:2px}
  .section{max-width:1100px;margin:0 auto 40px;padding:0 40px}
  .section-title{font-size:14px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:16px}
  .task-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
  .task-card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:24px;position:relative;overflow:hidden;transition:border-color .2s}
  .task-card:hover{border-color:var(--accent)}
  .task-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px}
  .task-card.easy::before{background:var(--green)}
  .task-card.medium::before{background:var(--yellow)}
  .task-card.hard::before{background:var(--red)}
  .task-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
  .task-name{font-size:15px;font-weight:700}
  .diff-badge{font-size:10px;font-weight:700;padding:3px 10px;border-radius:20px;text-transform:uppercase;letter-spacing:.5px}
  .diff-badge.easy{background:rgba(34,211,163,.15);color:var(--green)}
  .diff-badge.medium{background:rgba(251,191,36,.15);color:var(--yellow)}
  .diff-badge.hard{background:rgba(248,113,113,.15);color:var(--red)}
  .task-desc{font-size:13px;color:var(--muted);line-height:1.6;margin-bottom:16px}
  .task-meta{display:flex;gap:12px}
  .task-meta-item{font-size:12px;color:var(--muted);background:rgba(255,255,255,.05);padding:4px 10px;border-radius:6px}
  .task-score{font-size:20px;font-weight:800;color:var(--accent2);margin-top:12px}
  .task-score span{font-size:12px;color:var(--muted);font-weight:400}
  .endpoints-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
  .endpoint{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px;display:flex;align-items:flex-start;gap:14px}
  .method{font-size:11px;font-weight:800;padding:3px 8px;border-radius:5px;min-width:42px;text-align:center;flex-shrink:0;margin-top:1px}
  .method.get{background:rgba(34,211,163,.15);color:var(--green)}
  .method.post{background:rgba(108,99,255,.15);color:var(--accent)}
  .endpoint-path{font-family:'JetBrains Mono','Fira Code',monospace;font-size:13px;font-weight:600;color:var(--text)}
  .endpoint-desc{font-size:12px;color:var(--muted);margin-top:3px}
  .code-block{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:24px;margin-bottom:16px}
  .code-block pre{font-family:'JetBrains Mono','Fira Code',monospace;font-size:13px;color:#a8b2d8;line-height:1.7;overflow-x:auto;white-space:pre}
  .code-block .kw{color:#c792ea}.code-block .str{color:#c3e88d}.code-block .fn{color:#82aaff}.code-block .cm{color:#546e7a}.code-block .num{color:#f78c6c}
  .section-tabs{display:flex;gap:8px;margin-bottom:16px}
  .tab{font-size:13px;font-weight:500;padding:6px 16px;border-radius:8px;cursor:pointer;color:var(--muted);border:1px solid transparent;background:none}
  .tab.active{background:var(--card);border-color:var(--border);color:var(--text)}
  .reward-row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
  .reward-item{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 18px;flex:1;min-width:140px}
  .reward-item-name{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
  .reward-item-val{font-size:20px;font-weight:800;color:var(--accent2)}
  .footer{border-top:1px solid var(--border);padding:24px 40px;text-align:center;color:var(--muted);font-size:12px;margin-top:20px}
  @media(max-width:768px){.task-grid,.endpoints-grid{grid-template-columns:1fr}.stats-row{flex-wrap:wrap}.hero h2{font-size:26px}.top-bar{flex-direction:column;gap:12px;padding:16px}}
</style>
</head>
<body>

<div class="top-bar">
  <div class="logo">
    <div class="logo-icon">📧</div>
    <div class="logo-text">
      <h1>AI Email Triage Environment</h1>
      <p>OpenEnv-Compatible · Hugging Face Spaces</p>
    </div>
  </div>
  <div class="nav-links">
    <span class="badge">v1.0.0</span>
    <a href="/tasks">Tasks API</a>
    <a href="/openenv.yaml">openenv.yaml</a>
    <a href="/docs" class="btn-docs">Interactive Docs →</a>
  </div>
</div>

<div class="hero">
  <div class="hero-label">OpenEnv Environment</div>
  <h2>Real-World Email Triage<br/>for AI Agent Training</h2>
  <p>A complete, hackathon-ready environment where AI agents learn to triage corporate emails — classifying priority, routing to departments, detecting phishing, composing responses, and selecting follow-up actions.</p>

  <div class="stats-row">
    <div class="stat">
      <div class="stat-val">3</div>
      <div class="stat-label">Graded Tasks (Easy → Hard)</div>
    </div>
    <div class="stat">
      <div class="stat-val">22</div>
      <div class="stat-label">Realistic Email Scenarios</div>
    </div>
    <div class="stat">
      <div class="stat-val">5</div>
      <div class="stat-label">Reward Dimensions</div>
    </div>
    <div class="stat">
      <div class="stat-val">0.70</div>
      <div class="stat-label">GPT-4o-mini Baseline Score</div>
    </div>
  </div>
</div>

<div class="section">
  <div class="section-title">Tasks</div>
  <div class="task-grid">
    <div class="task-card easy">
      <div class="task-header">
        <span class="task-name">Priority Classification</span>
        <span class="diff-badge easy">Easy</span>
      </div>
      <div class="task-desc">Classify 5 emails as high, medium, or low priority. Focus on urgency signals. No routing or response needed.</div>
      <div class="task-meta">
        <span class="task-meta-item">5 emails</span>
        <span class="task-meta-item">task1</span>
      </div>
      <div class="task-score">0.82 <span>gpt-4o-mini baseline</span></div>
    </div>
    <div class="task-card medium">
      <div class="task-header">
        <span class="task-name">Priority + Routing</span>
        <span class="diff-badge medium">Medium</span>
      </div>
      <div class="task-desc">Triage 7 emails: assign priority, route to correct department, detect spam &amp; phishing. Includes VIP escalations and regulatory notices.</div>
      <div class="task-meta">
        <span class="task-meta-item">7 emails</span>
        <span class="task-meta-item">task2</span>
      </div>
      <div class="task-score">0.71 <span>gpt-4o-mini baseline</span></div>
    </div>
    <div class="task-card hard">
      <div class="task-header">
        <span class="task-name">Full Triage &amp; Response</span>
        <span class="diff-badge hard">Hard</span>
      </div>
      <div class="task-desc">Complete triage for 10 complex emails: priority, routing, spam detection, draft response, and follow-up actions. Security breaches, investor comms, legal audits.</div>
      <div class="task-meta">
        <span class="task-meta-item">10 emails</span>
        <span class="task-meta-item">task3</span>
      </div>
      <div class="task-score">0.63 <span>gpt-4o-mini baseline</span></div>
    </div>
  </div>
</div>

<div class="section">
  <div class="section-title">Reward Function</div>
  <div class="reward-row">
    <div class="reward-item"><div class="reward-item-name">Priority</div><div class="reward-item-val">Partial</div></div>
    <div class="reward-item"><div class="reward-item-name">Department</div><div class="reward-item-val">Group</div></div>
    <div class="reward-item"><div class="reward-item-name">Spam Detection</div><div class="reward-item-val">Binary</div></div>
    <div class="reward-item"><div class="reward-item-name">Response Quality</div><div class="reward-item-val">NLP</div></div>
    <div class="reward-item"><div class="reward-item-name">Follow-up F1</div><div class="reward-item-val">F1</div></div>
    <div class="reward-item"><div class="reward-item-name">Penalties</div><div class="reward-item-val">−0.3</div></div>
  </div>
  <p style="font-size:13px;color:var(--muted);line-height:1.7">Dense partial-credit signals at every step — not just binary end-of-episode rewards. Adjacent priorities get 0.5 credit. Same-domain department groups get 0.5. Missed phishing: −0.30 penalty. High priority demoted to low: −0.20 penalty.</p>
</div>

<div class="section">
  <div class="section-title">Quick Start</div>
  <div class="code-block"><pre><span class="cm"># 1. Reset an episode (task1 / task2 / task3)</span>
<span class="fn">curl</span> -X POST /env/task2/reset

<span class="cm"># 2. Submit a triage action</span>
<span class="fn">curl</span> -X POST /env/task2/step \
  -H <span class="str">"Content-Type: application/json"</span> \
  -d <span class="str">'{
    "priority": "high",
    "department": "support",
    "response": "Thank you — escalating this immediately.",
    "follow_up_actions": ["escalate", "create_ticket"],
    "is_spam": false,
    "confidence": 0.95,
    "reasoning": "Production outage with financial impact."
  }'</span>

<span class="cm"># Response includes per-dimension reward breakdown:</span>
<span class="cm"># { "reward": { "total": 0.92, "priority_score": 1.0, "department_score": 1.0, ... } }</span>

<span class="cm"># 3. Check state</span>
<span class="fn">curl</span> /env/task2/state

<span class="cm"># 4. Batch evaluate all actions at once (offline)</span>
<span class="fn">curl</span> -X POST /env/task1/evaluate -d <span class="str">'[{...}, {...}, ...]'</span></pre></div>

  <div class="code-block"><pre><span class="cm"># Run baseline inference (requires OPENAI_API_KEY)</span>
<span class="kw">export</span> OPENAI_API_KEY=sk-...
<span class="fn">python3</span> inference.py <span class="cm">           # all 3 tasks</span>
<span class="fn">python3</span> inference.py --task task3  <span class="cm"># single task</span></pre></div>
</div>

<div class="section">
  <div class="section-title">API Endpoints</div>
  <div class="endpoints-grid">
    <div class="endpoint">
      <span class="method get">GET</span>
      <div><div class="endpoint-path">/tasks</div><div class="endpoint-desc">List all task metadata</div></div>
    </div>
    <div class="endpoint">
      <span class="method post">POST</span>
      <div><div class="endpoint-path">/env/{task_id}/reset</div><div class="endpoint-desc">Start a new episode, get first email</div></div>
    </div>
    <div class="endpoint">
      <span class="method post">POST</span>
      <div><div class="endpoint-path">/env/{task_id}/step</div><div class="endpoint-desc">Submit triage action → reward + next email</div></div>
    </div>
    <div class="endpoint">
      <span class="method get">GET</span>
      <div><div class="endpoint-path">/env/{task_id}/state</div><div class="endpoint-desc">Current environment state</div></div>
    </div>
    <div class="endpoint">
      <span class="method post">POST</span>
      <div><div class="endpoint-path">/env/{task_id}/evaluate</div><div class="endpoint-desc">Batch offline evaluation grader</div></div>
    </div>
    <div class="endpoint">
      <span class="method get">GET</span>
      <div><div class="endpoint-path">/openenv.yaml</div><div class="endpoint-desc">OpenEnv specification file</div></div>
    </div>
    <div class="endpoint">
      <span class="method get">GET</span>
      <div><div class="endpoint-path">/docs</div><div class="endpoint-desc">Interactive Swagger API documentation</div></div>
    </div>
    <div class="endpoint">
      <span class="method get">GET</span>
      <div><div class="endpoint-path">/health</div><div class="endpoint-desc">Health check — {status: ok}</div></div>
    </div>
  </div>
</div>

<div class="footer">
  AI Email Triage Environment · OpenEnv Compatible · Deploy on Hugging Face Spaces · MIT License
</div>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root():
    """Landing page with environment overview and documentation."""
    return HTMLResponse(content=LANDING_HTML)


@app.get("/health")
def health():
    return {"status": "ok", "tasks_loaded": len(TASKS)}


@app.get("/tasks")
def list_tasks():
    return {task_id: task.model_dump() for task_id, task in TASKS.items()}


@app.get("/tasks/{task_id}")
def get_task(task_id: str = PathParam(..., description="Task ID: task1, task2, or task3")):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task: '{task_id}'")
    return TASKS[task_id].model_dump()


@app.post("/env/{task_id}/reset", response_model=Observation)
def reset_env(task_id: str = PathParam(..., description="Task ID to reset")):
    """Reset the environment and return the first email observation."""
    env = _get_env(task_id)
    try:
        obs = env.reset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return obs


@app.post("/env/{task_id}/step", response_model=StepResult)
def step_env(
    action: Action,
    task_id: str = PathParam(..., description="Task ID"),
):
    """
    Submit a triage action for the current email.
    Returns the next observation, reward, done flag, and grading details.
    """
    env = _get_env(task_id)
    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.get("/env/{task_id}/state", response_model=EnvState)
def get_state(task_id: str = PathParam(..., description="Task ID")):
    """Return the current state of the environment."""
    env = _get_env(task_id)
    return env.state()


@app.post("/env/{task_id}/evaluate")
def evaluate_episode(
    actions: list[Action],
    task_id: str = PathParam(..., description="Task ID"),
):
    """
    Batch evaluator — grade a complete list of actions offline without
    running the interactive step() loop. Useful for benchmarking.
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task: '{task_id}'")
    try:
        result = run_episode_grader(task_id, actions)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.get("/openenv.yaml", response_class=PlainTextResponse)
def get_openenv_yaml():
    """Return the OpenEnv specification YAML file."""
    spec_path = Path(__file__).parent / "openenv.yaml"
    if not spec_path.exists():
        raise HTTPException(status_code=404, detail="openenv.yaml not found")
    return spec_path.read_text()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
