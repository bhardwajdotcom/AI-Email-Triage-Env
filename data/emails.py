"""
Synthetic email dataset for the AI Email Triage & Response Environment.
Each email includes ground truth labels for evaluation.
"""

EMAILS = [
    # ─────────────────────────────────────────────────────────────────────────
    # TASK 1 — EASY: Priority Classification (5 emails)
    # ─────────────────────────────────────────────────────────────────────────
    {
        "task": "task1",
        "email_id": "t1_001",
        "from_address": "ceo@bigclient.com",
        "from_name": "Sarah Mitchell",
        "to_address": "support@company.com",
        "subject": "URGENT: Production system completely down — losing $50k/hour",
        "body": (
            "Our entire production system has crashed. We cannot process any orders. "
            "This is impacting ALL 200 of our stores nationwide. Every minute of downtime "
            "costs us approximately $50,000. I need your most senior engineer on this NOW. "
            "My board is demanding answers. Please call me immediately: +1-555-0192."
        ),
        "timestamp": "2024-01-15T09:02:00Z",
        "attachments": [],
        "thread_history": [],
        "labels": [],
        "is_reply": False,
        "urgency_indicators": ["URGENT", "completely down", "losing $50k/hour"],
        "is_spam": False,
        "ground_truth": {
            "priority": "high",
            "department": "support",
            "is_spam": False,
        },
    },
    {
        "task": "task1",
        "email_id": "t1_002",
        "from_address": "newsletter@techweekly.io",
        "from_name": "Tech Weekly",
        "to_address": "info@company.com",
        "subject": "This week in tech: AI trends, cloud news, and developer tips",
        "body": (
            "Welcome to this week's edition of Tech Weekly! In this issue:\n"
            "- The latest AI model benchmarks\n"
            "- Cloud cost optimization strategies\n"
            "- 10 developer productivity tips\n"
            "- Upcoming conferences and webinars\n\n"
            "Click to read the full newsletter online."
        ),
        "timestamp": "2024-01-15T08:00:00Z",
        "attachments": [],
        "thread_history": [],
        "labels": ["newsletter"],
        "is_reply": False,
        "urgency_indicators": [],
        "is_spam": False,
        "ground_truth": {
            "priority": "low",
            "department": "marketing",
            "is_spam": False,
        },
    },
    {
        "task": "task1",
        "email_id": "t1_003",
        "from_address": "james.wong@partner.com",
        "from_name": "James Wong",
        "to_address": "sales@company.com",
        "subject": "Q1 contract renewal — deadline end of this week",
        "body": (
            "Hi team,\n\n"
            "We need to finalize our Q1 contract renewal by Friday. "
            "The deal is worth $180,000 annually. Our procurement team has approved the budget, "
            "but we need the updated pricing proposal and SLA terms before we can sign. "
            "Please have someone reach out today so we can move forward.\n\n"
            "Best,\nJames"
        ),
        "timestamp": "2024-01-15T10:30:00Z",
        "attachments": [],
        "thread_history": [],
        "labels": [],
        "is_reply": False,
        "urgency_indicators": ["deadline end of this week"],
        "is_spam": False,
        "ground_truth": {
            "priority": "high",
            "department": "sales",
            "is_spam": False,
        },
    },
    {
        "task": "task1",
        "email_id": "t1_004",
        "from_address": "priya.sharma@company.com",
        "from_name": "Priya Sharma",
        "to_address": "hr@company.com",
        "subject": "Vacation request — March 18-22",
        "body": (
            "Hi HR team,\n\n"
            "I'd like to request vacation time from March 18 to 22 (5 days). "
            "I've already coordinated with my manager and the team has coverage. "
            "Please let me know if there's anything else needed.\n\n"
            "Thanks,\nPriya"
        ),
        "timestamp": "2024-01-15T11:00:00Z",
        "attachments": [],
        "thread_history": [],
        "labels": [],
        "is_reply": False,
        "urgency_indicators": [],
        "is_spam": False,
        "ground_truth": {
            "priority": "low",
            "department": "hr",
            "is_spam": False,
        },
    },
    {
        "task": "task1",
        "email_id": "t1_005",
        "from_address": "dev@startup-partner.com",
        "from_name": "Alex Rivera",
        "to_address": "engineering@company.com",
        "subject": "API rate limiting causing intermittent failures on our end",
        "body": (
            "Hello Engineering,\n\n"
            "We've been experiencing intermittent 429 (Too Many Requests) errors from your API "
            "over the past 3 days. It's happening roughly every 2 hours and affects about 15% "
            "of our API calls. We've already checked our request patterns and we're within the "
            "documented limits. Could someone look into this? It's starting to affect our SLAs.\n\n"
            "Thanks,\nAlex"
        ),
        "timestamp": "2024-01-15T09:45:00Z",
        "attachments": [],
        "thread_history": [],
        "labels": [],
        "is_reply": False,
        "urgency_indicators": ["intermittent failures", "affecting SLAs"],
        "is_spam": False,
        "ground_truth": {
            "priority": "medium",
            "department": "engineering",
            "is_spam": False,
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 2 — MEDIUM: Priority + Department Routing (7 emails)
    # ─────────────────────────────────────────────────────────────────────────
    {
        "task": "task2",
        "email_id": "t2_001",
        "from_address": "maria.jones@enterprise-client.com",
        "from_name": "Maria Jones",
        "to_address": "support@company.com",
        "subject": "Data export corrupted — audit deadline tomorrow",
        "body": (
            "Hi,\n\n"
            "I exported our transaction data for our annual audit and the CSV file is completely "
            "corrupted — all numeric values are showing as '#REF!' errors. The external auditors "
            "arrive tomorrow at 9 AM. We cannot provide corrupted financial records.\n\n"
            "This is account ID ENT-4492. Please treat this as emergency.\n\nMaria"
        ),
        "timestamp": "2024-01-15T16:00:00Z",
        "attachments": [{"filename": "export_jan.csv", "file_type": "csv", "size_kb": 2048}],
        "thread_history": [],
        "labels": [],
        "is_reply": False,
        "urgency_indicators": ["corrupted", "audit deadline tomorrow", "emergency"],
        "is_spam": False,
        "ground_truth": {
            "priority": "high",
            "department": "support",
            "is_spam": False,
            "follow_up_actions": ["escalate", "create_ticket"],
        },
    },
    {
        "task": "task2",
        "email_id": "t2_002",
        "from_address": "no-reply@verify-paypal-accounts.net",
        "from_name": "PayPal Security Team",
        "to_address": "finance@company.com",
        "subject": "⚠️ Your PayPal account has been limited — verify now",
        "body": (
            "Dear Customer,\n\n"
            "We have detected unusual activity on your PayPal account. "
            "Your account has been temporarily LIMITED.\n\n"
            "To restore full access, please verify your identity within 24 hours:\n"
            ">> Click here to verify: http://paypal-verify.suspicious-link.net/login\n\n"
            "Failure to verify will result in permanent account suspension.\n\n"
            "PayPal Security Department"
        ),
        "timestamp": "2024-01-15T07:22:00Z",
        "attachments": [],
        "thread_history": [],
        "labels": [],
        "is_reply": False,
        "urgency_indicators": ["unusual activity", "verify now", "24 hours"],
        "is_spam": True,
        "ground_truth": {
            "priority": "high",
            "department": "spam",
            "is_spam": True,
            "follow_up_actions": ["flag_review", "archive"],
        },
    },
    {
        "task": "task2",
        "email_id": "t2_003",
        "from_address": "legal@bigco-acquisition.com",
        "from_name": "Robert Chen, Esq.",
        "to_address": "ceo@company.com",
        "subject": "Confidential: Letter of Intent — Acquisition Discussion",
        "body": (
            "Dear Executive Team,\n\n"
            "I am writing on behalf of BigCo Ventures regarding our interest in exploring "
            "a potential acquisition of your company. We have reviewed your recent Series B "
            "materials and believe there is a strong strategic fit.\n\n"
            "We would like to schedule an initial conversation this week. "
            "Please treat this correspondence with strict confidentiality.\n\n"
            "Robert Chen, Partner — BigCo Legal\n+1-555-0148"
        ),
        "timestamp": "2024-01-15T08:30:00Z",
        "attachments": [{"filename": "LOI_Draft.pdf", "file_type": "pdf", "size_kb": 512}],
        "thread_history": [],
        "labels": ["confidential"],
        "is_reply": False,
        "urgency_indicators": ["this week", "confidential"],
        "is_spam": False,
        "ground_truth": {
            "priority": "high",
            "department": "executive",
            "is_spam": False,
            "follow_up_actions": ["escalate", "schedule_meeting"],
        },
    },
    {
        "task": "task2",
        "email_id": "t2_004",
        "from_address": "marcus.lee@company.com",
        "from_name": "Marcus Lee",
        "to_address": "hr@company.com",
        "subject": "Reporting a workplace harassment incident",
        "body": (
            "Hello HR,\n\n"
            "I need to formally report an incident that occurred last Tuesday. "
            "During a team meeting, a colleague made repeated inappropriate comments "
            "directed at me in front of the entire team. I have documented the incident "
            "with timestamps and have a witness.\n\n"
            "I'd like to schedule a confidential meeting as soon as possible.\n\nMarcus"
        ),
        "timestamp": "2024-01-15T12:00:00Z",
        "attachments": [],
        "thread_history": [],
        "labels": ["confidential"],
        "is_reply": False,
        "urgency_indicators": ["harassment", "formal report"],
        "is_spam": False,
        "ground_truth": {
            "priority": "high",
            "department": "hr",
            "is_spam": False,
            "follow_up_actions": ["schedule_meeting", "flag_review"],
        },
    },
    {
        "task": "task2",
        "email_id": "t2_005",
        "from_address": "survey@customer-insights.com",
        "from_name": "Customer Insights Team",
        "to_address": "info@company.com",
        "subject": "Would you like to participate in our quarterly survey?",
        "body": (
            "Hi there!\n\n"
            "We're conducting our quarterly industry survey and would love your participation. "
            "The survey takes about 10 minutes and covers industry trends, tools, and challenges. "
            "As a thank-you, participants receive a $10 Amazon gift card.\n\n"
            "Complete the survey here: https://survey-link.com/q123\n\nThank you!"
        ),
        "timestamp": "2024-01-15T10:00:00Z",
        "attachments": [],
        "thread_history": [],
        "labels": [],
        "is_reply": False,
        "urgency_indicators": [],
        "is_spam": False,
        "ground_truth": {
            "priority": "low",
            "department": "marketing",
            "is_spam": False,
            "follow_up_actions": ["archive"],
        },
    },
    {
        "task": "task2",
        "email_id": "t2_006",
        "from_address": "devops@company.com",
        "from_name": "DevOps Alerts",
        "to_address": "engineering@company.com",
        "subject": "[ALERT] Database CPU at 94% — immediate attention needed",
        "body": (
            "AUTOMATED ALERT — Database Server: prod-db-01\n\n"
            "CPU Usage: 94.2% (threshold: 85%)\n"
            "Memory: 87% utilized\n"
            "Active Connections: 1,847 (max: 2,000)\n"
            "Slow Queries: 47 in last 5 minutes\n\n"
            "The production database is approaching critical capacity. "
            "Without intervention, connections will be rejected within approximately 30 minutes.\n\n"
            "On-call engineer: Please escalate immediately."
        ),
        "timestamp": "2024-01-15T14:30:00Z",
        "attachments": [],
        "thread_history": [],
        "labels": ["automated-alert"],
        "is_reply": False,
        "urgency_indicators": ["ALERT", "94%", "immediate attention", "30 minutes"],
        "is_spam": False,
        "ground_truth": {
            "priority": "high",
            "department": "engineering",
            "is_spam": False,
            "follow_up_actions": ["escalate", "create_ticket"],
        },
    },
    {
        "task": "task2",
        "email_id": "t2_007",
        "from_address": "diana.chen@company.com",
        "from_name": "Diana Chen",
        "to_address": "engineering@company.com",
        "subject": "Request to update the onboarding documentation",
        "body": (
            "Hi team,\n\n"
            "The onboarding docs for new engineers are a bit outdated — "
            "they still reference the old CI/CD pipeline we deprecated in October. "
            "Could someone update the Getting Started section when they have time? "
            "No rush, just flagging it.\n\nThanks, Diana"
        ),
        "timestamp": "2024-01-15T13:00:00Z",
        "attachments": [],
        "thread_history": [],
        "labels": [],
        "is_reply": False,
        "urgency_indicators": [],
        "is_spam": False,
        "ground_truth": {
            "priority": "low",
            "department": "engineering",
            "is_spam": False,
            "follow_up_actions": ["create_ticket"],
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 3 — HARD: Full Triage (10 emails)
    # ─────────────────────────────────────────────────────────────────────────
    {
        "task": "task3",
        "email_id": "t3_001",
        "from_address": "cto@megacorp.com",
        "from_name": "Dr. Hannah Park",
        "to_address": "engineering@company.com",
        "subject": "Security breach — customer PII potentially exposed",
        "body": (
            "To the engineering leadership,\n\n"
            "We have detected what appears to be unauthorized access to our shared API integration "
            "with your platform. Our logs show 15,000 customer records may have been accessed "
            "between 2 AM - 4 AM this morning using what looks like a compromised API key.\n\n"
            "We need your security team involved IMMEDIATELY. We have GDPR obligations "
            "and a 72-hour reporting window that started 3 hours ago. "
            "This is a Severity 1 incident.\n\nDr. Hannah Park, CTO"
        ),
        "timestamp": "2024-01-15T07:00:00Z",
        "attachments": [{"filename": "access_logs.txt", "file_type": "txt", "size_kb": 1024}],
        "thread_history": [],
        "labels": ["urgent", "security"],
        "is_reply": False,
        "urgency_indicators": ["security breach", "PII", "IMMEDIATELY", "72-hour", "Severity 1"],
        "is_spam": False,
        "ground_truth": {
            "priority": "high",
            "department": "engineering",
            "is_spam": False,
            "follow_up_actions": ["escalate", "create_ticket", "schedule_meeting"],
            "response_keywords": ["security team", "immediately", "72 hours", "investigating", "contact"],
        },
    },
    {
        "task": "task3",
        "email_id": "t3_002",
        "from_address": "alerts@microsofft-security.com",
        "from_name": "Microsoft Security",
        "to_address": "it@company.com",
        "subject": "Your Microsoft 365 license expires in 24 hours — renew now",
        "body": (
            "IMPORTANT NOTICE — Microsoft 365 Account\n\n"
            "Your Microsoft 365 Business license is set to expire in 24 hours. "
            "All users will lose access to email, Teams, and OneDrive.\n\n"
            "Renew immediately: http://microsofft-security.com/renew?ref=urgent\n\n"
            "Enter your admin credentials to verify and renew your subscription.\n\n"
            "Microsoft Support Team\n1-800-MICROSOFFT"
        ),
        "timestamp": "2024-01-15T06:00:00Z",
        "attachments": [],
        "thread_history": [],
        "labels": [],
        "is_reply": False,
        "urgency_indicators": ["expires in 24 hours", "IMPORTANT NOTICE"],
        "is_spam": True,
        "ground_truth": {
            "priority": "high",
            "department": "spam",
            "is_spam": True,
            "follow_up_actions": ["flag_review", "archive"],
            "response_keywords": ["phishing", "spam", "do not click", "report"],
        },
    },
    {
        "task": "task3",
        "email_id": "t3_003",
        "from_address": "linda.torres@vip-customer.com",
        "from_name": "Linda Torres",
        "to_address": "support@company.com",
        "subject": "Re: Re: Re: Still not resolved after 3 weeks — escalating to CEO",
        "body": (
            "This is absolutely unacceptable.\n\n"
            "I have been trying to get this billing issue resolved for THREE WEEKS. "
            "I've spoken to 4 different agents. I'm a Platinum customer paying $24,000/year. "
            "Each time I call, I'm told it's been escalated, but nothing happens.\n\n"
            "I am formally notifying you that if this is not resolved by end of business TODAY, "
            "I will be contacting your CEO directly, posting on LinkedIn, and exploring "
            "legal options for breach of contract.\n\n"
            "— Linda Torres, VP Operations, VIP Customer Inc."
        ),
        "timestamp": "2024-01-15T11:30:00Z",
        "attachments": [],
        "thread_history": [
            {"sender": "support@company.com", "timestamp": "2024-01-08T09:00:00Z",
             "body": "Hi Linda, I've escalated this to our billing team. You'll hear back within 48 hours."},
            {"sender": "linda.torres@vip-customer.com", "timestamp": "2024-01-10T14:00:00Z",
             "body": "It's been 48 hours and still nothing. This is the 2nd agent I've spoken to."},
            {"sender": "support@company.com", "timestamp": "2024-01-12T10:00:00Z",
             "body": "Apologies Linda, escalating again to senior billing."},
        ],
        "labels": ["vip", "escalated"],
        "is_reply": True,
        "urgency_indicators": ["3 weeks", "escalating to CEO", "end of business TODAY", "legal options"],
        "is_spam": False,
        "ground_truth": {
            "priority": "high",
            "department": "executive",
            "is_spam": False,
            "follow_up_actions": ["escalate", "schedule_meeting", "reply"],
            "response_keywords": ["apologize", "personally", "today", "resolve", "priority"],
        },
    },
    {
        "task": "task3",
        "email_id": "t3_004",
        "from_address": "tom.baker@company.com",
        "from_name": "Tom Baker",
        "to_address": "hr@company.com",
        "subject": "Notice of resignation — 2 weeks",
        "body": (
            "Hi HR Team,\n\n"
            "I'm writing to formally notify you of my resignation from my position as "
            "Senior Software Engineer, effective two weeks from today (January 29, 2024).\n\n"
            "I've greatly enjoyed my time here and have learned a lot. "
            "I'm happy to help with knowledge transfer and document my current projects "
            "during my notice period. Please let me know what I need to complete.\n\n"
            "Best regards,\nTom Baker"
        ),
        "timestamp": "2024-01-15T09:00:00Z",
        "attachments": [],
        "thread_history": [],
        "labels": [],
        "is_reply": False,
        "urgency_indicators": ["2 weeks"],
        "is_spam": False,
        "ground_truth": {
            "priority": "medium",
            "department": "hr",
            "is_spam": False,
            "follow_up_actions": ["reply", "schedule_meeting"],
            "response_keywords": ["acknowledge", "offboarding", "knowledge transfer", "exit interview"],
        },
    },
    {
        "task": "task3",
        "email_id": "t3_005",
        "from_address": "compliance@regulator.gov",
        "from_name": "Regulatory Compliance Office",
        "to_address": "legal@company.com",
        "subject": "Notice of Audit — Response Required Within 10 Business Days",
        "body": (
            "Dear Compliance Officer,\n\n"
            "This notice informs you that your company has been selected for a routine "
            "data privacy compliance audit pursuant to Section 42(b) of the Data Protection Act.\n\n"
            "You are required to provide the following documentation within 10 business days:\n"
            "1. Data processing records for fiscal year 2023\n"
            "2. Current data retention policy documentation\n"
            "3. Records of data subject access requests (DSARs) and their resolution\n"
            "4. Evidence of staff data privacy training\n\n"
            "Failure to respond within the statutory timeframe may result in penalties "
            "of up to $50,000 per day.\n\nCompliance Office — Regulatory Authority"
        ),
        "timestamp": "2024-01-15T08:00:00Z",
        "attachments": [{"filename": "Audit_Notice_2024.pdf", "file_type": "pdf", "size_kb": 256}],
        "thread_history": [],
        "labels": ["legal", "compliance"],
        "is_reply": False,
        "urgency_indicators": ["10 business days", "penalties", "$50,000 per day", "required"],
        "is_spam": False,
        "ground_truth": {
            "priority": "high",
            "department": "legal",
            "is_spam": False,
            "follow_up_actions": ["escalate", "create_ticket", "schedule_meeting"],
            "response_keywords": ["acknowledge", "compliance", "10 days", "documentation", "legal team"],
        },
    },
    {
        "task": "task3",
        "email_id": "t3_006",
        "from_address": "sales@spammy-vendor.biz",
        "from_name": "TurboBoost SEO Solutions",
        "to_address": "info@company.com",
        "subject": "🚀 BOOST YOUR GOOGLE RANKINGS IN 48 HOURS — GUARANTEED!!!",
        "body": (
            "Hello Business Owner!!!\n\n"
            "Tired of being on page 2 of Google?? Our EXCLUSIVE AI-powered SEO system "
            "GUARANTEES you #1 rankings in 48 hours or YOUR MONEY BACK!!!\n\n"
            "✅ 10,000 BACKLINKS OVERNIGHT\n"
            "✅ DOUBLE YOUR TRAFFIC IN 1 WEEK\n"
            "✅ BEAT YOUR COMPETITORS FOREVER\n\n"
            "LIMITED TIME OFFER: 90% OFF — TODAY ONLY\n"
            "Click here: http://turboseo-spam.biz/buy-now\n\n"
            "To unsubscribe, reply STOP (but why would you??)"
        ),
        "timestamp": "2024-01-15T05:00:00Z",
        "attachments": [],
        "thread_history": [],
        "labels": [],
        "is_reply": False,
        "urgency_indicators": ["48 HOURS", "TODAY ONLY"],
        "is_spam": True,
        "ground_truth": {
            "priority": "low",
            "department": "spam",
            "is_spam": True,
            "follow_up_actions": ["archive"],
            "response_keywords": [],
        },
    },
    {
        "task": "task3",
        "email_id": "t3_007",
        "from_address": "investor@vcfirm.com",
        "from_name": "Katherine Walsh",
        "to_address": "ceo@company.com",
        "subject": "Board deck feedback — Series C prep",
        "body": (
            "Hi,\n\n"
            "I've reviewed the draft board deck for the Series C round. "
            "Overall it's solid, but I have a few concerns:\n\n"
            "1. The ARR growth chart on slide 7 looks inconsistent with the numbers in the appendix.\n"
            "2. The competitive landscape section needs to address the recent Y Combinator cohort.\n"
            "3. The burn rate assumptions seem optimistic given current market conditions.\n\n"
            "I'd like to schedule a call before you send this to the full board. "
            "I'm available Thursday or Friday afternoon.\n\nBest,\nKatherine Walsh, Managing Partner"
        ),
        "timestamp": "2024-01-15T10:00:00Z",
        "attachments": [{"filename": "board_deck_v3_comments.pdf", "file_type": "pdf", "size_kb": 3072}],
        "thread_history": [],
        "labels": ["investor", "confidential"],
        "is_reply": False,
        "urgency_indicators": ["Series C", "board"],
        "is_spam": False,
        "ground_truth": {
            "priority": "high",
            "department": "executive",
            "is_spam": False,
            "follow_up_actions": ["schedule_meeting", "reply"],
            "response_keywords": ["Thursday", "Friday", "schedule", "address", "thank"],
        },
    },
    {
        "task": "task3",
        "email_id": "t3_008",
        "from_address": "sofia.m@customer.com",
        "from_name": "Sofia Mendez",
        "to_address": "support@company.com",
        "subject": "How do I reset my 2FA?",
        "body": (
            "Hi,\n\n"
            "I got a new phone and can't access my authenticator app anymore. "
            "How do I reset my two-factor authentication? I've looked at the help docs "
            "but can't find the right page.\n\nThanks,\nSofia"
        ),
        "timestamp": "2024-01-15T13:45:00Z",
        "attachments": [],
        "thread_history": [],
        "labels": [],
        "is_reply": False,
        "urgency_indicators": [],
        "is_spam": False,
        "ground_truth": {
            "priority": "medium",
            "department": "support",
            "is_spam": False,
            "follow_up_actions": ["reply"],
            "response_keywords": ["account settings", "security", "verify identity", "steps", "help"],
        },
    },
    {
        "task": "task3",
        "email_id": "t3_009",
        "from_address": "procurement@bigretail.com",
        "from_name": "Procurement Team",
        "to_address": "sales@company.com",
        "subject": "RFP: Enterprise Software — Proposals due Feb 1",
        "body": (
            "Dear Sales Team,\n\n"
            "BigRetail Corp is issuing an RFP for enterprise software solutions. "
            "We are evaluating vendors for a 3-year contract valued at approximately $2M.\n\n"
            "Key requirements:\n"
            "- Integration with SAP ERP\n"
            "- SOC 2 Type II certification\n"
            "- 99.9% SLA guarantee\n"
            "- 24/7 support coverage\n\n"
            "Proposals are due February 1, 2024. Please see the attached RFP for full details.\n\n"
            "Procurement Team, BigRetail Corp"
        ),
        "timestamp": "2024-01-15T09:00:00Z",
        "attachments": [{"filename": "RFP_Enterprise_Software_2024.pdf", "file_type": "pdf", "size_kb": 4096}],
        "thread_history": [],
        "labels": ["rfp", "high-value"],
        "is_reply": False,
        "urgency_indicators": ["Feb 1 deadline", "$2M"],
        "is_spam": False,
        "ground_truth": {
            "priority": "high",
            "department": "sales",
            "is_spam": False,
            "follow_up_actions": ["reply", "schedule_meeting", "create_ticket"],
            "response_keywords": ["RFP", "February 1", "proposal", "requirements", "team"],
        },
    },
    {
        "task": "task3",
        "email_id": "t3_010",
        "from_address": "michael.grant@company.com",
        "from_name": "Michael Grant",
        "to_address": "engineering@company.com",
        "subject": "Idea: Add dark mode to the dashboard",
        "body": (
            "Hey team,\n\n"
            "Just a thought — a lot of our users have been asking for dark mode on the dashboard. "
            "Might be worth adding to the product roadmap discussion. "
            "I did a quick user survey and 67% said they'd use it.\n\n"
            "Not urgent at all, just wanted to flag it for the next sprint planning.\n\nMike"
        ),
        "timestamp": "2024-01-15T15:00:00Z",
        "attachments": [],
        "thread_history": [],
        "labels": [],
        "is_reply": False,
        "urgency_indicators": [],
        "is_spam": False,
        "ground_truth": {
            "priority": "low",
            "department": "engineering",
            "is_spam": False,
            "follow_up_actions": ["create_ticket"],
            "response_keywords": ["roadmap", "sprint", "noted", "feature request"],
        },
    },
]


def get_emails_for_task(task_id: str) -> list[dict]:
    return [e for e in EMAILS if e["task"] == task_id]
