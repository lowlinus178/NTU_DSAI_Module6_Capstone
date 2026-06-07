"""
eval_suite.py — 10 ground-truth Q&A pairs, each explicitly answerable
from the four policy documents.

Each entry has:
  question       : the question to ask the RAG system
  expected_answer: a concise reference answer (used by LLM judge)
  keywords       : terms the answer MUST contain for heuristic scoring
  source_doc     : which document the answer lives in (for faithfulness check)
"""

EVAL_QUESTIONS = [
    # ── HR Leave Policy ──────────────────────────────────────────────────────
    {
        "id": "hr_01",
        "question": "How many days of paid annual leave do full-time employees get per year?",
        "expected_answer": "Full-time employees accrue 20 days of paid annual leave per calendar year.",
        "keywords": ["20", "annual leave", "calendar year"],
        "source_doc": "hr_leave_policy",
    },
    {
        "id": "hr_02",
        "question": "How many days of sick leave are employees entitled to per year?",
        "expected_answer": "Employees are entitled to 10 days of paid sick leave per calendar year.",
        "keywords": ["10", "sick leave"],
        "source_doc": "hr_leave_policy",
    },
    {
        "id": "hr_03",
        "question": "How much paid parental leave does a primary caregiver receive?",
        "expected_answer": "Primary caregivers are entitled to 16 weeks of paid parental leave.",
        "keywords": ["16 weeks", "primary caregiver", "parental leave"],
        "source_doc": "hr_leave_policy",
    },
    {
        "id": "hr_04",
        "question": "How many days of unused annual leave can be carried over to the following year?",
        "expected_answer": "Up to 5 days of unused annual leave may be carried into the following year.",
        "keywords": ["5 days", "carried", "following year"],
        "source_doc": "hr_leave_policy",
    },

    # ── Expense Claims ───────────────────────────────────────────────────────
    {
        "id": "exp_01",
        "question": "What is the daily meal cap when travelling for business?",
        "expected_answer": "The daily meal cap while travelling is $80 covering breakfast, lunch, and dinner combined.",
        "keywords": ["80", "meal", "travelling"],
        "source_doc": "expense_claims",
    },
    {
        "id": "exp_02",
        "question": "How long should I keep original receipts after submitting an expense claim?",
        "expected_answer": "Original receipts should be kept for 90 days after submission.",
        "keywords": ["90 days", "receipt"],
        "source_doc": "expense_claims",
    },
    {
        "id": "exp_03",
        "question": "What is the annual office supplies budget for remote workers?",
        "expected_answer": "Remote workers can claim up to $200 per year for office supplies against a receipt.",
        "keywords": ["200", "office supplies", "remote"],
        "source_doc": "expense_claims",
    },

    # ── IT Support Policy ────────────────────────────────────────────────────
    {
        "id": "it_01",
        "question": "What is the first response target for a P1 IT support ticket?",
        "expected_answer": "P1 tickets (cannot work at all) have a first response target of within 30 minutes.",
        "keywords": ["30 minutes", "P1", "response"],
        "source_doc": "it_support_policy",
    },
    {
        "id": "it_02",
        "question": "How do I report a suspected security incident like a lost device?",
        "expected_answer": "Report it through the Security portal AND notify the IT team within 1 hour.",
        "keywords": ["Security portal", "IT team", "1 hour"],
        "source_doc": "it_support_policy",
    },

    # ── Incident Escalation ──────────────────────────────────────────────────
    {
        "id": "inc_01",
        "question": "How often should status updates be posted during a SEV1 incident?",
        "expected_answer": "Status updates should be posted every 15 minutes during a SEV1 incident.",
        "keywords": ["15 minutes", "SEV1", "update"],
        "source_doc": "incident_escalation",
    },
]
