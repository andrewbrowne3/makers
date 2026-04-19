"""Fallback agent — the router routes here when it's unsure.

For now it just formulates a clarifying question. Future: will actually
message the designer via Telegram / Slack / email.
"""
from __future__ import annotations

from app.agents.base import ReactAgent
from app.providers.llm import get_llm


def _noop(*, question: str = "", _ctx=None) -> str:
    """Record the clarifying question (stub for future chat integration)."""
    return f"queued question: {question}"


class QuestionAgent(ReactAgent):
    name = "question_agent"
    max_steps = 2
    system_prompt = (
        "The system needs more information to proceed. "
        "Formulate ONE concise clarifying question for the designer, then FINAL."
        "\n\nUse ACT: record_question(question=\"...\") once, then FINAL: the same question."
    )


def build_question_agent() -> QuestionAgent:
    return QuestionAgent(llm=get_llm(), tools={"record_question": _noop})
