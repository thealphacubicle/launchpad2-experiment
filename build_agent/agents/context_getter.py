"""Agent that surfaces follow-up questions for additional context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..llm_client import ChatMessage, LLMClient, LLMError, extract_json


@dataclass
class ContextOutput:
    questions: List[str]


class ContextGetterAgent:
    """Use an LLM to generate clarifying questions about a process description."""

    def __init__(self, llm: LLMClient, *, max_questions: int = 5) -> None:
        self._llm = llm
        self._max_questions = max_questions

    def generate_questions(self, statement: str) -> ContextOutput:
        system_prompt = (
            "You are a discovery consultant. Given a process description, generate targeted "
            "follow-up questions that would help someone map the process with more confidence."
        )
        user_prompt = (
            "Process description:\n" + statement.strip() + "\n\n"
            f"Return up to {self._max_questions} clarifying questions as a JSON array of strings."
            " Each question should focus on uncovering missing context such as triggers, owners,"
            " metrics, or pain points. Do not include any other commentary."
        )

        raw_response = self._llm.complete(
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ]
        )

        payload = extract_json(raw_response)
        if not isinstance(payload, list):
            raise LLMError("Context agent expected a JSON array of questions")

        questions: List[str] = []
        for item in payload:
            if isinstance(item, str):
                candidate = item.strip()
                if candidate:
                    questions.append(candidate)

        return ContextOutput(questions=questions[: self._max_questions])

