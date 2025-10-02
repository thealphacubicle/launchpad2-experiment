"""Agent that analyses DOWNTIME waste using an LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..llm_client import ChatMessage, LLMClient, LLMError, extract_json


@dataclass
class ReasoningOpportunity:
    category: str
    definition: str
    trigger: Optional[str]
    recommendation: str
    insight: str


@dataclass
class ReasoningOutput:
    opportunities: List[ReasoningOpportunity]


class ReasoningAgent:
    """Leverage an LLM to scan a process for DOWNTIME signals."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def analyse(self, statement: str) -> ReasoningOutput:
        system_prompt = (
            "You are a lean operations expert. Assess the process description using the"
            " DOWNTIME waste categories."
        )
        user_prompt = (
            "Process description:\n"
            f"{statement.strip()}\n\n"
            "Return JSON with the schema {\n"
            '  "opportunities": [\n'
            '    {"category": string, "definition": string, "trigger": string | null, '
            '"recommendation": string, "insight": string},\n'
            "    ...\n"
            "  ]\n"
            "}\n"
            "Include only categories with a reasonable signal. The definition should restate"
            " what that waste means for this context. The insight should explain the rationale"
            " referencing evidence from the description. The trigger can be null when no"
            " explicit phrase is available."
        )

        raw_response = self._llm.complete(
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ]
        )

        payload = extract_json(raw_response)
        if not isinstance(payload, dict):
            raise LLMError("Reasoning agent expected a JSON object")

        opportunities_raw = payload.get("opportunities")
        if opportunities_raw is None:
            raise LLMError("Reasoning agent response missing 'opportunities'")
        if not isinstance(opportunities_raw, list):
            raise LLMError("Reasoning agent opportunities must be a list")

        opportunities: List[ReasoningOpportunity] = []
        for item in opportunities_raw:
            if not isinstance(item, dict):
                raise LLMError("Reasoning agent opportunity must be an object")
            category = str(item.get("category", "")).strip()
            definition = str(item.get("definition", "")).strip()
            recommendation = str(item.get("recommendation", "")).strip()
            insight = str(item.get("insight", "")).strip()
            trigger_value = item.get("trigger")
            trigger = str(trigger_value).strip() if isinstance(trigger_value, str) else None

            if not category or not definition or not recommendation or not insight:
                raise LLMError("Reasoning agent produced an incomplete opportunity")

            opportunities.append(
                ReasoningOpportunity(
                    category=category,
                    definition=definition,
                    trigger=trigger,
                    recommendation=recommendation,
                    insight=insight,
                )
            )

        return ReasoningOutput(opportunities=opportunities)
