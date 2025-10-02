"""Agent that drafts process artefacts using an LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..llm_client import ChatMessage, LLMClient, LLMError, extract_json


@dataclass
class DrawerStage:
    order: int
    title: str
    focus: str
    question: str
    deliverable: str


@dataclass
class DrawerOutput:
    summary: str
    keywords: List[str]
    stages: List[DrawerStage]
    assumptions: List[str]
    risks: List[str]
    mermaid: str


class DrawerAgent:
    """Invoke an LLM to draft the process outline and flowchart."""

    _LEVEL_TO_COUNT: Dict[str, int] = {"compact": 4, "balanced": 5, "deep": 6}

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def draw(self, statement: str, *, detail_level: str) -> DrawerOutput:
        normalized_level = detail_level.lower().strip()
        target_steps = self._LEVEL_TO_COUNT.get(normalized_level, 5)

        system_prompt = (
            "You are a process mapping expert. Produce a structured playbook for the described"
            " process including a Mermaid flowchart."
        )
        user_prompt = (
            "Process description:\n" + statement.strip() + "\n\n"
            f"Produce EXACTLY {target_steps} stages. Respond strictly with JSON using the schema:\n"
            "{\n"
            "  \"summary\": string,\n"
            "  \"keywords\": [string, ...],\n"
            "  \"stages\": [\n"
            "    {\n"
            "      \"order\": integer starting at 1,\n"
            "      \"title\": string,\n"
            "      \"focus\": string,\n"
            "      \"question\": string,\n"
            "      \"deliverable\": string\n"
            "    }, ... (exactly the specified number of stages)\n"
            "  ],\n"
            "  \"assumptions\": [string, ...],\n"
            "  \"risks\": [string, ...],\n"
            "  \"mermaid\": string containing a flowchart in Mermaid syntax\n"
            "}\n"
            "Keep the flowchart consistent with the stages and use node identifiers S1..Sn."
        )

        raw_response = self._llm.complete(
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ]
        )

        payload = extract_json(raw_response)
        if not isinstance(payload, dict):
            raise LLMError("Drawer agent expected a JSON object")

        summary = str(payload.get("summary", "")).strip()
        keywords_raw = payload.get("keywords")
        assumptions_raw = payload.get("assumptions")
        risks_raw = payload.get("risks")
        stages_raw = payload.get("stages")
        mermaid = str(payload.get("mermaid", "")).strip()

        if not summary:
            raise LLMError("Drawer agent response missing summary")
        if not isinstance(stages_raw, list):
            raise LLMError("Drawer agent response missing stages list")
        if len(stages_raw) != target_steps:
            raise LLMError("Drawer agent returned incorrect number of stages")

        stages: List[DrawerStage] = []
        for idx, stage in enumerate(stages_raw, start=1):
            if not isinstance(stage, dict):
                raise LLMError("Drawer agent stage payload must be an object")
            order = int(stage.get("order", idx))
            title = str(stage.get("title", "")).strip()
            focus = str(stage.get("focus", "")).strip()
            question = str(stage.get("question", "")).strip()
            deliverable = str(stage.get("deliverable", "")).strip()

            if not title or not focus or not question or not deliverable:
                raise LLMError("Drawer agent produced a stage with missing fields")

            stages.append(
                DrawerStage(
                    order=order,
                    title=title,
                    focus=focus,
                    question=question,
                    deliverable=deliverable,
                )
            )

        for sequence, label in (
            (keywords_raw, "keywords"),
            (assumptions_raw, "assumptions"),
            (risks_raw, "risks"),
        ):
            if sequence is None:
                continue
            if not isinstance(sequence, list):
                raise LLMError(f"Drawer agent field '{label}' must be a list")

        keywords = _clean_list(keywords_raw)
        assumptions = _clean_list(assumptions_raw)
        risks = _clean_list(risks_raw)

        if "flowchart" not in mermaid.lower():
            raise LLMError("Drawer agent Mermaid output must define a flowchart")

        return DrawerOutput(
            summary=summary,
            keywords=keywords,
            stages=stages,
            assumptions=assumptions,
            risks=risks,
            mermaid=mermaid,
        )


def _clean_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    cleaned: List[str] = []
    for item in value:
        if isinstance(item, str):
            candidate = item.strip()
            if candidate:
                cleaned.append(candidate)
    return cleaned

