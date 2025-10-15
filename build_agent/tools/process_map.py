"""Documentation tools: process map generator.

This turns a natural language description of a process into a Mermaid flowchart
and a succinct textual summary. Implemented as a LangChain tool so supervisors
can call it directly.
"""

from __future__ import annotations

from typing import Optional

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os


SYSTEM = (
    "You convert process descriptions into structured Mermaid flowcharts. "
    "Always output a fenced mermaid 'flowchart TD' block followed by a short summary."
)


@tool("generate_process_map")
def generate_process_map(description: str, detail: str = "concise", model: Optional[str] = None) -> str:
    """Generate a Mermaid process map from a text description.

    - description: free text describing steps, actors, decisions.
    - detail: 'concise' | 'detailed' controls node granularity.
    - model: optional override of the chat model id.

    Returns Mermaid code and a short summary in plain text.
    """
    model_id = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model=model_id, temperature=0.1, api_key=api_key)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            (
                "human",
                "Create a {detail} Mermaid flowchart capturing the process below. "
                "Use clear node names and decision diamonds. Then provide a 3-5 bullet summary.\n\n"
                "Process:\n{description}",
            ),
        ]
    )

    chain = prompt | llm
    res = chain.invoke({"description": description, "detail": detail})
    return res.content if hasattr(res, "content") else str(res)


__all__ = ["generate_process_map"]

