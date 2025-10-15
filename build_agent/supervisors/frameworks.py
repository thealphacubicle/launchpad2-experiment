from __future__ import annotations

from typing import Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import tool


@tool("suggest_frameworks")
def suggest_frameworks(topic: str) -> str:
    """Suggest relevant frameworks, templates, or best practices for a topic.

    This is a lightweight placeholder to establish the Frameworks Supervisor. It
    produces quick, practical suggestions without deep analysis.
    """
    base = (
        "General guidance for frameworks. Tailor to org context: "
        "- RACI for roles\n- DACI for decisions\n- OKRs for outcomes\n"
        "- A3 or 5-Whys for problem-solving\n- Scrum/Kanban for delivery\n"
    )
    return f"Topic: {topic}\n\n{base}"


SYSTEM = (
    "You are the Frameworks Supervisor. Offer concise, actionable frameworks and "
    "templates to structure work. Prefer the `suggest_frameworks` tool to create "
    "immediately usable outputs."
)


def build_frameworks_supervisor(
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.2,
    api_key: Optional[str] = None,
) -> AgentExecutor:
    llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
    tools = [suggest_frameworks]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)


__all__ = ["build_frameworks_supervisor", "suggest_frameworks"]

