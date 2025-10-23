from __future__ import annotations

from typing import Optional

from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from build_agent.tools.process_map import generate_process_map


SYSTEM = (
    "You are the Documentation Supervisor. Your primary tool is "
    "`generate_process_map`, which turns text into Mermaid diagrams and a short summary. "
    "Use it whenever the user requests documentation, process flows, SOPs, or diagrams."
)


def build_documentation_supervisor(
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    api_key: Optional[str] = None,
) -> AgentExecutor:
    llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
    tools = [generate_process_map]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
    )


__all__ = ["build_documentation_supervisor"]
