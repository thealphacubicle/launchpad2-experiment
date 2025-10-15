from __future__ import annotations

from typing import Optional, Any, Dict

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from build_agent.tools.opendata_mcp import boston_opendata


SYSTEM = (
    "You are the Research Supervisor. Use the Boston OpenData MCP tool for Boston city "
    "data requests. If the question references city datasets, resource IDs, schemas, or "
    "DataStore querying, use the `boston_opendata` tool with a concise JSON args string."
)


def build_research_supervisor(
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.2,
    api_key: Optional[str] = None,
) -> AgentExecutor:
    llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)

    tools = [boston_opendata]

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


__all__ = ["build_research_supervisor"]
