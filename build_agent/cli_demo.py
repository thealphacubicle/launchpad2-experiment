"""Command line entry point for the LangChain agent demo."""

from __future__ import annotations

import argparse
from typing import Any, Dict

from .agent import build_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the demo agent once")
    parser.add_argument(
        "question",
        help="A natural language request for the agent, e.g. 'Summarize today's AI news'.",
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="The OpenAI chat model to use (defaults to gpt-3.5-turbo).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature passed to the chat model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = build_agent(model=args.model, temperature=args.temperature)
    result: Dict[str, Any] = agent.invoke({"input": args.question})

    print("\n=== Final answer ===")
    print(result["output"])


if __name__ == "__main__":
    main()
