"""Command line entry point for the mock agent system."""

from __future__ import annotations

import argparse
from typing import Iterable

from build_agent.agent import ProcessMappingAgent, ResearchLaunchAgent


def _print_lines(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)


def _handle_process(args: argparse.Namespace) -> None:
    agent = ProcessMappingAgent()
    plan = agent.map_process(args.problem, detail_level=args.detail)

    print("\n=== Process Overview ===")
    print(plan.summary)

    if plan.follow_up_questions:
        print("\nClarifying questions:")
        _print_lines(f"- {question}" for question in plan.follow_up_questions)

    print("\nMermaid diagram (paste into a Mermaid renderer):\n")
    print(plan.mermaid)

    print("\nAssumptions:")
    _print_lines(f"- {assumption}" for assumption in plan.assumptions)

    print("\nRisks:")
    _print_lines(f"- {risk}" for risk in plan.risks)

    if plan.downtime_opportunities:
        print("\nDOWNTIME opportunities:")
        for opportunity in plan.downtime_opportunities:
            print(f"- {opportunity.category}: {opportunity.insight}")
            print(f"  Why it matters: {opportunity.definition}")
            print(f"  Next move: {opportunity.recommendation}")
            if opportunity.trigger:
                print(f"  Signal: {opportunity.trigger}")


def _handle_research(args: argparse.Namespace) -> None:
    agent = ResearchLaunchAgent()
    report = agent.launch_brief(
        args.problem,
        budget=args.budget,
        scale=args.scale,
        constraints=args.constraints,
    )

    print("\n=== Research Brief ===")
    print(report.narrative)

    if report.portal_messages:
        print("\nPortal notes:")
        _print_lines(f"- {message}" for message in report.portal_messages)
        print(f"Source: {report.data_source}")

    if report.matches:
        print("\nRecommended datasets:")
        for match in report.matches:
            print(f"- {match.title} ({match.coverage}) -> {match.portal_url}")
            print(f"  Summary: {match.summary}")
            print(f"  Why useful: {match.why_useful}")
            print(f"  Score: {match.score:.2f}")
            if match.publisher:
                print(f"  Publisher: {match.publisher}")
            if match.last_updated:
                print(f"  Last updated: {match.last_updated}")
            if match.tags:
                print(f"  Tags: {', '.join(match.tags)}")

    if report.next_steps:
        print("\nNext steps:")
        _print_lines(f"- {step}" for step in report.next_steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interact with the mock agents from the CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser(
        "process", help="Generate a process map diagram for a problem statement"
    )
    process_parser.add_argument("problem", help="Problem statement to analyse")
    process_parser.add_argument(
        "--detail",
        default="balanced",
        choices=["compact", "balanced", "deep"],
        help="Level of detail to include in the generated process steps",
    )
    process_parser.set_defaults(func=_handle_process)

    research_parser = subparsers.add_parser(
        "research", help="Produce a research brief with dataset suggestions"
    )
    research_parser.add_argument("problem", help="Problem the launch system should explore")
    research_parser.add_argument(
        "--budget",
        type=float,
        help="Approximate budget ceiling in USD for the initiative",
    )
    research_parser.add_argument(
        "--scale",
        help="Target scale (e.g. city, national, global) to prioritise datasets",
    )
    research_parser.add_argument(
        "--constraints",
        help="Additional freeform constraints (sector, timeline, stakeholders)",
    )
    research_parser.set_defaults(func=_handle_research)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
