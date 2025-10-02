"""Streamlit UI showcasing the process mapping agent."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_agent.agent import ProcessMappingAgent

st.set_page_config(page_title="Launchpad Process Agent", page_icon="🛰️", layout="wide")
st.title("🛰️ Launchpad Process Agent")
st.caption(
    "Map the current state in plain language and the agent will surface the flow and improvement opportunities."
)


@st.cache_resource(show_spinner=False)
def get_process_agent() -> ProcessMappingAgent:
    return ProcessMappingAgent()


# Render Mermaid diagram via embedded component so users see the graph rather than the source.
def _render_mermaid_diagram(diagram_source: str) -> None:
    element_id = f"mermaid-{uuid.uuid4().hex}"
    mermaid_html = f"""
    <div id="{element_id}" class="mermaid">{diagram_source}</div>
    <script type="text/javascript">
    const renderMermaid = () => {{
        const el = document.getElementById('{element_id}');
        if (!el) {{
            return;
        }}
        mermaid.initialize({{ startOnLoad: false }});
        mermaid.init(undefined, el);
    }};
    if (window.mermaid) {{
        renderMermaid();
    }} else {{
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js';
        script.onload = renderMermaid;
        document.head.appendChild(script);
    }}
    </script>
    """
    components.html(mermaid_html, height=500, scrolling=True)



def _render_process_tab(agent: ProcessMappingAgent) -> None:
    st.subheader("Process Mapping Agent")
    st.write(
        "Describe the current state in plain language and the agent will map the flow."
        " The output includes a Mermaid diagram and a DOWNTIME waste scan to guide improvements."
    )

    with st.form("process_agent"):
        problem_statement = st.text_area(
            "Describe the current process",
            height=140,
            placeholder=(
                "Example: Service tickets arrive via web form, support triages for severity,"
                " escalates complex cases to specialists, and closes the loop with a customer survey."
            ),
        )
        detail_level = st.radio(
            "Depth of playbook",
            options=("compact", "balanced", "deep"),
            horizontal=True,
            index=1,
        )
        submitted = st.form_submit_button("Generate process diagram", type="primary")

    if submitted:
        try:
            plan = agent.map_process(problem_statement, detail_level=detail_level)
        except ValueError as exc:
            st.warning(str(exc))
            return

        st.success("Draft process ready for review.")

        st.markdown("#### Process diagram")
        _render_mermaid_diagram(plan.mermaid)

        overview_tab, stages_tab, assumptions_tab, risks_tab, downtime_tab = st.tabs(
            [
                "Overview",
                "Key stages",
                "Operating assumptions",
                "Watch-outs",
                "DOWNTIME opportunities",
            ]
        )

        with overview_tab:
            st.markdown("**Process summary**")
            st.write(plan.summary)
            if plan.keywords:
                st.caption(
                    "Frequent themes: "
                    + ", ".join(f"`{keyword}`" for keyword in plan.keywords)
                )

        with stages_tab:
            st.write("Open a stage to review the focus, guiding question, and deliverable.")
            for step in plan.steps:
                with st.expander(f"{step.order}. {step.title}", expanded=step.order == 1):
                    st.markdown(
                        f"- Focus: `{step.focus}`\n"
                        f"- Guiding question: {step.question}\n"
                        f"- Deliverable: {step.deliverable}"
                    )

        with assumptions_tab:
            if plan.assumptions:
                for assumption in plan.assumptions:
                    st.markdown(f"- {assumption}")
            else:
                st.info("No assumptions captured for this draft.")

        with risks_tab:
            if plan.risks:
                for risk in plan.risks:
                    st.markdown(f"- {risk}")
            else:
                st.info("No watch-outs flagged yet—pressure test the process with your team.")

        with downtime_tab:
            if plan.downtime_opportunities:
                downtime_letters = [
                    ("D", "Defects"),
                    ("O", "Overproduction"),
                    ("W", "Waiting"),
                    ("N", "Non-utilized talent"),
                    ("T", "Transportation"),
                    ("I", "Inventory"),
                    ("M", "Motion"),
                    ("E", "Excess processing"),
                ]
                grouped_opportunities = {letter: [] for letter, _ in downtime_letters}
                for opportunity in plan.downtime_opportunities:
                    matched = False
                    for letter, category_name in downtime_letters:
                        if opportunity.category.lower().startswith(category_name.lower()):
                            grouped_opportunities[letter].append(opportunity)
                            matched = True
                            break
                    if not matched:
                        key = opportunity.category[:1].upper()
                        grouped_opportunities.setdefault(key, []).append(opportunity)

                letter_tabs = st.tabs(
                    [f"{letter} – {category}" for letter, category in downtime_letters]
                )
                for letter_tab, (letter, category_name) in zip(letter_tabs, downtime_letters):
                    with letter_tab:
                        opportunities = grouped_opportunities.get(letter, [])
                        if opportunities:
                            for opportunity in opportunities:
                                st.markdown(
                                    f"**{opportunity.category}**\n"
                                    f"{opportunity.definition}"
                                )
                                st.write(opportunity.recommendation)
                                if opportunity.trigger:
                                    st.caption(f"Signal spotted: {opportunity.trigger}")
                        else:
                            st.write("No opportunities tagged for this category yet.")
            else:
                st.info("The agent did not spot DOWNTIME opportunities in this draft.")


process_agent = get_process_agent()
_render_process_tab(process_agent)

st.info(
    "Outputs are starting points — review the steps, assumptions, and DOWNTIME scan before implementation."
)
