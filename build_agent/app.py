"""Streamlit UI showcasing the process mapping agent."""

from __future__ import annotations

import uuid

import streamlit as st
import streamlit.components.v1 as components

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

        st.markdown("#### Overview")
        st.write(plan.summary)

        st.markdown("#### Process diagram")
        _render_mermaid_diagram(plan.mermaid)

        st.markdown("#### Key stages")
        for step in plan.steps:
            st.markdown(
                f"**{step.order}. {step.title}**\n"
                f"- Focus: `{step.focus}`\n"
                f"- Guiding question: {step.question}\n"
                f"- Deliverable: {step.deliverable}"
            )

        st.markdown("#### Operating assumptions")
        for assumption in plan.assumptions:
            st.markdown(f"- {assumption}")

        st.markdown("#### Watch-outs")
        for risk in plan.risks:
            st.markdown(f"- {risk}")

        if plan.downtime_opportunities:
            st.markdown("#### DOWNTIME opportunities")
            for opportunity in plan.downtime_opportunities:
                st.markdown(
                    f"**{opportunity.category}** - {opportunity.definition}\n"
                    f"{opportunity.recommendation}"
                )
                if opportunity.trigger:
                    st.caption(f"Signal spotted: {opportunity.trigger}")


process_agent = get_process_agent()
_render_process_tab(process_agent)

st.info(
    "Outputs are starting points — review the steps, assumptions, and DOWNTIME scan before implementation."
)
