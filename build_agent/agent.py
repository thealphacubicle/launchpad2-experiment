"""Mock agent implementations for the Streamlit demo."""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import itertools
import math
import re
from collections import Counter
from urllib.parse import urlparse

import requests
from requests import RequestException


_STOPWORDS = {
    "about",
    "after",
    "against",
    "all",
    "also",
    "and",
    "are",
    "around",
    "as",
    "at",
    "be",
    "been",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "create",
    "deliver",
    "design",
    "develop",
    "for",
    "from",
    "get",
    "had",
    "has",
    "have",
    "if",
    "in",
    "into",
    "improve",
    "increase",
    "it",
    "its",
    "like",
    "more",
    "need",
    "not",
    "optimize",
    "of",
    "on",
    "or",
    "our",
    "out",
    "over",
    "should",
    "so",
    "than",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "to",
    "transform",
    "under",
    "use",
    "we",
    "were",
    "what",
    "when",
    "which",
    "while",
    "will",
    "with",
    "within",
    "would",
    "upgrade",
}


def _extract_keywords(text: str, *, limit: int = 12) -> List[str]:
    """Return the most common non-stop words in ``text``."""

    words = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    counts: Counter[str] = Counter()
    for word in words:
        if word in _STOPWORDS or len(word) <= 2:
            continue
        counts[word] += 1
    return [word for word, _ in counts.most_common(limit)]


def _escape_mermaid(value: str) -> str:
    """Escape characters that conflict with Mermaid labels."""

    cleaned = value.replace("\n", " ").replace("\"", "'")
    return re.sub(r"\s+", " ", cleaned).strip()


@dataclass
class ProcessStep:
    order: int
    title: str
    focus: str
    question: str
    deliverable: str


@dataclass
class DowntimeOpportunity:
    category: str
    definition: str
    trigger: Optional[str]
    recommendation: str


@dataclass
class ProcessPlan:
    summary: str
    steps: List[ProcessStep]
    mermaid: str
    assumptions: List[str]
    risks: List[str]
    keywords: List[str]
    downtime_opportunities: List[DowntimeOpportunity]


class ProcessMappingAgent:
    """Deterministic agent that drafts process diagrams from a problem statement."""

    _fallback_templates: Sequence[Dict[str, str]] = (
        {
            "title": "Frame the challenge",
            "question": "Which stakeholders own {focus}?",
            "deliverable": "Problem brief that states desired outcomes and guardrails.",
        },
        {
            "title": "Map the current journey",
            "question": "What happens today when {focus} is addressed?",
            "deliverable": "Swimlane of the current process with latency hotspots highlighted.",
        },
        {
            "title": "Surface constraints",
            "question": "Which constraints or policies limit progress on {focus}?",
            "deliverable": "Constraint backlog prioritised by impact and confidence.",
        },
        {
            "title": "Generate solution paths",
            "question": "Which solution themes could move {focus} forward?",
            "deliverable": "Option matrix comparing feasibility, impact, and lift.",
        },
        {
            "title": "Prototype & test",
            "question": "How will we de-risk the top idea for {focus}?",
            "deliverable": "Rapid experiment charter with success metrics.",
        },
        {
            "title": "Plan rollout",
            "question": "What does adoption of the {focus} solution require?",
            "deliverable": "Rollout checklist with communications, training, and measurement.",
        },
    )

    _level_to_count: Dict[str, int] = {"compact": 4, "balanced": 5, "deep": 6}

    _step_questions: Tuple[str, ...] = (
        "What triggers this stage for {focus}, and who validates readiness?",
        "What information must be captured before handing off {focus}?",
        "How do we detect quality issues while {focus} is in motion?",
        "Who resolves blockers when {focus} stalls at this point?",
        "What metrics signal that work on {focus} can progress?",
        "How do we capture lessons from {focus} before closing the loop?",
    )

    _step_deliverables: Tuple[str, ...] = (
        "Clear intake checklist documented for {step}.",
        "Owners and collaborators aligned on {step} expectations.",
        "Quality controls defined for the {step} activity.",
        "Visible handoff criteria agreed for {step}.",
        "Instrumentation or KPIs established for {step}.",
        "Retrospective notes captured from {step}.",
    )

    _downtime_catalog: Tuple[Dict[str, object], ...] = (
        {
            "category": "Defects",
            "definition": "Rework or corrections caused by errors in outputs.",
            "keywords": ("defect", "error", "rework", "bug", "complaint", "return", "fail"),
            "recommendation": "Set explicit acceptance criteria or add automated checks for '{step}' to limit defects.",
            "fallback": "Add quality gates for {focus} to prevent defects and rework.",
        },
        {
            "category": "Overproduction",
            "definition": "Producing more than is needed or earlier than required.",
            "keywords": ("duplicate", "copy", "redundant", "extra", "batch", "over", "ahead"),
            "recommendation": "Right-size batch sizes or approvals around '{step}' so work aligns with actual demand.",
            "fallback": "Review where {focus} creates outputs that exceed downstream demand.",
        },
        {
            "category": "Waiting",
            "definition": "Idle time when people or systems wait for the next action.",
            "keywords": ("wait", "delay", "pending", "queue", "idle", "hold", "bottleneck"),
            "recommendation": "Expose queue length or SLA alerts for '{step}' to shorten waiting time.",
            "fallback": "Map bottlenecks so {focus} does not stall between stages.",
        },
        {
            "category": "Non-utilized talent",
            "definition": "Skills and knowledge not leveraged during the process.",
            "keywords": ("handoff", "manual", "approval", "review", "sign-off", "escalate", "specialist"),
            "recommendation": "Invite front-line insights into '{step}' to unlock latent expertise.",
            "fallback": "Check whether roles aligned to {focus} can contribute earlier or more fully.",
        },
        {
            "category": "Transportation",
            "definition": "Unnecessary movement of work, data, or materials.",
            "keywords": ("transfer", "move", "ship", "send", "transport", "handoff", "deliver"),
            "recommendation": "Consolidate handoffs or systems touched during '{step}' to cut transport waste.",
            "fallback": "Trace how {focus} moves between tools or teams to eliminate hops.",
        },
        {
            "category": "Inventory",
            "definition": "Work piling up without being processed.",
            "keywords": ("backlog", "queue", "stack", "accumulate", "pending", "pile", "cache"),
            "recommendation": "Limit work-in-progress around '{step}' to keep inventory lean.",
            "fallback": "Surface WIP limits for stages of {focus} so stock does not accumulate.",
        },
        {
            "category": "Motion",
            "definition": "Extra movement of people or tools beyond what is required.",
            "keywords": ("search", "lookup", "switch", "toggle", "walk", "navigate", "reopen"),
            "recommendation": "Streamline tooling for '{step}' to minimise motion and context switching.",
            "fallback": "Observe where {focus} forces people to switch tools or locations.",
        },
        {
            "category": "Excess processing",
            "definition": "Doing more work or complexity than customers need.",
            "keywords": ("complex", "custom", "over", "polish", "refine", "approval", "sign-off"),
            "recommendation": "Simplify criteria around '{step}' so the process meets expectations without over-processing.",
            "fallback": "Check if {focus} includes reviews or documentation that could be trimmed.",
        },
    )

    def map_process(self, problem_statement: str, *, detail_level: str = "balanced") -> ProcessPlan:
        statement = problem_statement.strip()
        if not statement:
            raise ValueError("A problem statement is required to map the process.")

        keywords = _extract_keywords(statement)
        focus = keywords[0] if keywords else "the process"
        target_steps = self._level_to_count.get(detail_level.lower(), 5)

        raw_actions = self._extract_actions(statement)
        action_view = self._limit_actions(raw_actions, target_steps)

        if action_view:
            steps = self._build_steps_from_actions(action_view, focus)
        else:
            steps = self._build_template_steps(focus, keywords, target_steps)
            action_view = [step.title for step in steps]

        mermaid = self._to_mermaid(steps)
        assumptions = self._draft_assumptions(focus, keywords, steps)
        risks = self._draft_risks(focus, keywords, steps)
        downtime = self._downtime_opportunities(focus, action_view, steps)
        summary = self._summarise(statement, focus, steps)

        return ProcessPlan(
            summary=summary,
            steps=steps,
            mermaid=mermaid,
            assumptions=assumptions,
            risks=risks,
            keywords=keywords,
            downtime_opportunities=downtime,
        )

    def _extract_actions(self, statement: str) -> List[str]:
        pattern = re.compile(
            r"(?:->|=>|\u2192|\u2794|\u279E|\u279C|\u27A1|\u27F6|,?\s*(?:and\s+)?then\b|,?\s*next\b|,?\s*after(?: that|wards)?\b|,?\s*subsequently\b|,?\s*finally\b|;)",
            flags=re.IGNORECASE,
        )
        actions: List[str] = []
        for raw_line in statement.replace("\r", "\n").split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            line = line.lstrip("-*\u2022 0123456789).")
            line = line.strip()
            if not line:
                continue
            normalized = pattern.sub(" | ", line)
            segments = re.split(r"[.|!?]|\|", normalized)
            for segment in segments:
                candidate = segment.strip(" \t-\u2022")
                if len(candidate) < 3:
                    continue
                actions.append(candidate)

        deduped: List[str] = []
        seen: set[str] = set()
        for action in actions:
            key = action.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(action)
        return deduped

    def _limit_actions(self, actions: Sequence[str], step_count: int) -> List[str]:
        if not actions:
            return []
        if step_count <= 0 or len(actions) <= step_count:
            return list(actions)

        limited: List[str] = []
        total = len(actions)
        for idx in range(step_count):
            start = math.floor(idx * total / step_count)
            end = math.floor((idx + 1) * total / step_count)
            chunk = [item for item in actions[start:end] if item]
            if chunk:
                limited.append(" -> ".join(chunk))
        return limited

    def _build_steps_from_actions(self, actions: Sequence[str], focus: str) -> List[ProcessStep]:
        steps: List[ProcessStep] = []
        default_focus = focus
        for idx, action in enumerate(actions, start=1):
            title = self._format_step_title(action)
            action_keywords = _extract_keywords(title)
            focus_word = action_keywords[0] if action_keywords else default_focus
            question_template = self._step_questions[(idx - 1) % len(self._step_questions)]
            deliverable_template = self._step_deliverables[(idx - 1) % len(self._step_deliverables)]
            steps.append(
                ProcessStep(
                    order=idx,
                    title=title,
                    focus=focus_word,
                    question=question_template.format(step=title.lower(), focus=focus_word),
                    deliverable=deliverable_template.format(step=title, focus=focus_word),
                )
            )
        return steps

    def _build_template_steps(
        self,
        focus: str,
        keywords: Sequence[str],
        step_count: int,
    ) -> List[ProcessStep]:
        steps: List[ProcessStep] = []
        templates = list(itertools.islice(self._fallback_templates, step_count))
        for idx, template in enumerate(templates, start=1):
            focus_word = keywords[idx - 1] if idx - 1 < len(keywords) else focus
            steps.append(
                ProcessStep(
                    order=idx,
                    title=template["title"].format(focus=focus_word.title()),
                    focus=focus_word,
                    question=template["question"].format(focus=focus_word),
                    deliverable=template["deliverable"],
                )
            )
        return steps

    def _format_step_title(self, action: str) -> str:
        cleaned = action.strip().strip(".")
        cleaned = re.sub(r"\s+", " ", cleaned)
        if len(cleaned) > 96:
            cleaned = cleaned[:93].rstrip(",; ") + "..."
        if not cleaned:
            return "Unnamed step"
        return cleaned[0].upper() + cleaned[1:]

    def _summarise(self, statement: str, focus: str, steps: Sequence[ProcessStep]) -> str:
        headline = statement.splitlines()[0].strip().rstrip(".")
        opening = steps[0].title if steps else focus
        closing = steps[-1].title if steps else focus
        opening_fragment = self._to_sentence_fragment(opening)
        closing_fragment = self._to_sentence_fragment(closing)
        if headline:
            if len(headline) > 180:
                headline = headline[:177].rstrip(",; ") + "..."
            return (
                f"{headline}. The mapped flow begins with {opening_fragment} and wraps with {closing_fragment}."
            )
        return (
            f"A {len(steps)}-stage outline for how {focus} currently runs, from {opening_fragment} to {closing_fragment}."
        )

    def _to_sentence_fragment(self, value: str) -> str:
        fragment = value.strip().rstrip(".")
        if not fragment:
            return "the process starting point"
        return fragment[0].lower() + fragment[1:]

    def _draft_assumptions(
        self,
        focus: str,
        keywords: Sequence[str],
        steps: Sequence[ProcessStep],
    ) -> List[str]:
        secondary = keywords[1] if len(keywords) > 1 else focus
        opening = steps[0].title if steps else focus
        closing = steps[-1].title if steps else focus
        return [
            f"Current-state documentation for '{opening}' through '{closing}' is accessible to the team.",
            f"Subject-matter experts on {secondary} can validate the mapped responsibilities.",
        ]

    def _draft_risks(
        self,
        focus: str,
        keywords: Sequence[str],
        steps: Sequence[ProcessStep],
    ) -> List[str]:
        anchor = keywords[2] if len(keywords) > 2 else focus
        closing = steps[-1].title if steps else focus
        pinch_point = steps[len(steps) // 2].title if steps else focus
        return [
            f"Hidden variations around '{pinch_point}' could invalidate the mapped flow.",
            f"Limited instrumentation near '{closing}' makes it difficult to prove improvements for {anchor}.",
        ]

    def _downtime_opportunities(
        self,
        focus: str,
        actions: Sequence[str],
        steps: Sequence[ProcessStep],
    ) -> List[DowntimeOpportunity]:
        if not actions:
            actions = [step.title for step in steps]
        if not actions:
            actions = [focus]

        focus_anchor = focus or "the process"
        lower_actions = [action.lower() for action in actions]

        opportunities: List[DowntimeOpportunity] = []
        for entry in self._downtime_catalog:
            category = str(entry["category"])
            definition = str(entry["definition"])
            keywords = tuple(entry.get("keywords", ()))
            recommendation_template = str(entry["recommendation"])
            fallback_template = str(entry["fallback"])

            trigger: Optional[str] = None
            recommendation = fallback_template.format(focus=focus_anchor)

            for action, action_lower in zip(actions, lower_actions):
                if any(keyword in action_lower for keyword in keywords):
                    trigger = action.strip()
                    recommendation = recommendation_template.format(
                        focus=focus_anchor,
                        step=trigger,
                    )
                    break

            opportunities.append(
                DowntimeOpportunity(
                    category=category,
                    definition=definition,
                    trigger=trigger,
                    recommendation=recommendation,
                )
            )

        return opportunities

    def _to_mermaid(self, steps: Sequence[ProcessStep]) -> str:
        lines = ["flowchart TD", "    START([\"Kickoff\"])" ]
        for step in steps:
            node_id = f"S{step.order}"
            label_title = _escape_mermaid(f"{step.order}. {step.title}")
            label_sub = _escape_mermaid(step.deliverable)
            lines.append(f"    {node_id}[\"{label_title}<br/>{label_sub}\"]")
        lines.append("    END([\"Validated plan\"])")

        if steps:
            lines.append("    START --> S1")
            for idx in range(1, len(steps)):
                lines.append(f"    S{idx} --> S{idx + 1}")
            lines.append(f"    S{steps[-1].order} --> END")
        else:
            lines.append("    START --> END")

        return "\n".join(lines)


def _clean_portal_text(value: Optional[str]) -> str:
    """Normalise text pulled from portal metadata."""

    if not value:
        return ""
    unescaped = html.unescape(value)
    no_markup = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", no_markup).strip()


class OpenDataPortalError(RuntimeError):
    """Raised when calls to the OpenData portal fail."""


@dataclass
class CatalogEntry:
    identifier: str
    title: str
    description: str
    link: str
    tags: Tuple[str, ...]
    categories: Tuple[str, ...]
    publisher: Optional[str]
    updated_at: Optional[str]


class OpenDataPortalClient:
    """Thin wrapper around Socrata's catalog API used by NYC OpenData."""

    def __init__(
        self,
        base_url: str = "https://data.cityofnewyork.us",
        *,
        timeout: int = 10,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()
        parsed = urlparse(self.base_url)
        self._domain = parsed.netloc or self.base_url

    def search_datasets(self, query_terms: Sequence[str], *, limit: int = 10) -> List[CatalogEntry]:
        query = " ".join(term for term in query_terms if term).strip()
        if not query:
            raise OpenDataPortalError("OpenData search requires at least one keyword.")

        params = {
            "search": query,
            "limit": str(limit),
            "search_context": self._domain,
        }
        url = f"{self.base_url}/api/catalog/v1"

        try:
            response = self._session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
        except RequestException as exc:
            raise OpenDataPortalError(f"OpenData request failed: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise OpenDataPortalError("OpenData response was not valid JSON.") from exc

        entries: List[CatalogEntry] = []
        for item in payload.get("results", []):
            resource = item.get("resource", {})
            if resource.get("type") not in {"dataset", "datalens"}:
                continue
            classification = item.get("classification", {})
            link = resource.get("permalink") or item.get("link") or ""
            tags_raw: Iterable[str] = (
                classification.get("domain_tags") or classification.get("tags") or []
            )
            tags = tuple(_clean_portal_text(tag) for tag in tags_raw if tag)
            categories_raw = classification.get("domain_category")
            if isinstance(categories_raw, str):
                categories = (_clean_portal_text(categories_raw),)
            else:
                categories = tuple(_clean_portal_text(cat) for cat in categories_raw or [] if cat)

            identifier = str(resource.get("id")) if resource.get("id") else ""
            entries.append(
                CatalogEntry(
                    identifier=identifier,
                    title=_clean_portal_text(resource.get("name")) or "Untitled dataset",
                    description=_clean_portal_text(resource.get("description")),
                    link=link or (f"{self.base_url}/d/{identifier}" if identifier else self.base_url),
                    tags=tags,
                    categories=categories,
                    publisher=_clean_portal_text(resource.get("attribution")) or None,
                    updated_at=_clean_portal_text(resource.get("updatedAt")) or None,
                )
            )

        return entries


_DEFAULT_DATASETS: Tuple[Dict[str, object], ...] = (
    {
        "title": "NYC 311 Service Requests",
        "portal_url": "https://data.cityofnewyork.us/Social-Services/311-Service-Requests/erm2-nwe9",
        "coverage": "Urban / municipal",
        "keywords": {"service", "maintenance", "complaint", "response", "city"},
        "budget_range": (0, 250_000),
        "summary": "Time-stamped incidents with location, category, and resolution notes.",
        "insight": "Helps prioritise neighbourhood investments based on volume and severity trends.",
    },
    {
        "title": "US Small Business Pulse Survey",
        "portal_url": "https://portal.census.gov/pulse/data",
        "coverage": "National / SME",
        "keywords": {"business", "revenue", "supply", "employment", "survey"},
        "budget_range": (50_000, 1_000_000),
        "summary": "Weekly indicators on operations, workforce, and financial stress for small firms.",
        "insight": "Benchmarks constraints for growth strategies targeting small enterprises.",
    },
    {
        "title": "Global Power Plant Database",
        "portal_url": "https://datasets.wri.org/dataset/globalpowerplantdatabase",
        "coverage": "Global / infrastructure",
        "keywords": {"energy", "power", "emissions", "capacity", "renewable"},
        "budget_range": (250_000, 5_000_000),
        "summary": "Facility-level attributes including fuel type, capacity, and estimated emissions.",
        "insight": "Supports modelling decarbonisation pathways and siting decisions.",
    },
    {
        "title": "Open Streets Foot Traffic",
        "portal_url": "https://opendata.cityofnewyork.us/dataset/Open-Streets-Foot-Traffic/ppuj-givf",
        "coverage": "Metropolitan / mobility",
        "keywords": {"mobility", "foot", "traffic", "pedestrian", "retail"},
        "budget_range": (0, 180_000),
        "summary": "Hourly pedestrian counts captured via sensors across commercial corridors.",
        "insight": "Quantifies demand patterns for placemaking or storefront activation projects.",
    },
    {
        "title": "Community Health Outcomes",
        "portal_url": "https://healthdata.gov/dataset/community-health-outcomes",
        "coverage": "County / public health",
        "keywords": {"health", "outcome", "equity", "hospital", "preventable"},
        "budget_range": (100_000, 750_000),
        "summary": "Rates of preventable hospitalisations and chronic conditions by demographic cohort.",
        "insight": "Reveals gaps in service delivery and priority groups for intervention.",
    },
)


@dataclass
class DatasetMatch:
    title: str
    portal_url: str
    summary: str
    why_useful: str
    coverage: str
    score: float
    tags: Tuple[str, ...] = ()
    publisher: Optional[str] = None
    last_updated: Optional[str] = None


@dataclass
class ResearchReport:
    narrative: str
    keywords: List[str]
    matches: List[DatasetMatch]
    next_steps: List[str]
    portal_messages: List[str]
    data_source: str


class ResearchLaunchAgent:
    """Agent that recommends open datasets aligned with a launch brief."""

    def __init__(
        self,
        portal_client: Optional[OpenDataPortalClient] = None,
        *,
        fallback_catalog: Optional[Sequence[Dict[str, object]]] = None,
    ) -> None:
        self._portal_client = portal_client or OpenDataPortalClient()
        self._fallback_catalog: Tuple[Dict[str, object], ...] = (
            tuple(fallback_catalog) if fallback_catalog is not None else _DEFAULT_DATASETS
        )

    def launch_brief(
        self,
        problem_statement: str,
        *,
        budget: Optional[float] = None,
        scale: Optional[str] = None,
        constraints: Optional[str] = None,
    ) -> ResearchReport:
        statement = problem_statement.strip()
        if not statement:
            raise ValueError("Describe the problem the launch system should investigate.")

        constraint_keywords = _extract_keywords(constraints or "", limit=8)
        base_keywords = _extract_keywords(statement)
        search_terms: List[str] = list(base_keywords)
        if scale:
            search_terms.append(scale)
        if constraints:
            search_terms.extend(_extract_keywords(constraints, limit=6))
        search_terms = search_terms[:10]

        portal_messages: List[str] = []
        portal_entries: List[CatalogEntry] = []
        if search_terms:
            try:
                portal_entries = self._portal_client.search_datasets(search_terms, limit=12)
                if not portal_entries:
                    portal_messages.append(
                        "OpenData search returned no datasets for the supplied terms."
                    )
            except OpenDataPortalError as exc:
                portal_messages.append(str(exc))
        else:
            portal_messages.append(
                "Add more detail so the OpenData search has keywords to work with."
            )

        matches = self._rank_portal_entries(
            portal_entries,
            base_keywords,
            constraint_keywords,
            scale,
        )
        data_source = "portal"

        if not matches:
            matches = self._fallback_matches(base_keywords, budget, scale)
            if matches:
                portal_messages.append(
                    "Showing curated fallback datasets while live search is empty."
                )
            else:
                portal_messages.append("No curated datasets are available for this focus.")
            data_source = "fallback"

        narrative = self._compose_narrative(
            statement,
            base_keywords,
            matches,
            budget,
            scale,
            data_source,
        )
        next_steps = self._next_steps(matches, budget, data_source)

        return ResearchReport(
            narrative=narrative,
            keywords=base_keywords,
            matches=matches,
            next_steps=next_steps,
            portal_messages=portal_messages,
            data_source=data_source,
        )

    def _rank_portal_entries(
        self,
        entries: Sequence[CatalogEntry],
        keywords: Sequence[str],
        constraint_keywords: Sequence[str],
        scale: Optional[str],
    ) -> List[DatasetMatch]:
        matches: List[DatasetMatch] = []
        focus = keywords[0] if keywords else "the initiative"

        for entry in entries:
            score, hits = self._score_entry(entry, keywords, constraint_keywords, scale)
            if score <= 0:
                continue
            coverage = ", ".join(entry.categories) if entry.categories else "OpenData"
            summary = entry.description or "No description provided by the portal."
            why_useful = self._build_usefulness_note(focus, hits, entry)
            matches.append(
                DatasetMatch(
                    title=entry.title,
                    portal_url=entry.link,
                    summary=summary,
                    why_useful=why_useful,
                    coverage=coverage,
                    score=score,
                    tags=entry.tags,
                    publisher=entry.publisher,
                    last_updated=entry.updated_at,
                )
            )

        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:3]

    def _score_entry(
        self,
        entry: CatalogEntry,
        keywords: Sequence[str],
        constraint_keywords: Sequence[str],
        scale: Optional[str],
    ) -> Tuple[float, List[str]]:
        score = 0.0
        hits: List[str] = []
        text_index = " ".join(
            part.lower()
            for part in [entry.title, entry.description, " ".join(entry.tags), " ".join(entry.categories)]
            if part
        )
        tags_lower = {tag.lower() for tag in entry.tags}

        for keyword in keywords:
            lowered = keyword.lower()
            if lowered in tags_lower:
                score += 2.5
                hits.append(keyword)
            elif lowered and lowered in text_index:
                score += 1.5
                hits.append(keyword)

        for extra in constraint_keywords:
            lowered = extra.lower()
            if lowered and lowered in text_index and extra not in hits:
                score += 0.75
                hits.append(extra)

        if scale:
            lowered_scale = scale.lower()
            if lowered_scale in text_index and lowered_scale not in hits:
                score += 1.5
                hits.append(lowered_scale)

        if entry.updated_at:
            score += 0.35

        if entry.tags:
            score += 0.2

        return score, hits

    def _build_usefulness_note(
        self,
        focus: str,
        hits: Sequence[str],
        entry: CatalogEntry,
    ) -> str:
        if hits:
            highlight = ", ".join(hits[:3])
            return f"Connects {highlight} data points that relate directly to {focus}."
        if entry.categories:
            category = entry.categories[0].lower()
            return f"Provides {category} context to ground early analysis."
        return "Adds supplementary signals from the OpenData portal relevant to the initiative."

    def _fallback_matches(
        self,
        keywords: Sequence[str],
        budget: Optional[float],
        scale: Optional[str],
    ) -> List[DatasetMatch]:
        scored: List[Tuple[float, Dict[str, object]]] = []
        for dataset in self._fallback_catalog:
            score = self._score_fallback(dataset, keywords, budget, scale)
            scored.append((score, dataset))
        scored.sort(key=lambda item: item[0], reverse=True)

        matches: List[DatasetMatch] = []
        for score, dataset in itertools.islice(scored, 3):
            matches.append(
                DatasetMatch(
                    title=str(dataset["title"]),
                    portal_url=str(dataset["portal_url"]),
                    summary=str(dataset["summary"]),
                    why_useful=str(dataset["insight"]),
                    coverage=str(dataset["coverage"]),
                    score=float(score),
                )
            )
        return matches

    def _score_fallback(
        self,
        dataset: Dict[str, object],
        keywords: Sequence[str],
        budget: Optional[float],
        scale: Optional[str],
    ) -> float:
        score = 0.0
        dataset_keywords = dataset["keywords"]
        description = str(dataset["summary"]).lower()

        for keyword in keywords:
            if keyword in dataset_keywords:
                score += 2.2
            elif keyword in description:
                score += 0.9

        if budget is not None:
            low, high = dataset["budget_range"]
            if low <= budget <= high:
                score += 1.8
            else:
                score -= 0.4

        if scale:
            lowered = scale.lower()
            coverage = str(dataset["coverage"]).lower()
            if lowered in coverage:
                score += 1.2

        return score if score else 0.4

    def _compose_narrative(
        self,
        statement: str,
        keywords: Sequence[str],
        matches: Sequence[DatasetMatch],
        budget: Optional[float],
        scale: Optional[str],
        data_source: str,
    ) -> str:
        focus = keywords[0] if keywords else "the launch brief"
        scale_fragment = f" for a {scale.lower()} initiative" if scale else ""
        budget_fragment = (
            f" with a working budget near ${budget:,.0f}" if budget is not None else ""
        )
        datasets_fragment = ", ".join(match.title for match in matches) or "open datasets"
        source_fragment = "from NYC OpenData" if data_source == "portal" else "from a curated fallback set"
        return (
            f"To accelerate work on {focus}{scale_fragment}{budget_fragment}, start by mining "
            f"{datasets_fragment} {source_fragment}. They align with the problem framing and surface "
            f"signals for early analysis."
        )

    def _next_steps(
        self,
        matches: Sequence[DatasetMatch],
        budget: Optional[float],
        data_source: str,
    ) -> List[str]:
        suggestions = [
            "Review metadata freshness and API documentation for shortlisted datasets.",
            "Spin up a collaborative notebook to profile schema, missing values, and joins.",
            "Outline stakeholder questions that the first-pass analysis should answer.",
        ]
        if matches and data_source == "portal":
            suggestions.insert(0, "Bookmark dataset API endpoints for scripted ingestion.")
        if matches and budget is not None:
            suggestions.append(
                f"Allocate ~${budget:,.0f} across ingestion, exploration, and presentation workloads."
            )
        return suggestions


__all__ = [
    "ProcessMappingAgent",
    "ProcessPlan",
    "DowntimeOpportunity",
    "ProcessStep",
    "ResearchLaunchAgent",
    "ResearchReport",
    "DatasetMatch",
    "OpenDataPortalClient",
    "OpenDataPortalError",
]
