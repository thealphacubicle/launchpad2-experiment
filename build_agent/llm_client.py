"""Lightweight wrapper for interacting with an LLM provider."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

try:
    from openai import OpenAI
except ModuleNotFoundError as exc:  # pragma: no cover - dependency injected at runtime
    OpenAI = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]


class LLMError(RuntimeError):
    """Raised when an LLM call cannot be completed."""


@dataclass
class ChatMessage:
    role: str
    content: str


class LLMClient:
    """Client that delegates chat-completion style prompts to an LLM provider."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.2,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> None:

        if load_dotenv is not None:
            load_dotenv()

        if OpenAI is None:
            raise LLMError(
                "The 'openai' package is required to use LLM-backed agents."
            ) from _IMPORT_ERROR

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise LLMError(
                "Set the OPENAI_API_KEY environment variable to call the LLM."
            )

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = OpenAI(api_key=resolved_key)

    def complete(self, *, messages: Iterable[ChatMessage]) -> str:
        """Return the content from the first assistant message."""

        payload: List[dict[str, str]] = [
            {"role": message.role, "content": message.content} for message in messages
        ]

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=payload,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:  # pragma: no cover - surfaces provider errors
            raise LLMError(f"LLM request failed: {exc}") from exc

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError("LLM response did not include choices")

        first_choice = choices[0]
        message = getattr(first_choice, "message", None) or {}
        content = getattr(message, "content", None)
        if not content:
            raise LLMError("LLM response did not include message content")

        return str(content)


def extract_json(payload: str) -> object:
    """Parse JSON from a raw LLM response.

    The helper tolerates fenced code blocks and surrounding commentary.
    """

    stripped = payload.strip()

    fenced = re.findall(
        r"```json\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL
    )
    if fenced:
        candidate = fenced[0].strip()
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end >= start:
            candidate = stripped[start : end + 1]
        else:
            candidate = stripped

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMError("LLM response was not valid JSON") from exc
