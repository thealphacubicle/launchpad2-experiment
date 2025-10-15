from __future__ import annotations

from typing import List
from langchain_core.messages import BaseMessage

_CHAT_HISTORY: List[BaseMessage] = []


def set_chat_history(history: List[BaseMessage]) -> None:
    global _CHAT_HISTORY
    _CHAT_HISTORY = history


def get_chat_history() -> List[BaseMessage]:
    return _CHAT_HISTORY


__all__ = ["set_chat_history", "get_chat_history"]

