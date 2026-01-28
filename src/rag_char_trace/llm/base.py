from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class LLMConfig:
    temperature: float = 0.0
    max_tokens: int = 128


class LLM:
    """Abstract LLM interface."""

    def generate(self, *, system: str, user: str, config: LLMConfig) -> str:
        raise NotImplementedError
