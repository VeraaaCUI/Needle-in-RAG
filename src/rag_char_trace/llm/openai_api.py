from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class OpenAIConfig:
    model: str
    temperature: float = 0.0
    max_tokens: int = 256
    timeout_s: float = 60.0
    max_retries: int = 6


class OpenAILLM:
    """
    Minimal OpenAI backend using the Responses API.

    Environment:
      - OPENAI_API_KEY (required)

    This uses the Responses API:
      client.responses.create(model=..., input=..., instructions=..., temperature=..., max_output_tokens=...)
    """

    def __init__(self, api_key: Optional[str] = None, timeout_s: float = 60.0):
        if OpenAI is None:
            raise RuntimeError("OpenAI SDK not installed. In your venv, run: pip install openai")

        # The official SDK reads OPENAI_API_KEY by default.
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.timeout_s = timeout_s

    def generate(self, system: str, user: str, config: OpenAIConfig) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(config.max_retries):
            try:
                resp = self.client.responses.create(
                    model=config.model,
                    input=user,
                    instructions=system if system else None,
                    temperature=config.temperature,
                    max_output_tokens=config.max_tokens,
                    timeout=self.timeout_s,
                    truncation="auto",
                    store=False,
                )
                out = getattr(resp, "output_text", None)
                if isinstance(out, str):
                    return out.strip()
                return str(resp).strip()
            except Exception as e:
                last_err = e
                sleep_s = min(2 ** attempt, 20) + (0.05 * attempt)
                time.sleep(sleep_s)

        raise RuntimeError(f"OpenAI request failed after retries: {last_err}")
