from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import requests

from rag_char_trace.llm.base import LLM, LLMConfig


@dataclass
class OllamaLLM(LLM):
    model: str
    base_url: str = "http://localhost:11434"
    timeout_s: int = 120

    def generate(self, *, system: str, user: str, config: LLMConfig) -> str:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": user,
            "system": system,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
            },
        }
        resp = requests.post(url, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
