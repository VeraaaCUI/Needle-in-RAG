from __future__ import annotations

import os
import time
from typing import Optional, Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


class OpenAILLM:
    """
    OpenAI backend compatible with this project's runner.

    scripts/02_run_rag.py expects:
        llm = OpenAILLM(model=args.openai_model)
        completion = llm.generate(system=..., user=..., config=LLMConfig)

    Requirements:
      - pip install openai
      - set OPENAI_API_KEY in your environment
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_s: float = 60.0,
        max_retries: int = 6,
    ):
        if OpenAI is None:
            raise RuntimeError("OpenAI SDK not installed. In your venv, run: pip install openai")

        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries

        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def generate(self, system: str, user: str, config: Any) -> str:
        temperature = float(getattr(config, "temperature", 0.0) or 0.0)
        max_tokens = int(getattr(config, "max_tokens", 256) or 256)

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=user,
                    instructions=system if system else None,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
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
