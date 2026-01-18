from __future__ import annotations

from dataclasses import dataclass

from rag_char_trace.llm.base import LLM, LLMConfig


@dataclass
class LlamaCppLLM(LLM):
    model_path: str
    n_ctx: int = 4096
    n_gpu_layers: int = 0

    def __post_init__(self) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it separately, e.g.\n"
                "  pip install llama-cpp-python\n"
                "and ensure you have a compatible wheel/build toolchain."
            ) from e
        self._llama = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
        )

    def generate(self, *, system: str, user: str, config: LLMConfig) -> str:
        prompt = f"[SYSTEM]\n{system}\n[/SYSTEM]\n\n{user}\n"
        out = self._llama(
            prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stop=["</s>", "[SYSTEM]", "[/SYSTEM]"],
        )
        return out["choices"][0]["text"]
