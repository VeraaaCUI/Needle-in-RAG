from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass(frozen=True)
class Span:
    """A character span in a specific prompt-used chunk and in the rendered prompt."""
    chunk_id: str
    rank: int

    # Chunk-local coordinates (0-based, within the chunk text as placed into the prompt)
    chunk_start: int
    chunk_end: int

    # Prompt-global coordinates (0-based, within generation.user)
    prompt_start: int
    prompt_end: int

    label: str = "candidate"     # e.g., "incorrect_answer", "numeric", ...
    score: float = 0.0           # higher => more suspicious


@dataclass
class Signals:
    pred: str
    gold: str
    target: str

    T_ASR: bool
    PAR: bool
    P0_ASR: bool
    UNK: bool
    CONFUSION: bool
    EM: bool
    EM_contains: bool

    pool_incorrect_answers: List[str]


def signals_from_row(row: Dict[str, Any]) -> Signals:
    gen = (row.get("generation", {}) or {})
    sig = (gen.get("signals", {}) or {})
    return Signals(
        pred=str(sig.get("pred", "") or ""),
        gold=str(sig.get("gold", "") or ""),
        target=str(sig.get("target", "") or ""),
        T_ASR=bool(sig.get("T_ASR", False)),
        PAR=bool(sig.get("PAR", False)),
        P0_ASR=bool(sig.get("P0_ASR", False)),
        UNK=bool(sig.get("UNK", False)),
        CONFUSION=bool(sig.get("CONFUSION", False)),
        EM=bool(sig.get("EM", False)),
        EM_contains=bool(sig.get("EM_contains", False)),
        pool_incorrect_answers=list(sig.get("pool_incorrect_answers", []) or []),
    )
