from __future__ import annotations

from typing import Any, Dict


def should_trigger(row: Dict[str, Any], trigger: str) -> bool:
    gen = (row.get("generation", {}) or {})
    sig = (gen.get("signals", {}) or {})
    t = trigger.upper()

    if t == "ALWAYS":
        return True
    if t == "PAR":
        return bool(sig.get("PAR", False))
    if t == "T_ASR":
        return bool(sig.get("T_ASR", False))
    if t == "ANY":
        return bool(sig.get("PAR", False) or sig.get("T_ASR", False) or sig.get("CONFUSION", False))
    if t == "CONFUSION":
        return bool(sig.get("CONFUSION", False))
    return bool(sig.get("PAR", False))
