from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

from rag_char_trace.pass1.textutil import apply_masks


def mask_chunk_text_regions(
    *,
    user_prompt: str,
    prompt_blocks: List[Dict[str, Any]],
    ranks_to_mask: List[int],
) -> Tuple[str, List[Tuple[int, int]]]:
    """Mask the entire text region (not header) of selected prompt-used passages."""
    regions: List[Tuple[int, int]] = []
    by_rank = {int(b.get("rank", -1)): b for b in prompt_blocks}
    for r in ranks_to_mask:
        b = by_rank.get(int(r))
        if not b:
            continue
        a = int(b.get("prompt_text_start", -1))
        e = int(b.get("prompt_text_end", -1))
        if a >= 0 and e >= 0 and e > a:
            regions.append((a, e))
    return apply_masks(user_prompt, regions), regions


def mask_prompt_regions(user_prompt: str, regions: List[Tuple[int, int]]) -> str:
    return apply_masks(user_prompt, regions)
