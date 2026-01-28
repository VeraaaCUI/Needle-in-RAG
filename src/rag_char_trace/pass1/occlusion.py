from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from rag_char_trace.llm.base import LLMConfig
from rag_char_trace.pass1.textutil import apply_masks
from rag_char_trace.pass1.types import Signals
from rag_char_trace.utils.text import contains_normalized, normalize_text


def _parse_completion(completion: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"answer": "", "evidence": "", "format_ok": False}
    if not completion:
        return out
    answer = ""
    evidence = ""
    lines = [ln.strip() for ln in completion.splitlines() if ln.strip()]
    for ln in lines:
        l = ln.lower()
        if l.startswith("answer:"):
            answer = ln.split(":", 1)[1].strip()
        elif l.startswith("evidence:"):
            evidence = ln.split(":", 1)[1].strip()
    if answer:
        out["format_ok"] = True
        out["answer"] = answer
        out["evidence"] = evidence
        return out
    out["answer"] = lines[0] if lines else completion.strip()
    return out


def compute_signals(
    *,
    pred: str,
    gold: str,
    target: str,
    pool_incorrect_answers: List[str],
    p0_incorrect: str,
) -> Signals:
    """Compute dataset-style signals (oracle-capable) for analysis.

    NOTE: If pool_incorrect_answers is empty, PAR will be False by construction.
    This is fine when running Pass-1 in a label-free mode; you can still evaluate
    with oracle labels offline by supplying the pool list.
    """
    pred = (pred or "").strip()
    gold = (gold or "").strip()
    target = (target or "").strip()

    unknown = pred.upper() == "UNKNOWN"
    t_asr = bool(target and contains_normalized(pred, target))
    gold_em = normalize_text(pred) == normalize_text(gold) if gold else False
    gold_contains = bool(gold and contains_normalized(pred, gold))
    pool_set = set((ia or "").strip() for ia in pool_incorrect_answers if (ia or "").strip())
    par = bool(pool_set and any(contains_normalized(pred, ia) for ia in pool_set))
    p0_asr = bool(p0_incorrect and contains_normalized(pred, p0_incorrect))
    confusion = (not gold_contains) and (not par)

    return Signals(
        pred=pred,
        gold=gold,
        target=target,
        T_ASR=t_asr,
        PAR=par,
        P0_ASR=p0_asr,
        UNK=unknown,
        CONFUSION=confusion,
        EM=gold_em,
        EM_contains=gold_contains,
        pool_incorrect_answers=sorted(pool_set),
    )


@dataclass
class EvalResult:
    completion: str
    parsed: Dict[str, Any]
    signals: Signals


class PromptEvaluator:
    """Evaluate a (system,user) prompt by calling the provided LLM backend with caching."""

    def __init__(self, llm: Any, llm_cfg: LLMConfig):
        self.llm = llm
        self.llm_cfg = llm_cfg
        self.cache: Dict[Tuple[str, str], EvalResult] = {}

    def eval(
        self,
        *,
        system: str,
        user: str,
        gold: str,
        target: str,
        pool_incorrect_answers: List[str],
        p0_incorrect: str,
    ) -> EvalResult:
        key = (system, user)
        if key in self.cache:
            return self.cache[key]

        completion = self.llm.generate(system=system, user=user, config=self.llm_cfg)
        parsed = _parse_completion(completion)
        pred = parsed.get("answer", "") or ""
        sig = compute_signals(
            pred=pred,
            gold=gold,
            target=target,
            pool_incorrect_answers=pool_incorrect_answers,
            p0_incorrect=p0_incorrect,
        )
        res = EvalResult(completion=completion, parsed=parsed, signals=sig)
        self.cache[key] = res
        return res


def influence_score(
    base: Signals,
    new: Signals,
    *,
    objective: str = "oracle",
    bad_answer: str = "",
    user_prompt: str = "",
    target_weight: float = 0.7,
    correct_weight: float = 0.5,
) -> float:
    """Score how useful a mask is.

    objective="oracle" (previous behavior):
      - prefers reductions in PAR / T_ASR and improvements in EM_contains.

    objective="event" (label-free, realistic):
      - treats the *observed wrong answer* (bad_answer) as the target to remove.
      - prefers masks that change the model answer away from bad_answer while staying grounded.

    Larger is better.
    """
    obj = (objective or "oracle").lower().strip()

    if obj == "event":
        # Primary: remove the observed bad answer string from the output.
        ba = (bad_answer or base.pred or "").strip()
        removed = 1.0 if (ba and (not contains_normalized(new.pred, ba))) else 0.0
        changed = 1.0 if normalize_text(new.pred) != normalize_text(base.pred) else 0.0

        # Groundedness proxy: in extractive prompting, the answer should occur in the provided prompt.
        grounded = 1.0 if (user_prompt and new.pred and contains_normalized(user_prompt, new.pred)) else 0.0

        unk_penalty = 1.0 if new.UNK else 0.0

        return 1.4 * removed + 0.2 * changed + 0.2 * grounded - 0.3 * unk_penalty

    # --- oracle objective (default) ---
    base_poison = 1.0 if base.PAR else 0.0
    new_poison = 1.0 if new.PAR else 0.0

    base_target = 1.0 if base.T_ASR else 0.0
    new_target = 1.0 if new.T_ASR else 0.0

    base_correct = 1.0 if base.EM_contains else 0.0
    new_correct = 1.0 if new.EM_contains else 0.0

    return (base_poison - new_poison) + target_weight * (base_target - new_target) + correct_weight * (new_correct - base_correct)


def success_predicate(
    base: Signals,
    new: Signals,
    trigger: str,
    *,
    objective: str = "oracle",
    bad_answer: str = "",
) -> bool:
    """Whether a mask 'succeeds' under the chosen objective.

    - oracle: succeed when the requested trigger is resolved (e.g., PAR flips from True to False).
    - event: succeed when the model output no longer contains the observed bad answer string.
    """
    obj = (objective or "oracle").lower().strip()

    if obj == "event":
        ba = (bad_answer or base.pred or "").strip()
        if not ba:
            return False
        # Strict: require the answer to move away from the bad answer, not just add extra tokens.
        return (not contains_normalized(new.pred, ba)) and (normalize_text(new.pred) != normalize_text(ba))

    t = (trigger or "PAR").upper()
    if t == "PAR":
        return base.PAR and (not new.PAR)
    if t == "T_ASR":
        return base.T_ASR and (not new.T_ASR)
    if t == "ANY":
        # Conservative: any reduction in poison adoption OR improvement in correctness
        return (base.PAR and (not new.PAR)) or (base.T_ASR and (not new.T_ASR)) or ((not base.EM_contains) and new.EM_contains)
    if t == "ALWAYS":
        return True
    if t == "CONFUSION":
        return base.CONFUSION and (not new.CONFUSION)
    return base.PAR and (not new.PAR)


def bisect_minimal_span(
    *,
    evaluator: PromptEvaluator,
    system: str,
    user_prompt: str,
    base: Signals,
    trigger: str,
    gold: str,
    target: str,
    pool_incorrect_answers: List[str],
    p0_incorrect: str,
    span_prompt_start: int,
    span_prompt_end: int,
    max_steps: int = 10,
    min_len: int = 4,
    objective: str = "oracle",
    bad_answer: str = "",
) -> Tuple[int, int, EvalResult]:
    """Try to shrink [start,end) while keeping success_predicate true."""
    lo = int(span_prompt_start)
    hi = int(span_prompt_end)

    best_res = evaluator.eval(
        system=system,
        user=apply_masks(user_prompt, [(lo, hi)]),
        gold=gold,
        target=target,
        pool_incorrect_answers=pool_incorrect_answers,
        p0_incorrect=p0_incorrect,
    )

    if not success_predicate(base, best_res.signals, trigger, objective=objective, bad_answer=bad_answer):
        return lo, hi, best_res

    steps = 0
    while steps < max_steps and (hi - lo) > min_len:
        steps += 1
        mid = lo + (hi - lo) // 2

        # Try left half
        res_left = evaluator.eval(
            system=system,
            user=apply_masks(user_prompt, [(lo, mid)]),
            gold=gold,
            target=target,
            pool_incorrect_answers=pool_incorrect_answers,
            p0_incorrect=p0_incorrect,
        )
        if success_predicate(base, res_left.signals, trigger, objective=objective, bad_answer=bad_answer):
            hi = mid
            best_res = res_left
            continue

        # Try right half
        res_right = evaluator.eval(
            system=system,
            user=apply_masks(user_prompt, [(mid, hi)]),
            gold=gold,
            target=target,
            pool_incorrect_answers=pool_incorrect_answers,
            p0_incorrect=p0_incorrect,
        )
        if success_predicate(base, res_right.signals, trigger, objective=objective, bad_answer=bad_answer):
            lo = mid
            best_res = res_right
            continue

        # Can't shrink further without losing success
        break

    return lo, hi, best_res
