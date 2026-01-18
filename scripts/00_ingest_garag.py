from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def slugify(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-\.]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "unnamed"


def iter_candidate_files(dataset_dir: Path) -> Iterable[Path]:
    # Scan dataset dir and subfolders; ignore our standard files.
    exclude = {"chunks.jsonl", "answers.jsonl"}
    for ext in (".json", ".jsonl"):
        for p in dataset_dir.rglob(f"*{ext}"):
            if p.name in exclude:
                continue
            yield p


def try_load_json(path: Path) -> Optional[Any]:
    try:
        if path.suffix.lower() == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return rows
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_garag_payload(obj: Any) -> bool:
    items = None
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        for k in ("data", "items", "examples", "records"):
            if isinstance(obj.get(k), list):
                items = obj.get(k)
                break
    if not items:
        return False
    first = items[0]
    if not isinstance(first, dict):
        return False
    return ("adv_texts" in first) and ("question" in first) and (("id" in first) or ("qid" in first) or ("orig_id" in first))


def extract_items(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for k in ("data", "items", "examples", "records"):
            v = obj.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def normalize_answer(ans: Any) -> str:
    if ans is None:
        return ""
    if isinstance(ans, str):
        return ans.strip()
    if isinstance(ans, list):
        for x in ans:
            if isinstance(x, str) and x.strip():
                return x.strip()
        return ""
    return str(ans).strip()


def normalize_incorrect(ans: Any) -> str:
    if ans is None:
        return ""
    if isinstance(ans, str):
        return ans.strip()
    if isinstance(ans, list):
        for x in ans:
            if isinstance(x, str) and x.strip():
                return x.strip()
        return ""
    return str(ans).strip()


def normalize_adv_texts(v: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(v, list):
        return out
    for it in v:
        if isinstance(it, dict):
            ctx = it.get("context", "")
            score = it.get("score", None)
        else:
            ctx = str(it)
            score = None
        if not isinstance(ctx, str):
            ctx = str(ctx)
        out.append({"context": ctx, "score": score})
    return out


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ingest GARAG poisoned datasets into data_splits/<dataset>/attack=GARAG/ (answers.jsonl + chunks.jsonl)."
    )
    ap.add_argument("--data-root", default="data", help="Root containing dataset folders (e.g., data/nq, data/msmarco)")
    ap.add_argument("--out-root", default="data_splits", help="Output root for split datasets")
    ap.add_argument("--datasets", nargs="+", default=["nq", "msmarco"])
    ap.add_argument("--attack", default="GARAG", help="Attack name to store in meta.attack and directory name")
    ap.add_argument("--model", default="nomodel", help="Model tag to store in qid/meta.model")
    ap.add_argument("--dry-run", action="store_true", help="Scan and report counts without writing files")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    attack = args.attack
    model = args.model

    total_files = 0
    total_items = 0
    total_chunks = 0

    for dataset in args.datasets:
        ds_dir = data_root / dataset
        if not ds_dir.exists():
            print(f"[WARN] dataset dir not found: {ds_dir}")
            continue

        out_dir = out_root / dataset / f"attack={slugify(attack)}"
        answers_out = out_dir / "answers.jsonl"
        chunks_out = out_dir / "chunks.jsonl"

        answers_rows: List[Dict[str, Any]] = []
        chunks_rows: List[Dict[str, Any]] = []
        used_sources: List[str] = []

        for fp in iter_candidate_files(ds_dir):
            obj = try_load_json(fp)
            if obj is None or (not is_garag_payload(obj)):
                continue

            total_files += 1
            used_sources.append(str(fp))
            variant = f"{attack}__{slugify(fp.stem)}"

            for ex in extract_items(obj):
                orig_id = ex.get("id", ex.get("orig_id", ex.get("qid", "")))
                if not isinstance(orig_id, str):
                    orig_id = str(orig_id)

                question = ex.get("question", "")
                if not isinstance(question, str):
                    question = str(question)

                answer = normalize_answer(ex.get("answer"))
                incorrect = normalize_incorrect(ex.get("incorrect_answer"))
                qid = f"{orig_id}::{attack}::{model}::{variant}"

                answers_rows.append({
                    "qid": qid,
                    "orig_id": orig_id,
                    "dataset": dataset,
                    "attack": attack,
                    "model": model,
                    "variant": variant,
                    "question": question,
                    "answer": answer,
                    "incorrect_answer": incorrect,
                    "source_file": str(fp).replace("/", "\\"),
                })
                total_items += 1

                adv_texts = normalize_adv_texts(ex.get("adv_texts", []))
                for j, adv in enumerate(adv_texts):
                    ctx = adv.get("context", "")
                    score = adv.get("score", None)
                    chunk_id = f"poison::{attack}::{model}::{qid}::{j}"

                    chunks_rows.append({
                        "chunk_id": chunk_id,
                        "text": ctx,
                        "guilty_spans": [],
                        "meta": {
                            "poisoned": True,
                            "dataset": dataset,
                            "attack": attack,
                            "model": model,
                            "variant": variant,
                            "target_qid": qid,
                            "orig_id": orig_id,
                            "incorrect_answer": incorrect,
                            "answer": answer,
                            "adv_rank": j,
                            "adv_score": score,
                            "span_weak": False,
                            "source_file": str(fp).replace("/", "\\"),
                        }
                    })
                    total_chunks += 1

        if not answers_rows:
            print(f"[INFO] No GARAG-like files found in {ds_dir}")
            continue

        # Sort for reproducibility
        answers_rows.sort(key=lambda r: (r.get("variant", ""), r.get("orig_id", "")))
        chunks_rows.sort(key=lambda r: r.get("chunk_id", ""))

        print(f"[OK] dataset={dataset} files={len(used_sources)} qas={len(answers_rows)} chunks={len(chunks_rows)} -> {out_dir}")

        if args.dry_run:
            continue

        write_jsonl(answers_out, answers_rows)
        write_jsonl(chunks_out, chunks_rows)

        manifest = {
            "dataset": dataset,
            "attack": attack,
            "model": model,
            "out_dir": str(out_dir).replace("/", "\\"),
            "answers": str(answers_out).replace("/", "\\"),
            "chunks": str(chunks_out).replace("/", "\\"),
            "source_files": [s.replace("/", "\\") for s in used_sources],
            "n_qas": len(answers_rows),
            "n_chunks": len(chunks_rows),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "attack": attack,
        "datasets": args.datasets,
        "total_files_ingested": total_files,
        "total_qas": total_items,
        "total_chunks": total_chunks,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
