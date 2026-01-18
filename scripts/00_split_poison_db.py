from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON in {path} at line {line_no}: {e}")


_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def slug(s: str) -> str:
    s = (s or '').strip()
    if not s:
        return 'unknown'
    s = _SAFE_CHARS.sub('_', s)
    s = s.strip('._-')
    return s or 'unknown'


def key_tuple(obj: dict, fields: List[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for f in fields:
        v = obj.get(f)
        if v is None:
            # chunks.jsonl stores group attrs under meta
            v = (obj.get('meta') or {}).get(f)
        out.append(str(v) if v is not None else '')
    return tuple(out)


def group_dir(base: Path, dataset: str, fields: List[str], key: Tuple[str, ...]) -> Path:
    parts = [dataset]
    for f, v in zip(fields, key):
        parts.append(f"{f}={slug(v)}")
    return base.joinpath(*parts)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Split poisoned RAG corpora into separate per-attack (and optionally per-variant/model) databases.\n\n"
            "Inputs per dataset: data/<dataset>/answers.jsonl and data/<dataset>/chunks.jsonl\n"
            "Outputs: <out_root>/<dataset>/<field>=<value>/answers.jsonl + chunks.jsonl + manifest.json"
        )
    )
    ap.add_argument('--data-root', default='data', help='Root directory containing dataset subfolders (default: data)')
    ap.add_argument('--datasets', nargs='*', default=['nq', 'msmarco'], help='Datasets to process (default: nq msmarco)')
    ap.add_argument(
        '--split-by',
        default='attack',
        help=(
            "Comma-separated grouping fields. Common: attack or attack,variant or attack,variant,model. "
            "(default: attack)"
        ),
    )
    ap.add_argument('--out-root', default='data_splits', help='Output root directory (default: data_splits)')
    ap.add_argument(
        '--mode',
        choices=['by_answers_qid', 'by_chunk_meta'],
        default='by_answers_qid',
        help=(
            "How to assign chunks to groups. "
            "by_answers_qid: use answers.jsonl qid->group mapping (recommended to avoid competing-attacks mixing). "
            "by_chunk_meta: group chunks solely by their meta fields." 
        ),
    )
    ap.add_argument(
        '--keep-unmatched-chunks',
        action='store_true',
        help=(
            "Only relevant for mode=by_chunk_meta. If set, writes chunks even if their target_qid isn't present in answers."
        ),
    )
    ap.add_argument('--min-answers', type=int, default=1, help='Skip groups with fewer than this many answers (default: 1)')
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    fields = [f.strip() for f in args.split_by.split(',') if f.strip()]
    if not fields:
        raise SystemExit('Invalid --split-by: must contain at least one field (e.g., attack)')

    summary_all = {}

    for dataset in args.datasets:
        ds_dir = data_root / dataset
        answers_path = ds_dir / 'answers.jsonl'
        chunks_path = ds_dir / 'chunks.jsonl'

        if not answers_path.exists():
            print(f"[WARN] Missing {answers_path}; skipping dataset={dataset}")
            continue
        if not chunks_path.exists():
            print(f"[WARN] Missing {chunks_path}; skipping dataset={dataset}")
            continue

        # 1) Read answers; build qid->group map and write grouped answers
        qid_to_key: Dict[str, Tuple[str, ...]] = {}
        key_to_answers: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)

        for a in iter_jsonl(answers_path):
            key = key_tuple(a, fields)
            qid = str(a.get('qid', '')).strip()
            if not qid:
                continue
            qid_to_key[qid] = key
            key_to_answers[key].append(a)

        # Optionally prune tiny groups
        key_to_answers = {k: v for k, v in key_to_answers.items() if len(v) >= args.min_answers}

        # Write answers per group
        for key, rows in key_to_answers.items():
            gdir = group_dir(out_root, dataset, fields, key)
            write_jsonl(gdir / 'answers.jsonl', rows)

        # 2) Stream chunks and assign to groups
        key_to_chunks_count: Dict[Tuple[str, ...], int] = defaultdict(int)
        key_to_written: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)

        if args.mode == 'by_answers_qid':
            # Only keep chunks whose target_qid is in answers.jsonl, and assign group by that qid.
            for c in iter_jsonl(chunks_path):
                meta = c.get('meta') or {}
                tq = str(meta.get('target_qid', '')).strip()
                if not tq:
                    continue
                key = qid_to_key.get(tq)
                if key is None:
                    continue
                if key not in key_to_answers:
                    continue  # group pruned
                key_to_written[key].append(c)
        else:
            # Group chunks purely by their meta fields.
            for c in iter_jsonl(chunks_path):
                key = key_tuple(c, fields)
                if key not in key_to_answers and not args.keep_unmatched_chunks:
                    continue
                key_to_written[key].append(c)

        for key, rows in key_to_written.items():
            gdir = group_dir(out_root, dataset, fields, key)
            cnt = write_jsonl(gdir / 'chunks.jsonl', rows)
            key_to_chunks_count[key] = cnt

        # 3) Write manifest per dataset
        manifest = {
            'dataset': dataset,
            'split_by': fields,
            'mode': args.mode,
            'groups': [],
        }

        for key, a_rows in sorted(key_to_answers.items(), key=lambda kv: str(kv[0])):
            gdir = group_dir(out_root, dataset, fields, key)
            group_entry = {
                'group': {f: v for f, v in zip(fields, key)},
                'dir': str(gdir).replace('\\', '/'),
                'answers': len(a_rows),
                'chunks': int(key_to_chunks_count.get(key, 0)),
            }
            manifest['groups'].append(group_entry)

        (out_root / dataset).mkdir(parents=True, exist_ok=True)
        with (out_root / dataset / 'manifest.json').open('w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        summary_all[dataset] = {
            'groups': len(manifest['groups']),
            'total_answers': sum(g['answers'] for g in manifest['groups']),
            'total_chunks': sum(g['chunks'] for g in manifest['groups']),
            'manifest': str((out_root / dataset / 'manifest.json')).replace('\\', '/'),
        }

        print(f"[OK] dataset={dataset} groups={summary_all[dataset]['groups']} "
              f"answers={summary_all[dataset]['total_answers']} chunks={summary_all[dataset]['total_chunks']}")

    # global summary
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary_all, f, ensure_ascii=False, indent=2)
    print(f"Wrote summary to: {(out_root / 'summary.json')}")


if __name__ == '__main__':
    main()
