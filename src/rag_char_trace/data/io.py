from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

import jsonlines


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    meta: Dict[str, Any]
    guilty_spans: Optional[List[Dict[str, int]]] = None


@dataclass(frozen=True)
class QAItem:
    qid: str
    question: str
    answer: str
    incorrect_answer: str
    meta: Dict[str, Any]


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with jsonlines.open(path, mode="r") as r:
        for obj in r:
            yield obj


def load_chunks(path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    for obj in read_jsonl(path):
        chunks.append(
            Chunk(
                chunk_id=obj["chunk_id"],
                text=obj.get("text", ""),
                meta=obj.get("meta", {}),
                guilty_spans=obj.get("guilty_spans"),
            )
        )
    return chunks


def load_answers(path: str) -> List[QAItem]:
    items: List[QAItem] = []
    for obj in read_jsonl(path):
        meta = {
            k: v
            for k, v in obj.items()
            if k not in {"qid", "question", "answer", "incorrect_answer"}
        }
        items.append(
            QAItem(
                qid=obj["qid"],
                question=obj["question"],
                answer=obj.get("answer", ""),
                incorrect_answer=obj.get("incorrect_answer", ""),
                meta=meta,
            )
        )
    return items
