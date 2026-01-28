from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import orjson


@dataclass
class RetrievalHit:
    rank: int
    score: float
    chunk_id: str
    text: str
    meta: Dict[str, Any]


@dataclass
class TraceRecord:
    qid: str
    question: str
    retrieval: Dict[str, Any]
    generation: Dict[str, Any]
    meta: Dict[str, Any]


class JsonlTraceWriter:
    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._fp = open(path, "ab")

    def write(self, record: TraceRecord) -> None:
        self._fp.write(orjson.dumps(asdict(record)))
        self._fp.write(b"\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()
