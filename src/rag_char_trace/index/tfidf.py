from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from rag_char_trace.data.io import Chunk


@dataclass
class TfidfIndex:
    vectorizer: TfidfVectorizer
    matrix: sparse.csr_matrix
    chunks: List[Chunk]

    def query(self, text: str, top_k: int = 10) -> List[Tuple[int, float]]:
        q = self.vectorizer.transform([text])
        scores = (self.matrix @ q.T).toarray().reshape(-1)
        if top_k >= len(scores):
            idx = np.argsort(-scores)
        else:
            # argpartition for speed
            idx_part = np.argpartition(-scores, top_k)[:top_k]
            idx = idx_part[np.argsort(-scores[idx_part])]
        return [(int(i), float(scores[i])) for i in idx[:top_k]]


def build_tfidf_index(chunks: Sequence[Chunk], *, max_features: int = 200_000) -> TfidfIndex:
    texts = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform(texts).tocsr()
    return TfidfIndex(vectorizer=vectorizer, matrix=matrix, chunks=list(chunks))


def save_tfidf_index(index: TfidfIndex, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(index.vectorizer, os.path.join(out_dir, "vectorizer.joblib"))
    sparse.save_npz(os.path.join(out_dir, "matrix.npz"), index.matrix)
    joblib.dump(index.chunks, os.path.join(out_dir, "chunks.joblib"))


def load_tfidf_index(index_dir: str) -> TfidfIndex:
    vectorizer = joblib.load(os.path.join(index_dir, "vectorizer.joblib"))
    matrix = sparse.load_npz(os.path.join(index_dir, "matrix.npz")).tocsr()
    chunks = joblib.load(os.path.join(index_dir, "chunks.joblib"))
    return TfidfIndex(vectorizer=vectorizer, matrix=matrix, chunks=chunks)
