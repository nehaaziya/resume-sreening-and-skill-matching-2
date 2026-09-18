"""
Step 4: Vector store (FAISS)
------------------------------
Stores resume embeddings in a FAISS index for fast similarity search,
and keeps a parallel id <-> metadata mapping (FAISS itself only knows
about integer row positions, not your string ids).

- IndexFlatIP: exact cosine similarity (since vectors are L2-normalized,
  inner product == cosine similarity). Fine up to tens of thousands of
  vectors; simplest and always exact — the safe default for a 1000-resume
  batch.
- IndexIVFFlat: approximate nearest neighbor, clusters vectors into
  `nlist` cells and only searches the closest few cells. Use once your
  candidate pool grows into the tens/hundreds of thousands and exact
  search becomes the bottleneck.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissVectorStore:
    def __init__(self, dim: int, use_ivf: bool = False, nlist: int = 100):
        self.dim = dim
        self.use_ivf = use_ivf
        self.id_map: list[str] = []          # row index -> resume_id
        self.metadata: dict[str, dict] = {}  # resume_id -> arbitrary metadata

        if use_ivf:
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self._trained = False
        else:
            self.index = faiss.IndexFlatIP(dim)
            self._trained = True

    def add(self, ids: list[str], vectors: np.ndarray, metadata: list[dict] | None = None):
        vectors = np.ascontiguousarray(vectors.astype("float32"))

        if self.use_ivf and not self._trained:
            # IVF indexes need a representative sample to learn cluster centroids
            self.index.train(vectors)
            self._trained = True

        self.index.add(vectors)
        self.id_map.extend(ids)

        if metadata:
            for rid, meta in zip(ids, metadata):
                self.metadata[rid] = meta

        logger.info("Added %d vectors. Index now holds %d total.", len(ids), self.index.ntotal)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        query_vector = np.ascontiguousarray(query_vector.reshape(1, -1).astype("float32"))
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS pads with -1 if fewer than top_k results exist
            resume_id = self.id_map[idx]
            results.append((resume_id, float(score)))
        return results

    def save(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "id_map.json").write_text(json.dumps(self.id_map))
        (directory / "metadata.json").write_text(json.dumps(self.metadata))
        (directory / "config.json").write_text(json.dumps({"dim": self.dim, "use_ivf": self.use_ivf}))
        logger.info("Saved index (%d vectors) to %s", self.index.ntotal, directory)

    @classmethod
    def load(cls, directory: str | Path) -> "FaissVectorStore":
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        store = cls(dim=config["dim"], use_ivf=config["use_ivf"])
        store.index = faiss.read_index(str(directory / "index.faiss"))
        store.id_map = json.loads((directory / "id_map.json").read_text())
        store.metadata = json.loads((directory / "metadata.json").read_text())
        store._trained = True
        logger.info("Loaded index (%d vectors) from %s", store.index.ntotal, directory)
        return store