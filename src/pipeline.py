"""
Orchestrates the full flow:
  ingestion -> preprocessing -> embedding -> vector store -> similarity/ranking

Used by both run_demo.py (CLI) and src/api.py (REST API), so the logic
lives in exactly one place.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .embedding import EmbeddingEngine
from .evaluation import processing_time_report
from .ingestion import RawResume, batched, ingest_directory
from .preprocessing import StructuredProfile, build_profile
from .similarity import RankedCandidate, rank_candidates
from .vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


class ResumeScreeningPipeline:
    def __init__(self, model_name: str = "all-mpnet-base-v2", batch_size: int = 128, use_ivf: bool = False):
        self.embedder = EmbeddingEngine(model_name=model_name, batch_size=batch_size)
        self.vector_store: FaissVectorStore | None = None
        self.profiles: dict[str, StructuredProfile] = {}
        self.use_ivf = use_ivf

    # ---------------- Indexing (build the candidate pool) ----------------

    def index_directory(self, directory: str | Path, batch_size: int = 150) -> dict:
        """
        Full ingest -> preprocess -> embed -> index flow for up to ~1000+
        resumes in a directory. Processes in batches of `batch_size`
        (100-200 recommended) so memory/GPU usage stays bounded.
        """
        raw_resumes = ingest_directory(directory)
        valid = [r for r in raw_resumes if not r.parse_error]
        failed = [r for r in raw_resumes if r.parse_error]
        logger.info("%d resumes parsed, %d failed", len(valid), len(failed))

        per_item_times: list[float] = []

        for batch in batched(valid, batch_size):
            ids = [r.resume_id for r in batch]
            profiles = [build_profile(r.resume_id, r.raw_text) for r in batch]
            texts = [p.clean_text for p in profiles]

            embedding_result = self.embedder.encode(ids, texts)
            per_item_times.extend([embedding_result.per_item_seconds] * len(ids))

            if self.vector_store is None:
                dim = embedding_result.vectors.shape[1]
                self.vector_store = FaissVectorStore(dim=dim, use_ivf=self.use_ivf)

            metadata = [{"num_skills": len(p.skills)} for p in profiles]
            self.vector_store.add(ids, embedding_result.vectors, metadata)

            for p in profiles:
                self.profiles[p.resume_id] = p

        return {
            "indexed": len(valid),
            "failed": [r.resume_id for r in failed],
            "performance": processing_time_report(per_item_times),
        }

    def save_index(self, directory: str | Path):
        if self.vector_store is None:
            raise RuntimeError("Nothing indexed yet.")
        self.vector_store.save(directory)

    def load_index(self, directory: str | Path):
        self.vector_store = FaissVectorStore.load(directory)

    # ---------------- Matching (query with a job description) ----------------

    def match(
        self,
        job_description_text: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
        min_years_experience: float | None = None,
        min_education: str | None = None,
    ) -> list[RankedCandidate]:
        if self.vector_store is None:
            raise RuntimeError("No resumes indexed yet. Call index_directory() first.")

        job_profile = build_profile("job_description", job_description_text)
        query_vector = self.embedder.encode_single(job_profile.clean_text)

        # Over-fetch from FAISS since post-filtering (experience/education)
        # may drop some hits; fetch a wider pool then trim to top_k after ranking.
        fetch_k = max(top_k * 3, 30)
        hits = self.vector_store.search(query_vector, top_k=fetch_k)

        ranked = rank_candidates(
            faiss_hits=hits,
            job_profile=job_profile,
            candidate_profiles=self.profiles,
            min_similarity=min_similarity,
            min_years_experience=min_years_experience,
            min_education=min_education,
        )
        return ranked[:top_k]
