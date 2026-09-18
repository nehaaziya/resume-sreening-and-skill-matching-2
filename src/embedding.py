"""
Step 3: Embedding generation
------------------------------
Wraps a SentenceTransformer model to turn resume / job-description text
into dense vectors, batched (100-200 per batch) and GPU-accelerated when
available. Also includes an optional fine-tuning entry point for when
labeled (resume, job, match-label) triples exist.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-mpnet-base-v2"  # strong general-purpose SBERT model
# Alternatives: "all-MiniLM-L6-v2" (faster, smaller, good for >100k docs)


@dataclass
class EmbeddingResult:
    ids: list[str]
    vectors: np.ndarray          # shape (N, dim), L2-normalized
    seconds_elapsed: float
    per_item_seconds: float


class EmbeddingEngine:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 128,   # within the 100-200 recommended range
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        logger.info("Loading SBERT model '%s' on device '%s'", model_name, self.device)
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, ids: list[str], texts: list[str]) -> EmbeddingResult:
        """
        Encode a list of texts in batches. Safe to call with up to
        ~1000+ items; internally chunks by `self.batch_size` so memory
        stays bounded and GPU utilization stays high.
        """
        assert len(ids) == len(texts)
        start = time.perf_counter()

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
            normalize_embeddings=True,   # so dot product == cosine similarity
            device=self.device,
        )

        elapsed = time.perf_counter() - start
        per_item = elapsed / max(len(texts), 1)
        logger.info(
            "Encoded %d texts in %.2fs (%.4fs/item) on %s",
            len(texts), elapsed, per_item, self.device,
        )
        return EmbeddingResult(ids=ids, vectors=vectors, seconds_elapsed=elapsed, per_item_seconds=per_item)

    def encode_in_batches(self, ids: list[str], texts: list[str], batch_size: int | None = None):
        """
        Generator variant: yields an EmbeddingResult per batch instead of
        one giant call. Useful when you want to stream results into the
        vector store as they're produced (e.g. progress bars, checkpointing,
        or pipelines that write-through to disk incrementally).
        """
        bs = batch_size or self.batch_size
        for i in range(0, len(texts), bs):
            batch_ids = ids[i : i + bs]
            batch_texts = texts[i : i + bs]
            yield self.encode(batch_ids, batch_texts)

    def encode_single(self, text: str) -> np.ndarray:
        return self.model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True, device=self.device
        )[0]


# ---------------------------------------------------------------------
# Fine-tuning hook: use this only if you have labeled domain data, e.g.
# pairs of (resume_text, job_text, similarity_score in [0,1]) collected
# from historical recruiter decisions (hired/interviewed vs rejected).
# ---------------------------------------------------------------------
def fine_tune_on_domain_data(
    base_model_name: str,
    training_triples: list[tuple[str, str, float]],
    output_path: str,
    epochs: int = 4,
    batch_size: int = 16,
):
    """
    training_triples: list of (resume_text, job_text, label 0.0-1.0)
    label = 1.0 for confirmed strong matches (e.g., hired/interviewed),
    0.0 for confirmed poor matches (e.g., rejected at screening).
    """
    model = SentenceTransformer(base_model_name)
    examples = [
        InputExample(texts=[resume, job], label=float(label))
        for resume, job, label in training_triples
    ]
    dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=int(0.1 * len(dataloader) * epochs),
        output_path=output_path,
        show_progress_bar=True,
    )
    logger.info("Fine-tuned model saved to %s", output_path)
    return output_path
