"""
Step 6: Evaluation
---------------------
Computes the metrics required to judge screening quality and system
performance, given a labeled evaluation set of the form:

    ground_truth = {
        "job_1": {"resume_12", "resume_47", "resume_88"},   # relevant resumes
        "job_2": {"resume_3"},
        ...
    }
    predictions = {
        "job_1": ["resume_12", "resume_5", "resume_47", ...],  # ranked, best first
        "job_2": ["resume_9", "resume_3", ...],
        ...
    }
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int


def precision_recall_f1_at_k(relevant: set[str], ranked: list[str], k: int) -> tuple[float, float, float]:
    top_k = ranked[:k]
    if not top_k:
        return 0.0, 0.0, 0.0
    hits = sum(1 for r in top_k if r in relevant)
    precision = hits / len(top_k)
    recall = hits / len(relevant) if relevant else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def reciprocal_rank(relevant: set[str], ranked: list[str]) -> float:
    for i, resume_id in enumerate(ranked, start=1):
        if resume_id in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    score = 0.0
    for i, resume_id in enumerate(ranked[:k], start=1):
        rel = 1.0 if resume_id in relevant else 0.0
        score += rel / math.log2(i + 1)
    return score


def ndcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    actual_dcg = dcg_at_k(relevant, ranked, k)
    ideal_ranked = list(relevant) + [r for r in ranked if r not in relevant]
    ideal_dcg = dcg_at_k(relevant, ideal_ranked, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_all(
    ground_truth: dict[str, set[str]],
    predictions: dict[str, list[str]],
    k: int = 10,
) -> RetrievalMetrics:
    precisions, recalls, f1s, rrs, ndcgs = [], [], [], [], []

    for job_id, relevant in ground_truth.items():
        ranked = predictions.get(job_id, [])
        p, r, f1 = precision_recall_f1_at_k(relevant, ranked, k)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        rrs.append(reciprocal_rank(relevant, ranked))
        ndcgs.append(ndcg_at_k(relevant, ranked, k))

    return RetrievalMetrics(
        precision_at_k=statistics.mean(precisions) if precisions else 0.0,
        recall_at_k=statistics.mean(recalls) if recalls else 0.0,
        f1_at_k=statistics.mean(f1s) if f1s else 0.0,
        mrr=statistics.mean(rrs) if rrs else 0.0,
        ndcg_at_k=statistics.mean(ndcgs) if ndcgs else 0.0,
        k=k,
    )


def similarity_distribution(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {}
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    return {
        "min": sorted_scores[0],
        "max": sorted_scores[-1],
        "mean": statistics.mean(sorted_scores),
        "median": statistics.median(sorted_scores),
        "stdev": statistics.stdev(sorted_scores) if n > 1 else 0.0,
        "p25": sorted_scores[int(0.25 * (n - 1))],
        "p75": sorted_scores[int(0.75 * (n - 1))],
    }


def processing_time_report(per_item_seconds: list[float]) -> dict[str, float]:
    if not per_item_seconds:
        return {}
    return {
        "mean_seconds_per_resume": statistics.mean(per_item_seconds),
        "p95_seconds_per_resume": sorted(per_item_seconds)[int(0.95 * (len(per_item_seconds) - 1))],
        "total_seconds": sum(per_item_seconds),
        "resumes_per_second": len(per_item_seconds) / sum(per_item_seconds) if sum(per_item_seconds) > 0 else 0.0,
    }


def robustness_report(profiles: list, noisy_flag_fn=None) -> dict[str, float]:
    """
    Basic robustness signal: fraction of resumes that failed to yield any
    extracted skills, education, or experience (a proxy for "the parser
    couldn't get meaningful signal out of this document" — e.g. scanned
    images, heavily-formatted PDFs, corrupted files).
    """
    if not profiles:
        return {}
    empty = sum(
        1 for p in profiles
        if not p.skills and not p.education and p.years_experience is None
    )
    return {
        "total_resumes": len(profiles),
        "low_signal_resumes": empty,
        "low_signal_rate": empty / len(profiles),
    }
