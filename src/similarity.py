"""
Step 5: Similarity scoring & ranking
--------------------------------------
Combines the FAISS semantic-similarity score with structured signals
(skill overlap, years of experience, education level) into a final
weighted score, then ranks and filters candidates. Also produces a
simple explainability payload (matched/missing skills) per candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .preprocessing import StructuredProfile

EDUCATION_RANK = {"PhD": 4, "Masters": 3, "Bachelors": 2, "Associate": 1, "High School": 0}


@dataclass
class RankedCandidate:
    resume_id: str
    semantic_score: float          # cosine similarity from FAISS, in [-1, 1] (~[0,1] in practice)
    skill_overlap_score: float     # jaccard-style overlap in [0, 1]
    final_score: float
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    years_experience: float | None = None
    education: list[str] = field(default_factory=list)


def skill_overlap(job_skills: set[str], candidate_skills: set[str]) -> tuple[float, list[str], list[str]]:
    if not job_skills:
        return 0.0, [], []
    matched = sorted(job_skills & candidate_skills)
    missing = sorted(job_skills - candidate_skills)
    score = len(matched) / len(job_skills)
    return score, matched, missing


def meets_min_education(candidate_education: list[str], min_level: str | None) -> bool:
    if not min_level:
        return True
    if not candidate_education:
        return False
    candidate_max = max(EDUCATION_RANK.get(e, -1) for e in candidate_education)
    return candidate_max >= EDUCATION_RANK.get(min_level, -1)


def rank_candidates(
    faiss_hits: list[tuple[str, float]],
    job_profile: StructuredProfile,
    candidate_profiles: dict[str, StructuredProfile],
    semantic_weight: float = 0.7,
    skill_weight: float = 0.3,
    min_similarity: float = 0.0,
    min_years_experience: float | None = None,
    min_education: str | None = None,
) -> list[RankedCandidate]:
    """
    faiss_hits: output of FaissVectorStore.search() -> [(resume_id, cosine_score), ...]
    Returns candidates sorted by final_score descending, after applying
    hard filters (min similarity / experience / education).
    """
    results: list[RankedCandidate] = []

    for resume_id, semantic_score in faiss_hits:
        if semantic_score < min_similarity:
            continue

        profile = candidate_profiles.get(resume_id)
        if profile is None:
            continue

        if min_years_experience is not None:
            years = profile.years_experience or 0.0
            if years < min_years_experience:
                continue

        if not meets_min_education(profile.education, min_education):
            continue

        overlap_score, matched, missing = skill_overlap(job_profile.skills, profile.skills)
        final_score = semantic_weight * semantic_score + skill_weight * overlap_score

        results.append(
            RankedCandidate(
                resume_id=resume_id,
                semantic_score=semantic_score,
                skill_overlap_score=overlap_score,
                final_score=final_score,
                matched_skills=matched,
                missing_skills=missing,
                years_experience=profile.years_experience,
                education=profile.education,
            )
        )

    results.sort(key=lambda c: c.final_score, reverse=True)
    return results
