"""
Test cases with sample inputs/outputs covering preprocessing, ranking,
and evaluation metrics. Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    dcg_at_k,
    evaluate_all,
    ndcg_at_k,
    precision_recall_f1_at_k,
    reciprocal_rank,
    similarity_distribution,
)
from src.preprocessing import (
    build_profile,
    extract_education,
    extract_skills,
    extract_years_experience,
)
from src.similarity import rank_candidates, skill_overlap


SAMPLE_RESUME = """
Jane Doe
jane.doe@example.com | (555) 123-4567

SUMMARY
Software engineer with 5 years of experience building backend systems.

SKILLS
Python, SQL, AWS, Docker, Machine Learning, Communication

EDUCATION
M.S. in Computer Science, State University

CERTIFICATIONS
AWS Certified Solutions Architect
"""

SAMPLE_JD = """
We are hiring a backend engineer with strong Python and AWS skills.
Requires at least 3 years of experience and a Bachelor's degree.
Docker and SQL experience preferred.
"""


def test_extract_skills_finds_expected_terms():
    skills = extract_skills(SAMPLE_RESUME)
    assert "python" in skills
    assert "sql" in skills
    assert "aws" in skills
    assert "docker" in skills


def test_extract_education_detects_masters():
    edu = extract_education(SAMPLE_RESUME)
    assert "Masters" in edu


def test_extract_years_experience():
    years = extract_years_experience(SAMPLE_RESUME)
    assert years == 5.0


def test_build_profile_end_to_end():
    profile = build_profile("resume_1", SAMPLE_RESUME)
    assert profile.email == "jane.doe@example.com"
    assert profile.years_experience == 5.0
    assert "python" in profile.skills
    assert len(profile.certifications) >= 1


def test_skill_overlap_computes_jaccard_style_score():
    job_skills = {"python", "aws", "docker", "sql"}
    candidate_skills = {"python", "aws", "communication"}
    score, matched, missing = skill_overlap(job_skills, candidate_skills)
    assert matched == ["aws", "python"]
    assert missing == ["docker", "sql"]
    assert score == 2 / 4


def test_rank_candidates_filters_by_min_similarity():
    job_profile = build_profile("job", SAMPLE_JD)
    candidate_profiles = {"resume_1": build_profile("resume_1", SAMPLE_RESUME)}
    faiss_hits = [("resume_1", 0.9)]

    ranked = rank_candidates(
        faiss_hits, job_profile, candidate_profiles, min_similarity=0.5
    )
    assert len(ranked) == 1
    assert ranked[0].resume_id == "resume_1"
    assert ranked[0].final_score > 0


def test_rank_candidates_excludes_below_min_similarity():
    job_profile = build_profile("job", SAMPLE_JD)
    candidate_profiles = {"resume_1": build_profile("resume_1", SAMPLE_RESUME)}
    faiss_hits = [("resume_1", 0.2)]

    ranked = rank_candidates(
        faiss_hits, job_profile, candidate_profiles, min_similarity=0.5
    )
    assert len(ranked) == 0


def test_precision_recall_f1_at_k():
    relevant = {"r1", "r2", "r3"}
    ranked = ["r1", "r9", "r2", "r8", "r3"]
    p, r, f1 = precision_recall_f1_at_k(relevant, ranked, k=3)
    assert p == 2 / 3
    assert r == 2 / 3
    assert round(f1, 4) == round(2 / 3, 4)


def test_reciprocal_rank():
    relevant = {"r3"}
    ranked = ["r1", "r2", "r3", "r4"]
    assert reciprocal_rank(relevant, ranked) == 1 / 3


def test_ndcg_perfect_ranking_is_one():
    relevant = {"r1", "r2"}
    ranked = ["r1", "r2", "r3", "r4"]
    assert round(ndcg_at_k(relevant, ranked, k=4), 4) == 1.0


def test_evaluate_all_aggregates_across_jobs():
    ground_truth = {"job_1": {"r1", "r2"}, "job_2": {"r5"}}
    predictions = {"job_1": ["r1", "r3", "r2"], "job_2": ["r5", "r6"]}
    metrics = evaluate_all(ground_truth, predictions, k=3)
    assert metrics.mrr > 0
    assert 0 <= metrics.ndcg_at_k <= 1
    assert 0 <= metrics.precision_at_k <= 1


def test_similarity_distribution_basic_stats():
    stats = similarity_distribution([0.1, 0.5, 0.9, 0.3, 0.7])
    assert stats["min"] == 0.1
    assert stats["max"] == 0.9
    assert round(stats["mean"], 2) == 0.5
