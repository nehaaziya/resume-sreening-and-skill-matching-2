"""
REST API for recruiter integration.

Run with:
    uvicorn src.api:app --reload --port 8000

Endpoints:
    POST /resumes/upload   - upload & index a batch of resume files
    POST /match            - rank indexed resumes against a job description
    GET  /health           - liveness probe
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field

from .pipeline import ResumeScreeningPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Resume Screening & Skill Matching API", version="1.0.0")

# NOTE: a single global pipeline instance keeps the SBERT model loaded once
# and the FAISS index resident in memory across requests. For multi-tenant
# / multi-recruiter production use, key this by workspace/org id instead.
pipeline = ResumeScreeningPipeline()


class MatchRequest(BaseModel):
    job_description: str
    top_k: int = Field(default=10, ge=1, le=200)
    min_similarity: float = Field(default=0.0, ge=-1.0, le=1.0)
    min_years_experience: float | None = None
    min_education: str | None = None  # one of: High School, Associate, Bachelors, Masters, PhD


class MatchResponseItem(BaseModel):
    resume_id: str
    final_score: float
    semantic_score: float
    skill_overlap_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    years_experience: float | None
    education: list[str]


class MatchResponse(BaseModel):
    results: list[MatchResponseItem]


class UploadResponse(BaseModel):
    indexed: int
    failed: list[str]
    performance: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/resumes/upload", response_model=UploadResponse)
async def upload_resumes(files: list[UploadFile] = File(...)):
    """
    Accepts a batch of resume files (pdf/docx/txt), writes them to a
    temp directory, and runs the full ingest -> embed -> index pipeline.
    Safe to call repeatedly to grow the indexed candidate pool.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for upload in files:
            dest = tmp_path / upload.filename
            with dest.open("wb") as f:
                shutil.copyfileobj(upload.file, f)

        result = pipeline.index_directory(tmp_path)

    return UploadResponse(**result)


@app.post("/match", response_model=MatchResponse)
def match_candidates(request: MatchRequest):
    ranked = pipeline.match(
        job_description_text=request.job_description,
        top_k=request.top_k,
        min_similarity=request.min_similarity,
        min_years_experience=request.min_years_experience,
        min_education=request.min_education,
    )
    return MatchResponse(
        results=[
            MatchResponseItem(
                resume_id=c.resume_id,
                final_score=round(c.final_score, 4),
                semantic_score=round(c.semantic_score, 4),
                skill_overlap_score=round(c.skill_overlap_score, 4),
                matched_skills=c.matched_skills,
                missing_skills=c.missing_skills,
                years_experience=c.years_experience,
                education=c.education,
            )
            for c in ranked
        ]
    )
