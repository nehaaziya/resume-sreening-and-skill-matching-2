"""
Step 2: Preprocessing
----------------------
Cleans raw resume/JD text and extracts structured fields:
  - skills (via a taxonomy lookup + noun-phrase fallback)
  - years of experience
  - education level / degrees
  - certifications

This is intentionally rule-based + lightweight-NLP (spaCy) rather than a
heavy trained extraction model, so it works out of the box. Swap in a
fine-tuned NER model here later if domain-specific labeled data becomes
available (see `embedding.py` docstring for the fine-tuning hook).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------
# Skill taxonomy: replace / extend with a domain-specific list (O*NET,
# ESCO, LinkedIn Skills Graph, or an internal HR taxonomy).
# ---------------------------------------------------------------------
DEFAULT_SKILLS_PATH = Path(__file__).parent.parent / "data" / "skills_taxonomy.txt"

EDUCATION_PATTERNS = [
    (r"\bph\.?d\.?\b|\bdoctorate\b", "PhD"),
    (r"\bm\.?s\.?c?\.?\b|\bmaster'?s\b|\bm\.?tech\b|\bmba\b", "Masters"),
    (r"\bb\.?s\.?c?\.?\b|\bbachelor'?s\b|\bb\.?tech\b|\bb\.?e\.?\b", "Bachelors"),
    (r"\bassociate'?s? degree\b", "Associate"),
    (r"\bhigh school\b|\bdiploma\b", "High School"),
]

CERTIFICATION_HINTS = [
    "certified", "certification", "certificate", "pmp", "aws certified",
    "azure certified", "google cloud certified", "cissp", "cfa", "cpa",
    "scrum master", "six sigma", "comptia", "itil",
]

EXPERIENCE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\s*(?:of)?\s*(?:experience|exp)?",
    re.IGNORECASE,
)

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_PATTERN = re.compile(r"(\+?\d[\d\-\s()]{8,}\d)")

SECTION_HEADERS = [
    "experience", "work experience", "employment history", "education",
    "skills", "technical skills", "certifications", "projects", "summary",
]


@lru_cache(maxsize=1)
def load_skill_taxonomy(path: str | None = None) -> set[str]:
    p = Path(path) if path else DEFAULT_SKILLS_PATH
    if not p.exists():
        # Minimal built-in fallback so the module works with zero setup.
        return {
            "python", "java", "c++", "sql", "machine learning", "deep learning",
            "nlp", "pytorch", "tensorflow", "aws", "azure", "gcp", "docker",
            "kubernetes", "react", "node.js", "excel", "project management",
            "data analysis", "communication", "leadership", "agile", "scrum",
            "tableau", "power bi", "spark", "hadoop", "git", "linux",
        }
    return {line.strip().lower() for line in p.read_text().splitlines() if line.strip()}


@dataclass
class StructuredProfile:
    resume_id: str
    raw_text: str
    clean_text: str
    skills: set[str] = field(default_factory=set)
    education: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    years_experience: float | None = None
    email: str | None = None
    phone: str | None = None


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_skills(text: str, taxonomy: set[str] | None = None) -> set[str]:
    taxonomy = taxonomy or load_skill_taxonomy()
    text_lower = f" {text.lower()} "
    found = set()
    for skill in taxonomy:
        # word-boundary match so "r" doesn't match inside "car", etc.
        pattern = r"(?<![a-zA-Z0-9+#.])" + re.escape(skill) + r"(?![a-zA-Z0-9+#])"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


def extract_education(text: str) -> list[str]:
    text_lower = text.lower()
    found = []
    for pattern, label in EDUCATION_PATTERNS:
        if re.search(pattern, text_lower):
            found.append(label)
    # Highest level first, de-duplicated, preserving rank order
    rank = ["PhD", "Masters", "Bachelors", "Associate", "High School"]
    return sorted(set(found), key=lambda x: rank.index(x))


def extract_certifications(text: str) -> list[str]:
    text_lower = text.lower()
    lines = text.split("\n")
    hits = []
    for line in lines:
        line_l = line.lower()
        if any(hint in line_l for hint in CERTIFICATION_HINTS):
            hits.append(line.strip())
    return hits[:20]  # cap to avoid pulling in noise from garbled PDFs


def extract_years_experience(text: str) -> float | None:
    matches = EXPERIENCE_PATTERN.findall(text)
    if not matches:
        return None
    values = [float(m) for m in matches]
    return max(values)  # assume the most-cited figure is the headline total


def extract_contact_info(text: str) -> tuple[str | None, str | None]:
    email_match = EMAIL_PATTERN.search(text)
    phone_match = PHONE_PATTERN.search(text)
    email = email_match.group(0) if email_match else None
    phone = phone_match.group(0) if phone_match else None
    return email, phone


def build_profile(resume_id: str, raw_text: str, taxonomy: set[str] | None = None) -> StructuredProfile:
    text = clean_text(raw_text)
    email, phone = extract_contact_info(text)
    return StructuredProfile(
        resume_id=resume_id,
        raw_text=raw_text,
        clean_text=text,
        skills=extract_skills(text, taxonomy),
        education=extract_education(text),
        certifications=extract_certifications(text),
        years_experience=extract_years_experience(text),
        email=email,
        phone=phone,
    )


def redact_pii(text: str) -> str:
    """Optional privacy step: strip email/phone before the text is embedded
    or logged, if policy requires it. Structured fields (email/phone) are
    still retained separately in StructuredProfile for recruiter contact."""
    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    text = PHONE_PATTERN.sub("[PHONE]", text)
    return text
