# Resume Screening & Skill Matching System

An NLP-based system that parses resumes (PDF/DOCX/TXT), extracts structured
information (skills, experience, education, certifications), generates
semantic embeddings with Sentence-BERT, indexes them in FAISS for fast
vector search, and ranks candidates against a job description using a
weighted combination of semantic similarity and skill overlap.

---

## Table of Contents

- [Architecture](#architecture)
- [Methodology](#methodology)
  - [1. Ingestion](#1-ingestion)
  - [2. Preprocessing](#2-preprocessing)
  - [3. Embedding](#3-embedding)
  - [4. Vector Store](#4-vector-store)
  - [5. Similarity & Ranking](#5-similarity--ranking)
  - [6. Evaluation](#6-evaluation)
- [Scoring Formula](#scoring-formula)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [CLI Demo](#cli-demo)
  - [REST API](#rest-api)
- [Scaling to 1,000+ Resumes](#scaling-to-1000-resumes)
- [Data Privacy](#data-privacy)
- [Evaluation Metrics](#evaluation-metrics)


---

## Architecture

The system is a linear pipeline with six stages, orchestrated end-to-end by
`src/pipeline.py`. That module is the single entry point used by both the
CLI demo (`run_demo.py`) and the REST API (`src/api.py`), so ingestion,
scoring, and ranking logic live in exactly one place.

```
                ┌──────────────┐
Resumes (PDF/   │  Ingestion   │  src/ingestion.py
DOCX/TXT)  ───► │  (text       │  - ThreadPoolExecutor, parallel I/O
                │  extraction) │  - per-file error capture
                └──────┬───────┘
                       │  RawResume (id, path, raw_text, parse_error)
                ┌──────▼───────┐
                │Preprocessing │  src/preprocessing.py
                │(clean, parse │  - skill taxonomy lookup (regex, word-boundary)
                │ structure)   │  - education / years-experience / certification
                │              │    extraction (regex heuristics)
                │              │  - email / phone extraction (for PII handling)
                └──────┬───────┘
                       │  StructuredProfile (skills, education, years, ...)
                ┌──────▼───────┐
                │  Embedding   │  src/embedding.py
                │ (SBERT,      │  - all-mpnet-base-v2 by default
                │  batched,    │  - batched (100–200/batch), GPU if available
                │  GPU)        │  - L2-normalized vectors (dot product = cosine)
                └──────┬───────┘
                       │  EmbeddingResult (ids, vectors, timing)
                ┌──────▼───────┐
                │ Vector Store │  src/vector_store.py
                │ (FAISS)      │  - IndexFlatIP (exact) or IndexIVFFlat (ANN)
                └──────┬───────┘
                       │  [(resume_id, cosine_score), ...]
                ┌──────▼───────┐
                │  Similarity  │  src/similarity.py
                │  & Ranking   │  - weighted score = 0.7*semantic + 0.3*skill overlap
                │              │  - hard filters: min similarity / experience / education
                │              │  - explainability: matched vs. missing skills
                └──────┬───────┘
                       │  RankedCandidate[]
                ┌──────▼───────┐
                │  Evaluation  │  src/evaluation.py
                │  (offline)   │  - Precision/Recall/F1, MRR, NDCG @k
                │              │  - similarity distribution, latency, robustness
                └──────┬───────┘
                       │
                ┌──────▼───────┐
                │   REST API   │  src/api.py (FastAPI)
                │              │  - POST /resumes/upload, POST /match, GET /health
                └──────────────┘
```

Both consumers of `ResumeScreeningPipeline` call the same two methods:

- `index_directory(dir)` — ingest → preprocess → embed → add to FAISS
- `match(job_description_text, ...)` — embed the JD → FAISS search → rank/filter

---

## Methodology

### 1. Ingestion

`src/ingestion.py` walks a directory (`Path.rglob("*")`), keeps files with
a supported extension (`.pdf`, `.docx`, `.txt`), and extracts raw text:

| Format | Method |
|---|---|
| `.pdf` | `pdfplumber` — text extracted per page and joined |
| `.docx` | `python-docx` — paragraph text **and** table cell text (skills are often laid out in tables) |
| `.txt` | Plain read, UTF-8, invalid bytes ignored |

Parsing is parallelized with a `ThreadPoolExecutor` (I/O-bound work, so
threads — not processes — are appropriate). Each file either returns a
`RawResume` with text, or one with `parse_error` set (unsupported type,
empty/unreadable content, or an extraction exception) — failures are
captured per-file rather than aborting the whole batch, and reported back
in `index_directory()`'s return value.

### 2. Preprocessing

`src/preprocessing.py` turns raw text into a `StructuredProfile`. This
stage is deliberately **rule-based + lightweight regex**, not a trained
extraction model — it works out of the box with no labeled data, at the
cost of being less robust to unusual resume formats than a trained NER
model would be.

- **Text cleaning** — normalizes line endings, collapses repeated
  whitespace/blank lines.
- **Skills** — looked up against a taxonomy (`data/skills_taxonomy.txt` if
  present, otherwise a small built-in fallback list) using
  word-boundary-safe regex matching, so `"r"` doesn't false-match inside
  `"car"`. This is a lookup against a closed vocabulary, not open-ended
  skill discovery — see [Known Limitations](#known-limitations).
- **Education** — regex patterns for PhD / Masters / Bachelors /
  Associate / High School, deduplicated and sorted by level (highest first).
- **Years of experience** — regex over phrases like `"7+ years of
  experience"`; if multiple mentions are found, the **maximum** value is
  taken as the headline figure.
- **Certifications** — lines containing hint keywords (`"certified"`,
  `"pmp"`, `"aws certified"`, etc.), capped at 20 lines to avoid pulling
  noise out of garbled PDF extractions.
- **Contact info** — email and phone extracted via regex into structured
  fields, so they can be redacted from the free-text body independently
  (see [Data Privacy](#data-privacy)).

The same `build_profile()` function is applied to job descriptions, so job
and resume text are structured identically before scoring.

### 3. Embedding

`src/embedding.py` wraps a `SentenceTransformer` model
(`all-mpnet-base-v2` by default — a strong general-purpose SBERT model;
`all-MiniLM-L6-v2` is noted as a faster/smaller alternative for pools
above ~100k documents).

- Encoding runs on GPU automatically if `torch.cuda.is_available()`,
  otherwise CPU, with no code changes required.
- Text is encoded in configurable batches (default 128, within the
  recommended 100–200 range) so memory stays bounded regardless of pool
  size.
- Vectors are **L2-normalized** at encode time, so a FAISS inner-product
  index is mathematically equivalent to cosine similarity — avoiding a
  separate normalization step at search time.
- `encode_in_batches()` is available as a generator variant for streaming
  results into the vector store incrementally (progress bars,
  checkpointing) instead of one large blocking call.
- An optional `fine_tune_on_domain_data()` hook is provided for teams that
  accumulate labeled `(resume, job, match-label)` triples from historical
  recruiter decisions — it fine-tunes the base model with a cosine-
  similarity loss. Not required for the system to function.

### 4. Vector Store

`src/vector_store.py` wraps FAISS and keeps a parallel `id_map` (row index
→ resume ID) and `metadata` dict, since FAISS itself only knows integer
row positions.

- **`IndexFlatIP`** (default) — exact cosine similarity via inner product
  on normalized vectors. Simple, always exact, and fine up to tens of
  thousands of vectors — the safe default for a ~1,000-resume batch.
- **`IndexIVFFlat`** (opt-in via `use_ivf=True`) — approximate nearest
  neighbor search; clusters vectors into `nlist` cells and searches only
  the nearest few, trading a small amount of recall for speed once the
  pool grows into the tens/hundreds of thousands.
- The index (plus `id_map.json`, `metadata.json`, `config.json`) can be
  persisted to disk with `.save()` / `.load()`, so a screening run doesn't
  require re-embedding on every restart.

### 5. Similarity & Ranking

`src/similarity.py` combines two signals into one `final_score`:

1. **Semantic score** — cosine similarity between the job-description
   embedding and each resume embedding, from FAISS.
2. **Skill overlap score** — a Jaccard-style overlap between the job's
   extracted skill set and the candidate's: `|matched skills| / |job
   skills|`.

Hard filters are applied **before** scoring is finalized, so filtered-out
candidates never appear in results:

- `min_similarity` — drop hits below a cosine-similarity floor
- `min_years_experience` — drop candidates below a years-of-experience floor
- `min_education` — drop candidates below a minimum education rank
  (`High School < Associate < Bachelors < Masters < PhD`)

Every returned `RankedCandidate` includes **matched** and **missing**
skills relative to the job description, giving a recruiter an explainable
reason for the score rather than an opaque number.

`pipeline.match()` over-fetches from FAISS (`top_k * 3`, minimum 30)
before filtering, since post-hoc experience/education filters can remove
hits — over-fetching keeps the final result set close to the requested
`top_k` even after filtering.

### 6. Evaluation

`src/evaluation.py` is an **offline** evaluation module — it operates on a
labeled dataset you supply (`ground_truth`: job → set of relevant resume
IDs; `predictions`: job → ranked list of resume IDs), not on live
screening runs. It computes:

- **Precision / Recall / F1 @ k**
- **MRR** (Mean Reciprocal Rank) — how high the first relevant result lands
- **NDCG @ k** — rewards relevant results appearing earlier in the ranking
- **Similarity-score distribution** (min/max/mean/median/stdev/p25/p75) —
  useful for sanity-checking whether your score thresholds are calibrated
  to the actual score distribution the model produces
- **Processing-time report** — mean/p95 seconds per resume, throughput
  (resumes/sec)
- **Robustness report** — fraction of resumes that yielded *no* extracted
  skills, education, or experience at all — a proxy for "the parser
  couldn't get meaningful signal out of this document" (scanned images,
  heavily-styled PDFs, corrupted files)

See `tests/test_pipeline.py` for example usage.

---

## Scoring Formula

```
final_score = (semantic_weight × semantic_score) + (skill_weight × skill_overlap_score)
```

Defaults: `semantic_weight = 0.7`, `skill_weight = 0.3` — both configurable
per-call in `rank_candidates()`.

The CLI demo (`run_demo.py`) buckets the resulting `final_score` into a
recruiter-facing decision:

| Score | Decision |
|---|---|
| ≥ 0.70 | **SELECTED** — strong overall match |
| 0.60 – 0.69 | **REVIEW** — moderate match, recruiter should review |
| < 0.60 | **REJECTED** — low overall match |

These thresholds are a starting point, not a statistically derived cutoff
— tune them against your own labeled outcomes using the evaluation module
above.

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── run_demo.py              # CLI entry point + Excel report generation
├── src/
│   ├── __init__.py
│   ├── ingestion.py          # Step 1
│   ├── preprocessing.py      # Step 2
│   ├── embedding.py          # Step 3
│   ├── vector_store.py       # Step 4
│   ├── similarity.py         # Step 5
│   ├── evaluation.py         # Step 6
│   ├── pipeline.py           # Orchestrator used by CLI + API
│   └── api.py                # FastAPI REST layer
├── data/
│   ├── resumes/               # Drop resume files here (gitignored)
│   └── skills_taxonomy.txt   # Optional custom skill vocabulary
└── tests/
    └── test_pipeline.py
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

GPU is used automatically for embedding if `torch.cuda.is_available()` is
`True`; otherwise the system falls back to CPU with no code changes.

---

## Usage

### CLI Demo

```bash
# 1. Put resumes in ./data/resumes  (pdf, docx, or txt)
# 2. Run the end-to-end demo
python run_demo.py \
    --resumes_dir data/resumes \
    --job_description data/jd.txt \
    --top_k 10
```

Optional flags:

| Flag | Purpose |
|---|---|
| `--min_similarity` | Floor on semantic similarity (default `0.0`) |
| `--min_years_experience` | Floor on extracted years of experience |
| `--min_education` | Minimum education level (`High School` … `PhD`) |
| `--save_index_to` | Directory to persist the FAISS index for reuse |

The run prints a ranked terminal summary and writes a formatted
`resume_screening_results.xlsx` (rank, scores, decision, reason, matched/
missing skills, experience, education — with frozen header, autofilter,
and column widths preset for readability).

### REST API

```bash
uvicorn src.api:app --reload --port 8000
```

| Endpoint | Purpose |
|---|---|
| `POST /resumes/upload` | Upload a batch of resume files; ingests and indexes them. Safe to call repeatedly to grow the candidate pool. |
| `POST /match` | Submit a job description (+ optional filters); returns ranked candidates as JSON. |
| `GET /health` | Liveness probe. |

The API holds a single global `ResumeScreeningPipeline` instance so the
SBERT model and FAISS index stay resident in memory across requests. For
multi-tenant use (multiple recruiters/orgs with separate candidate pools),
key the pipeline instance by workspace/org ID rather than using one
global instance.

---

## Scaling to 1,000+ Resumes

- **Embedding**: batched in configurable chunks (default 128, recommended
  100–200) so memory stays bounded regardless of pool size.
- **Vector search**: `IndexFlatIP` for pools under ~5k (exact, fully
  accurate); switches to `IndexIVFFlat` (approximate, clustered) above
  that via `FaissVectorStore(use_ivf=True)`.
- **I/O vs. GPU work kept separate**: file parsing runs in a
  `ThreadPoolExecutor` (I/O-bound); embedding runs in large batches to
  maximize GPU throughput per forward pass.

---

## Data Privacy

- No resume text or embeddings are sent to third-party APIs — everything
  runs on local/self-hosted models (SentenceTransformers + FAISS).
- PII fields (name, email, phone) are extracted into structured fields
  during preprocessing so they can be redacted or hashed independently of
  the free-text body before storage, if required by policy
  (`preprocessing.redact_pii()` strips email/phone from text).
- **Not yet implemented, recommended before production use:** encryption
  at rest for the FAISS index file and metadata store, and access control
  on the API layer.

---

## Evaluation Metrics

To evaluate ranking quality, supply a labeled set of `(job, relevant
resume ids)` pairs and your system's ranked predictions to
`evaluate_all()` in `src/evaluation.py`:

```python
from src.evaluation import evaluate_all

ground_truth = {"job_1": {"resume_12", "resume_47"}}
predictions = {"job_1": ["resume_12", "resume_5", "resume_47"]}

metrics = evaluate_all(ground_truth, predictions, k=10)
# RetrievalMetrics(precision_at_k=..., recall_at_k=..., f1_at_k=...,
#                   mrr=..., ndcg_at_k=..., k=10)
```

See `tests/test_pipeline.py` for a fuller example, including the
similarity-distribution, processing-time, and robustness reports.

---

