# Resume Screening & Skill Matching System

An NLP-based system that parses resumes (PDF/DOCX/TXT), extracts structured
information (skills, experience, education, certifications), generates
semantic embeddings with Sentence-BERT, indexes them in FAISS for
fast vector search, and ranks candidates against a job description.

## Architecture

```
                ┌──────────────┐
Resumes (PDF/   │  Ingestion   │  src/ingestion.py
DOCX/TXT)  ───► │  (text       │
                │  extraction) │
                └──────┬───────┘
                       │
                ┌──────▼───────┐
                │Preprocessing │  src/preprocessing.py
                │(clean, parse │  - skills / education / experience /
                │ structure)   │    certifications extraction
                └──────┬───────┘
                       │
                ┌──────▼───────┐
                │  Embedding   │  src/embedding.py
                │ (SBERT,      │  - batch encode (100-200/batch)
                │  batched,    │  - GPU-accelerated
                │  GPU)        │
                └──────┬───────┘
                       │
                ┌──────▼───────┐
                │ Vector Store │  src/vector_store.py
                │ (FAISS)      │  - IndexFlatIP / IndexIVFFlat
                └──────┬───────┘
                       │
                ┌──────▼───────┐
                │  Similarity  │  src/similarity.py
                │  & Ranking   │  - cosine similarity, weighted score,
                │              │    explainability, filtering
                └──────┬───────┘
                       │
                ┌──────▼───────┐
                │  Evaluation  │  src/evaluation.py
                │              │  - Precision/Recall/F1, MRR, NDCG,
                │              │    latency, similarity distribution
                └──────┬───────┘
                       │
                ┌──────▼───────┐
                │   REST API   │  src/api.py (FastAPI)
                └──────────────┘
```

Everything is orchestrated by `src/pipeline.py`, which is the single entry
point used by both the CLI demo (`run_demo.py`) and the API layer
(`src/api.py`).

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

GPU is used automatically if `torch.cuda.is_available()` is True; otherwise
the system falls back to CPU without any code changes.

## Quick start

```bash
# 1. Put resumes in ./data/resumes  (pdf, docx, or txt)
# 2. Run the end-to-end demo
python run_demo.py --resumes_dir data/resumes --job_description data/jd.txt --top_k 10
```

## Run as an API

```bash
uvicorn src.api:app --reload --port 8000
```

Endpoints:
- `POST /resumes/upload` — ingest and index a batch of resumes
- `POST /match` — submit a job description, get ranked candidates
- `GET  /health` — liveness check

## Scaling to 1000 resumes

- `embedding.py` batches encoding in configurable chunks (default 128,
  recommended 100–200) so memory stays bounded regardless of pool size.
- `vector_store.py` uses `IndexFlatIP` for pools under ~5k (exact search,
  fully accurate) and switches to `IndexIVFFlat` (approximate, clustered)
  above that, which is configurable via `FaissVectorStore(use_ivf=True)`.
- All I/O-bound steps (file parsing) can run in a `ThreadPoolExecutor`;
  all GPU-bound steps (embedding) run in large batches to maximize
  throughput per forward pass.

## Data privacy notes

- No resume text or embeddings are sent to third-party APIs — everything
  runs on local/self-hosted models (SentenceTransformers + FAISS).
- PII fields (name, email, phone) are extracted into structured fields so
  they can be redacted/hashed independently of the free-text body before
  storage, if required by policy.
- Add an encryption-at-rest layer (e.g., encrypt the FAISS index file and
  metadata store) and access-control on the API layer before production use.

## Evaluation

See `src/evaluation.py` and `tests/test_pipeline.py` for how to compute
Precision/Recall/F1, MRR, NDCG, similarity-score distributions, and
per-resume processing time, given a labeled set of (job, relevant resume
ids) pairs.
