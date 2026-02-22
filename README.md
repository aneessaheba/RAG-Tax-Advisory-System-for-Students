# RAG Tax Advisor for International Students

A chatbot that helps international students in the U.S. understand their tax obligations. It uses **RAG** (Retrieval-Augmented Generation) — meaning it searches through real IRS publications, tax treaties, and university guides to give grounded answers instead of making things up.



---

## How It Works

```
PDF Documents ──> Extract Text ──> Clean ──> Chunk ──> Embed ──> Store in ChromaDB
                                                                        │
User asks a question ──> Embed question ──> Search ChromaDB ──> Top matches
                                                                        │
                              Student profile + matching docs + question │
                                                                        ▼
                                                              Gemini (free LLM)
                                                                        │
                                                                        ▼
                                                                 Answer to user
```

1. **Data pipeline** (run once): Extracts text from ~40 tax PDFs, cleans it, splits into small chunks, embeds each chunk into a vector, and stores everything in a local vector database (ChromaDB).
2. **Chat app**: Asks the student a few profile questions (visa type, country, income, etc.), then for each tax question, searches the database for the most relevant chunks and sends them along with the question to Google Gemini (free) to generate an answer.

---

## Project Structure

```
RAG-Tax-Advisor/
│
├── app.py                      # Main chatbot — run this to ask tax questions
├── run_pipeline.py             # Runs all 5 data pipeline steps in order
├── requirements.txt            # Python dependencies (all free)
├── .env                        # Your Gemini API key (not committed to git)
├── .env.example                # Template for .env
├── user_profile.json           # Saved student profile from last session
│
├── tax_rag_data/               # Data + pipeline scripts
│   ├── document_manifest.csv   # List of all PDFs with metadata (doc_id, type, title, etc.)
│   │
│   ├── irs_publications/       # IRS publications (PDFs)
│   │   ├── p519.pdf            #   Pub 519 - U.S. Tax Guide for Aliens (the main one)
│   │   ├── p901.pdf            #   Pub 901 - U.S. Tax Treaties
│   │   ├── p970.pdf            #   Pub 970 - Tax Benefits for Education
│   │   └── p17.pdf             #   Pub 17 - Your Federal Income Tax
│   │
│   ├── irs_forms/              # IRS forms and instructions (PDFs)
│   │   ├── i1040nr.pdf         #   1040-NR instructions (nonresident tax return)
│   │   ├── f8843.pdf           #   Form 8843 (exempt individual statement)
│   │   ├── f8233.pdf           #   Form 8233 (treaty exemption from withholding)
│   │   ├── fw8ben.pdf          #   Form W-8BEN (foreign status certificate)
│   │   ├── fw2.pdf             #   Form W-2 (wage statement)
│   │   ├── f1098t.pdf          #   Form 1098-T (tuition statement)
│   │   └── i1098et.pdf         #   Instructions for 1098-E and 1098-T
│   │
│   ├── treaties/               # U.S. tax treaties with common countries (PDFs)
│   │   ├── india.pdf, china.pdf, korea.pdf, canada.pdf, etc.
│   │   └── inditech.pdf, chintech.pdf, etc. (technical explanations)
│   │
│   ├── university_guides/      # University tax guides for international students (PDFs)
│   │   ├── International-Student-Tax-FactSheet.pdf
│   │   ├── International+Student+Tax+Filing+Guide.pdf
│   │   ├── F-1-OPT-and-CPT-Info.pdf
│   │   └── ... (20+ guides from various universities)
│   │
│   ├── extract_pdfs_to_json.py     # Step 1: Extract text from each PDF page → JSON
│   ├── clean_parsed_json.py        # Step 2: Clean text (fix line breaks, spacing, etc.)
│   ├── split_clean_json_to_chunks.py # Step 3: Split into 500-word chunks with overlap
│   ├── embed_chunks.py             # Step 4: Embed chunks using sentence-transformers (free)
│   ├── upload_to_chromadb.py       # Step 5: Load embedded chunks into ChromaDB
│   │
│   └── data_work/              # Generated data (not committed to git)
│       ├── parsed_docs/        #   Raw extracted JSON from PDFs
│       ├── clean_docs/         #   Cleaned JSON
│       ├── chunks/             #   500-word text chunks
│       ├── embedded_chunks/    #   Chunks with embedding vectors
│       └── chroma_db/          #   ChromaDB vector database
│
└── tiered-support-tax-rag/     # Config files
    ├── .env.example
    └── .gitignore
```

### What Each Pipeline Script Does

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `extract_pdfs_to_json.py` | Reads each PDF listed in `document_manifest.csv`, extracts text page-by-page using PyMuPDF, saves as JSON |
| 2 | `clean_parsed_json.py` | Fixes hyphenated line breaks, removes extra whitespace, normalizes formatting |
| 3 | `split_clean_json_to_chunks.py` | Splits each page into ~500-word chunks with 100-word overlap so no information falls between cracks |
| 4 | `embed_chunks.py` | Converts each text chunk into a 384-dimensional vector using `all-MiniLM-L6-v2` (free, runs locally) |
| 5 | `upload_to_chromadb.py` | Loads all chunks + embeddings into ChromaDB (a local vector database, no server needed) |

### What the App Does (`app.py`)

1. Asks the student 7 profile questions (visa type, home country, first entry year, tax year, income types, state, SSN/ITIN)
2. Enters a chat loop where the student can ask tax questions
3. For each question: embeds the query → searches ChromaDB for top 5 matching chunks → sends profile + chunks + question to Gemini → prints the answer

---

## Features

### Feature 1 — Hybrid Retrieval (BM25 + Vector + RRF)

Plain vector search misses exact keyword matches — e.g. searching "8843" might not surface the Form 8843 chunk if the embedding isn't close enough. BM25 (keyword search) fills that gap.

**How it works (`retriever.py`):**
1. **Vector search** — ChromaDB finds the top 20 semantically similar chunks using `all-MiniLM-L6-v2` embeddings
2. **BM25 search** — `rank_bm25` scores all chunks by keyword overlap with the query; top 20 taken
3. **Reciprocal Rank Fusion (RRF)** — merges both ranked lists: score = Σ 1/(60 + rank). Top 5 returned.

```
Query ──> Vector Search (top 20) ──┐
                                   ├──> RRF merge ──> Top 5 chunks
Query ──> BM25 Search   (top 20) ──┘
```

**Result:** Hit rate improved from **70% → 100%** on the evaluation set.

---

### Feature 2 — Confidence Threshold + Refusal Policy

The bot refuses to answer when it shouldn't — preventing hallucinations and off-topic responses.

**Two-layer refusal (`app.py`):**

| Layer | Check | What triggers refusal |
|-------|-------|-----------------------|
| 1 | Keyword filter | Question contains none of ~40 tax-related keywords |
| 2 | Confidence threshold | Best vector similarity score < 0.70 |

Layer 1 catches completely off-topic questions (e.g. "What's the weather?") before even hitting the database. Layer 2 catches tax-sounding questions where the database has no relevant documents.

```python
# Layer 1 — fast keyword check
if not is_tax_question(question):
    print("This doesn't appear to be a tax question.")

# Layer 2 — retrieval confidence
chunks, confidence = retriever.retrieve(query)
if confidence < 0.70:
    print(f"[Low confidence: {confidence:.2f}] Couldn't find reliable info.")
```

---

### Feature 3 — LLM-as-a-Judge Evaluation

Cosine similarity can't tell if an answer is actually correct — two texts can be similar in embedding space but factually wrong. LLM-as-a-Judge uses Gemini itself to score answer quality.

**How it works (`evaluate.py`):**

For each test question, after generating the answer, a second Gemini call scores it on 3 dimensions:

| Dimension | What it checks |
|-----------|---------------|
| Correctness | Is the answer factually accurate for U.S. tax law? |
| Completeness | Does it fully address the question? |
| Groundedness | Is it based on the retrieved context (no hallucination)? |

Each score is 0.0–1.0. The overall Judge score is their average.

**Result:** Judge score = **0.770** — identified 2 weak answers that cosine metrics rated as acceptable but were actually vague or off-topic.

---

### Feature 4 — Latency and Token Tracking

Every query logs how long each step takes and estimates token usage. Printed after each answer and saved to `query_log.jsonl`.

**What's tracked:**

| Field | Description |
|-------|-------------|
| `retrieval_latency_s` | Time for hybrid search (BM25 + vector + RRF) |
| `llm_latency_s` | Time for Gemini API call |
| `total_latency_s` | End-to-end time |
| `input_tokens_est` | Estimated prompt tokens (chars ÷ 4) |
| `output_tokens_est` | Estimated answer tokens (chars ÷ 4) |
| `used_fallback` | Whether Gemini was unavailable |

**Sample output after each answer:**
```
[Retrieval: 0.12s | LLM: 2.34s | Total: 2.46s | ~1823 in / 142 out tokens]
```

**Sample `query_log.jsonl` entry:**
```json
{"timestamp": "2026-02-21T17:30:00", "question": "Do I need to file Form 8843?",
 "confidence": 0.84, "retrieval_latency_s": 0.12, "llm_latency_s": 2.34,
 "total_latency_s": 2.46, "input_tokens_est": 1823, "output_tokens_est": 142, "used_fallback": false}
```

---

### Feature 5 — Human Feedback Loop

After every answer, the user can rate it helpful or not. Ratings are saved to `feedback_log.jsonl` for future analysis — e.g. identifying which questions the bot consistently gets wrong.

**In the chat:**
```
Was this helpful? (y/n, or press Enter to skip): y
  Feedback recorded: 👍 Helpful
```

**Sample `feedback_log.jsonl` entry:**
```json
{"timestamp": "2026-02-21T17:31:00", "question": "Do I need to file Form 8843?",
 "answer_snippet": "Yes, as an F-1 student you must file...", "rating": 1}
```

Pressing Enter skips without recording. Rating `1` = helpful, `0` = not helpful.

---

## Tech Stack (All Free)

| Component | Tool | Why |
|-----------|------|-----|
| PDF extraction | PyMuPDF | Fast, reliable PDF text extraction |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Free, runs locally, good quality |
| Vector database | ChromaDB | Free, embedded (no server), just works |
| LLM | Google Gemini 2.0 Flash (free tier) | Free API with generous limits |

---

## Setup & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get a free Gemini API key
#    Go to: https://aistudio.google.com/apikey
#    Copy your key into .env:
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. Run the data pipeline (only needed once, takes a few minutes)
python run_pipeline.py

# 4. Start the chatbot
python app.py
```

---

## Example Questions You Can Ask

- "Do I need to file taxes if I had no income?"
- "What forms do I need to file as an F-1 student?"
- "Does India have a tax treaty with the U.S. for students?"
- "I worked on campus — how do I report that income?"
- "What is the substantial presence test?"
- "Can I claim the standard deduction as a nonresident?"
- "Do I need to file state taxes in California?"

---

## Evaluation Results

Evaluated on 10 international student tax questions across 5 metrics (0.0–1.0, higher is better).

### v1 — Vector-only retrieval (baseline)

**Setup:** `all-MiniLM-L6-v2` embeddings · `gemini-2.0-flash` · Top-5 vector search · 2,247 chunks from 41 PDFs

| # | Question | Ctx Rel | Hit | Ans Rel | Faith |
|---|----------|---------|-----|---------|-------|
| 1 | Do F-1 students need to file Form 8843? | 0.610 | ❌ | 0.624 | 0.727 |
| 2 | Can nonresident aliens claim the standard deduction? | 0.543 | ✅ | 0.757 | 0.758 |
| 3 | What tax return form do nonresident aliens file? | 0.690 | ✅ | 0.797 | 0.770 |
| 4 | Are F-1 students exempt from FICA taxes? | 0.611 | ✅ | 0.866 | 0.811 |
| 5 | What is the substantial presence test? | 0.498 | ✅ | 0.605 | 0.872 |
| 6 | Does the US-India tax treaty benefit students? | 0.600 | ✅ | 0.487 | 0.682 |
| 7 | What is Form 1098-T used for? | 0.621 | ✅ | 0.784 | 0.868 |
| 8 | Do international students on OPT need to pay taxes? | 0.637 | ❌ | 0.707 | 0.707 |
| 9 | What is Form W-8BEN used for? | 0.395 | ✅ | 0.797 | 0.752 |
| 10 | When is the tax filing deadline for nonresident aliens? | 0.637 | ❌ | 0.504 | 0.433 |
| | **AVERAGE** | **0.584** | **0.70** | **0.693** | **0.738** |

### v2 — Hybrid retrieval (vector + BM25 + RRF)

**Setup:** Same as v1 but retrieval upgraded to hybrid: vector search + BM25 keyword search merged with Reciprocal Rank Fusion

| # | Question | Ctx Rel | Hit | Ans Rel | Faith |
|---|----------|---------|-----|---------|-------|
| 1 | Do F-1 students need to file Form 8843? | 0.560 | ✅ | 0.864 | 0.707 |
| 2 | Can nonresident aliens claim the standard deduction? | 0.518 | ✅ | 0.806 | 0.654 |
| 3 | What tax return form do nonresident aliens file? | 0.666 | ✅ | 0.802 | 0.698 |
| 4 | Are F-1 students exempt from FICA taxes? | 0.611 | ✅ | 0.841 | 0.797 |
| 5 | What is the substantial presence test? | 0.457 | ✅ | 0.605 | 0.884 |
| 6 | Does the US-India tax treaty benefit students? | 0.547 | ✅ | 0.794 | 0.655 |
| 7 | What is Form 1098-T used for? | 0.585 | ✅ | 0.784 | 0.849 |
| 8 | Do international students on OPT need to pay taxes? | 0.602 | ✅ | 0.565 | 0.511 |
| 9 | What is Form W-8BEN used for? | 0.363 | ✅ | 0.774 | 0.722 |
| 10 | When is the tax filing deadline for nonresident aliens? | 0.582 | ✅ | 0.569 | 0.514 |
| | **AVERAGE** | **0.549** | **1.00** | **0.740** | **0.699** |

### v3 — LLM-as-a-Judge added

**Setup:** Same as v2 + Gemini now also rates each answer on correctness, completeness, and groundedness (0–1 each), averaged into a single Judge score.

| # | Question | Ctx Rel | Hit | Ans Rel | Faith | Judge |
|---|----------|---------|-----|---------|-------|-------|
| 1 | Do F-1 students need to file Form 8843? | 0.560 | ✅ | 0.864 | 0.707 | 1.000 |
| 2 | Can nonresident aliens claim the standard deduction? | 0.518 | ✅ | 0.818 | 0.671 | 1.000 |
| 3 | What tax return form do nonresident aliens file? | 0.666 | ✅ | 0.802 | 0.698 | 1.000 |
| 4 | Are F-1 students exempt from FICA taxes? | 0.611 | ✅ | 0.867 | 0.790 | 0.700 |
| 5 | What is the substantial presence test? | 0.457 | ✅ | 0.597 | 0.882 | 1.000 |
| 6 | Does the US-India tax treaty benefit students? | 0.547 | ✅ | 0.668 | 0.550 | 0.000 |
| 7 | What is Form 1098-T used for? | 0.585 | ✅ | 0.792 | 0.874 | 1.000 |
| 8 | Do international students on OPT need to pay taxes? | 0.602 | ✅ | 0.493 | 0.406 | 0.000 |
| 9 | What is Form W-8BEN used for? | 0.363 | ✅ | 0.863 | 0.668 | 1.000 |
| 10 | When is the tax filing deadline for nonresident aliens? | 0.582 | ✅ | 0.528 | 0.595 | 1.000 |
| | **AVERAGE** | **0.549** | **1.00** | **0.729** | **0.684** | **0.770** |

**LLM Judge findings:** 7 of 10 answers scored 1.0, identifying two weak answers — Q6 (India treaty answer too vague to be actionable) and Q8 (OPT answer referenced tax software instead of explaining the tax obligation directly). Cosine similarity alone would not have caught these gaps.

### v1 → v2 → v3 Comparison

| Metric | v1 (vector) | v2 (hybrid) | v3 (+ LLM Judge) | Change v1→v3 |
|--------|------------|-------------|-----------------|-------------|
| Context Relevance | 0.584 | 0.549 | 0.549 | -0.035 |
| **Hit Rate** | **0.70** | **1.00** | **1.00** | **+0.30 ✅** |
| **Answer Relevance** | **0.693** | **0.740** | **0.729** | **+0.036 ✅** |
| Faithfulness | 0.738 | 0.699 | 0.684 | -0.054 |
| **LLM Judge** | — | — | **0.770** | **new ✅** |

**Key improvement (v1→v2):** Hit rate jumped from 70% → **100%** — BM25 catches exact form names and tax terms (like "8843", "FICA", "OPT") that vector search can miss.

**Key improvement (v2→v3):** LLM-as-a-Judge adds a human-like quality signal. It identified two answers that cosine metrics rated as acceptable but were actually weak or off-topic — a gap that embedding similarity can't detect.

### Metric Definitions

| Metric | What it measures |
|--------|-----------------|
| **Context Relevance** | Cosine similarity between the question and retrieved chunks — are we fetching the right docs? |
| **Hit Rate** | Did retrieved chunks collectively contain all expected answer keywords? |
| **Answer Relevance** | Cosine similarity between the question and the generated answer — does it address what was asked? |
| **Faithfulness** | Cosine similarity between the generated answer and retrieved context — is the answer grounded? |
| **LLM Judge** | Gemini scores correctness + completeness + groundedness (avg of 3 sub-scores, 0–1 each) |

---

## Unused/Legacy Scripts

These files in `tax_rag_data/` are from an earlier version and are **not used** by the current pipeline:

| File | What it was | Why unused |
|------|-------------|-----------|
| `parse_pdfs.py` | Older PDF extractor | Replaced by `extract_pdfs_to_json.py` |
| `hybrid_retrieval.py` | Elasticsearch-based retrieval | Replaced by ChromaDB in `app.py` |
| `intake_cli.py` | Standalone profile intake | Merged into `app.py` |
| `rag_generation.py` | Prompt builder for LLM | Merged into `app.py` |
| `verify_manifest_vs_files.py` | Manifest vs files checker | Was useful during setup, not needed to run |

You can safely delete these if you want to keep things clean.
