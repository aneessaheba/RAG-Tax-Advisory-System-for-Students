# RAG Tax Advisor for International Students

A chatbot that helps international students in the U.S. understand their tax obligations. It uses **RAG** (Retrieval-Augmented Generation) — meaning it searches through real IRS publications, tax treaties, and university guides to give grounded answers instead of making things up.

**All free** — no paid APIs or servers needed.

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
