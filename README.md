# Infosys Financial Analyst — Chatbot

A conversational analyst over seven Infosys financial documents. Ask a
question, get an answer with citations; ask for a "side-by-side table"
or a "one-page report" and you get a downloadable `.xlsx` or `.pdf`
instead of just text.

Built on a conversational RAG with metadata aware re-ranking and format aware response generation.

The seven sources:

| # | Source | File |
|---|---|---|
| 1 | Annual Report FY25 | `infosys-ar-25.pdf` |
| 2 | Q1 FY26 Press Release | `ifrs-usd-press-release_q1.pdf` |
| 3 | Q2 FY26 Press Release | `ifrs-usd-press-release_q2.pdf` |
| 4 | Q3 FY26 Press Release | `ifrs-usd-press-release_q3.pdf` |
| 5 | Q4 FY26 Press Release | `ifrs-usd-press-release_q4.pdf` |
| 6 | Investor Sheet (Multi-year P&L, BS, employee data) | `investor-sheet.xls` |
| 7 | BSE 500209 Stock Price FY26 | `500209.csv` |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  data/   (7 source files: PDFs, .xls, .csv)                        │
└────────────────────────────────────────────────────────────────────┘
                │
                │  ingest.py
                ▼
┌────────────────────────────────────────────────────────────────────┐
│  document_loader.py                                                │
│   - load_pdf:   page-by-page, tables extracted separately          │
│   - load_excel: each sheet → "Column: value" text, 50-row batches  │
│   - load_csv:   monthly chunks + an overall summary chunk          │
│   Every chunk carries (source, file_name, location, text)          │
└────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────────────┐
│  vector_store.py  →  FAISS IndexFlatIP (cosine via L2-normalised)  │
│   Embeddings:                                                      │
│     - Ollama:  nomic-embed-text  (local, default)                  │
│     - Gemini:  models/gemini-embedding-001                         │
│   Retrieval: pull 4× top_k, then re-rank with a small boost for    │
│   chunks whose source name overlaps query tokens (so "Q1 FY26"     │
│   actually surfaces Q1 chunks instead of being drowned by the      │
│   500-page Annual Report).                                         │
└────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────────────┐
│  rag_engine.py                                                     │
│   ask()  /  ask_stream()                                           │
│     1. _rewrite_question  — collapse the chat history into a       │
│        standalone question so retrieval has full context           │
│     2. _preroute_format   — decide pdf / excel / markdown from     │
│        the *original* question, BEFORE generation                  │
│     3. store.search       — top-k chunks                           │
│     4. LLM call           — format-aware prompt:                   │
│           excel → MUST produce a markdown table                    │
│           pdf   → sectioned narrative with headings                │
│           md    → terse                                            │
│     5. _scrub_refusals    — strip "I can't generate a PDF" style   │
│        sentences if the model slips                                │
│     6. _maybe_upgrade_format — promote md → pdf/excel if the       │
│        answer ended up long or table-heavy                         │
│   LLM backend is pluggable: Ollama (llama3.1:8b) or Gemini         │
│   (gemini-2.5-flash).                                              │
└────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────────────┐
│  exporters.py                                                      │
│   export_pdf   — ReportLab; tables in answer render as styled      │
│                  tables, sources go on a separate page             │
│   export_excel — pandas + openpyxl; one sheet per markdown table;  │
│                  fallback: extract Metric/Value/Citation rows      │
│                  from bullet/key-value prose so the Data sheet is  │
│                  never just a wall of text                         │
└────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────────────┐
│  app.py  (Streamlit chat UI)   OR   cli.py                         │
│   - streams tokens, shows each pipeline step live                  │
│   - persistent chat history, source expander per message           │
│   - download button for the PDF / Excel when one is produced       │
└────────────────────────────────────────────────────────────────────┘
```


---

## Setup (local, no credit card needed)

You can run the bot in two modes. Pick one:

### Option A — Local LLM via Ollama (default, fully offline, slow)

1. **Install Ollama** from <https://ollama.com>, start it.
2. **Pull the models**:
   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```
3. **Python deps**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Confirm `config.py` is on Ollama** (default):
   ```python
   LLM_PROVIDER = "ollama"
   ```

### Option B — Free Gemini API (faster, cloud)

1. Get a free key at <https://aistudio.google.com/apikey>.
2. Create `.env`:
   ```
   GEMINI_API_KEY=your_key_here
   ```
3. In `config.py`, switch:
   ```python
   LLM_PROVIDER = "gemini"
   ```
4. Same `pip install -r requirements.txt`.

Gemini's free tier has tight rate limits — embedding 500+ chunks takes
several minutes because the loader pauses 7s between batches to stay
under the per-minute quota. Ollama is slower per query but has no
quota.

---

## Run

```bash
# one-time: build the FAISS index from /data
python ingest.py

# option 1 — Streamlit chat UI
streamlit run app.py

# option 2 — CLI
python cli.py

# regenerate the 6 sample conversations + PDFs/Excels
python generate_samples.py
```

`ingest.py` reads every file in `data/`, chunks each (2000 chars,
200-char overlap), embeds the chunks, and writes
`vectorstore/index.faiss` + `vectorstore/metadata.pkl`. Re-run only if
the source documents change.

---

## Files

```
config.py            tunables: provider, chunk size, top_k, paths
document_loader.py   PDF / Excel / CSV → Chunk objects
vector_store.py      embeddings + FAISS index + retrieval
rag_engine.py        question rewriting, format routing, LLM call, refusal scrub
exporters.py         PDF (ReportLab) and Excel (pandas/openpyxl) builders
ingest.py            one-shot indexer
app.py               Streamlit UI
cli.py               terminal UI
generate_samples.py  produces the 6 canned sample conversations
data/                source documents 
vectorstore/         FAISS index + chunk metadata
outputs/             every generated PDF/Excel from interactive sessions
sample_conversations/   6 transcripts + 6 exported files + index.json
```

---

## Sample conversations

`sample_conversations/` contains 3 PDF and 3 Excel examples produced by
`generate_samples.py` script to run cases and store the response in md file along with pdf and excel generated

---

## Configuration knobs (`config.py`)

| Knob | Default | What it controls |
|---|---|---|
| `LLM_PROVIDER` | `"ollama"` | `"ollama"` or `"gemini"` |
| `OLLAMA_CHAT_MODEL` | `"llama3.1:8b"` | local generation model |
| `CHAT_MODEL` | `"gemini-2.5-flash"` | cloud generation model |
| `CHUNK_SIZE` | `2000` | chars per chunk |
| `CHUNK_OVERLAP` | `200` | chars of overlap between chunks |
| `TOP_K` | `10` | chunks fed to the LLM per query |
| `MAX_HISTORY_TURNS` | `6` | how many past turns the rewriter sees |
