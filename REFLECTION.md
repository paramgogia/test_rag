# Reflection

## 1. What makes it feel intelligent rather than keyword search?

Three things, in order of how much they mattered.

**Follow-ups actually work.** Without anything special, asking "and Q2?"
after a Q1 question just embeds the literal string "and Q2?" and you
get garbage back. So before retrieval I run the question through a
rewriter pass that uses the chat history to produce a self-contained
version"and Q2?" becomes "What was Infosys's Q2 FY26 revenue?".
Costs an extra LLM call per turn but it's the difference between a
chatbot and a search box.

**Retrieval doesn't get drowned by the annual report.** The FY25
report is 500 pages, so on pure cosine it wins against every short
press release. I pull a 4× wider candidate pool and add a small score
boost to chunks whose source name overlaps with the question tokens
("Q1 FY26" in the query lifts "Q1 FY26 Press Release" chunks).
Half a day of fiddling, very high payoff on quarter-specific questions.

Smaller stuff that helps: citations are mandatory on every factual
claim, PDF tables are extracted as their own structured chunks so
column relationships survive, and the CSV gets a hand-rolled summary
chunk so "how did the stock do" doesn't have to retrieve 250 daily
rows.

Also since it is compatible to local Ollama and uses llama 3.1 8B param model 
that any 16gb laptop can run perfectly given model is pulled and also
no tension for api key rate limiting or request quota reached.

## 2. Where it still falls short

- **It hallucinates when retrieval misses.** the local model hallucinates 
 for some cases if the system prompt tells it
 not to do this, the 8B model does it anyway.

- **Sometimes the Citations aren't verified.** The model claims "[… | Page 1]" and
  the bot trusts it. On Llama 3.1 8B specifically, few citations
  land on Page 1 even when the data is from a later page. Grepping the
  actual chunk for the cited number post-hoc would catch this.

- **No diversity in retrieval.** "Margin across all four quarters"
  ideally returns one chunk per quarter; in practice you sometimes get
  three Q1 chunks because Q1 phrasing happened to score highest.
  Standard MMR or a cross-encoder would help.

- **`pdfplumber` struggles with the annual report's multi-row
  headers.** Balance-sheet line items come through fragmented.
  https://www.npmjs.com/package/doc-to-md-rag
  This was built by me months ago for JS as there was no existing package 
  supporting proper md multi-row formatting for RAG.


## 3. AI tools used, and what I had to fix

Used Gemini (70%) and Claude (30%) for suggestion and fixes,
application llm generation runs on
**Ollama (llama3.1:8b)** locally with a **Gemini 2.5 Flash** path
also wired up — embeddings are `nomic-embed-text` locally. I started
on Gemini, hit the free-tier rate limit during ingestion, added the
Ollama path, kept it as default.

Things I had to fix or override:

- **Chunking** — first draft chunked entire PDFs as one blob, so page
  citations were meaningless. Rewrote `load_pdf` to chunk within each
  page.

- **Gemini rate limits** — initial code sent one chunk per embed call,
  immediate 429s. Batched to 10, added a backoff that parses the
  "retry in X.Xs" hint out of the error message, and a 7-second sleep
  between batches.

- **Pipe-in-citation bug** — only caught this when I actually opened
  the generated Excels. Citations look like
  `[Q1 FY26 Press Release | Page 1]` with a literal `|` inside, and my
  table parser was treating that pipe as a column separator. Four-column
  tables came out as eight columns with citations sprawled across
  cells. Fixed with a bracket-depth-aware splitter
  (`_split_table_row`) and re-exported the samples.

- **LLM refusing to "generate" files** — Llama 3.1 8B would mid-answer
  say things like "Unfortunately I cannot create an Excel
  spreadsheet", which then got dumped verbatim into the spreadsheet.
  Added `_scrub_refusals` to regex out those sentences plus connecting
  filler ("Unfortunately,", "As an AI language model,") without
  touching the surrounding facts.

- **Excel fallback for non-table answers.** If the LLM returned
  bullets instead of a markdown table, the Excel was a single "Answer"
  column of prose. Added `_structured_rows_from_prose` to mine
  `- Label: value [citation]` lines into proper Metric / Value /
  Citation rows.

- **Rewriter prompt** — first draft returned "Here is the rewritten
  question: …" instead of just the question. Tightened the prompt and
  added prefix-stripping as a safety net.
