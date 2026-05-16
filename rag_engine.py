"""
Conversational RAG engine with pluggable LLM backend.

Switch backends via config.LLM_PROVIDER:
- "ollama": local llama / qwen / etc via Ollama
- "gemini": Google's Gemini API
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Iterator

from config import (
    LLM_PROVIDER,
    GEMINI_API_KEY, CHAT_MODEL,
    OLLAMA_HOST, OLLAMA_CHAT_MODEL,
    TOP_K, MAX_HISTORY_TURNS
)
from vector_store import VectorStore


# ---------- Prompts (same as before) ----------

QUESTION_REWRITER_PROMPT = """You rewrite follow-up questions into standalone questions.

Given the chat history and the latest user question, produce a SINGLE
self-contained question that captures all necessary context from the
conversation. If the latest question is already standalone, return it
unchanged.

Do not answer the question. Output only the rewritten question, nothing else.

Chat history:
{history}

Latest question: {question}

Standalone question:"""


SYSTEM_PROMPT = """You are an experienced financial analyst chatbot specialising in Infosys Ltd.

You have access to:
- Infosys Integrated Annual Report FY25
- Quarterly press releases for Q1-Q4 FY26
- A multi-year investor sheet (P&L, balance sheet, employee data)
- BSE 500209 stock price history for FY26

Rules you MUST follow:
1. Answer ONLY using the context provided below. If the answer cannot be
   found in the context, say so honestly — do not invent numbers.
2. Every factual claim must end with a citation tag in square brackets
   matching one of the sources listed in the context, e.g. [Q1 FY26 Press
   Release | Page 7]. Multiple sources can be cited together.
3. When the user asks for trends, comparisons, or multi-quarter analysis,
   actively pull numbers from multiple chunks and reason about them.
4. Prefer clean structured output: tables for comparisons, bullet points
   for lists, short paragraphs for explanations. Use markdown.
5. If numeric figures appear in different units (USD vs INR, millions vs
   crores), state the unit explicitly.
6. Never speculate beyond the documents. Do not give investment advice.

IMPORTANT — file generation:
PDF and Excel files are produced AUTOMATICALLY by the surrounding Python
application after you respond. You only need to supply the content in the
right shape; you must NEVER say things like "I cannot generate a PDF",
"I am unable to create an Excel file", "I don't have the ability to
produce downloadable files", or any similar refusal. The download is
handled for you. Just write the answer — a downloadable file will be
attached to the chat automatically.

You are talking to someone who understands finance. Be precise, concise,
and analyst-grade."""


ANSWER_PROMPT = """{system}

The text between === CONTEXT === markers below contains verbatim excerpts
from the Infosys financial documents. Each excerpt starts with a citation
tag in square brackets like [Q1 FY26 Press Release | Page 1]. THESE TAGS
IDENTIFY REAL SOURCES that you DO have access to — never say the document
"is not in the context" if a chunk from that document appears below.

=== CONTEXT (retrieved from the documents) ===
{context}
=== END CONTEXT ===

Chat history (for tone and continuity, not as a source of facts):
{history}

User question: {question}

Output format requested by the system: {format_hint}
{format_directive}

Instructions:
- Quote specific numbers, percentages, and dates directly from the excerpts
  above. Do NOT round or paraphrase numerical values.
- End every factual statement with its source tag, e.g. "Revenue was
  $4,941M [Q1 FY26 Press Release | Page 1]".
- If multiple excerpts from different sources contain the answer, prefer
  the most specific (e.g. a press release for a quarterly number).
- Only say "the answer is not in the documents" if NO excerpt above
  mentions the topic at all.
- Do NOT refuse to "generate a PDF" or "generate an Excel" — file
  rendering is done by the application after your response. Just write
  the content.

Write the analyst-grade answer below."""


FORMAT_DIRECTIVES = {
    "excel": (
        "Because the user wants a data export, your answer MUST include at "
        "least one markdown table with a header row and a separator line "
        "(e.g. `| Metric | Q1 | Q2 | Q3 | Q4 |` then `|---|---|---|---|---|`). "
        "Put EVERY numeric data point into the table — one metric per row, "
        "one period/category per column. Keep prose to a 1-2 sentence intro "
        "before the table and a brief note after. Cite sources after the "
        "table or in a footer line."
    ),
    "pdf": (
        "Because the user wants a report, structure the answer with clear "
        "markdown headings (## Section), a short executive summary at the "
        "top, then sections covering the key dimensions, and a closing "
        "'Key takeaways' bullet list. Use tables where they help. Aim for "
        "a thorough multi-section narrative."
    ),
    "markdown": (
        "Keep the answer concise — a short paragraph or a small bullet "
        "list. Use a table only if the data genuinely benefits from one."
    ),
}


FORMAT_ROUTER_PROMPT = """Decide the best output format for this financial analyst question.

Question: {question}

Choose ONE format:
- "markdown": short factual answers, single-quarter lookups, definitions
- "pdf": multi-section reports, executive summaries, narrative analyses
- "excel": data-heavy answers, multi-row tables, time-series comparisons

Respond with ONLY a JSON object like: {{"format": "markdown"}}"""


# Strong keyword triggers — if any of these appear in the user's question,
# we hard-route to that format regardless of the LLM's opinion.
_EXCEL_KEYWORDS = (
    "excel", "spreadsheet", ".xlsx", "xlsx", "csv", "table of",
    "export", "download as a table", "downloadable table", "data dump",
    "side-by-side", "side by side",
)
_PDF_KEYWORDS = (
    "pdf", ".pdf", "report", "summary", "summarise", "summarize",
    "executive summary", "one-pager", "one-page", "writeup", "write-up",
    "memo", "brief",
)


# Patterns that indicate the LLM is refusing to "produce" a file. If we
# see any of these in the answer, we strip them — the file IS produced,
# by external Python code, so the refusal is incorrect and confusing.
_REFUSAL_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(?:unfortunately,?\s+)?i (?:can(?:not|'t)|am unable to|won'?t be able to) (?:generate|create|produce|build|render|output|export|provide|attach|deliver|make) (?:a |an |the )?(?:pdf|excel|spreadsheet|xlsx|file|download|attachment|report)[^.\n]*\.?",
        r"(?:unfortunately,?\s+)?i do(?:n'?t| not) have the (?:ability|capability|tools?|means) to (?:generate|create|produce|build|render|output|export|provide|attach|deliver|make)[^.\n]*\.?",
        r"as an? (?:ai|language model)[^.\n]*?(?:cannot|can'?t|unable)[^.\n]*?(?:pdf|excel|file|download)[^.\n]*\.?",
        r"(?:unfortunately,?\s+)?(?:please note(?: that)?,?\s+)?(?:i|my response) (?:can(?:not|'t)|am unable|are unable)[^.\n]*?(?:pdf|excel|file|download|attach)[^.\n]*\.?",
    ]
]


def _scrub_refusals(answer: str) -> str:
    """Remove any sentences where the LLM claims it can't make a file."""
    cleaned = answer
    for pat in _REFUSAL_PATTERNS:
        cleaned = pat.sub("", cleaned)
    # Clean up orphan prefixes / connective tissue that referred to the
    # refusal sentence we just removed.
    orphan_patterns = [
        r"\bAs an? (?:AI language model|language model|AI)\s*[,.]?\s*",
        r"\bUnfortunately\s*,?\s*(?=[A-Z])",
        r"\bPlease note(?: that)?\s*[:,]?\s*(?=[A-Z])",
        r"\bNote\s*:\s*(?=[A-Z])",
    ]
    for op in orphan_patterns:
        cleaned = re.sub(op, "", cleaned)
    # Collapse the empty lines / double spaces we leave behind.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ---------- Unified LLM caller ----------

class LLMClient:
    """Thin wrapper that calls either Gemini or Ollama transparently."""

    def __init__(self):
        self.provider = LLM_PROVIDER
        if self.provider == "gemini":
            import google.generativeai as genai
            if not GEMINI_API_KEY:
                raise RuntimeError("GEMINI_API_KEY not set in .env")
            genai.configure(api_key=GEMINI_API_KEY)
            self._gemini = genai.GenerativeModel(CHAT_MODEL)
        elif self.provider == "ollama":
            import ollama
            self._ollama = ollama.Client(host=OLLAMA_HOST)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        if self.provider == "gemini":
            resp = self._gemini.generate_content(
                prompt,
                generation_config={"temperature": temperature},
            )
            return (resp.text or "").strip()
        else:  # ollama
            resp = self._ollama.generate(
                model=OLLAMA_CHAT_MODEL,
                prompt=prompt,
                options={"temperature": temperature},
                keep_alive="30m",
            )
            return (resp.get("response") or "").strip()

    def generate_stream(self, prompt: str, temperature: float = 0.2) -> Iterator[str]:
        """Yield response tokens as they arrive from the LLM."""
        if self.provider == "gemini":
            resp = self._gemini.generate_content(
                prompt,
                generation_config={"temperature": temperature},
                stream=True,
            )
            for chunk in resp:
                text = getattr(chunk, "text", "") or ""
                if text:
                    yield text
        else:  # ollama
            for part in self._ollama.generate(
                model=OLLAMA_CHAT_MODEL,
                prompt=prompt,
                options={"temperature": temperature},
                keep_alive="30m",
                stream=True,
            ):
                piece = part.get("response") or ""
                if piece:
                    yield piece


# ---------- Engine ----------

@dataclass
class ChatTurn:
    role: str
    content: str


@dataclass
class RagAnswer:
    answer: str
    sources: List[Dict[str, Any]]
    format: str
    rewritten_question: str


class RagEngine:
    def __init__(self, store: VectorStore):
        self.llm = LLMClient()
        self.store = store
        self.history: List[ChatTurn] = []

    def reset(self):
        self.history.clear()

    def ask(self, question: str) -> RagAnswer:
        standalone = self._rewrite_question(question)
        # Decide format FIRST, off the original question, so we can steer
        # the LLM toward table-shaped or report-shaped content.
        fmt = self._preroute_format(question)
        results = self.store.search(standalone, top_k=TOP_K)
        context_block, sources = self._format_context(results)
        answer = self._generate_answer(standalone, context_block, fmt)
        answer = _scrub_refusals(answer)

        # Light post-check: if format is "markdown" but the answer is huge
        # or table-heavy, upgrade it so users still get a file.
        fmt = self._maybe_upgrade_format(fmt, answer)

        self.history.append(ChatTurn("user", question))
        self.history.append(ChatTurn("assistant", answer))
        self.history = self.history[-2 * MAX_HISTORY_TURNS:]

        return RagAnswer(
            answer=answer,
            sources=sources,
            format=fmt,
            rewritten_question=standalone,
        )

    def ask_stream(self, question: str) -> Iterator[tuple]:
        """
        Streamed variant of ask(). Yields events so the UI can show each
        pipeline step as it happens, plus answer tokens as they arrive.

        Event shapes:
          ("step", {"name": <str>, "status": "start"|"done", ...payload})
          ("token", {"text": <str>})
          ("final", RagAnswer)
        """
        # 1. Question rewrite (only meaningful when there's prior history)
        has_history = bool(self.history)
        yield ("step", {"name": "rewrite", "status": "start",
                        "has_history": has_history})
        standalone = self._rewrite_question(question)
        yield ("step", {"name": "rewrite", "status": "done",
                        "standalone": standalone,
                        "rewritten": has_history and standalone != question})

        # 2. Format pre-routing — done BEFORE generation so we can steer the LLM.
        yield ("step", {"name": "format", "status": "start"})
        fmt = self._preroute_format(question)
        yield ("step", {"name": "format", "status": "done", "format": fmt,
                        "stage": "preroute"})

        # 3. Retrieval
        yield ("step", {"name": "retrieve", "status": "start", "top_k": TOP_K})
        results = self.store.search(standalone, top_k=TOP_K)
        context_block, sources = self._format_context(results)
        yield ("step", {"name": "retrieve", "status": "done",
                        "count": len(results), "sources": sources})

        # 4. Answer generation (streamed) — pass format hint into the prompt
        yield ("step", {"name": "generate", "status": "start",
                        "provider": self.llm.provider,
                        "model": OLLAMA_CHAT_MODEL if self.llm.provider == "ollama" else CHAT_MODEL})
        prompt = ANSWER_PROMPT.format(
            system=SYSTEM_PROMPT,
            context=context_block,
            history=self._history_text(),
            question=standalone,
            format_hint=fmt,
            format_directive=FORMAT_DIRECTIVES.get(fmt, FORMAT_DIRECTIVES["markdown"]),
        )
        parts: List[str] = []
        for token in self.llm.generate_stream(prompt, temperature=0.2):
            parts.append(token)
            yield ("token", {"text": token})
        answer = _scrub_refusals("".join(parts).strip())
        yield ("step", {"name": "generate", "status": "done",
                        "length": len(answer)})

        # 5. Possibly upgrade format if answer shape demands it
        new_fmt = self._maybe_upgrade_format(fmt, answer)
        if new_fmt != fmt:
            fmt = new_fmt
            yield ("step", {"name": "format", "status": "done", "format": fmt,
                            "stage": "upgrade"})

        # 5. Persist history
        self.history.append(ChatTurn("user", question))
        self.history.append(ChatTurn("assistant", answer))
        self.history = self.history[-2 * MAX_HISTORY_TURNS:]

        yield ("final", RagAnswer(
            answer=answer,
            sources=sources,
            format=fmt,
            rewritten_question=standalone,
        ))

    # ---------- internals ----------

    def _history_text(self) -> str:
        if not self.history:
            return "(no prior turns)"
        return "\n".join(f"{t.role.capitalize()}: {t.content}" for t in self.history)

    def _rewrite_question(self, question: str) -> str:
        if not self.history:
            return question
        prompt = QUESTION_REWRITER_PROMPT.format(
            history=self._history_text(),
            question=question,
        )
        try:
            rewritten = self.llm.generate(prompt, temperature=0.0).strip()
            if not rewritten or len(rewritten) > 500:
                return question
            # Strip common LLM preambles
            for prefix in ("Standalone question:", "Rewritten question:", "Question:"):
                if rewritten.lower().startswith(prefix.lower()):
                    rewritten = rewritten[len(prefix):].strip()
            return rewritten
        except Exception:
            return question

    def _format_context(self, results) -> tuple[str, List[Dict[str, Any]]]:
        blocks, sources = [], []
        seen = set()
        for chunk, score in results:
            tag = chunk.cite_tag()
            blocks.append(f"{tag}\n{chunk.text}")
            key = (chunk.source, chunk.location)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": chunk.source,
                    "location": chunk.location,
                    "snippet": chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
                    "score": round(score, 3),
                })
        return "\n\n---\n\n".join(blocks), sources

    def _generate_answer(self, standalone: str, context: str, fmt: str) -> str:
        prompt = ANSWER_PROMPT.format(
            system=SYSTEM_PROMPT,
            context=context,
            history=self._history_text(),
            question=standalone,
            format_hint=fmt,
            format_directive=FORMAT_DIRECTIVES.get(fmt, FORMAT_DIRECTIVES["markdown"]),
        )
        return self.llm.generate(prompt, temperature=0.2)

    def _preroute_format(self, question: str) -> str:
        """
        Decide the output format from the QUESTION alone, before we call
        the LLM, so we can steer the answer's shape. Excel beats PDF when
        both keyword groups match (an explicit "excel" wins over a generic
        "summary"). Falls back to "markdown".
        """
        q_lower = question.lower()
        excel_hit = any(k in q_lower for k in _EXCEL_KEYWORDS)
        pdf_hit = any(k in q_lower for k in _PDF_KEYWORDS)
        if excel_hit:
            return "excel"
        if pdf_hit:
            return "pdf"

        # Question shapes that strongly imply a table even without keywords
        table_signals = (
            "compare", "comparison", "across all", "across each", "by quarter",
            "quarter-over-quarter", "qoq", "yoy", "year-over-year",
            "month by month", "per quarter", "per month", "every quarter",
            "build a table", "give me a table", "list out",
        )
        if any(s in q_lower for s in table_signals):
            return "excel"

        return "markdown"

    def _maybe_upgrade_format(self, current: str, answer: str) -> str:
        """
        Safety net for when the question didn't look special but the answer
        came out long/tabular — promote markdown to pdf/excel so the user
        still gets a downloadable file.
        """
        if current != "markdown":
            return current
        # Lots of pipes + a separator row = a real markdown table -> Excel
        if answer.count("|") > 12 and re.search(r"\|\s*-{2,}", answer):
            return "excel"
        if len(answer) > 1500:
            return "pdf"
        return current