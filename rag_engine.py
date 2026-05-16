"""
Conversational RAG engine with pluggable LLM backend.

Switch backends via config.LLM_PROVIDER:
- "ollama": local llama / qwen / etc via Ollama
- "gemini": Google's Gemini API
"""
from __future__ import annotations
import json
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

Instructions:
- Quote specific numbers, percentages, and dates directly from the excerpts
  above. Do NOT round or paraphrase numerical values.
- End every factual statement with its source tag, e.g. "Revenue was
  $4,941M [Q1 FY26 Press Release | Page 1]".
- If multiple excerpts from different sources contain the answer, prefer
  the most specific (e.g. a press release for a quarterly number).
- Only say "the answer is not in the documents" if NO excerpt above
  mentions the topic at all.

Write the analyst-grade answer below."""


FORMAT_ROUTER_PROMPT = """Decide the best output format for this financial analyst answer.

Question: {question}

Answer preview (first 800 chars):
{preview}

Choose ONE format:
- "markdown": short factual answers, single-quarter lookups, definitions
- "pdf": multi-section reports, executive summaries, narrative analyses
- "excel": data-heavy answers, multi-row tables, time-series comparisons

Respond with ONLY a JSON object like: {{"format": "markdown"}}"""


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
        results = self.store.search(standalone, top_k=TOP_K)
        context_block, sources = self._format_context(results)
        answer = self._generate_answer(standalone, context_block)
        # Route format off the *original* question — the user's "table of",
        # "summary", "report" hints are stripped by the standalone rewriter.
        fmt = self._route_format(question, answer)

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

        # 2. Retrieval
        yield ("step", {"name": "retrieve", "status": "start", "top_k": TOP_K})
        results = self.store.search(standalone, top_k=TOP_K)
        context_block, sources = self._format_context(results)
        yield ("step", {"name": "retrieve", "status": "done",
                        "count": len(results), "sources": sources})

        # 3. Answer generation (streamed)
        yield ("step", {"name": "generate", "status": "start",
                        "provider": self.llm.provider,
                        "model": OLLAMA_CHAT_MODEL if self.llm.provider == "ollama" else CHAT_MODEL})
        prompt = ANSWER_PROMPT.format(
            system=SYSTEM_PROMPT,
            context=context_block,
            history=self._history_text(),
            question=standalone,
        )
        parts: List[str] = []
        for token in self.llm.generate_stream(prompt, temperature=0.2):
            parts.append(token)
            yield ("token", {"text": token})
        answer = "".join(parts).strip()
        yield ("step", {"name": "generate", "status": "done",
                        "length": len(answer)})

        # 4. Format routing (heuristic-only here for speed; UI shows it as a step)
        # Use the original question — the standalone rewriter strips format hints.
        yield ("step", {"name": "format", "status": "start"})
        fmt = self._route_format_heuristic(question, answer)
        yield ("step", {"name": "format", "status": "done", "format": fmt})

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

    def _generate_answer(self, standalone: str, context: str) -> str:
        prompt = ANSWER_PROMPT.format(
            system=SYSTEM_PROMPT,
            context=context,
            history=self._history_text(),
            question=standalone,
        )
        return self.llm.generate(prompt, temperature=0.2)

    def _route_format_heuristic(self, question: str, answer: str) -> str:
        """Decide output format from keywords + answer shape. No extra LLM call."""
        q_lower = question.lower()
        if any(k in q_lower for k in ["report", "summary", "summarise", "summarize", "overview", "pdf"]):
            return "pdf"
        if any(k in q_lower for k in ["excel", "spreadsheet", "table of", "export", "download"]):
            return "excel"
        if answer.count("|") > 12 and "---" in answer:
            return "excel"
        if len(answer) > 1500:
            return "pdf"
        return "markdown"

    def _route_format(self, question: str, answer: str) -> str:
        """Heuristic format routing with an LLM tie-breaker for ambiguous cases."""
        fmt = self._route_format_heuristic(question, answer)
        if fmt != "markdown":
            return fmt

        prompt = FORMAT_ROUTER_PROMPT.format(question=question, preview=answer[:800])
        try:
            text = self.llm.generate(prompt, temperature=0.0)
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                f = data.get("format", "markdown").lower()
                if f in {"markdown", "pdf", "excel"}:
                    return f
        except Exception:
            pass
        return "markdown"