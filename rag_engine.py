"""
Conversational RAG engine.

Pipeline per user message:
  1. Rewrite the question using chat history so follow-ups become standalone
     queries ("how did it compare to Q2?" -> "How did Infosys Q3 FY26 revenue
     compare to Q2 FY26?"). This is what makes follow-ups actually work.
  2. Retrieve top-K chunks from the vector store.
  3. Build a strict prompt that forces citation and admits unknowns.
  4. Route the output: a separate small LLM call decides if the answer is
     better delivered as markdown, PDF report, or Excel sheet.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any

import google.generativeai as genai

from config import (
    GEMINI_API_KEY, CHAT_MODEL, TOP_K, MAX_HISTORY_TURNS
)
from vector_store import VectorStore


# ---------- Prompts ----------

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
   actively pull numbers from multiple chunks and reason about them. Do
   not just list paragraphs.
4. Prefer clean structured output: tables for comparisons, bullet points
   for lists, short paragraphs for explanations. Use markdown.
5. If numeric figures appear in different units (USD vs INR, millions vs
   crores), state the unit explicitly.
6. Never speculate beyond the documents. Do not give investment advice.

You are talking to someone who understands finance. Be precise, concise,
and analyst-grade."""


ANSWER_PROMPT = """{system}

=== CONTEXT (retrieved from the documents) ===
{context}
=== END CONTEXT ===

Chat history (for tone and continuity, not as a source of facts):
{history}

User question: {question}

Write the analyst-grade answer below. Remember: cite every fact with the
bracketed source tag. If you genuinely cannot answer from the context,
say so plainly."""


FORMAT_ROUTER_PROMPT = """Decide the best output format for this financial analyst answer.

Question: {question}

Answer preview (first 800 chars):
{preview}

Choose ONE format:
- "markdown": short factual answers, single-quarter lookups, definitions,
  simple comparisons that fit comfortably in a chat bubble.
- "pdf": multi-section reports, executive summaries, analyses spanning
  several quarters or topics, narrative-heavy answers, anything the user
  would want to save and share.
- "excel": data-heavy answers — multi-row tables, time-series data,
  side-by-side numeric comparisons, anything that screams "spreadsheet".

Respond with ONLY a JSON object: {{"format": "markdown"}} or
{{"format": "pdf"}} or {{"format": "excel"}}. No other text."""


# ---------- Engine ----------

@dataclass
class ChatTurn:
    role: str    # "user" or "assistant"
    content: str


@dataclass
class RagAnswer:
    answer: str                     # the markdown answer
    sources: List[Dict[str, Any]]   # list of {source, location, snippet}
    format: str                     # "markdown" | "pdf" | "excel"
    rewritten_question: str         # for debugging / sample logs


class RagEngine:
    def __init__(self, store: VectorStore):
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set in .env")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(CHAT_MODEL)
        self.store = store
        self.history: List[ChatTurn] = []

    # ---------- public ----------

    def reset(self):
        self.history.clear()

    def ask(self, question: str) -> RagAnswer:
        # 1. rewrite for follow-ups
        standalone = self._rewrite_question(question)

        # 2. retrieve
        results = self.store.search(standalone, top_k=TOP_K)
        context_block, sources = self._format_context(results)

        # 3. generate answer
        answer = self._generate_answer(question, standalone, context_block)

        # 4. route output format
        fmt = self._route_format(standalone, answer)

        # 5. update history
        self.history.append(ChatTurn("user", question))
        self.history.append(ChatTurn("assistant", answer))
        self.history = self.history[-2 * MAX_HISTORY_TURNS:]

        return RagAnswer(
            answer=answer,
            sources=sources,
            format=fmt,
            rewritten_question=standalone,
        )

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
            resp = self.model.generate_content(prompt)
            rewritten = (resp.text or "").strip()
            # safety: if model returned empty or absurd, fall back
            if not rewritten or len(rewritten) > 500:
                return question
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

    def _generate_answer(self, question: str, standalone: str, context: str) -> str:
        prompt = ANSWER_PROMPT.format(
            system=SYSTEM_PROMPT,
            context=context,
            history=self._history_text(),
            question=standalone,
        )
        resp = self.model.generate_content(prompt)
        return (resp.text or "").strip()

    def _route_format(self, question: str, answer: str) -> str:
        # Fast heuristics first — saves an API call most of the time.
        q_lower = question.lower()
        if any(k in q_lower for k in ["report", "summary", "summarise", "summarize", "overview", "pdf"]):
            return "pdf"
        if any(k in q_lower for k in ["excel", "spreadsheet", "table of", "export", "download"]):
            return "excel"
        # Detect markdown table in answer => likely tabular
        if answer.count("|") > 12 and "---" in answer:
            return "excel"
        if len(answer) > 1500:
            return "pdf"

        # Otherwise ask the router LLM
        prompt = FORMAT_ROUTER_PROMPT.format(
            question=question, preview=answer[:800]
        )
        try:
            resp = self.model.generate_content(prompt)
            text = (resp.text or "").strip()
            # tolerant JSON parse
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                fmt = data.get("format", "markdown").lower()
                if fmt in {"markdown", "pdf", "excel"}:
                    return fmt
        except Exception:
            pass
        return "markdown"
