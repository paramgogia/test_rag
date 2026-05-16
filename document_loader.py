"""
Loads heterogeneous documents (PDF, Excel, CSV) into uniform text chunks.

Each chunk carries metadata so the LLM can cite which file (and page/sheet)
it came from. We intentionally keep tabular data as structured text rather
than free-flowing prose - the LLM handles tables much better when row/column
relationships are preserved.
"""
from __future__ import annotations
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

import pdfplumber
import pandas as pd


@dataclass
class Chunk:
    """A single retrievable chunk with citation metadata."""
    text: str
    source: str          # Friendly source name e.g. "Q1 FY26 Press Release"
    file_name: str       # Original filename
    location: str        # e.g. "Page 3" or "Sheet: P&L" or "Rows 1-50"
    chunk_id: int = 0

    def to_dict(self):
        return asdict(self)

    def cite_tag(self) -> str:
        """Compact tag used when feeding context to the LLM."""
        return f"[{self.source} | {self.location}]"


# ---------- Text chunking ----------

def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into chunks of ~chunk_size characters, breaking on sentence
    boundaries when possible. Overlap preserves context across chunk borders.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # split on sentence-ish boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= chunk_size:
            current = f"{current} {sent}".strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk with tail of previous for overlap
            if overlap and chunks:
                tail = chunks[-1][-overlap:]
                current = f"{tail} {sent}".strip()
            else:
                current = sent
    if current:
        chunks.append(current)
    return chunks


# ---------- PDF loader ----------

def load_pdf(path: Path, source_name: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """
    Extract text from a PDF page-by-page. We chunk WITHIN each page so the
    page citation stays accurate. Tables are extracted separately so column
    relationships survive into the chunk text.
    """
    chunks: List[Chunk] = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""

            # Extract tables and append in a readable form
            tables = page.extract_tables() or []
            for t_idx, table in enumerate(tables):
                table_lines = []
                for row in table:
                    cells = [str(c).strip() if c else "" for c in row]
                    if any(cells):
                        table_lines.append(" | ".join(cells))
                if table_lines:
                    page_text += (
                        f"\n\n[Table {t_idx+1} on page {page_num}]\n"
                        + "\n".join(table_lines)
                    )

            for part in _split_text(page_text, chunk_size, overlap):
                chunks.append(Chunk(
                    text=part,
                    source=source_name,
                    file_name=path.name,
                    location=f"Page {page_num}",
                ))
    return chunks


# ---------- Excel loader ----------

def load_excel(path: Path, source_name: str) -> List[Chunk]:
    """
    Each sheet becomes one or more chunks. Sheets are loaded with pandas
    and serialised as text in a "Column: value" friendly format, since
    that's far easier for an LLM to reason over than raw CSV.
    """
    chunks: List[Chunk] = []
    # Try xlrd for old .xls, openpyxl for .xlsx
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    try:
        xls = pd.ExcelFile(path, engine=engine)
    except Exception:
        xls = pd.ExcelFile(path)

    for sheet_name in xls.sheet_names:
        try:
            df = xls.parse(sheet_name)
        except Exception:
            continue
        if df.empty:
            continue

        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue

        # Build a header + rows representation
        header = f"Sheet: {sheet_name}\nColumns: {', '.join(str(c) for c in df.columns)}\n"
        rows_text = []
        for idx, row in df.iterrows():
            row_str = " | ".join(f"{c}={row[c]}" for c in df.columns if pd.notna(row[c]))
            if row_str:
                rows_text.append(f"Row {idx}: {row_str}")

        # batch rows so each chunk stays manageable (50 rows / chunk)
        batch_size = 50
        for i in range(0, len(rows_text), batch_size):
            batch = rows_text[i:i + batch_size]
            text = header + "\n".join(batch)
            loc = (
                f"Sheet: {sheet_name}"
                if len(rows_text) <= batch_size
                else f"Sheet: {sheet_name}, Rows {i}-{i + len(batch) - 1}"
            )
            chunks.append(Chunk(
                text=text,
                source=source_name,
                file_name=path.name,
                location=loc,
            ))
    return chunks


# ---------- CSV loader ----------

def load_csv(path: Path, source_name: str) -> List[Chunk]:
    """
    CSVs (here: stock prices) are usually time-series. We summarise into
    monthly chunks so retrieval can grab the right time window without
    blowing the context with thousands of daily rows.
    """
    chunks: List[Chunk] = []
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return [Chunk(
            text=f"Failed to parse CSV: {e}",
            source=source_name,
            file_name=path.name,
            location="parse-error",
        )]

    # Try to detect a date column
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)
        df["_month"] = df[date_col].dt.to_period("M")

        # 1) Overall summary chunk (very valuable for "how did the stock do" questions)
        try:
            close_col = next((c for c in df.columns if c.lower() in {"close", "close price"}), None)
            if close_col:
                summary = (
                    f"Stock price summary from CSV ({path.name}):\n"
                    f"Date range: {df[date_col].min().date()} to {df[date_col].max().date()}\n"
                    f"Trading days: {len(df)}\n"
                    f"Opening price (first day): {df[close_col].iloc[0]}\n"
                    f"Closing price (last day): {df[close_col].iloc[-1]}\n"
                    f"Period high: {df[close_col].max()}\n"
                    f"Period low: {df[close_col].min()}\n"
                    f"Average close: {df[close_col].mean():.2f}\n"
                    f"Columns available: {', '.join(df.columns)}"
                )
                chunks.append(Chunk(
                    text=summary,
                    source=source_name,
                    file_name=path.name,
                    location="Summary",
                ))
        except Exception:
            pass

        # 2) Per-month chunks
        for month, group in df.groupby("_month"):
            rows = []
            for _, r in group.iterrows():
                parts = [f"{c}={r[c]}" for c in df.columns if c != "_month" and pd.notna(r[c])]
                rows.append(" | ".join(parts))
            text = f"Stock data for {month}:\n" + "\n".join(rows)
            chunks.append(Chunk(
                text=text,
                source=source_name,
                file_name=path.name,
                location=f"Month: {month}",
            ))
    else:
        # Fallback: chunk by 50 rows
        for i in range(0, len(df), 50):
            batch = df.iloc[i:i + 50]
            text = batch.to_csv(index=False)
            chunks.append(Chunk(
                text=text,
                source=source_name,
                file_name=path.name,
                location=f"Rows {i}-{i + len(batch) - 1}",
            ))

    return chunks


# ---------- Dispatcher ----------

def load_document(path: Path, source_name: str, chunk_size: int, overlap: int) -> List[Chunk]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path, source_name, chunk_size, overlap)
    if ext in {".xlsx", ".xls"}:
        return load_excel(path, source_name)
    if ext == ".csv":
        return load_csv(path, source_name)
    raise ValueError(f"Unsupported file type: {ext}")
