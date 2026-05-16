"""
Turn a RagAnswer into a downloadable PDF report or Excel sheet.

The Excel exporter is markdown-table aware: if the LLM's answer contains
a pipe-delimited markdown table, we extract it into a real sheet. If not,
we fall back to a single 'Answer' column.
"""
from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)

from config import OUTPUTS_DIR


# ---------- helpers ----------

def _safe_filename(question: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", question.strip().lower())[:40].strip("_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{clean}_{ts}" if clean else f"answer_{ts}"


def _extract_markdown_tables(text: str) -> List[List[List[str]]]:
    """Find markdown tables in the answer. Returns list of tables (each is list of rows)."""
    tables = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # markdown table: at least 2 pipes, next line is separator
        if line.count("|") >= 2 and i + 1 < len(lines) and re.match(r"^\s*\|?[\s\-:|]+\|?\s*$", lines[i + 1]):
            rows = []
            header = [c.strip() for c in line.strip("|").split("|")]
            rows.append(header)
            j = i + 2
            while j < len(lines) and lines[j].count("|") >= 2:
                row = [c.strip() for c in lines[j].strip("|").split("|")]
                # pad/truncate to header width
                if len(row) < len(header):
                    row += [""] * (len(header) - len(row))
                elif len(row) > len(header):
                    row = row[:len(header)]
                rows.append(row)
                j += 1
            if len(rows) >= 2:
                tables.append(rows)
            i = j
        else:
            i += 1
    return tables


def _strip_markdown(text: str) -> str:
    """Light markdown-to-plain conversion for PDF body. Keeps line breaks."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"^#{1,6}\s*(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)
    return text


# ---------- PDF ----------

def export_pdf(question: str, answer: str, sources: List[Dict[str, Any]]) -> Path:
    """Build a nicely-formatted PDF report."""
    filename = _safe_filename(question) + ".pdf"
    path = OUTPUTS_DIR / filename

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )

    styles = getSampleStyleSheet()
    h_style = ParagraphStyle(
        "header", parent=styles["Heading1"],
        fontSize=16, textColor=colors.HexColor("#0a4d8c"), spaceAfter=12,
    )
    sub_style = ParagraphStyle(
        "sub", parent=styles["Normal"],
        fontSize=9, textColor=colors.grey, spaceAfter=14,
    )
    q_style = ParagraphStyle(
        "q", parent=styles["Heading3"],
        fontSize=12, textColor=colors.HexColor("#222222"), spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "body", parent=styles["BodyText"],
        fontSize=10, leading=14, spaceAfter=6,
    )

    story = []
    story.append(Paragraph("Infosys Financial Analyst Report", h_style))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}", sub_style
    ))
    story.append(Paragraph("Question", q_style))
    story.append(Paragraph(question, body_style))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Answer", q_style))

    # Mix tables and paragraphs
    tables = _extract_markdown_tables(answer)
    if tables:
        # Render text WITHOUT the markdown tables (replace them with placeholders)
        text_no_tables = answer
        for t in tables:
            md_lines = []
            for row in t:
                md_lines.append("| " + " | ".join(row) + " |")
            md = "\n".join(md_lines)
            text_no_tables = text_no_tables.replace(md, "{{TABLE}}", 1)

        parts = text_no_tables.split("{{TABLE}}")
        for i, part in enumerate(parts):
            for para in part.strip().split("\n"):
                if para.strip():
                    story.append(Paragraph(_strip_markdown(para), body_style))
            if i < len(tables):
                t = tables[i]
                tbl = Table(t, repeatRows=1, hAlign="LEFT")
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0a4d8c")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.white, colors.HexColor("#f4f7fb")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]))
                story.append(Spacer(1, 0.1 * inch))
                story.append(tbl)
                story.append(Spacer(1, 0.1 * inch))
    else:
        for para in answer.split("\n"):
            if para.strip():
                story.append(Paragraph(_strip_markdown(para), body_style))

    # Sources
    if sources:
        story.append(PageBreak())
        story.append(Paragraph("Sources", q_style))
        for s in sources:
            line = f"<b>{s['source']}</b> — {s['location']} (relevance: {s.get('score', '?')})"
            story.append(Paragraph(line, body_style))
            story.append(Paragraph(
                f"<i>{s['snippet']}</i>",
                ParagraphStyle("snip", parent=body_style, fontSize=9,
                               textColor=colors.HexColor("#555555"), spaceAfter=8),
            ))

    doc.build(story)
    return path


# ---------- Excel ----------

def export_excel(question: str, answer: str, sources: List[Dict[str, Any]]) -> Path:
    """
    Build a multi-sheet Excel: (1) Question & narrative answer, (2) each
    extracted table on its own sheet, (3) sources.
    """
    filename = _safe_filename(question) + ".xlsx"
    path = OUTPUTS_DIR / filename

    tables = _extract_markdown_tables(answer)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # Sheet 1: Summary
        summary_df = pd.DataFrame({
            "Field": ["Question", "Generated", "Answer (narrative)"],
            "Value": [
                question,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                answer,
            ],
        })
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Sheet(s): Tables
        for i, t in enumerate(tables, start=1):
            header, *rows = t
            df = pd.DataFrame(rows, columns=header)
            sheet = f"Table_{i}"[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)

        # If no tables were found, write the answer paragraphs as a Data sheet
        if not tables:
            paragraphs = [p.strip() for p in answer.split("\n") if p.strip()]
            df = pd.DataFrame({"Answer": paragraphs})
            df.to_excel(writer, sheet_name="Data", index=False)

        # Sources sheet
        if sources:
            src_df = pd.DataFrame(sources)
            src_df.to_excel(writer, sheet_name="Sources", index=False)

        # Auto-widen columns
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = 10
                col_letter = col[0].column_letter
                for cell in col:
                    if cell.value is not None:
                        max_len = max(max_len, min(60, len(str(cell.value))))
                ws.column_dimensions[col_letter].width = max_len + 2

    return path
