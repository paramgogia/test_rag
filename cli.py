"""
CLI version of the Infosys analyst — useful for quick testing without
spinning up Streamlit.

Usage:
    python cli.py
"""
from __future__ import annotations
import sys

from vector_store import VectorStore
from rag_engine import RagEngine
from exporters import export_pdf, export_excel


def main():
    print("=" * 60)
    print("Infosys Financial Analyst (CLI)")
    print("Type 'exit' to quit, 'reset' to clear conversation history.")
    print("=" * 60)

    store = VectorStore()
    if not store.load():
        print("ERROR: vector store missing. Run: python ingest.py")
        sys.exit(1)

    engine = RagEngine(store)

    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "reset":
            engine.reset()
            print("(conversation reset)")
            continue

        try:
            res = engine.ask(q)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        print(f"\nAssistant:\n{res.answer}")
        print(f"\nFormat suggested: {res.format}")
        print(f"Sources ({len(res.sources)}):")
        for s in res.sources:
            print(f"  - {s['source']} | {s['location']} (score={s.get('score')})")

        if res.format == "pdf":
            path = export_pdf(q, res.answer, res.sources)
            print(f"PDF written to: {path}")
        elif res.format == "excel":
            path = export_excel(q, res.answer, res.sources)
            print(f"Excel written to: {path}")


if __name__ == "__main__":
    main()
