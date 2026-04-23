import argparse
import os
import re
import sys
from pathlib import Path

import openai

from config.settings import (
    BM25_INDEX_PATH,
    CHUNKS_PATH,
    CHUNKING_VERSION,
    INGESTION_STATE_PATH,
    PDF_PATH,
)
from src.ingest.chunker import chunk_pages
from src.ingest.embedder import load_ingestion_state, run_ingestion, save_ingestion_state
from src.ingest.pdf_parser import compute_pdf_hash, parse_pdf
from src.generation.answer_generator import generate_answer
from src.generation.grounding_checker import check_grounding
from src.generation.prompt_builder import build_messages
from src.generation.query_rewriter import rewrite_query
from src.retrieval.context_assembler import assemble_context, resolve_citations, validate_index_sync
from src.retrieval.hybrid_retriever import hybrid_retrieve
from src.store.bm25_manager import build_bm25_index, load_bm25_index, save_bm25_index
from src.store.chroma_manager import initialize_chroma


def _require_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return key


# ---------------------------------------------------------------------------
# ingest command
# ---------------------------------------------------------------------------

def _run_ingest(args: argparse.Namespace) -> None:
    api_key = _require_api_key()

    pdf_path = Path(args.pdf) if args.pdf else PDF_PATH
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # --force: reset append-only files before starting
    if args.force:
        if CHUNKS_PATH.exists():
            CHUNKS_PATH.write_bytes(b"")
            print("Truncated chunks.jsonl.")
        if INGESTION_STATE_PATH.exists():
            INGESTION_STATE_PATH.unlink()
            print("Deleted ingestion_state.json.")
        if BM25_INDEX_PATH.exists():
            BM25_INDEX_PATH.unlink()
            print("Deleted bm25_index.pkl.")

    print("Computing PDF hash...")
    pdf_hash = compute_pdf_hash(pdf_path)

    state = load_ingestion_state(INGESTION_STATE_PATH)

    # Skip entirely if this exact PDF + chunking config was already fully ingested
    if (
        not args.force
        and state.get("status") == "complete"
        and state.get("pdf_hash") == pdf_hash
        and state.get("chunking_version") == CHUNKING_VERSION
    ):
        print("Ingestion already complete. Use --force to re-ingest.")
        return

    # Stamp current run's fingerprints into state before any work begins
    state["pdf_hash"] = pdf_hash
    state["chunking_version"] = CHUNKING_VERSION
    state["status"] = "in_progress"
    save_ingestion_state(state, INGESTION_STATE_PATH)

    openai_client = openai.OpenAI(api_key=api_key)
    chroma_manager = initialize_chroma()

    try:
        # Step 1: Parse + chunk
        # Skip if chunks.jsonl already has content (embedding may be partially done).
        if not CHUNKS_PATH.exists() or CHUNKS_PATH.stat().st_size == 0:
            print("Parsing and chunking PDF...")
            total = chunk_pages(parse_pdf(pdf_path))
            print(f"Chunking complete: {total} chunks written.")
        else:
            print("chunks.jsonl already has content — skipping chunking, resuming embedding.")

        # Step 2: Embed (handles its own resume via last_embedded_chunk_id)
        print("Embedding chunks...")
        run_ingestion(CHUNKS_PATH, INGESTION_STATE_PATH, chroma_manager, openai_client)
        state = load_ingestion_state(INGESTION_STATE_PATH)
        print(f"Embedding complete: {state['embedded_chunk_count']} chunks embedded.")

        # Step 3: Build BM25 index
        if not state.get("bm25_built"):
            print("Building BM25 index...")
            bm25_manager = build_bm25_index(CHUNKS_PATH)
            save_bm25_index(bm25_manager, BM25_INDEX_PATH)
            state = load_ingestion_state(INGESTION_STATE_PATH)
            state["bm25_built"] = True
            state["bm25_chunk_count"] = bm25_manager.corpus_size()
            save_ingestion_state(state, INGESTION_STATE_PATH)
            print(f"BM25 index built: {bm25_manager.corpus_size()} chunks indexed.")
        else:
            print("BM25 index already built — skipping.")

        print("Ingestion complete.")

    except KeyboardInterrupt:
        print("\nInterrupted — progress saved. Re-run to resume.", file=sys.stderr)
        sys.exit(0)


# ---------------------------------------------------------------------------
# query pipeline
# ---------------------------------------------------------------------------

_REFUSAL_PREFIX = "The provided sources do not contain sufficient information"


def _process_query(
    question: str,
    chroma_manager,
    bm25_manager,
    openai_client,
) -> str:
    fused_results = hybrid_retrieve(question, chroma_manager, bm25_manager, openai_client)

    if not fused_results:
        return "Could not find relevant passages in the textbook."

    context_block, citation_map = assemble_context(fused_results)
    num_sources = len(fused_results)

    messages = build_messages(context_block, question, num_sources)
    raw_answer = generate_answer(messages, openai_client)

    if raw_answer.startswith(_REFUSAL_PREFIX):
        rewritten = rewrite_query(question, fused_results, openai_client)
        second_results = hybrid_retrieve(rewritten, chroma_manager, bm25_manager, openai_client)

        if second_results:
            context_block2, citation_map2 = assemble_context(second_results)
            num_sources2 = len(second_results)
            messages2 = build_messages(context_block2, rewritten, num_sources2)
            raw_answer2 = generate_answer(messages2, openai_client)

            if raw_answer2.startswith(_REFUSAL_PREFIX):
                return raw_answer + f"\n\nTry asking: {rewritten}"

            fused_results = second_results
            citation_map = citation_map2
            num_sources = num_sources2
            raw_answer = raw_answer2
        else:
            return raw_answer + f"\n\nTry asking: {rewritten}"

    resolved_answer, sources_used_block, invalid_citations = resolve_citations(
        raw_answer, citation_map, num_sources
    )

    if invalid_citations:
        tags = ", ".join(f"[SOURCE {n}]" for n in sorted(invalid_citations))
        invalid_warning = (
            f"Warning: answer referenced {tags} which were not in the retrieved "
            f"context. Those citations have been removed.\n\n"
        )
        resolved_answer = re.sub(
            r"\[SOURCE (" + "|".join(str(n) for n in invalid_citations) + r")\]",
            "",
            resolved_answer,
        ).strip()
    else:
        invalid_warning = ""

    grounding_report = check_grounding(resolved_answer, fused_results, citation_map)
    score = grounding_report.grounding_score

    final_output = invalid_warning + resolved_answer + "\n\n" + sources_used_block

    if score < 0.50:
        return (
            "Very low confidence: this answer may not be well supported by the retrieved "
            "sources. Verify carefully against the original text.\n\n"
            + final_output
        )

    if score < 0.80:
        return (
            "Low confidence: portions of this answer may not be fully supported by "
            "the retrieved sources. Verify against the original text.\n\n"
            + final_output
        )

    return final_output


# ---------------------------------------------------------------------------
# query command
# ---------------------------------------------------------------------------

def _run_query(args: argparse.Namespace) -> None:
    api_key = _require_api_key()

    if not BM25_INDEX_PATH.exists():
        print("Error: BM25 index not found. Run 'ingest' first.", file=sys.stderr)
        sys.exit(1)

    print("Loading indexes...")
    openai_client = openai.OpenAI(api_key=api_key)
    chroma_manager = initialize_chroma()
    bm25_manager = load_bm25_index(BM25_INDEX_PATH)

    # Sync check — count-based only, warns on gross divergence
    if CHUNKS_PATH.exists():
        is_synced, counts = validate_index_sync(CHUNKS_PATH, chroma_manager, bm25_manager)
        if not is_synced:
            print(
                f"Warning: Index mismatch detected — consider re-running ingestion.\n"
                f"  chunks.jsonl: {counts['jsonl']}  "
                f"Chroma: {counts['chroma']}  "
                f"BM25: {counts['bm25']}"
            )

    print("Ready. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower().startswith("query:"):
            question = question[6:].strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            break

        result = _process_query(question, chroma_manager, bm25_manager, openai_client)
        print(f"\n{result}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG History Textbook Q&A")
    parser.set_defaults(command="query")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest textbook PDF into indexes")
    ingest_parser.add_argument(
        "--pdf",
        type=str,
        help="Path to textbook PDF (default: data/raw/textbook.pdf)",
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest from scratch, discarding all existing state",
    )

    subparsers.add_parser("query", help="Interactive query REPL")

    args = parser.parse_args()

    if args.command == "ingest":
        _run_ingest(args)
    elif args.command == "query":
        _run_query(args)


if __name__ == "__main__":
    main()
