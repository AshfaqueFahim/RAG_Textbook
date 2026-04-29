import json
import os
import re
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import fitz
import openai
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import (
    BM25_INDEX_PATH,
    CHROMA_DB_PATH,
    CHUNKS_PATH,
    CHUNKING_VERSION,
    INGESTION_STATE_PATH,
    PDF_PATH,
)
from src.generation.answer_generator import generate_answer
from src.generation.grounding_checker import check_grounding
from src.generation.prompt_builder import build_messages
from src.generation.query_rewriter import rewrite_query
from src.ingest.chunker import chunk_pages
from src.ingest.embedder import (
    load_ingestion_state,
    run_ingestion,
    save_ingestion_state,
)
from src.ingest.pdf_parser import compute_pdf_hash, parse_pdf
from src.retrieval.context_assembler import assemble_context, resolve_citations
from src.retrieval.hybrid_retriever import hybrid_retrieve
from src.store.bm25_manager import build_bm25_index, load_bm25_index, save_bm25_index
from src.store.chroma_manager import initialize_chroma

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI()

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_chroma = None
_bm25 = None
_openai_client = None

ingestion_status = {
    "state": "idle",
    "progress": 0,
    "message": "",
    "error": None,
}
_ingest_lock = threading.Lock()


@app.on_event("startup")
def startup():
    global _chroma, _bm25, _openai_client
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    _openai_client = openai.OpenAI(api_key=api_key)
    _chroma = initialize_chroma()
    _bm25 = load_bm25_index(BM25_INDEX_PATH)
    ingestion_status["state"] = "idle"


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
_REFUSAL_PREFIX = "The provided sources do not contain sufficient information"


class QueryRequest(BaseModel):
    question: str


@app.post("/api/query")
def query(req: QueryRequest):
    if ingestion_status["state"] == "ingesting":
        raise HTTPException(status_code=503, detail="Ingestion in progress — try again shortly.")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    fused = hybrid_retrieve(question, _chroma, _bm25, _openai_client)
    if not fused:
        return {"answer": "Could not find relevant passages in the textbook.", "sources": [], "confidence": "none"}

    context_block, citation_map = assemble_context(fused)
    messages = build_messages(context_block, question, len(fused))
    raw = generate_answer(messages, _openai_client)

    if raw.startswith(_REFUSAL_PREFIX):
        rewritten = rewrite_query(question, fused, _openai_client)
        fused2 = hybrid_retrieve(rewritten, _chroma, _bm25, _openai_client)
        if fused2:
            ctx2, cmap2 = assemble_context(fused2)
            raw2 = generate_answer(build_messages(ctx2, rewritten, len(fused2)), _openai_client)
            if not raw2.startswith(_REFUSAL_PREFIX):
                fused, citation_map, raw = fused2, cmap2, raw2

    resolved, sources_block, invalid = resolve_citations(raw, citation_map, len(fused))
    for n in invalid:
        resolved = re.sub(rf"\[SOURCE {n}\]", "", resolved).strip()

    score = check_grounding(resolved, fused, citation_map).grounding_score
    confidence = "high" if score >= 0.80 else ("low" if score >= 0.50 else "very_low")

    sources = []
    for line in sources_block.splitlines():
        m = re.match(r"\[SOURCE \d+\] (.+)", line)
        if m:
            sources.append(m.group(1))

    return {"answer": resolved.strip(), "sources": sources, "confidence": confidence}


# ---------------------------------------------------------------------------
# Upload & re-ingest
# ---------------------------------------------------------------------------
def _validate_pdf(pdf_path: Path) -> tuple[bool, str]:
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return False, "Not a valid PDF file."
    if doc.page_count == 0:
        return False, "PDF has no pages."
    text_pages = sum(
        1 for i in range(min(10, doc.page_count))
        if len(doc[i].get_text("text").strip()) >= 50
    )
    doc.close()
    if text_pages == 0:
        return False, (
            "PDF appears to be image-only and contains no extractable text. "
            "Run OCR first: ocrmypdf --force-ocr input.pdf output.pdf"
        )
    return True, "ok"


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in open(path, encoding="utf-8") if line.strip())


def _run_ingestion_background(pdf_path: Path):
    global _chroma, _bm25

    def update(state=None, progress=None, message=None, error=None):
        if state is not None:
            ingestion_status["state"] = state
        if progress is not None:
            ingestion_status["progress"] = progress
        if message is not None:
            ingestion_status["message"] = message
        if error is not None:
            ingestion_status["error"] = error

    try:
        update(state="ingesting", progress=0, message="Resetting indexes…")

        # Full reset (including Chroma)
        if CHUNKS_PATH.exists():
            CHUNKS_PATH.write_bytes(b"")
        if INGESTION_STATE_PATH.exists():
            INGESTION_STATE_PATH.unlink()
        if BM25_INDEX_PATH.exists():
            BM25_INDEX_PATH.unlink()
        if CHROMA_DB_PATH.exists():
            shutil.rmtree(CHROMA_DB_PATH)

        # Move PDF into place
        PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(pdf_path), str(PDF_PATH))

        # Phase 1: chunking (indeterminate)
        update(progress=2, message="Chunking pages…")
        chunk_pages(parse_pdf(PDF_PATH))

        # Count total chunks so we can show real % during embedding
        total_chunks = _count_jsonl_lines(CHUNKS_PATH)

        # Phase 2: embedding (real percentage)
        new_chroma = initialize_chroma()
        pdf_hash = compute_pdf_hash(PDF_PATH)
        state = {
            "pdf_hash": pdf_hash,
            "chunking_version": CHUNKING_VERSION,
            "status": "in_progress",
            "embedded_chunk_count": 0,
            "last_embedded_chunk_id": None,
            "bm25_built": False,
            "bm25_chunk_count": 0,
        }
        save_ingestion_state(state, INGESTION_STATE_PATH)

        def on_batch(embedded_count):
            pct = int(embedded_count / total_chunks * 90) if total_chunks > 0 else 50
            update(progress=pct, message=f"Embedding chunks… ({embedded_count}/{total_chunks})")

        run_ingestion(CHUNKS_PATH, INGESTION_STATE_PATH, new_chroma, _openai_client, on_batch=on_batch)

        # Phase 3: BM25 (fast, indeterminate)
        update(progress=92, message="Building search index…")
        new_bm25 = build_bm25_index(CHUNKS_PATH)
        save_bm25_index(new_bm25, BM25_INDEX_PATH)

        # Mark complete in state file
        final_state = load_ingestion_state(INGESTION_STATE_PATH)
        final_state["bm25_built"] = True
        final_state["bm25_chunk_count"] = new_bm25.corpus_size()
        final_state["status"] = "complete"
        save_ingestion_state(final_state, INGESTION_STATE_PATH)

        # Swap singletons
        _chroma = new_chroma
        _bm25 = new_bm25

        update(state="ready", progress=100, message="Ready!")

    except Exception as exc:
        update(state="error", message=f"Error: {exc}", error=str(exc))

    finally:
        try:
            pdf_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    with _ingest_lock:
        if ingestion_status["state"] == "ingesting":
            raise HTTPException(status_code=409, detail="Ingestion already in progress.")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save upload to a temp file
    suffix = ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp.close()
        tmp_path = Path(tmp.name)

        valid, msg = _validate_pdf(tmp_path)
        if not valid:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=msg)

        # Kick off background ingestion
        ingestion_status.update({"state": "ingesting", "progress": 0, "message": "Starting…", "error": None})
        t = threading.Thread(target=_run_ingestion_background, args=(tmp_path,), daemon=True)
        t.start()

        return {"state": "ingesting", "message": "Ingestion started."}

    except HTTPException:
        raise
    except Exception as exc:
        Path(tmp.name).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
@app.get("/api/status")
def status():
    return ingestion_status


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
