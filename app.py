import os
import re
import sys
from pathlib import Path

import openai
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import BM25_INDEX_PATH
from src.generation.answer_generator import generate_answer
from src.generation.grounding_checker import check_grounding
from src.generation.prompt_builder import build_messages
from src.generation.query_rewriter import rewrite_query
from src.retrieval.context_assembler import assemble_context, resolve_citations
from src.retrieval.hybrid_retriever import hybrid_retrieve
from src.store.bm25_manager import load_bm25_index
from src.store.chroma_manager import initialize_chroma

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once on startup, reused across requests
_chroma = None
_bm25 = None
_openai_client = None


@app.on_event("startup")
def startup():
    global _chroma, _bm25, _openai_client
    api_key = os.environ.get("OPENAI_API_KEY", "")
    _openai_client = openai.OpenAI(api_key=api_key)
    _chroma = initialize_chroma()
    _bm25 = load_bm25_index(BM25_INDEX_PATH)


_REFUSAL_PREFIX = "The provided sources do not contain sufficient information"


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/query")
def query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    fused = hybrid_retrieve(question, _chroma, _bm25, _openai_client)
    if not fused:
        return {"answer": "Could not find relevant passages in the textbook.", "sources": [], "confidence": "none", "warning": None}

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

    return {"answer": resolved.strip(), "sources": sources, "confidence": confidence, "warning": None}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
