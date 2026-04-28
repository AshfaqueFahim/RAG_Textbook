import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openai

from config.settings import BM25_INDEX_PATH, CHUNKS_PATH
from src.generation.answer_generator import generate_answer
from src.generation.grounding_checker import check_grounding
from src.generation.prompt_builder import build_messages
from src.generation.query_rewriter import rewrite_query
from src.retrieval.context_assembler import assemble_context, resolve_citations
from src.retrieval.hybrid_retriever import hybrid_retrieve
from src.store.bm25_manager import load_bm25_index
from src.store.chroma_manager import initialize_chroma

# ---------------------------------------------------------------------------
# Module-level singletons — reused across warm invocations
# ---------------------------------------------------------------------------
_chroma = None
_bm25 = None
_openai_client = None


def _init():
    global _chroma, _bm25, _openai_client
    if _chroma is not None:
        return
    api_key = os.environ.get("OPENAI_API_KEY", "")
    _openai_client = openai.OpenAI(api_key=api_key)
    _chroma = initialize_chroma()
    _bm25 = load_bm25_index(BM25_INDEX_PATH)


# ---------------------------------------------------------------------------
# Pipeline (mirrors cli/main.py _process_query)
# ---------------------------------------------------------------------------
_REFUSAL_PREFIX = "The provided sources do not contain sufficient information"


def _process_query(question: str) -> dict:
    _init()
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
            context_block2, citation_map2 = assemble_context(fused2)
            messages2 = build_messages(context_block2, rewritten, len(fused2))
            raw2 = generate_answer(messages2, _openai_client)
            if not raw2.startswith(_REFUSAL_PREFIX):
                fused, citation_map, raw = fused2, citation_map2, raw2

    resolved, sources_block, invalid = resolve_citations(raw, citation_map, len(fused))

    if invalid:
        for n in invalid:
            resolved = re.sub(rf"\[SOURCE {n}\]", "", resolved).strip()

    grounding = check_grounding(resolved, fused, citation_map)
    score = grounding.grounding_score

    confidence = "high" if score >= 0.80 else ("low" if score >= 0.50 else "very_low")

    # Parse sources_block into structured list
    sources = []
    for line in sources_block.splitlines():
        m = re.match(r"\[SOURCE \d+\] (.+)", line)
        if m:
            sources.append(m.group(1))

    return {
        "answer": resolved.strip(),
        "sources": sources,
        "confidence": confidence,
        "warning": None,
    }


# ---------------------------------------------------------------------------
# Vercel handler
# ---------------------------------------------------------------------------
class handler(BaseHTTPRequestHandler):
    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            question = (body.get("question") or "").strip()
            if not question:
                raise ValueError("Missing 'question'")

            result = _process_query(question)
            payload = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(payload)

        except Exception as exc:
            error = json.dumps({"error": str(exc)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(error)

    def log_message(self, *_):
        pass
