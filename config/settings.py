import hashlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — all paths are relative to this
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "textbook.pdf"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
INGESTION_STATE_PATH = PROJECT_ROOT / "data" / "processed" / "ingestion_state.json"
BM25_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "bm25_index.pkl"
CHROMA_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"

# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------
CHROMA_COLLECTION_NAME = "textbook_chunks"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_TARGET_MIN = 700     # Preferred lower bound (tokens)
CHUNK_TARGET_MAX = 800     # Preferred upper bound (tokens)
CHUNK_MAX_TOKENS = 1024    # Hard ceiling — never exceeded
CHUNK_MIN_TOKENS = 512     # Hard floor — chunks below this are merged or discarded

OVERLAP_FRACTION = 0.22          # Primary overlap fraction (last N% of previous chunk)
OVERLAP_FRACTION_FALLBACK = 0.15 # Used when primary overlap would push chunk over CHUNK_MAX_TOKENS

# ---------------------------------------------------------------------------
# Parser version
# Bump this integer whenever pdf_parser.py logic changes in a way that affects
# PageRecord output — e.g., chapter detection regex changes, heading heuristic
# threshold changes, or block parsing strategy changes. Changing this value
# causes CHUNKING_VERSION to change automatically, which triggers a --force
# requirement on next ingestion to prevent stale vectors.
# ---------------------------------------------------------------------------
PARSER_VERSION = 1

# ---------------------------------------------------------------------------
# Chunking version — auto-computed, do not set manually.
# Changes whenever any chunking-relevant constant above changes.
# Stored in ingestion_state.json and compared on each ingestion startup.
# ---------------------------------------------------------------------------
_chunking_version_input = "|".join(str(v) for v in [
    CHUNK_TARGET_MIN,
    CHUNK_TARGET_MAX,
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    OVERLAP_FRACTION,
    PARSER_VERSION,
])
CHUNKING_VERSION = hashlib.sha256(_chunking_version_input.encode()).hexdigest()[:16]

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 100

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
GENERATION_MODEL = "gpt-4o-mini"
GENERATION_MAX_TOKENS = 1024
GENERATION_TEMPERATURE = 0

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
VECTOR_SEARCH_TOP_K = 20   # Candidates fetched from Chroma before fusion
BM25_SEARCH_TOP_K = 20     # Candidates fetched from BM25 before fusion
FINAL_TOP_K = 10           # Final number of chunks passed to the LLM after RRF fusion
RRF_K_CONSTANT = 60        # Standard RRF constant; reduces sensitivity to rank-1 outliers

# ---------------------------------------------------------------------------
# Chapter map — single source of truth for chapter titles and start pages.
# Used by pdf_parser.py to attach metadata to PageRecords without doing any
# title extraction from the PDF itself.
# start_page is the 1-indexed PDF page number where the chapter begins.
# ---------------------------------------------------------------------------
CHAPTER_MAP = {}
