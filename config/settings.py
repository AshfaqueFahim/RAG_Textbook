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
CHAPTER_MAP = {
    1:  {"title": "The Meeting of Cultures",                          "start_page": 2},
    2:  {"title": "Transplantations and Borderlands",                 "start_page": 32},
    3:  {"title": "Society and Culture in Provincial America",        "start_page": 64},
    4:  {"title": "The Empire in Transition",                         "start_page": 98},
    5:  {"title": "The American Revolution",                          "start_page": 124},
    6:  {"title": "The Constitution and the New Republic",            "start_page": 158},
    7:  {"title": "The Jeffersonian Era",                             "start_page": 180},
    8:  {"title": "Varieties of American Nationalism",                "start_page": 216},
    9:  {"title": "Jacksonian America",                               "start_page": 234},
    10: {"title": "America's Economic Revolution",                    "start_page": 260},
    11: {"title": "Cotton, Slavery, and the Old South",               "start_page": 296},
    12: {"title": "Antebellum Culture and Reform",                    "start_page": 318},
    13: {"title": "The Impending Crisis",                             "start_page": 342},
    14: {"title": "The Civil War",                                    "start_page": 370},
    15: {"title": "Reconstruction and the New South",                 "start_page": 406},
    16: {"title": "The Conquest of the Far West",                     "start_page": 440},
    17: {"title": "Industrial Supremacy",                             "start_page": 472},
    18: {"title": "The Age of the City",                              "start_page": 498},
    19: {"title": "From Stalemate to Crisis",                         "start_page": 530},
    20: {"title": "The Imperial Republic",                            "start_page": 552},
    21: {"title": "The Rise of Progressivism",                        "start_page": 552},
    22: {"title": "The Battle for National Reform",                   "start_page": 600},
    23: {"title": "America and the Great War",                        "start_page": 620},
    24: {"title": "The New Era",                                      "start_page": 648},
    25: {"title": "The Great Depression",                             "start_page": 0},
    26: {"title": "The New Deal",                                     "start_page": 702},
    27: {"title": "The Global Crisis",                                "start_page": 728},
    28: {"title": "America in a World at War",                        "start_page": 748},
    29: {"title": "The Cold War",                                     "start_page": 776},
    30: {"title": "The Affluent Society",                             "start_page": 798},
    31: {"title": "The Ordeal of Liberalism",                         "start_page": 830},
    32: {"title": "The Crisis of Authority",                          "start_page": 858},
    33: {"title": "From the Age of Limits to the Age of Reagan",      "start_page": 892},
    34: {"title": "The Age of Globalization",                         "start_page": 918},
}
