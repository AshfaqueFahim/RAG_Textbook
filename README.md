# RAG History Q&A

A command-line tool that answers questions about a scanned history textbook PDF. You ask a question in plain English, and it finds the relevant passages, generates an answer, and cites the exact chapter, section, and page number for every claim. Runs locally on a MacBook with no GPU required.

---

## Requirements

- Python 3.10+
- An OpenAI API key

**Python dependencies:**

```
openai
chromadb
pymupdf
rank-bm25
tiktoken
```

---

## Setup

**1. Clone the repo**

```bash
git clone <repo-url>
cd RAGPROJECT
```

**2. Install dependencies**

```bash
pip install openai chromadb pymupdf rank-bm25 tiktoken
```

**3. Add your OpenAI API key**

```bash
export OPENAI_API_KEY="sk-..."
```

Add this to your shell profile (`~/.zshrc` or `~/.bashrc`) to avoid setting it each session.

**4. Place the PDF**

Copy your textbook PDF to:

```
data/raw/textbook.pdf
```

If the PDF has garbled or unreadable text (common with older scanned books), run OCR on it first:

```bash
ocrmypdf --force-ocr data/raw/textbook.pdf data/raw/textbook.pdf
```

**5. Configure chapter information**

Open `config/settings.py` and fill in `CHAPTER_MAP` with the chapters from your textbook. Each entry maps a chapter number to its title and the printed page number where it starts:

```python
CHAPTER_MAP = {
    1: {"title": "Chapter Title Here", "start_page": 1},
    2: {"title": "Another Chapter",    "start_page": 24},
    # ...
}
```

---

## Usage

**First run — ingest the textbook**

Before you can query anything, the tool needs to parse the PDF, split it into chunks, embed them, and build search indexes. This only needs to be done once (or again if the PDF changes):

```bash
python3 -m src.cli.main ingest
```

This takes several minutes depending on the size of the PDF. Progress is saved, so if it's interrupted you can re-run the same command to resume.

**Query the textbook**

Once ingestion is complete, start the interactive query REPL:

```bash
python3 -m src.cli.main query
```

Type a question and press Enter. Type `quit` or `exit` to stop.

---

## How it works

**PDF parsing** — The PDF is read page by page. Each page's text is extracted and tagged with its chapter, section header, and printed page number using the chapter map you configured.

**Chunking** — The extracted text is split into chunks of roughly 700–800 tokens each. Chunks are sized to fit comfortably within the embedding model's input limit while keeping related sentences together.

**Hybrid search** — When you ask a question, the tool searches the chunks using two methods in parallel: dense vector search (semantic similarity via embeddings) and BM25 keyword search. The results from both are merged and ranked, which handles both conceptual questions and specific name/date lookups better than either method alone.

**Answer generation** — The top-ranked chunks are assembled into a prompt and sent to GPT-4o-mini, which is instructed to answer using only those passages and to cite every factual claim with a source tag. The tool then validates that all cited sources were actually retrieved, checks how well the answer is grounded in the source text, and formats the final output with a sources block showing the chapter, section, and page for each citation.
