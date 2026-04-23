import re
from dataclasses import dataclass, field


@dataclass
class GroundingReport:
    is_grounded: bool
    flagged_sentences: list[tuple[str, str]]
    grounding_score: float


def extract_key_terms(text: str) -> set[str]:
    key_terms = set()
    for token in text.split():
        cleaned = re.sub(r"[^\w]", "", token)
        if len(cleaned) > 3 and (cleaned[0].isupper() or cleaned.isdigit()):
            key_terms.add(cleaned.lower())
    return key_terms


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _extract_cited_source(sentence: str) -> int | None:
    matches = re.findall(r"\[SOURCE (\d+)\]", sentence)
    return int(matches[0]) if matches else None


def check_grounding(
    answer_text: str,
    retrieved_chunks: list[dict],
    citation_map: dict,
) -> GroundingReport:
    """
    Per-source grounding check: a sentence citing [SOURCE N] is verified against
    SOURCE N's text only. Uncited sentences fall back to the full corpus.
    grounding_score is a weak heuristic — word overlap, not factual accuracy.
    """
    source_terms = {
        n: extract_key_terms(entry["chunk_text"])
        for n, entry in citation_map.items()
    }
    corpus_terms = extract_key_terms(" ".join(c["text"] for c in retrieved_chunks))

    sentences = _split_sentences(answer_text)
    if not sentences:
        return GroundingReport(is_grounded=True, flagged_sentences=[], grounding_score=1.0)

    flagged: list[tuple[str, str]] = []
    grounded_count = 0

    for sentence in sentences:
        if len(sentence.split()) < 5:
            grounded_count += 1
            continue

        sentence_terms = extract_key_terms(sentence)
        if not sentence_terms:
            grounded_count += 1
            continue

        cited_n = _extract_cited_source(sentence)

        if cited_n is not None and cited_n in citation_map:
            if sentence_terms & source_terms[cited_n]:
                grounded_count += 1
            else:
                flagged.append((
                    sentence,
                    f"cites SOURCE {cited_n} but terms not found in SOURCE {cited_n}",
                ))
        else:
            if sentence_terms & corpus_terms:
                grounded_count += 1
            else:
                flagged.append((sentence, "uncited and terms not found in any source"))

    score = grounded_count / len(sentences)
    return GroundingReport(
        is_grounded=score >= 0.80,
        flagged_sentences=flagged,
        grounding_score=score,
    )
