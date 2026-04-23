_REWRITE_TEMPLATE = """\
Original question: {question}

Passages retrieved from the textbook (may be only partially relevant):
{passages}

The original question did not match the textbook's vocabulary well enough to retrieve \
a useful answer. Using the names, terms, dates, events, and phrasing found in the \
passages above, write a single improved question that is more likely to retrieve \
relevant information from the same textbook.

Output only the improved question — no explanation, no quotes, no preamble."""


def rewrite_query(original_question: str, retrieved_chunks: list, openai_client) -> str:
    try:
        passage_texts = "\n\n".join(
            f"[{i + 1}] {chunk['text'][:400]}"
            for i, chunk in enumerate(retrieved_chunks[:5])
        )
        prompt = _REWRITE_TEMPLATE.format(
            question=original_question,
            passages=passage_texts,
        )
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=128,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else original_question
    except Exception:
        return original_question
