from config.settings import (
    GENERATION_MAX_TOKENS,
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
)


def generate_answer(messages: list[dict], openai_client) -> str:
    response = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=GENERATION_TEMPERATURE,
        max_tokens=GENERATION_MAX_TOKENS,
    )
    return response.choices[0].message.content
