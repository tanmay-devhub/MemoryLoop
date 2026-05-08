import ollama
from memory import get_failure_count, get_recent_failures, store_lesson


def should_reflect() -> bool:
    count = get_failure_count()
    return count > 0 and count % 5 == 0


def reflect(error_type: str = "factual_error") -> str | None:
    failures = get_recent_failures(n=10)
    if len(failures) < 3:
        return None

    cases = "\n\n".join(
        f"Question: {f['query']}\n"
        f"Wrong answer: {f['response']}\n"
        f"Correct answer: {f['correction']}"
        for f in failures
    )

    error_label = error_type.replace("_", " ")
    prompt = (
        "You are a learning AI reviewing your own past mistakes. "
        f"These are all {error_label} errors. "
        "Write a lesson specifically about avoiding this type of mistake. "
        "Write ONE concise lesson (2 sentences max). "
        "Write only the lesson — no preamble, no explanation.\n\n"
        f"{cases}"
    )

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
        )
        lesson_text = response["message"]["content"].strip()
    except Exception:
        return None

    store_lesson(content=lesson_text, error_type=error_type)
    return lesson_text
