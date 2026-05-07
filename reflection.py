import ollama
from memory import get_recent_failures, get_interaction_count, store_lesson


def should_reflect() -> bool:
    try:
        from memory import _interactions_col
        if _interactions_col.count() == 0:
            return False
        all_items = _interactions_col.get(include=["metadatas"])
        failure_count = sum(
            1 for m in all_items["metadatas"]
            if m.get("outcome") in ("incorrect", "corrected")
        )
        return failure_count > 0 and failure_count % 5 == 0
    except Exception:
        return False


def reflect() -> str | None:
    failures = get_recent_failures(n=10)
    if len(failures) < 3:
        return None

    cases = "\n\n".join(
        f"Question: {f['query']}\n"
        f"Wrong answer: {f['response']}\n"
        f"Correct answer: {f['correction']}"
        for f in failures
    )

    prompt = (
        "You are a learning AI reviewing your own past mistakes. "
        "Based on the following failure cases, write ONE concise lesson (2 sentences max) "
        "that would help you avoid similar mistakes in the future. "
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

    store_lesson(content=lesson_text, error_type="general")
    return lesson_text
