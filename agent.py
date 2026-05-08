import ollama
from memory import retrieve_lessons, store_interaction

_BASE_INSTRUCTIONS = (
    "You are a helpful assistant. Be concise. Answer in 2-3 sentences max."
)

_CONFIDENCE_INSTRUCTION = (
    "\n\nAfter your answer, on a NEW LINE, output exactly this format and nothing else after it:\n"
    "CONFIDENCE: 0.85\n"
    "where the number is your confidence in your answer from 0.0 (no idea) to 1.0 "
    "(completely certain). Do not add any text after the number."
)


def _parse_confidence(raw: str) -> tuple[str, float]:
    """Split raw LLM output into (clean_answer, confidence_float)."""
    try:
        if "CONFIDENCE:" in raw:
            parts = raw.split("CONFIDENCE:", 1)
            answer = parts[0].strip()
            conf_str = parts[1].strip().split()[0].rstrip(".")
            confidence = max(0.0, min(1.0, float(conf_str)))
            return answer, confidence
    except Exception:
        pass
    return raw.strip(), 0.5


def run_agent(query: str) -> dict:
    lessons = retrieve_lessons(query, n=3)

    if lessons:
        lessons_block = "\n".join(f"- {l['content']}" for l in lessons)
        system_prompt = (
            f"{_BASE_INSTRUCTIONS}\n\n"
            f"You have learned the following lessons from past mistakes — apply them:\n"
            f"{lessons_block}"
            f"{_CONFIDENCE_INSTRUCTION}"
        )
    else:
        system_prompt = _BASE_INSTRUCTIONS + _CONFIDENCE_INSTRUCTION

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        )
        raw_answer = response["message"]["content"]
        answer, confidence = _parse_confidence(raw_answer)
    except Exception:
        return {
            "answer": "Ollama is not responding. Please ensure Ollama is running in background.",
            "lessons_used": [],
            "lesson_ids_used": [],
            "interaction_id": None,
            "confidence": 0.5,
        }

    interaction_id = store_interaction(
        query=query,
        response=answer,
        outcome="pending",
        confidence=confidence,
    )

    return {
        "answer": answer,
        "lessons_used": lessons,
        "lesson_ids_used": [l["id"] for l in lessons],
        "interaction_id": interaction_id,
        "confidence": confidence,
    }
