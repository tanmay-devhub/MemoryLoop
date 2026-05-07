import ollama
from memory import retrieve_lessons, store_interaction

_BASE_SYSTEM = (
    "You are a helpful assistant. Be concise. Answer in 2-3 sentences max."
)


def run_agent(query: str) -> dict:
    lessons = retrieve_lessons(query, n=3)

    if lessons:
        lessons_block = "\n".join(f"- {l}" for l in lessons)
        system_prompt = (
            f"{_BASE_SYSTEM}\n\n"
            f"You have learned the following lessons from past mistakes — apply them:\n"
            f"{lessons_block}"
        )
    else:
        system_prompt = _BASE_SYSTEM

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        )
        answer = response["message"]["content"]
    except Exception:
        return {
            "answer": "Ollama is not responding. Please ensure Ollama is running in background.",
            "lessons_used": [],
            "interaction_id": None,
        }

    interaction_id = store_interaction(
        query=query,
        response=answer,
        outcome="pending",
    )

    return {
        "answer": answer,
        "lessons_used": lessons,
        "interaction_id": interaction_id,
    }
