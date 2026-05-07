import json
from datetime import datetime
from pathlib import Path

import ollama

from agent import run_agent

_EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
_HISTORY_PATH = Path(__file__).parent / "eval_history.json"


def load_eval_questions() -> list[dict]:
    with open(_EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_single(q: dict) -> dict:
    result = run_agent(q["question"])
    answer = result["answer"]

    judge_prompt = (
        "Does this answer correctly answer the question? Reply YES or NO only.\n"
        f"Question: {q['question']} | Expected: {q['expected_answer']} | Agent answer: {answer}"
    )

    correct = False
    try:
        judge_response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": judge_prompt}],
        )
        verdict = judge_response["message"]["content"].strip().upper()
        correct = verdict.startswith("YES")
    except Exception:
        correct = False

    return {
        "id": q["id"],
        "question": q["question"],
        "answer": answer,
        "expected": q["expected_answer"],
        "correct": correct,
        "category": q.get("category", ""),
    }


def compile_results(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def run_eval() -> dict:
    questions = load_eval_questions()
    results = [evaluate_single(q) for q in questions]
    final = compile_results(results)
    save_eval_result(final)
    return final


def get_eval_history() -> list[dict]:
    try:
        if not _HISTORY_PATH.exists():
            return []
        with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_eval_result(result: dict) -> None:
    try:
        history = get_eval_history()
        history.append(result)
        with open(_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass
