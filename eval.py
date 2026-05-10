import json
from datetime import datetime
from pathlib import Path

import ollama

from agent import run_agent

_HISTORY_PATH = Path(__file__).parent / "eval_history.json"

EVAL_SETS = {
    "General (50 questions)": "eval_set.json",
    "Algorithms (20 questions)": "eval_set_algorithms.json",
    "System Design (20 questions)": "eval_set_systems.json",
    "Python Deep Dive (20 questions)": "eval_set_python.json",
}


def load_eval_set(filename: str) -> list[dict]:
    path = Path(__file__).parent / filename
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: eval set file not found: {path}")
        return []
    except Exception as e:
        print(f"Warning: could not load {filename}: {e}")
        return []


def load_eval_questions(eval_set_name: str = "General (50 questions)") -> list[dict]:
    filename = EVAL_SETS.get(eval_set_name, "eval_set.json")
    return load_eval_set(filename)


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


def compile_results(
    results: list, eval_set_name: str = "General (50 questions)"
) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "eval_set_name": eval_set_name,
    }


def run_eval(eval_set_name: str = "General (50 questions)") -> dict:
    questions = load_eval_questions(eval_set_name)
    results = [evaluate_single(q) for q in questions]
    final = compile_results(results, eval_set_name)
    save_eval_result(final)
    return final


def get_eval_history() -> list[dict]:
    try:
        if not _HISTORY_PATH.exists():
            return []
        with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for record in data:
            if "eval_set_name" not in record:
                record["eval_set_name"] = "General (50 questions)"
        return data
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
