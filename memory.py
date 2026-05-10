import uuid
from datetime import datetime
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

_DB_PATH = Path(__file__).parent / "memory_agent_db"
_client = chromadb.PersistentClient(path=str(_DB_PATH))
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

_lessons_col = _client.get_or_create_collection("lessons")
_interactions_col = _client.get_or_create_collection("interactions")


# ── helpers ───────────────────────────────────────────────────────────────────


def _embed(text: str) -> list[float]:
    return _embedder.encode(text).tolist()


def _safe_float(value) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _safe_int(value) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


# ── lessons ───────────────────────────────────────────────────────────────────


def store_lesson(content: str, error_type: str) -> str:
    lesson_id = str(uuid.uuid4())
    try:
        _lessons_col.add(
            ids=[lesson_id],
            embeddings=[_embed(content)],
            documents=[content],
            metadatas=[
                {
                    "error_type": error_type,
                    "last_used_at": datetime.now().isoformat(),
                    "usefulness_score": "0.0",
                    "is_active": "true",
                    "times_retrieved": "0",
                }
            ],
        )
    except Exception:
        pass
    return lesson_id


def retrieve_lessons(query: str, n: int = 3) -> list[dict]:
    """Returns list of dicts: {id, content, usefulness_score, times_retrieved, last_used_at}."""
    try:
        total = _lessons_col.count()
        if total == 0:
            return []
        results = _lessons_col.query(
            query_embeddings=[_embed(query)],
            n_results=total,
            include=["documents", "metadatas"],
        )
        if not results["documents"] or not results["documents"][0]:
            return []

        lessons: list[dict] = []
        for i, (doc, meta) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        ):
            if meta.get("is_active", "true") != "true":
                continue
            lesson_id = results["ids"][0][i]
            lessons.append(
                {
                    "id": lesson_id,
                    "content": doc,
                    "usefulness_score": _safe_float(
                        meta.get("usefulness_score", "0.0")
                    ),
                    "times_retrieved": _safe_int(meta.get("times_retrieved", "0")),
                    "last_used_at": meta.get("last_used_at", ""),
                }
            )
            update_lesson_usage(lesson_id, was_helpful=None)
            if len(lessons) >= n:
                break

        return lessons
    except Exception:
        return []


def update_lesson_usage(lesson_id: str, was_helpful) -> None:
    """was_helpful: True / False / None (None = retrieval only, no score change)."""
    try:
        existing = _lessons_col.get(ids=[lesson_id], include=["metadatas"])
        if not existing["ids"]:
            return
        meta = existing["metadatas"][0]
        meta["last_used_at"] = datetime.now().isoformat()
        meta["times_retrieved"] = str(_safe_int(meta.get("times_retrieved", "0")) + 1)

        if was_helpful is True:
            score = min(_safe_float(meta.get("usefulness_score", "0.0")) + 0.1, 1.0)
            meta["usefulness_score"] = str(round(score, 4))
        elif was_helpful is False:
            score = max(_safe_float(meta.get("usefulness_score", "0.0")) - 0.05, 0.0)
            meta["usefulness_score"] = str(round(score, 4))

        _lessons_col.update(ids=[lesson_id], metadatas=[meta])
    except Exception:
        pass


def run_decay_check() -> dict:
    result = {"archived": 0, "kept": 0, "checked": 0}
    try:
        if _lessons_col.count() == 0:
            return result
        all_items = _lessons_col.get(include=["metadatas"])
        now = datetime.now()
        for lesson_id, meta in zip(all_items["ids"], all_items["metadatas"]):
            if meta.get("is_active", "true") != "true":
                continue
            result["checked"] += 1
            try:
                raw_ts = meta.get("last_used_at", "")
                last_used = datetime.fromisoformat(raw_ts) if raw_ts else now
                days_since = (now - last_used).days
                score = _safe_float(meta.get("usefulness_score", "0.0"))
                if days_since > 30 and score < 0.2:
                    meta["is_active"] = "false"
                    _lessons_col.update(ids=[lesson_id], metadatas=[meta])
                    result["archived"] += 1
                else:
                    result["kept"] += 1
            except Exception:
                result["kept"] += 1
    except Exception:
        pass
    return result


def get_active_lessons_count() -> int:
    try:
        if _lessons_col.count() == 0:
            return 0
        all_items = _lessons_col.get(include=["metadatas"])
        return sum(
            1 for m in all_items["metadatas"] if m.get("is_active", "true") == "true"
        )
    except Exception:
        return 0


def get_archived_lessons_count() -> int:
    try:
        if _lessons_col.count() == 0:
            return 0
        all_items = _lessons_col.get(include=["metadatas"])
        return sum(
            1 for m in all_items["metadatas"] if m.get("is_active", "true") == "false"
        )
    except Exception:
        return 0


def get_all_lessons() -> list[dict]:
    try:
        if _lessons_col.count() == 0:
            return []
        all_items = _lessons_col.get(include=["documents", "metadatas"])
        return [
            {
                "id": lid,
                "content": doc,
                "error_type": meta.get("error_type", "general"),
                "usefulness_score": _safe_float(meta.get("usefulness_score", "0.0")),
                "times_retrieved": _safe_int(meta.get("times_retrieved", "0")),
                "last_used_at": meta.get("last_used_at", ""),
                "is_active": meta.get("is_active", "true") == "true",
            }
            for lid, doc, meta in zip(
                all_items["ids"], all_items["documents"], all_items["metadatas"]
            )
        ]
    except Exception:
        return []


def get_lessons_by_error_type(error_type: str) -> list[dict]:
    try:
        return [l for l in get_all_lessons() if l["error_type"] == error_type]
    except Exception:
        return []


def get_lesson_count() -> int:
    try:
        return _lessons_col.count()
    except Exception:
        return 0


# ── interactions ──────────────────────────────────────────────────────────────


def store_interaction(
    query: str,
    response: str,
    outcome: str,
    correction: str | None = None,
    confidence: float = 0.5,
    error_type: str = "factual_error",
) -> str:
    interaction_id = str(uuid.uuid4())
    try:
        meta: dict = {
            "outcome": outcome,
            "query": query,
            "confidence": str(confidence),
            "error_type": error_type,
        }
        if correction:
            meta["correction"] = correction
        _interactions_col.add(
            ids=[interaction_id],
            embeddings=[_embed(query)],
            documents=[response],
            metadatas=[meta],
        )
    except Exception:
        pass
    return interaction_id


def update_interaction_outcome(
    interaction_id: str,
    outcome: str,
    correction: str | None = None,
    error_type: str | None = None,
) -> None:
    try:
        existing = _interactions_col.get(
            ids=[interaction_id], include=["documents", "metadatas", "embeddings"]
        )
        if not existing["ids"]:
            return
        meta = existing["metadatas"][0]
        meta["outcome"] = outcome
        if correction:
            meta["correction"] = correction
        if error_type:
            meta["error_type"] = error_type
        _interactions_col.update(ids=[interaction_id], metadatas=[meta])
    except Exception:
        pass


def get_recent_failures(n: int = 10) -> list[dict]:
    try:
        if _interactions_col.count() == 0:
            return []
        all_items = _interactions_col.get(include=["documents", "metadatas"])
        failures = []
        for doc, meta in zip(all_items["documents"], all_items["metadatas"]):
            if meta.get("outcome") in ("incorrect", "corrected") and meta.get(
                "correction"
            ):
                failures.append(
                    {
                        "query": meta.get("query", ""),
                        "response": doc,
                        "correction": meta.get("correction", ""),
                    }
                )
        return failures[-n:]
    except Exception:
        return []


def get_failure_count() -> int:
    try:
        if _interactions_col.count() == 0:
            return 0
        all_items = _interactions_col.get(include=["metadatas"])
        return sum(
            1
            for m in all_items["metadatas"]
            if m.get("outcome") in ("incorrect", "corrected")
        )
    except Exception:
        return 0


def get_interaction_count() -> int:
    try:
        return _interactions_col.count()
    except Exception:
        return 0


def get_confidence_stats() -> dict:
    empty = {
        "avg_confidence": 0.0,
        "avg_confidence_correct": 0.0,
        "avg_confidence_incorrect": 0.0,
        "overconfident_count": 0,
        "underconfident_count": 0,
        "total_scored": 0,
    }
    try:
        if _interactions_col.count() == 0:
            return empty
        all_items = _interactions_col.get(include=["metadatas"])
        all_conf: list[float] = []
        correct_conf: list[float] = []
        incorrect_conf: list[float] = []
        overconfident = underconfident = scored = 0

        for meta in all_items["metadatas"]:
            outcome = meta.get("outcome", "pending")
            if outcome == "pending":
                continue
            conf = _safe_float(meta.get("confidence", "0.5"))
            if conf == 0.0 and meta.get("confidence", "0.5") not in ("0.0", "0"):
                conf = 0.5
            all_conf.append(conf)
            scored += 1
            if outcome == "correct":
                correct_conf.append(conf)
                if conf < 0.4:
                    underconfident += 1
            elif outcome in ("incorrect", "corrected"):
                incorrect_conf.append(conf)
                if conf > 0.7:
                    overconfident += 1

        return {
            "avg_confidence": (
                round(sum(all_conf) / len(all_conf), 4) if all_conf else 0.0
            ),
            "avg_confidence_correct": (
                round(sum(correct_conf) / len(correct_conf), 4) if correct_conf else 0.0
            ),
            "avg_confidence_incorrect": (
                round(sum(incorrect_conf) / len(incorrect_conf), 4)
                if incorrect_conf
                else 0.0
            ),
            "overconfident_count": overconfident,
            "underconfident_count": underconfident,
            "total_scored": scored,
        }
    except Exception:
        return empty


def get_error_breakdown() -> dict:
    breakdown = {
        "factual_error": 0,
        "incomplete_answer": 0,
        "wrong_complexity": 0,
        "hallucination": 0,
    }
    try:
        if _interactions_col.count() == 0:
            return breakdown
        all_items = _interactions_col.get(include=["metadatas"])
        for meta in all_items["metadatas"]:
            if meta.get("outcome") == "corrected":
                et = meta.get("error_type", "factual_error")
                if et in breakdown:
                    breakdown[et] += 1
    except Exception:
        pass
    return breakdown
