import uuid
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

_DB_PATH = Path(__file__).parent / "memory_agent_db"
_client = chromadb.PersistentClient(path=str(_DB_PATH))
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

_lessons_col = _client.get_or_create_collection("lessons")
_interactions_col = _client.get_or_create_collection("interactions")


def _embed(text: str) -> list[float]:
    return _embedder.encode(text).tolist()


def store_lesson(content: str, error_type: str) -> str:
    lesson_id = str(uuid.uuid4())
    try:
        _lessons_col.add(
            ids=[lesson_id],
            embeddings=[_embed(content)],
            documents=[content],
            metadatas=[{"error_type": error_type}],
        )
    except Exception:
        pass
    return lesson_id


def retrieve_lessons(query: str, n: int = 3) -> list[str]:
    try:
        if _lessons_col.count() == 0:
            return []
        results = _lessons_col.query(
            query_embeddings=[_embed(query)],
            n_results=min(n, _lessons_col.count()),
        )
        return results["documents"][0] if results["documents"] else []
    except Exception:
        return []


def store_interaction(
    query: str,
    response: str,
    outcome: str,
    correction: str | None = None,
) -> str:
    interaction_id = str(uuid.uuid4())
    try:
        meta: dict = {"outcome": outcome, "query": query}
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


def get_recent_failures(n: int = 10) -> list[dict]:
    try:
        if _interactions_col.count() == 0:
            return []
        all_items = _interactions_col.get(include=["documents", "metadatas"])
        failures = []
        for doc, meta in zip(all_items["documents"], all_items["metadatas"]):
            if meta.get("outcome") in ("incorrect", "corrected") and meta.get("correction"):
                failures.append({
                    "query": meta.get("query", ""),
                    "response": doc,
                    "correction": meta.get("correction", ""),
                })
        return failures[-n:]
    except Exception:
        return []


def get_all_lessons() -> list[dict]:
    try:
        if _lessons_col.count() == 0:
            return []
        all_items = _lessons_col.get(include=["documents", "metadatas"])
        return [
            {"content": doc, "error_type": meta.get("error_type", "general")}
            for doc, meta in zip(all_items["documents"], all_items["metadatas"])
        ]
    except Exception:
        return []


def get_interaction_count() -> int:
    try:
        return _interactions_col.count()
    except Exception:
        return 0


def get_lesson_count() -> int:
    try:
        return _lessons_col.count()
    except Exception:
        return 0


def update_interaction_outcome(interaction_id: str, outcome: str, correction: str | None = None) -> None:
    try:
        existing = _interactions_col.get(ids=[interaction_id], include=["documents", "metadatas", "embeddings"])
        if not existing["ids"]:
            return
        meta = existing["metadatas"][0]
        meta["outcome"] = outcome
        if correction:
            meta["correction"] = correction
        _interactions_col.update(
            ids=[interaction_id],
            metadatas=[meta],
        )
    except Exception:
        pass
