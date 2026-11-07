"""
Correction log — RAG memory layer for Phase 3.

Stores (question, bad SQL/code, rejection reason, corrected SQL/code) records
so that future calls can retrieve similar past corrections and avoid the same
mistakes on the first generation attempt.

Storage (dual-write):
  ChromaDB  → ./chroma_db    vector-searchable by Voyage AI question embedding
  JSONL     → evals/results/correction_log.jsonl   human-readable backup
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import chromadb
    import voyageai
    CHROMADB_AVAILABLE = True
except Exception:
    CHROMADB_AVAILABLE = False

CHROMA_PATH = str(Path(__file__).parent.parent / "chroma_db")
JSONL_PATH = Path(__file__).parent.parent / "evals" / "results" / "correction_log.jsonl"
COLLECTION_NAME = "corrections"
VOYAGE_MODEL = "voyage-3"


def _voyage_client() -> "voyageai.Client":
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY is not set.")
    return voyageai.Client(api_key=api_key)


def _chroma_collection() -> "chromadb.Collection":
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def log_correction(
    question: str,
    agent_type: str,
    generated_code: str,
    error_or_reason: str,
    corrected_code: str,
) -> None:
    """Embeds question and persists the correction to ChromaDB + JSONL. No-op if chromadb unavailable."""
    if not CHROMADB_AVAILABLE:
        return
    vo = _voyage_client()
    embedding = vo.embed([question], model=VOYAGE_MODEL, input_type="document").embeddings[0]

    record_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    metadata = {
        "agent_type": agent_type,
        "generated_code": generated_code,
        "error_or_reason": error_or_reason,
        "corrected_code": corrected_code,
        "timestamp": timestamp,
    }

    _chroma_collection().add(
        ids=[record_id],
        embeddings=[embedding],
        documents=[question],
        metadatas=[metadata],
    )

    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps({"id": record_id, "question": question, **metadata}) + "\n")


def retrieve_similar_corrections(
    question: str,
    agent_type: str,
    k: int = 3,
) -> list[dict]:
    """
    Returns up to k past corrections matching agent_type, ordered by
    cosine similarity to question. Returns [] when the log is empty or chromadb unavailable.
    """
    if not CHROMADB_AVAILABLE:
        return []
    collection = _chroma_collection()
    if collection.count() == 0:
        return []

    vo = _voyage_client()
    embedding = vo.embed([question], model=VOYAGE_MODEL, input_type="query").embeddings[0]

    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(k, collection.count()),
            where={"agent_type": agent_type},
        )
    except Exception:
        return []

    if not results["ids"] or not results["ids"][0]:
        return []

    return [
        {
            "question": results["documents"][0][i],
            "generated_code": meta["generated_code"],
            "error_or_reason": meta["error_or_reason"],
            "corrected_code": meta["corrected_code"],
        }
        for i, meta in enumerate(results["metadatas"][0])
    ]
