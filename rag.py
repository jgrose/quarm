"""
rag.py — Local RAG with Qdrant + e5-base-v2 embeddings.
Provides vector search across agent artifacts, task outputs, and web content.
Persists across runs so future projects benefit from past knowledge.
"""

import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

# Suppress noisy HTTP logs from HuggingFace/sentence-transformers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

log = logging.getLogger("quarm.rag")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "quarm_artifacts"
EMBED_MODEL = "intfloat/e5-base-v2"
EMBED_DIM = 768
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Lazy-loaded singletons ───────────────────────────────────────────────────

_client: Optional[QdrantClient] = None
_embedder = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
        # Create collection if it doesn't exist
        collections = [c.name for c in _client.get_collections().collections]
        if COLLECTION not in collections:
            _client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
            log.info(f"Created Qdrant collection: {COLLECTION}")
    return _client


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
        log.info(f"Loaded embedding model: {EMBED_MODEL}")
    return _embedder


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Prepends 'query: ' for e5 models."""
    model = _get_embedder()
    # e5 models expect 'query: ' prefix for queries and 'passage: ' for documents
    return model.encode(texts, normalize_embeddings=True).tolist()


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── Public API ───────────────────────────────────────────────────────────────

def embed_and_store(text: str, metadata: dict) -> int:
    """Chunk text, embed, and upsert to Qdrant. Returns number of chunks stored."""
    if not text.strip():
        return 0
    client = _get_client()
    chunks = _chunk_text(text)
    # Prepend 'passage: ' for e5 document embedding
    passages = [f"passage: {c}" for c in chunks]
    vectors = _embed(passages)
    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        point_id = str(uuid.uuid4())
        payload = {
            **metadata,
            "text": chunk,
            "chunk_index": i,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        points.append(PointStruct(id=point_id, vector=vec, payload=payload))
    client.upsert(collection_name=COLLECTION, points=points)
    return len(points)


def search(query: str, top_k: int = 5, filters: Optional[dict] = None) -> list[dict]:
    """Search the knowledge base. Returns list of {text, score, metadata}."""
    client = _get_client()
    # Prepend 'query: ' for e5 query embedding
    vec = _embed([f"query: {query}"])[0]
    qfilter = None
    if filters:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        qfilter = Filter(must=conditions)
    results = client.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=top_k,
        query_filter=qfilter,
        with_payload=True,
    )
    return [
        {
            "text": r.payload.get("text", ""),
            "score": r.score,
            "source": r.payload.get("source", ""),
            "content_type": r.payload.get("content_type", ""),
            "agent": r.payload.get("agent", ""),
            "plan_id": r.payload.get("plan_id", ""),
            "task_id": r.payload.get("task_id", ""),
        }
        for r in results.points
    ]


def ingest_text(text: str, source: str, content_type: str = "output",
                plan_id: str = "", task_id: str = "", agent: str = "",
                tags: Optional[list[str]] = None) -> int:
    """Convenience wrapper: ingest text with standard metadata."""
    metadata = {
        "source": source,
        "content_type": content_type,
        "plan_id": plan_id,
        "task_id": task_id,
        "agent": agent,
        "tags": tags or [],
    }
    return embed_and_store(text, metadata)


def ingest_url(url: str, content: str, plan_id: str = "", task_id: str = "",
               agent: str = "") -> int:
    """Ingest fetched web content into RAG."""
    return ingest_text(content, source=url, content_type="web",
                       plan_id=plan_id, task_id=task_id, agent=agent)


def get_stats() -> dict:
    """Get collection stats for the dashboard."""
    try:
        client = _get_client()
        info = client.get_collection(COLLECTION)
        return {
            "total_points": info.points_count,
            "status": str(info.status),
        }
    except Exception as e:
        return {"total_points": 0, "vectors_count": 0, "status": f"error: {e}"}
