import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from fastapi import HTTPException
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from loguru import logger


AGENTS_DIR = Path(os.environ.get("VRAG_AGENT_DIR", Path(__file__).resolve().parent.parent / "agents"))
HASH_DIMENSION = 256


class HashEmbeddings(Embeddings):
    def _vectorize(self, text: str) -> list[float]:
        vector = [0.0] * HASH_DIMENSION
        tokens = [token for token in text.lower().split() if token]
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % HASH_DIMENSION
            vector[index] += 1.0

        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)


@lru_cache(maxsize=32)
def _load_documents(db_path: str) -> list[Document]:
    chunk_file = Path(db_path) / "chunks.json"
    if not chunk_file.exists():
        raise HTTPException(status_code=404, detail=f"Chunk store not found at {chunk_file}")

    with chunk_file.open("r", encoding="utf-8") as handle:
        chunk_data = json.load(handle)

    if not chunk_data:
        raise HTTPException(status_code=404, detail=f"No chunks found at {chunk_file}")

    return [
        Document(page_content=item["content"], metadata=item.get("metadata", {}))
        for item in chunk_data
    ]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(l * r for l, r in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return dot_product / (left_norm * right_norm)


def fetch_relevant_chunks(tenant_id: str, query: Any, num_retrieval: int = 3):
    """Vector retrieval backed by local chunks.json and hash embeddings."""

    try:
        chunks_path = AGENTS_DIR / tenant_id / query.eRep_id / "chunks"
        if not chunks_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Chunk store not found at {chunks_path}"
            )

        documents = _load_documents(str(chunks_path))
        embeddings = HashEmbeddings()
        query_vector = embeddings.embed_query(query.question)
        scored_documents = [
            (_cosine_similarity(query_vector, embeddings.embed_query(doc.page_content)), doc)
            for doc in documents
        ]
        scored_documents.sort(key=lambda item: item[0], reverse=True)
        relevant_docs = [doc for _, doc in scored_documents[:num_retrieval]]
        logger.info(relevant_docs)
        return [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in relevant_docs
        ]

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))