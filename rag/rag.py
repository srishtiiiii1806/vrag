import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from loguru import logger

from rag.bm25_retriever import load_bm25_retriever


AGENTS_DIR = Path(os.environ.get("VRAG_AGENT_DIR", Path(__file__).resolve().parent.parent / "agents"))


@lru_cache(maxsize=32)
def get_cached_bm25(db_path: str, k: int):
    return load_bm25_retriever(db_path=db_path, k=k)


def _chunk_store_path(tenant_id: str, eRep_id: str) -> Path:
    return AGENTS_DIR / tenant_id / eRep_id / "chunks"


def _load_chunk_preview(chunks_path: Path) -> list[dict[str, Any]]:
    chunk_file = chunks_path / "chunks.json"
    if not chunk_file.exists():
        raise HTTPException(status_code=404, detail=f"Chunk store not found at {chunk_file}")
    with chunk_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fetch_relevant_chunks(tenant_id: str, query: Any, num_retrieval: int = 3):
    """Vectorless retrieval backed by the local chunks.json store."""

    try:
        chunks_path = _chunk_store_path(tenant_id, query.eRep_id)
        if not chunks_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Chunk store not found at {chunks_path}"
            )

        retriever = get_cached_bm25(db_path=str(chunks_path), k=num_retrieval)
        relevant_docs = retriever.invoke(query.question)
        logger.info(relevant_docs)
        return [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in relevant_docs
        ]

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))