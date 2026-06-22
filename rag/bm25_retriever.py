import json
import os

from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever


def load_bm25_retriever(
    db_path,
    k=3,
):

    chunk_file = os.path.join(
        db_path,
        "chunks.json",
    )


    with open(
        chunk_file,
        "r",
        encoding="utf-8",
    ) as f:
        chunk_data = json.load(f)

    if not chunk_data:
        raise ValueError(f"No chunks found in {chunk_file}")

    docs = [
        Document(
            page_content=item["content"],
            metadata=item.get("metadata", {}),
        )
        for item in chunk_data
    ]

    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k

    return retriever