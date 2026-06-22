"""Utilities for evaluating RAG quality on a labeled sample set.

The expected input is a JSON file containing a list of objects with at least:

    {
        "tenant_id": "...",
        "eRep_id": "...",
        "question": "...",
        "expected_answer": "...",  # optional
        "expected_context": "...",  # optional
        "expected_context_keywords": ["...", "..."]  # optional
    }

This module reports retrieval hit rate: whether at least one retrieved chunk
matches the expected context or expected keywords.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from loguru import logger


@dataclass
class RAGEvalSample:
    tenant_id: str
    eRep_id: str
    question: str
    expected_context: str | None = None
    expected_context_keywords: list[str] = field(default_factory=list)
    method: str = "basic"
    backend: str = "vectorless"


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _get_doc_content(doc: Any) -> str:
    if isinstance(doc, dict):
        return str(doc.get("content", ""))
    return str(getattr(doc, "page_content", ""))


def _matches_expected_context(retrieved_docs: Sequence[Any], sample: RAGEvalSample) -> bool:
    if sample.expected_context:
        target = _normalize_text(sample.expected_context)
        return any(target in _normalize_text(_get_doc_content(doc)) for doc in retrieved_docs)

    if sample.expected_context_keywords:
        keywords = [_normalize_text(keyword) for keyword in sample.expected_context_keywords if keyword.strip()]
        if not keywords:
            return False

        for doc in retrieved_docs:
            content = _normalize_text(_get_doc_content(doc))
            if all(keyword in content for keyword in keywords):
                return True

    return False


def _load_backend_module(backend: str):
    if backend == "vectorless":
        return importlib.import_module("rag.rag")
    if backend == "vector":
        return importlib.import_module("rag.rag2")
    raise ValueError(f"Unsupported backend: {backend}")


def evaluate_rag_samples(
    samples: Iterable[RAGEvalSample],
    *,
    num_retrieval: int = 3,
    backend: str = "vectorless",
) -> dict[str, Any]:
    """Evaluate a labeled sample set and return aggregate metrics."""

    rag_module = _load_backend_module(backend)
    fetch_relevant_chunks = rag_module.fetch_relevant_chunks

    total_samples = 0
    retrieval_hits = 0
    failures: list[dict[str, Any]] = []

    for sample in samples:
        total_samples += 1
        query = type("Query", (), {"question": sample.question, "eRep_id": sample.eRep_id, "method": sample.method})()

        try:
            retrieved_docs = fetch_relevant_chunks(
                sample.tenant_id,
                query,
                num_retrieval=num_retrieval,
            )
        except Exception as exc:
            failures.append(
                {
                    "question": sample.question,
                    "stage": "retrieval",
                    "error": str(exc),
                }
            )
            continue

        if _matches_expected_context(retrieved_docs, sample):
            retrieval_hits += 1

    metrics: dict[str, Any] = {
        "backend": backend,
        "total_samples": total_samples,
        "retrieval_hit_rate": (retrieval_hits / total_samples) if total_samples else 0.0,
        "retrieval_hits": retrieval_hits,
        "failures": failures,
    }

    return metrics


def _load_samples(file_path: Path) -> list[RAGEvalSample]:
    with file_path.open("r", encoding="utf-8") as handle:
        raw_samples = json.load(handle)

    if not isinstance(raw_samples, list):
        raise ValueError("Evaluation file must contain a JSON list of samples")

    samples: list[RAGEvalSample] = []
    for index, item in enumerate(raw_samples, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Sample at index {index} must be a JSON object")

        samples.append(
            RAGEvalSample(
                tenant_id=str(item["tenant_id"]),
                eRep_id=str(item["eRep_id"]),
                question=str(item["question"]),
                expected_context=item.get("expected_context"),
                expected_context_keywords=list(item.get("expected_context_keywords", []) or []),
                method=str(item.get("method", "basic")),
                backend=str(item.get("backend", "vectorless")),
            )
        )

    return samples


def _print_comparison(metrics_by_backend: dict[str, dict[str, Any]]) -> None:
    for backend, metrics in metrics_by_backend.items():
        print(json.dumps(metrics, indent=2, ensure_ascii=True))
        print()

    if {"vector", "vectorless"}.issubset(metrics_by_backend):
        vector_metrics = metrics_by_backend["vector"]
        vectorless_metrics = metrics_by_backend["vectorless"]
        print(
            json.dumps(
                {
                    "comparison": {
                        "retrieval_hit_rate_delta": vector_metrics.get("retrieval_hit_rate", 0.0)
                        - vectorless_metrics.get("retrieval_hit_rate", 0.0),
                    }
                },
                indent=2,
                ensure_ascii=True,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and answer accuracy")
    parser.add_argument("input", type=Path, help="Path to a JSON file with labeled evaluation samples")
    parser.add_argument("--num-retrieval", type=int, default=3, help="Number of chunks to retrieve per question")
    parser.add_argument(
        "--backend",
        choices=["vectorless", "vector", "both"],
        default="both",
        help="Which RAG backend to evaluate",
    )
    args = parser.parse_args()

    samples = _load_samples(args.input)
    if args.backend == "both":
        metrics_by_backend: dict[str, dict[str, Any]] = {}
        for backend_name in ("vectorless", "vector"):
            backend_samples = [sample for sample in samples if sample.backend in {backend_name, "both"}]
            if not backend_samples:
                continue
            metrics_by_backend[backend_name] = evaluate_rag_samples(
                backend_samples,
                num_retrieval=args.num_retrieval,
                backend=backend_name,
            )

        logger.info(json.dumps(metrics_by_backend, indent=2, ensure_ascii=True))
        _print_comparison(metrics_by_backend)
    else:
        backend_samples = [sample for sample in samples if sample.backend in {args.backend, "both"}]
        metrics = evaluate_rag_samples(
            backend_samples,
            num_retrieval=args.num_retrieval,
            backend=args.backend,
        )

        logger.info(json.dumps(metrics, indent=2, ensure_ascii=True))
        print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()