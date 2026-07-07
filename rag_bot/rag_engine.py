from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable, List, Sequence


@dataclass(frozen=True)
class CourseDocument:
    title: str
    source_url: str
    content: str


@dataclass(frozen=True)
class RetrievalHit:
    document: CourseDocument
    score: float


class Embedder:
    """Embedding adapter.

    Uses sentence-transformers when available and falls back to a deterministic
    in-process embedder for local/dev testing.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self._encoder: Callable[[Sequence[str]], List[List[float]]] | None = None
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)

            def encode(texts: Sequence[str]) -> List[List[float]]:
                vectors = model.encode(list(texts), normalize_embeddings=True)
                return vectors.tolist()

            self._encoder = encode
        except Exception:
            self._encoder = self._fallback_encode

    @staticmethod
    def _fallback_encode(texts: Sequence[str]) -> List[List[float]]:
        vocab = ["практика", "отчет", "telegram", "rag", "llm", "урфу"]
        vectors: List[List[float]] = []
        for text in texts:
            lowered = text.lower()
            vector = [float(lowered.count(token)) for token in vocab]
            norm = sum(v * v for v in vector) ** 0.5
            vectors.append([v / norm if norm else 0.0 for v in vector])
        return vectors

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        return self._encoder(texts) if self._encoder else self._fallback_encode(texts)


class RAGEngine:
    """Minimal RAG pipeline for Telegram bot backend.

    Pipeline: docs -> embeddings -> vector search -> citation-aware answer.
    """

    def __init__(self, documents: Sequence[CourseDocument], embedder: Embedder | None = None) -> None:
        if not documents:
            raise ValueError("At least one document is required")
        self._documents = list(documents)
        self._embedder = embedder or Embedder()
        self._doc_vectors = self._embedder.encode([d.content for d in self._documents])

    @staticmethod
    def _dot(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def retrieve(self, question: str, top_k: int = 3) -> List[RetrievalHit]:
        query_vector = self._embedder.encode([question])[0]
        scored = [
            RetrievalHit(document=doc, score=self._dot(query_vector, vector))
            for doc, vector in zip(self._documents, self._doc_vectors)
        ]
        return sorted(scored, key=lambda hit: hit.score, reverse=True)[:top_k]

    def answer(self, question: str, top_k: int = 3) -> dict:
        start = perf_counter()
        hits = self.retrieve(question, top_k=top_k)
        bullets = []
        for index, hit in enumerate(hits, start=1):
            snippet = hit.document.content.strip().split(".")[0].strip()
            bullets.append(f"[{index}] {snippet}")

        sources = [hit.document.source_url for hit in hits]
        answer = "\n".join(
            [
                "Краткий ответ по материалам курса:",
                *bullets,
                "",
                "Источники:",
                *[f"[{i}] {url}" for i, url in enumerate(sources, start=1)],
            ]
        )
        elapsed_ms = (perf_counter() - start) * 1000
        return {
            "answer": answer,
            "sources": sources,
            "latency_ms": round(elapsed_ms, 2),
        }


def evaluate_relevance(engine: RAGEngine, test_questions: Iterable[tuple[str, str]], top_k: int = 1) -> dict:
    cases = list(test_questions)
    if not cases:
        return {"relevance_percent": 0.0, "avg_latency_ms": 0.0, "total_cases": 0}

    relevant = 0
    latencies: List[float] = []
    for question, expected_source in cases:
        result = engine.answer(question, top_k=top_k)
        latencies.append(float(result["latency_ms"]))
        if expected_source in result["sources"]:
            relevant += 1

    relevance_percent = (relevant / len(cases)) * 100
    avg_latency = sum(latencies) / len(latencies)
    return {
        "relevance_percent": round(relevance_percent, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "total_cases": len(cases),
    }
