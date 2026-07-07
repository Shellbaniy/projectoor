import unittest

from rag_bot import CourseDocument, RAGEngine, evaluate_relevance


class DummyEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([
                float("практик" in lowered),
                float("отчет" in lowered or "отчёт" in lowered),
                float("telegram" in lowered or "телеграм" in lowered),
            ])
        return vectors


class RAGEngineTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            CourseDocument(
                title="Практика УрФУ",
                source_url="https://urfu.example/practice-guide",
                content="Отчет по практике оформляется по шаблону кафедры. Срок сдачи — последняя неделя семестра.",
            ),
            CourseDocument(
                title="Telegram бот",
                source_url="https://urfu.example/tg-bot-method",
                content="Для Telegram-бота используйте webhook или long polling и журналируйте ошибки.",
            ),
        ]
        self.engine = RAGEngine(self.documents, embedder=DummyEmbedder())

    def test_answer_contains_source_citations(self):
        result = self.engine.answer("Как оформить отчет по практике?", top_k=1)
        self.assertIn("Источники:", result["answer"])
        self.assertIn("[1]", result["answer"])
        self.assertEqual(result["sources"][0], "https://urfu.example/practice-guide")

    def test_relevance_and_latency_metrics(self):
        metrics = evaluate_relevance(
            self.engine,
            [
                ("Где взять требования к отчету по практике?", "https://urfu.example/practice-guide"),
                ("Как разворачивать телеграм-бота?", "https://urfu.example/tg-bot-method"),
            ],
        )
        self.assertEqual(metrics["total_cases"], 2)
        self.assertGreaterEqual(metrics["relevance_percent"], 50.0)
        self.assertGreaterEqual(metrics["avg_latency_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
