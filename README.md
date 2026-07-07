# projectoor

RAG-чатбот для Telegram: отвечает на вопросы по документам курса УрФУ и методичкам практики.

## Что реализовано
- Индексация документов курса (`CourseDocument`)
- Embeddings через `sentence-transformers` (с безопасным fallback для локального теста)
- Векторный поиск по embeddings
- RAG-ответ с обязательными ссылками на источники
- Метрики для резюме:
  - `% релевантных ответов` на тестовом наборе вопросов
  - `avg latency` (среднее время ответа, мс)

## Ключевой стек
- Python
- sentence-transformers
- RAG pipeline (retrieval + citation-aware answer generation)

## Быстрый пример
```python
from rag_bot import CourseDocument, RAGEngine, evaluate_relevance

docs = [
    CourseDocument(
        title="Практика УрФУ",
        source_url="https://urfu.example/practice-guide",
        content="Отчет по практике оформляется по шаблону кафедры.",
    )
]

engine = RAGEngine(docs)
print(engine.answer("Как оформить отчет по практике?"))

metrics = evaluate_relevance(
    engine,
    [("Где требования к отчету?", "https://urfu.example/practice-guide")],
)
print(metrics)
```

## Для резюме (готовая формулировка)
Собрал Telegram RAG-чатбота для учебных материалов УрФУ: построил пайплайн загрузки документов, embeddings (sentence-transformers), векторный поиск и генерацию ответов с цитированием источников; добавил оценку качества (% релевантных ответов) и времени ответа.
