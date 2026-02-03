"""Reusable components for HyperGraph demo notebooks."""

from __future__ import annotations

import numpy as np
from openai import AsyncOpenAI, OpenAI


class LLM:
    """Thin wrapper around OpenAI chat completions."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.client = AsyncOpenAI()
        self.model = model
        self.temperature = temperature

    async def generate(self, messages: list[dict]) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""

    async def answer_with_context(self, query: str, documents: list[str]) -> str:
        context = "\n".join(f"- {d}" for d in documents)
        return await self.generate([
            {"role": "system", "content": "Answer based ONLY on the provided context. Be concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ])

    async def judge(self, query: str, expected: str, actual: str) -> str:
        return await self.generate([
            {
                "role": "system",
                "content": (
                    "You are a judge. Given a question, expected answer, and actual answer, "
                    "rate the actual answer from 1-5 and explain why. "
                    "Respond in the format: SCORE: X\nREASON: ..."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\nExpected: {expected}\nActual: {actual}",
            },
        ])


class Embedder:
    """Wraps OpenAI embeddings API."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(input=texts, model=self.model)
        return [d.embedding for d in resp.data]


class VectorStore:
    """In-memory vector store using cosine similarity."""

    def __init__(self, documents: list[str], embedder: Embedder):
        self.documents = documents
        self.embeddings = embedder.embed_batch(documents)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        scores = [_cosine_similarity(query_embedding, de) for de in self.embeddings]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [{"text": self.documents[i], "score": scores[i]} for i in ranked[:top_k]]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

DOCUMENTS = [
    "HyperGraph is a Python framework for building AI/ML workflows using explicit graphs.",
    "HyperGraph supports hierarchy — graphs that contain graphs — for managing complexity.",
    "The .map() feature lets you write code for a single item and scale to many automatically.",
    "HyperGraph unifies DAGs and cycles in one framework, enabling both pipelines and agents.",
    "Nodes in HyperGraph are pure functions. Edges are inferred from matching output→input names.",
    "Python was created by Guido van Rossum and first released in 1991.",
    "The capital of France is Paris.",
]