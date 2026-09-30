from functools import lru_cache

import chromadb

from src.config import settings
from src.embeddings import embed_query
from src.index_baseline import COLLECTION
from src.llm import chat

TOP_K = 5

SYSTEM_PROMPT = """You answer questions about SEC 10-K filings using ONLY the numbered context passages.
- Cite the passages you used, like [1] or [2].
- Tables are usually in millions of dollars. Always include the unit in your answer.
- If the passages do not contain the answer, reply exactly: NOT_FOUND"""


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_collection(COLLECTION)


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    res = _collection().query(query_embeddings=[embed_query(question)], n_results=k)
    return [
        {"text": doc, "company": meta["company"], "ticker": meta["ticker"]}
        for doc, meta in zip(res["documents"][0], res["metadatas"][0])
    ]


def answer(question: str) -> dict:
    hits = retrieve(question)
    context = "\n\n".join(f"[{i}] ({h['company']})\n{h['text']}" for i, h in enumerate(hits, 1))
    text = chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context passages:\n{context}\n\nQuestion: {question}"},
        ]
    )
    return {"answer": text, "contexts": [h["text"] for h in hits]}