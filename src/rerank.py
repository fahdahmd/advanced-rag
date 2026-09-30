from functools import lru_cache

from sentence_transformers import CrossEncoder

from src import hybrid
from src.config import settings
from src.rag_baseline import TOP_K, generate

POOL = 30  # candidates handed to the reranker


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    pool = hybrid.retrieve(question, k=POOL, candidates=POOL)
    scores = _reranker().predict([(question, hit["text"]) for hit in pool], batch_size=8)
    ranked = sorted(zip(scores, pool), key=lambda pair: pair[0], reverse=True)
    return [hit for _, hit in ranked[:k]]


def answer(question: str) -> dict:
    return generate(question, retrieve(question))