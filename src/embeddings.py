from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import settings

# bge models work better when queries (not documents) get this prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_documents(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    return _model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 100
    ).tolist()


def embed_query(text: str) -> list[float]:
    return _model().encode(QUERY_PREFIX + text, normalize_embeddings=True).tolist()