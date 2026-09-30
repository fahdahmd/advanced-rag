import re
from functools import lru_cache

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from src.config import settings
from src.embeddings import embed_query
from src.index_baseline import COLLECTION
from src.rag_baseline import TOP_K, generate

CANDIDATES = 20  # how many results each search hands to the merge step
RRF_K = 60  # standard RRF constant; higher = flatter scores

# Keeps numbers whole ("416,161", "10.23") and words with digits ("h20", "1a")
TOKEN = re.compile(r"\d[\d,.]*\d|[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@lru_cache(maxsize=1)
def _index() -> dict:
    """Load the baseline chunks and build the keyword index (takes a second)."""
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    collection = client.get_collection(COLLECTION)
    data = collection.get(include=["documents", "metadatas"])
    return {
        "collection": collection,
        "bm25": BM25Okapi([tokenize(d) for d in data["documents"]]),
        "ids": {cid: i for i, cid in enumerate(data["ids"])},
        "docs": data["documents"],
        "metas": data["metadatas"],
    }


def _keyword_positions(question: str, n: int) -> list[int]:
    scores = _index()["bm25"].get_scores(tokenize(question))
    return [int(i) for i in np.argsort(scores)[::-1][:n]]


def _vector_positions(question: str, n: int) -> list[int]:
    ix = _index()
    res = ix["collection"].query(query_embeddings=[embed_query(question)], n_results=n)
    return [ix["ids"][cid] for cid in res["ids"][0]]


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Merge ranked lists: each appearance earns 1/(k + rank) points."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, pos in enumerate(ranking, start=1):
            scores[pos] = scores.get(pos, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def _hits(positions: list[int], k: int) -> list[dict]:
    ix = _index()
    return [
        {"text": ix["docs"][i], "company": ix["metas"][i]["company"], "ticker": ix["metas"][i]["ticker"]}
        for i in positions[:k]
    ]


def retrieve_keyword(question: str, k: int = TOP_K) -> list[dict]:
    return _hits(_keyword_positions(question, k), k)


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    fused = rrf([_vector_positions(question, CANDIDATES), _keyword_positions(question, CANDIDATES)])
    return _hits(fused, k)


def answer(question: str) -> dict:
    return generate(question, retrieve(question))


def answer_keyword(question: str) -> dict:
    return generate(question, retrieve_keyword(question))