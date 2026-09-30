import chromadb
from tqdm import tqdm

from src.chunking import chunk_text
from src.config import settings
from src.corpus import full_text, load_filings
from src.embeddings import embed_documents

COLLECTION = "baseline"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH = 128


def main():
    ids, docs, metas = [], [], []
    for ticker, filing in load_filings().items():
        for i, (start, chunk) in enumerate(chunk_text(full_text(filing), CHUNK_SIZE, CHUNK_OVERLAP)):
            ids.append(f"{ticker}-{i}")
            docs.append(chunk)
            metas.append({"ticker": ticker, "company": filing["company"], "start": start})
    print(f"{len(docs)} chunks to embed")

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        client.delete_collection(COLLECTION)  # rebuild from scratch each run
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for i in tqdm(range(0, len(docs), BATCH), desc="indexing"):
        batch = docs[i : i + BATCH]
        collection.add(
            ids=ids[i : i + BATCH],
            documents=batch,
            embeddings=embed_documents(batch),
            metadatas=metas[i : i + BATCH],
        )
    print(f"Done: {collection.count()} chunks in '{COLLECTION}'")


if __name__ == "__main__":
    main()