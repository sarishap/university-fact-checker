"""
vectorstore.py
--------------
Wraps ChromaDB so the rest of the application never has to touch
ChromaDB's API directly. Embeddings are generated with the free,
open-source HuggingFace model 'all-MiniLM-L6-v2' (no paid API keys
required), consistent with the rest of this project.
"""

import chromadb
from chromadb.utils import embedding_functions

from chunker import Chunk

COLLECTION_NAME = "university_pages"


class ChromaVectorStore:
    def __init__(self, persist_directory: str = "university_vectorstore", model_name: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},  # use cosine similarity
        )

    def reset(self):
        """Wipe the collection before re-indexing a new site."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]):
        """Embed and store chunks, batching to keep memory usage reasonable."""
        if not chunks:
            return

        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            self.collection.add(
                ids=[f"chunk-{start + i}" for i in range(len(batch))],
                documents=[c.text for c in batch],
                metadatas=[{"source_url": c.source_url, "source_title": c.source_title} for c in batch],
            )

    def query(self, question: str, top_k: int = 3):
        """
        Returns a list of dicts: {text, source_url, source_title, similarity}
        'similarity' is cosine similarity in [0, 1], where 1 = identical.
        """
        results = self.collection.query(query_texts=[question], n_results=top_k)

        if not results["documents"] or not results["documents"][0]:
            return []

        output = []
        for doc, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            # ChromaDB returns cosine *distance* (1 - similarity) when using cosine space.
            similarity = 1 - distance
            output.append({
                "text": doc,
                "source_url": metadata["source_url"],
                "source_title": metadata["source_title"],
                "similarity": similarity,
            })

        return output

    def count(self) -> int:
        return self.collection.count()
