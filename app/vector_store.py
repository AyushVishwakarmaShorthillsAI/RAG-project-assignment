import abc
import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import pickle
import os

# Ensure the logs directory exists
logs_dir = 'logs'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Configure logging
logging.basicConfig(filename=os.path.join(logs_dir, 'rag_project.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class BaseVectorStore(abc.ABC):
    """Abstract base class for vector stores."""
    @abc.abstractmethod
    def store(self, texts: List[str], embeddings: List[List[float]]):
        pass

    @abc.abstractmethod
    def query(self, query_embedding: List[float], top_k: int, use_mmr: bool) -> List[str]:
        pass

class FAISSVectorStore(BaseVectorStore):
    """Concrete vector store using FAISS for faster retrieval."""
    def __init__(self, embedding_model: SentenceTransformer, dimension: int = 384):
        logging.info("Initializing FAISSVectorStore")
        self.embedding_model = embedding_model
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
        self.embeddings = []

    def store(self, texts: List[str], embeddings: List[List[float]]):
        if not texts:
            logging.warning("No texts provided for storing.")
            return
        embeddings_np = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_np)
        self.index.add(embeddings_np)
        self.texts.extend(texts)
        self.embeddings.extend(embeddings_np.tolist())
        logging.info(f"Stored {len(texts)} documents in FAISS")

    def query(self, query_embedding: List[float], top_k: int = 5, use_mmr: bool = False) -> List[str]:
        query_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_np)

        _, indices = self.index.search(query_np, top_k * 2 if use_mmr else top_k)
        selected_texts = [self.texts[i] for i in indices[0]]

        if use_mmr:
            selected_embeddings = [self.embeddings[i] for i in indices[0]]
            return self._mmr(query_embedding, selected_embeddings, selected_texts, top_k)

        return selected_texts

    def _mmr(self, query_embedding, doc_embeddings, texts, k=5, lambda_param=0.5) -> List[str]:
        selected = []
        selected_indices = []
        # Use pre-stored embeddings for MMR
        similarity_to_query = np.dot(doc_embeddings, np.array(query_embedding).reshape(-1, 1)).flatten()

        while len(selected) < k and len(selected) < len(texts):
            scores = []
            for i in range(len(texts)):
                if i in selected_indices:
                    continue
                relevance = similarity_to_query[i]
                diversity = max(
                    np.dot(doc_embeddings[i], doc_embeddings[j])
                    for j in selected_indices
                ) if selected_indices else 0
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
                scores.append((mmr_score, i))

            scores.sort(reverse=True)
            _, best_idx = scores[0]
            selected.append(texts[best_idx])
            selected_indices.append(best_idx)

        return selected

    def save(self, index_path: str = "faiss_index.bin", texts_path: str = "texts.pkl", emb_path: str = "embeddings.pkl"):
        faiss.write_index(self.index, index_path)
        with open(texts_path, 'wb') as f:
            pickle.dump(self.texts, f)
        with open(emb_path, 'wb') as f:
            pickle.dump(self.embeddings, f)
        logging.info(f"Saved FAISS index to {index_path}, texts to {texts_path}, and embeddings to {emb_path}")

    @classmethod
    def load(cls, embedding_model: SentenceTransformer,
             index_path: str = "faiss_index.bin",
             texts_path: str = "texts.pkl",
             emb_path: str = "embeddings.pkl"):
        if not (os.path.exists(index_path) and os.path.exists(texts_path)):
            raise FileNotFoundError("FAISS index or texts file not found")

        index = faiss.read_index(index_path)
        with open(texts_path, 'rb') as f:
            texts = pickle.load(f)

        instance = cls(embedding_model)
        instance.index = index
        instance.texts = texts

        if os.path.exists(emb_path):
            with open(emb_path, 'rb') as f:
                instance.embeddings = pickle.load(f)
            logging.info("Loaded embeddings for MMR retrieval.")
        else:
            instance.embeddings = []
            logging.warning("Embeddings file not found. MMR will not be available unless re-stored.")

        logging.info(f"Loaded FAISS index from {index_path} and texts from {texts_path}")
        return instance
