import abc
import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import pickle
import os

script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)

# === Configure Logging ===
logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(logs_dir, "rag_project.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === Vector Store Implementation ===
class BaseVectorStore(abc.ABC):
    """Abstract base class for vector stores."""
    @abc.abstractmethod
    def store(self, texts: List[str], embeddings: List[List[float]]):
        pass

    @abc.abstractmethod
    def query(self, query_embedding: List[float], top_k: int) -> List[str]:
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

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        if self.index.ntotal == 0:
            logging.warning("FAISS index is empty. No vectors to search.")
            return []

        query_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_np)

        _, indices = self.index.search(query_np, top_k)
        valid_indices = [i for i in indices[0] if 0 <= i < len(self.texts)]

        if not valid_indices:
            logging.warning("No valid indices found in FAISS search result.")
            return []

        selected_texts = [self.texts[i] for i in valid_indices]
        return selected_texts

    def save(self, index_path: str = "faiss_index.bin", texts_path: str = "texts.pkl", emb_path: str = "embeddings.pkl"):
        full_index_path = index_path
        full_texts_path = texts_path
        full_emb_path = emb_path

        faiss.write_index(self.index, full_index_path)
        with open(full_texts_path, 'wb') as f:
            pickle.dump(self.texts, f)
        with open(full_emb_path, 'wb') as f:
            pickle.dump(self.embeddings, f)

        logging.info(f"Saved FAISS index to {full_index_path}, texts to {full_texts_path}, and embeddings to {full_emb_path}")

    @classmethod
    def load(cls, embedding_model: SentenceTransformer,
             index_path: str = "faiss_index.bin",
             texts_path: str = "texts.pkl",
             emb_path: str = "embeddings.pkl"):

        if not (os.path.exists(index_path) and os.path.exists(texts_path)):
            raise FileNotFoundError(f"FAISS index or texts file not found. Checked:\n{index_path}\n{texts_path}")

        index = faiss.read_index(index_path)
        with open(texts_path, 'rb') as f:
            texts = pickle.load(f)

        instance = cls(embedding_model)
        instance.index = index
        instance.texts = texts

        if os.path.exists(emb_path):
            with open(emb_path, 'rb') as f:
                instance.embeddings = pickle.load(f)
            logging.info("Loaded embeddings.")
        else:
            instance.embeddings = []
            logging.warning("Embeddings file not found.")

        logging.info(f"Loaded FAISS index from {index_path} and texts from {texts_path}")
        return instance