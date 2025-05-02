import logging
import pickle
import numpy as np
import faiss
from abc import ABC, abstractmethod
from typing import List, Optional

class BaseVectorStore(ABC):
    @abstractmethod
    def store(self, texts: List[str], embeddings: List[List[float]]):
        pass
    
    @abstractmethod
    def query(self, query_embedding: List[float], top_k: int, use_mmr: bool = True) -> List[str]:
        pass
    
    @abstractmethod
    def save(self, index_path: str, texts_path: str):
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, embedding_model, index_path: str, texts_path: str):
        pass

class FAISSVectorStore(BaseVectorStore):
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.texts = []
        self.dimension = None
        logging.info("FAISSVectorStore initialized.")
    
    def store(self, texts: List[str], embeddings: List[List[float]]):
        self.texts = texts
        embeddings_np = np.array(embeddings, dtype=np.float32)
        self.dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_np)
        logging.info(f"Stored {len(texts)} texts and embeddings in FAISS index.")
    
    def query(self, query_embedding: List[float], top_k: int, use_mmr: bool = True) -> List[str]:
        if not self.index:
            logging.error("FAISS index is not initialized.")
            return []
        
        # Convert query embedding to numpy array
        query_np = np.array([query_embedding], dtype=np.float32)
        
        # Check dimensions
        query_dim = query_np.shape[1]
        if query_dim != self.dimension:
            logging.error(f"Dimension mismatch: Query embedding has dimension {query_dim}, but FAISS index expects {self.dimension}.")
            raise ValueError(f"Dimension mismatch: Query embedding has dimension {query_dim}, but FAISS index expects {self.dimension}.")
        
        # Perform search
        if use_mmr:
            # Fetch more candidates for MMR
            _, indices = self.index.search(query_np, top_k * 2)
            indices = indices[0]
            
            # Apply MMR (simplified version for diversity)
            selected_indices = []
            query_embedding_np = np.array(query_embedding, dtype=np.float32)
            
            for idx in indices:
                if len(selected_indices) >= top_k:
                    break
                if idx in selected_indices:
                    continue
                
                # Compute similarity with already selected embeddings
                selected_embeddings = np.array([self.index.reconstruct(int(i)) for i in selected_indices], dtype=np.float32)
                if selected_embeddings.size == 0:
                    selected_indices.append(idx)
                    continue
                
                # Compute MMR score: balance relevance and diversity
                sim_to_query = np.dot(query_embedding_np, self.index.reconstruct(int(idx))) / (
                    np.linalg.norm(query_embedding_np) * np.linalg.norm(self.index.reconstruct(int(idx)))
                )
                sim_to_selected = np.max(np.dot(selected_embeddings, self.index.reconstruct(int(idx))) / (
                    np.linalg.norm(selected_embeddings, axis=1) * np.linalg.norm(self.index.reconstruct(int(idx)))
                )) if selected_embeddings.size > 0 else 0
                mmr_score = 0.7 * sim_to_query - 0.3 * sim_to_selected
                
                if mmr_score > 0:
                    selected_indices.append(idx)
        else:
            _, indices = self.index.search(query_np, top_k)
            selected_indices = indices[0]
        
        # Retrieve texts for the selected indices
        results = []
        for idx in selected_indices:
            if idx < len(self.texts):
                results.append(self.texts[idx])
        
        logging.info(f"Retrieved {len(results)} contexts for query.")
        return results
    
    def save(self, index_path: str, texts_path: str):
        if self.index:
            faiss.write_index(self.index, index_path)
            with open(texts_path, "wb") as f:
                pickle.dump(self.texts, f)
            logging.info(f"Saved FAISS index to {index_path} and texts to {texts_path}.")
        else:
            logging.warning("No FAISS index to save.")
    
    @classmethod
    def load(cls, embedding_model, index_path: str, texts_path: str):
        vector_store = cls(embedding_model)
        vector_store.index = faiss.read_index(index_path)
        with open(texts_path, "rb") as f:
            vector_store.texts = pickle.load(f)
        vector_store.dimension = vector_store.index.d
        logging.info(f"Loaded FAISS index from {index_path} and texts from {texts_path}.")
        return vector_store