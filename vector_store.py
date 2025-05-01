import pickle
import faiss
import logging
import numpy as np

class FAISSVectorStore:
    """Vector store using FAISS for efficient similarity search."""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.dimension = 384  # Dimension for 'all-MiniLM-L6-v2'
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        logging.info("FAISSVector Ferrari initialized.")
    
    def store(self, texts: list, embeddings: list):
        """Store texts and their embeddings in the vector store."""
        if not texts or not embeddings or len(texts) != len(embeddings):
            logging.warning("No texts or embeddings provided, or mismatch in lengths.")
            return
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings_array)
        self.texts.extend(texts)
        logging.info(f"Stored {len(texts)} texts and embeddings.")
    
    def query(self, query_embedding: list, top_k: int = 5) -> list:
        """Query the vector store for the top_k most similar texts."""
        if self.index.ntotal == 0:
            logging.warning("Vector store index is empty. Cannot query.")
            return []
        
        query_array = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_array, top_k)
        
        # Filter out invalid indices
        valid_indices = [i for i in indices[0] if i >= 0 and i < len(self.texts)]
        if not valid_indices:
            logging.warning("No valid indices returned from FAISS search.")
            return []
        
        selected_texts = [self.texts[i] for i in valid_indices]
        logging.info(f"Queried vector store: Retrieved {len(selected_texts)} contexts.")
        return selected_texts
    
    def save(self, index_path: str, texts_path: str):
        """Save the FAISS index and texts to disk."""
        faiss.write_index(self.index, index_path)
        with open(texts_path, 'wb') as f:
            pickle.dump(self.texts, f)
        logging.info(f"Saved FAISS index to {index_path} and texts to {texts_path}.")
    
    @staticmethod
    def load(embedding_model, index_path: str, texts_path: str):
        """Load a FAISSVectorStore from disk."""
        vector_store = FAISSVectorStore(embedding_model)
        vector_store.index = faiss.read_index(index_path)
        with open(texts_path, 'rb') as f:
            vector_store.texts = pickle.load(f)
        logging.info(f"Loaded FAISS index from {index_path} and texts from {texts_path}.")
        return vector_store