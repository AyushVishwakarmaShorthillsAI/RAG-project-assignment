import logging
from functools import lru_cache
from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    from .vector_store import BaseVectorStore
    from .llm import BaseLLM
except ImportError:
    from vector_store import BaseVectorStore
    from llm import BaseLLM

# Configure logging
logging.basicConfig(filename='rag_project.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class RAGPipeline:
    """RAG pipeline with caching and optimized processing."""
    def __init__(self, vector_store: BaseVectorStore, llm: BaseLLM, embedding_model: SentenceTransformer):
        self.vector_store = vector_store
        self.llm = llm
        self.embedding_model = embedding_model
        self.query_cache: Dict[str, str] = {}  # Cache for query results
    
    @lru_cache(maxsize=1000)
    def process(self, question: str) -> str:
        if question in self.query_cache:
            logging.info(f"Cache hit for question: {question}")
            return self.query_cache[question]
        
        try:
            query_embedding = self.embedding_model.encode(question, show_progress_bar=False).tolist()
            retrieved_docs = self.vector_store.query(query_embedding, top_k=10)
            query_vec = np.array([query_embedding], dtype=np.float32)
            doc_embeddings = self.embedding_model.encode(retrieved_docs, show_progress_bar=False)
            similarities = np.dot(doc_embeddings, query_vec.T).flatten()
            top_indices = [i for i, sim in enumerate(similarities) if sim > 0.6][-3:][::-1]
            if not top_indices:
                top_indices = similarities.argsort()[-3:][::-1]  # Fallback if no high similarity
            relevant_docs = [retrieved_docs[i] for i in top_indices]
            # Use the full top document instead of trimming to first sentence
            context = relevant_docs[0] if relevant_docs else "No relevant context found."
            logging.info(f"Retrieved context for {question}: {relevant_docs}")
            prompt = f"Using the context, Answer the question directly and do not suggest any other questions or clarifications. Do not repeat the context.\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
            answer = self.llm.generate(prompt)
            self.query_cache[question] = answer
            logging.info(f"Question: {question}\nAnswer: {answer}")
            return answer
        except Exception as e:
            logging.error(f"Error processing question: {str(e)}")
            return "Error processing question."