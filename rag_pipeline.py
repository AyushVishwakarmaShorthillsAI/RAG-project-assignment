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
    def __init__(self, vector_store: BaseVectorStore, llm: BaseLLM, embedding_model: SentenceTransformer):
        self.vector_store = vector_store
        self.llm = llm
        self.embedding_model = embedding_model
        self.query_cache: Dict[str, str] = {}
        self.prev_contexts: Dict[str, List[str]] = {}

    @lru_cache(maxsize=1000)
    def process(self, question: str) -> str:
        if question in self.query_cache:
            logging.info(f"Cache hit for question: {question}")
            return self.query_cache[question]

        try:
            # Embed the query to get its vector representation
            query_embedding = self.embedding_model.encode(question, show_progress_bar=False, normalize_embeddings=True).tolist()
            
            # Retrieve the top 5 documents from the vector store
            retrieved_docs = self.vector_store.query(query_embedding, top_k=5)

            if not retrieved_docs:
                logging.warning("No documents retrieved. Returning fallback response.")
                return "No relevant documents found for the question."

            # Check if the retrieved documents are the same as the previous context
            previous_docs = list(self.prev_contexts.values())[-1] if self.prev_contexts else []
            is_same_context = (retrieved_docs == previous_docs)
            logging.info(f"Context same as previous: {is_same_context}")

            # Update the previous contexts for the current question
            self.prev_contexts[question] = retrieved_docs

            # Use all retrieved documents as context
            context = "\n\n".join(retrieved_docs) if retrieved_docs else "No relevant context found."

            # Log the full context with the question
            logging.info(f"Question: {question}\nFull Retrieved Context:\n{context}")

            # Enhanced prompt to enforce accuracy and conciseness
            prompt = (
                "You are a knowledgeable assistant. Provide a single, concise, and accurate answer to the question based solely on the provided context.\n"
                "Do not include additional information, avoid repetition, and focus only on the exact question asked.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )

            # Get the answer from the LLM
            answer = self.llm.generate(prompt)
            self.query_cache[question] = answer
            logging.info(f"Answer: {answer}")
            return answer

        except Exception as e:
            logging.error(f"Error processing question: {str(e)}")
            return "An error occurred while processing your question."