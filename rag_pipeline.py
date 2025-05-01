import logging
from functools import lru_cache
from typing import List

class RAGPipeline:
    """RAG pipeline integrating vector store and LLM for question answering."""
    
    def __init__(self, vector_store, llm, embedding_model):
        self.vector_store = vector_store
        self.llm = llm
        self.embedding_model = embedding_model
        logging.info("RAGPipeline initialized.")
    
    @lru_cache(maxsize=128)
    def process(self, question: str) -> str:
        """Process a question through the RAG pipeline."""
        logging.info(f"Question: {question}")
        
        # Encode the question
        query_embedding = self.embedding_model.encode([question]).tolist()[0]
        
        # Retrieve relevant contexts
        contexts = self.vector_store.query(query_embedding, top_k=5)
        if not contexts:
            logging.warning("No relevant contexts found.")
            return "No relevant information found to answer the question."
        
        # Combine contexts into a single string
        context_str = "\n".join(contexts)
        logging.info(f"Full Retrieved Context:\n{context_str}")
        
        # Prepare the prompt with context and question, requesting only a summary
        prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n"
            f"Provide a brief answer in not more than 3 sentences that directly addresses the question using scientific reasoning. "
            f"The answer must be based on established scientific principles or historical scientific developments relevant to the question. "
            f"Do not include additional details, historical context, or unrelated information beyond the core scientific answer."
        )
        
        # Generate response using the LLM
        answer = self.llm.generate(prompt)
        logging.info(f"Answer: {answer}")
        return answer.strip()