import logging
from functools import lru_cache

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
        logging.info(f"Processing question: {question}")
        
        # Encode the question
        query_embedding = self.embedding_model.encode([question]).tolist()[0]
        
        # Retrieve relevant contexts
        contexts = self.vector_store.query(query_embedding, top_k=5)
        if not contexts:
            logging.warning("No relevant contexts found.")
            return "No relevant information found to answer the question."
        
        # Combine contexts into a single string
        context_str = "\n".join(contexts)
        
        # Prepare the prompt with context and question
        prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n"
            f"Based on the provided context, answer the question in brief."
        )
        
        # Generate response using the LLM
        answer = self.llm.generate(prompt)
        logging.info(f"Generated answer: {answer}")
        return answer