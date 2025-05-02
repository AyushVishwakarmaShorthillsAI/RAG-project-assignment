import streamlit as st
import logging
import os
import torch
import time  # Added for measuring response time
from datetime import datetime  # Added for timestamp in logs
from scraper import WebScraper
from vector_store import FAISSVectorStore
from llm import OllamaLLM
from rag_pipeline import RAGPipeline
from data_processing import scrape_and_store
from sentence_transformers import SentenceTransformer
from all_Urls import URLS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File paths for FAISS index, texts, and QA logs
INDEX_PATH = "faiss_index.bin"
TEXTS_PATH = "texts.pkl"
QA_LOGS_PATH = "qa_logs.txt"  # Path for saving question-answer logs

# URLs to scrape 
URLS = URLS

# Initialize the RAG pipeline
@st.cache_resource
def initialize_pipeline():
    logger.info("Initializing RAG pipeline...")
    embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = embedding_model.to(device)
    
    vector_store = None
    if os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH):
        vector_store = FAISSVectorStore.load(embedding_model, INDEX_PATH, TEXTS_PATH)
        logger.info("Loaded existing FAISS index and texts.")
    else:
        scraper = WebScraper(embedding_model)
        vector_store = FAISSVectorStore(embedding_model)
        scrape_and_store(scraper, vector_store, URLS, INDEX_PATH, TEXTS_PATH)
    
    llm = OllamaLLM()
    pipeline = RAGPipeline(vector_store, llm, embedding_model)
    return pipeline

# Function to log question, answer, and response time to qa_logs.txt
def log_qa(question, answer, response_time):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"Timestamp: {timestamp}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Response Time: {response_time:.2f} seconds\n"
        f"{'-'*50}\n"
    )
    with open(QA_LOGS_PATH, "a", encoding="utf-8") as f:
        f.write(log_entry)

# Streamlit UI
def run_ui(rag_pipeline):
    st.title("RAG-LLM Q&A System")
    
    question = st.text_input("Enter your question:")
    if st.button("Submit"):
        if question:
            with st.spinner("Processing..."):
                # Measure the start time
                start_time = time.time()
                
                # Process the question
                answer = rag_pipeline.process(question)
                
                # Calculate response time
                response_time = time.time() - start_time
                
                # Display the answer
                st.write("**Answer:**")
                st.write(answer)
                
                # Display the response time
                st.write(f"**Response Time:** {response_time:.2f} seconds")
                
                # Log the question, answer, and response time
                log_qa(question, answer, response_time)
        else:
            st.error("Please enter a question.")

# Main execution
if __name__ == "__main__":
    # Initialize the pipeline
    rag_pipeline = initialize_pipeline()
    
    # Run the Streamlit UI
    run_ui(rag_pipeline)