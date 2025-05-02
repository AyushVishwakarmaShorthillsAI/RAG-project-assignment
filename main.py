import logging
import asyncio
import streamlit as st
import sys
import torch
import time
import os
from all_Urls import URLS


try:
    from .scraper import WikipediaScraper
    from .vector_store import FAISSVectorStore
    from .llm import OllamaLLM
    from .rag_pipeline import RAGPipeline
    from .data_processing import scrape_and_store
except ImportError:
    from scraper import WikipediaScraper
    from vector_store import FAISSVectorStore
    from llm import OllamaLLM
    from rag_pipeline import RAGPipeline
    from data_processing import scrape_and_store

from sentence_transformers import SentenceTransformer

# Logging setup for general application logs
logging.basicConfig(
    filename='rag_project.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))

# Separate logger for Q&A interactions
qa_logger = logging.getLogger('qa_interactions')
qa_handler = logging.FileHandler('qa_interactions.log')
qa_handler.setFormatter(logging.Formatter('%(asctime)s - Question: %(message)s'))
qa_logger.addHandler(qa_handler)
qa_logger.setLevel(logging.INFO)

# Logger for answers to ensure proper formatting
qa_answer_logger = logging.getLogger('qa_interactions_answer')
qa_answer_handler = logging.FileHandler('qa_interactions.log', mode='a')
qa_answer_handler.setFormatter(logging.Formatter('%(asctime)s - Answer: %(message)s'))
qa_answer_logger.addHandler(qa_answer_handler)
qa_answer_logger.setLevel(logging.INFO)

RUN_FLAG = False  # Prevents multiple Streamlit executions

def run_ui(rag_pipeline: RAGPipeline):
    """Streamlit UI for querying the RAG pipeline."""
    st.title("RAG-LLM Q&A System")

    # Initialize session state
    if 'last_question' not in st.session_state:
        st.session_state.last_question = None
    if 'answer' not in st.session_state:
        st.session_state.answer = None
    if 'response_time' not in st.session_state:
        st.session_state.response_time = None
    if 'query_processed' not in st.session_state:
        st.session_state.query_processed = False
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False

    logger.info("Launching Streamlit UI...")

    with st.form("query_form"):
        question = st.text_input("Enter your question:")
        submitted = st.form_submit_button("Get Answer")

        if submitted and question.strip():
            logger.info(f"Form submitted with question: {question}")
            # Check if the question has changed
            if question != st.session_state.last_question:
                logger.info(f"New question detected: {question}")
                rag_pipeline.process.cache_clear()
                start = time.time()
                answer = rag_pipeline.process(question)
                duration = time.time() - start
                # Log Q&A pair to qa_interactions.log
                qa_logger.info(question)
                qa_answer_logger.info(answer)
                # Update session state
                st.session_state.last_question = question
                st.session_state.answer = answer
                st.session_state.response_time = duration
                st.session_state.query_processed = True
                st.session_state.form_submitted = True
            else:
                logger.info(f"Question same as previous: {question}")

        # Display the result only if the form was submitted and processed
        if st.session_state.form_submitted and st.session_state.answer:
            st.markdown(f"**Answer:** {st.session_state.answer}")
            st.markdown(f"**Response Time:** {st.session_state.response_time:.2f} seconds")
            # Reset form_submitted to prevent re-display on re-render
            st.session_state.form_submitted = False
        elif submitted and not question.strip():
            st.warning("Please enter a valid question.")

async def main_async():
    global RUN_FLAG
    if RUN_FLAG:
        return
    RUN_FLAG = True

    logger.info("Initializing components...")

    try:
        # Load models
        scraper = WikipediaScraper()
        embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        embedding_model = embedding_model.to(device)

        index_path = "faiss_index.bin"
        texts_path = "texts.pkl"

        # Load or create vector store
        if os.path.exists(index_path) and os.path.exists(texts_path):
            logger.info("Loading existing FAISS index and texts.")
            vector_store = FAISSVectorStore.load(embedding_model, index_path, texts_path)
        else:
            logger.info("Creating new FAISS index from scratch.")
            vector_store = FAISSVectorStore(embedding_model)

        llm = OllamaLLM()
        rag_pipeline = RAGPipeline(vector_store, llm, embedding_model)

        # Only scrape if files don't exist
        if not os.path.exists(index_path) or not os.path.exists(texts_path):
            urls = URLS
            logger.info("Scraping and storing content...")
            await scrape_and_store(scraper, vector_store, urls, index_path, texts_path)
            logger.info("Scraping completed.")

        logger.info("Launching Streamlit UI...")
        run_ui(rag_pipeline)

    except Exception as e:
        logger.exception(f"Application failed: {e}")

# Run the async function within Streamlit's event loop
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main_async())
    loop.close()