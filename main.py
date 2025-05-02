import logging
import streamlit as st
import sys
import torch
import time
import os
import json
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

# Logging setup
logging.basicConfig(
    filename='rag_project.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))

RUN_FLAG = False

def log_interaction(question, answer):
    log_entry = {
        "question": question.strip(),
        "answer": answer.strip()
    }
    try:
        with open("qa_interactions.log", "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")
        logger.info(f"Logged interaction: Question='{question}', Answer='{answer}'")
    except Exception as e:
        logger.error(f"Error logging interaction: {e}")

def display_history():
    st.sidebar.subheader("Previous Q&A History")
    st.sidebar.markdown("---")

    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)

    if not os.path.exists("qa_interactions.log"):
        st.sidebar.info("No previous interactions found. (File does not exist)")
        logger.info("qa_interactions.log does not exist.")
        return

    try:
        with open("qa_interactions.log", "r", encoding="utf-8") as f:
            lines = f.readlines()
        logger.info(f"Read {len(lines)} lines from qa_interactions.log")
    except Exception as e:
        logger.error(f"Error reading qa_interactions.log: {e}")
        st.sidebar.error("Error reading interaction history.")
        return

    if not lines:
        st.sidebar.info("No previous interactions found. (File is empty)")
        logger.info("qa_interactions.log is empty.")
        return

    if debug_mode:
        st.sidebar.subheader("Debug: Raw Log Content")
        st.sidebar.code("\n".join(lines), language="json")

    # Create an expander for the history to make it collapsible
    with st.sidebar.expander("View History", expanded=True):
        # Display entries in reverse order (most recent first)
        history_entries = []
        for idx, entry in enumerate(reversed(lines)):
            try:
                qa = json.loads(entry.strip())
                question = qa.get("question", "").strip()
                answer = qa.get("answer", "").strip()
                if question and answer:
                    history_entries.append((question, answer))
                else:
                    logger.warning(f"Skipping entry with missing question or answer: {entry.strip()}")
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed log entry: {entry.strip()} - Error: {e}")
                continue

        if not history_entries:
            st.info("No valid interactions found in history.")
            logger.info("No valid interactions found after parsing qa_interactions.log.")
            return

        # Display each entry using Streamlit components
        for idx, (question, answer) in enumerate(history_entries):
            st.markdown(f"**Q{len(history_entries) - idx}:** {question}")
            st.markdown(f"**A{len(history_entries) - idx}:** {answer}")
            st.markdown("---")
        logger.info(f"Displayed {len(history_entries)} history entries.")

def run_ui(rag_pipeline: RAGPipeline):
    """Streamlit UI for querying the RAG pipeline."""
    st.title("RAG-LLM Q&A System")

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

    # Display the history in the sidebar
    display_history()

    st.subheader("Ask a Question")
    with st.form("query_form"):
        question = st.text_input("Enter your question:")
        submitted = st.form_submit_button("Get Answer")

        if submitted and question.strip():
            logger.info(f"Form submitted with question: {question}")
            if question != st.session_state.last_question:
                rag_pipeline.process.cache_clear()
                start = time.time()
                answer = rag_pipeline.process(question)
                duration = time.time() - start
                log_interaction(question, answer)
                st.session_state.last_question = question
                st.session_state.answer = answer
                st.session_state.response_time = duration
                st.session_state.query_processed = True
                st.session_state.form_submitted = True
            else:
                logger.info("Question same as previous.")

        if st.session_state.form_submitted and st.session_state.answer:
            st.markdown("---")
            st.markdown(f"**Answer:** {st.session_state.answer}")
            st.markdown(f"**Response Time:** {st.session_state.response_time:.2f} seconds")
            st.session_state.form_submitted = False
        elif submitted and not question.strip():
            st.warning("Please enter a valid question.")

def main():
    global RUN_FLAG
    if RUN_FLAG:
        return
    RUN_FLAG = True

    logger.info("Initializing components...")

    try:
        scraper = WikipediaScraper()
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        embedding_model = embedding_model.to(device)

        index_path = "faiss_index.bin"
        texts_path = "texts.pkl"

        if os.path.exists(index_path) and os.path.exists(texts_path):
            logger.info("Loading existing FAISS index and texts.")
            vector_store = FAISSVectorStore.load(embedding_model, index_path, texts_path)
        else:
            logger.info("Creating new FAISS index from scratch.")
            vector_store = FAISSVectorStore(embedding_model)

        llm = OllamaLLM()
        rag_pipeline = RAGPipeline(vector_store, llm, embedding_model)

        if not os.path.exists(index_path) or not os.path.exists(texts_path):
            urls = URLS
            logger.info("Scraping and storing content...")
            scrape_and_store(scraper, vector_store, urls, index_path, texts_path)
            logger.info("Scraping completed.")

        logger.info("Launching Streamlit UI...")
        run_ui(rag_pipeline)

    except Exception as e:
        logger.exception(f"Application failed: {e}")

if __name__ == "__main__":
    main()