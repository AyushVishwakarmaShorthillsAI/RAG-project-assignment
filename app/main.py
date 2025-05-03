import logging
import streamlit as st
import torch
import time
import os
import json
from all_Urls import URLS
from scraper import WikipediaScraper
from vector_store import FAISSVectorStore
from llm import OllamaLLM
from rag_pipeline import RAGPipeline
from data_processing import scrape_and_store
from sentence_transformers import SentenceTransformer

# === Logging Setup ===
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
os.makedirs(LOG_DIR, exist_ok=True)

RAG_LOG_PATH = os.path.join(LOG_DIR, "rag_project.log")
QA_LOG_PATH = os.path.join(LOG_DIR, "qa_interactions.log")

# Remove existing logging handlers to avoid Streamlit conflicts
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging manually
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(RAG_LOG_PATH, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# === Logging Interaction ===
def log_interaction(question, answer):
    log_entry = {"question": question.strip(), "answer": answer.strip()}
    try:
        with open(QA_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")
        logger.info(f"Logged interaction: Question='{question}', Answer='{answer}'")
    except Exception as e:
        logger.error(f"Error logging interaction: {e}")

# === Display History ===
def display_history():
    st.sidebar.subheader("Previous Q&A History")
    st.sidebar.markdown("---")

    if not os.path.exists(QA_LOG_PATH):
        st.sidebar.info("No previous interactions found.")
        return

    try:
        with open(QA_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Error reading qa_interactions.log: {e}")
        st.sidebar.error("Error reading interaction history.")
        return

    if not lines:
        st.sidebar.info("No previous interactions found.")
        return

    history_entries = []
    for entry in reversed(lines):
        try:
            qa = json.loads(entry.strip())
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "").strip()
            if question and answer:
                history_entries.append((question, answer))
        except json.JSONDecodeError as e:
            logger.warning(f"Skipping malformed log entry: {entry.strip()} - Error: {e}")
            continue

    if not history_entries:
        st.info("No valid interactions found in history.")
        return

    for idx, (question, answer) in enumerate(history_entries):
        with st.sidebar.expander(f"**Q{len(history_entries) - idx}:** {question}"):
            st.markdown(f"**A{len(history_entries) - idx}:** {answer}")

# === Streamlit UI ===
def run_ui(rag_pipeline: RAGPipeline):
    st.title("RAG-LLM Q&A System")

    if 'last_question' not in st.session_state:
        st.session_state.last_question = None
    if 'answer' not in st.session_state:
        st.session_state.answer = None
    if 'response_time' not in st.session_state:
        st.session_state.response_time = None

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

            st.markdown("---")
            st.markdown(f"**Answer:** {st.session_state.answer}")
            st.markdown(f"**Response Time:** {st.session_state.response_time:.2f} seconds")
        elif submitted:
            st.warning("Please enter a valid question.")

# === Main Function ===
def main():
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

        run_ui(rag_pipeline)
    except Exception as e:
        logger.exception(f"Application failed: {e}")

if __name__ == "__main__":
    main()
