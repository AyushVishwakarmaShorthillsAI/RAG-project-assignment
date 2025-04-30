import logging
import asyncio
import streamlit as st
import sys
import torch
import time
import os

try:
    from .scraper import WikipediaScraper
    from .vector_store import FAISSVectorStore
    from .llm import HuggingFaceLLM
    from .rag_pipeline import RAGPipeline
    from .data_processing import scrape_and_store
except ImportError:
    from scraper import WikipediaScraper
    from vector_store import FAISSVectorStore
    from llm import HuggingFaceLLM
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

    with st.form("query_form"):
        question = st.text_input("Enter your question:")
        submitted = st.form_submit_button("Get Answer")

        if submitted and question.strip():
            # Only process if the question is new
            if question != st.session_state.last_question:
                start = time.time()
                answer = rag_pipeline.process(question)
                duration = time.time() - start
                # Update session state
                st.session_state.last_question = question
                st.session_state.answer = answer
                st.session_state.response_time = duration
            # Display the result
            if st.session_state.answer:
                st.markdown(f"**Answer:** {st.session_state.answer}")
                st.markdown(f"**Response Time:** {st.session_state.response_time:.2f} seconds")
            else:
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
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
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

        llm = HuggingFaceLLM()
        rag_pipeline = RAGPipeline(vector_store, llm, embedding_model)

        # Only scrape if files don't exist
        if not os.path.exists(index_path) or not os.path.exists(texts_path):
            urls = [
                "https://en.wikipedia.org/wiki/Evolution",
                "https://en.wikipedia.org/wiki/Genetics",
                "https://en.wikipedia.org/wiki/Periodic_table",
                "https://en.wikipedia.org/wiki/Chemical_bonding",
                "https://en.wikipedia.org/wiki/Plate_tectonics",
                "https://en.wikipedia.org/wiki/Big_Bang",
                "https://en.wikipedia.org/wiki/Black_hole",
                "https://en.wikipedia.org/wiki/Photosynthesis",
                "https://en.wikipedia.org/wiki/Quantum_mechanics",
                "https://en.wikipedia.org/wiki/Relativity_theory",
                "https://en.wikipedia.org/wiki/Climate_change",
                "https://en.wikipedia.org/wiki/Ecosystem",
                "https://en.wikipedia.org/wiki/Neuroscience",
                "https://en.wikipedia.org/wiki/Immunology",
                "https://en.wikipedia.org/wiki/DNA",
                "https://en.wikipedia.org/wiki/RNA",
                "https://en.wikipedia.org/wiki/Photosynthetic_pigment",
                "https://en.wikipedia.org/wiki/Atomic_structure",
                "https://en.wikipedia.org/wiki/Thermodynamics",
                "https://en.wikipedia.org/wiki/Astrophysics",
                "https://en.wikipedia.org/wiki/World_War_I",
                "https://en.wikipedia.org/wiki/World_War_II",
                "https://en.wikipedia.org/wiki/Renaissance",
                "https://en.wikipedia.org/wiki/Industrial_Revolution",
                "https://en.wikipedia.org/wiki/French_Revolution",
                "https://en.wikipedia.org/wiki/American_Revolution",
                "https://en.wikipedia.org/wiki/Cold_War",
                "https://en.wikipedia.org/wiki/Ancient_Egypt",
                "https://en.wikipedia.org/wiki/Roman_Empire",
                "https://en.wikipedia.org/wiki/Middle_Ages",
                "https://en.wikipedia.org/wiki/Great_Depression",
                "https://en.wikipedia.org/wiki/Civil_Rights_Movement",
                "https://en.wikipedia.org/wiki/Space_Race",
                "https://en.wikipedia.org/wiki/Fall_of_the_Berlin_Wall",
                "https://en.wikipedia.org/wiki/Colonization_of_Africa",
                "https://en.wikipedia.org/wiki/Indian_Independence_Movement",
                "https://en.wikipedia.org/wiki/Byzantine_Empire",
                "https://en.wikipedia.org/wiki/Mongol_Empire",
                "https://en.wikipedia.org/wiki/History_of_China",
                "https://en.wikipedia.org/wiki/Vietnam_War",
                "https://en.wikipedia.org/wiki/Calculus",
                "https://en.wikipedia.org/wiki/Algebra",
                "https://en.wikipedia.org/wiki/Geometry",
                "https://en.wikipedia.org/wiki/Trigonometry",
                "https://en.wikipedia.org/wiki/Number_theory",
                "https://en.wikipedia.org/wiki/Probability_theory",
                "https://en.wikipedia.org/wiki/Statistics",
                "https://en.wikipedia.org/wiki/Set_theory",
                "https://en.wikipedia.org/wiki/Linear_algebra",
                "https://en.wikipedia.org/wiki/Differential_equations",
                "https://en.wikipedia.org/wiki/Game_theory",
                "https://en.wikipedia.org/wiki/Topology",
                "https://en.wikipedia.org/wiki/Chaos_theory",
                "https://en.wikipedia.org/wiki/Graph_theory",
                "https://en.wikipedia.org/wiki/Mathematical_logic",
                "https://en.wikipedia.org/wiki/William_Shakespeare",
                "https://en.wikipedia.org/wiki/Homer",
                "https://en.wikipedia.org/wiki/Iliad",
            ]
            logger.info("Scraping and storing Wikipedia content...")
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