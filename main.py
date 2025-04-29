import abc
import logging
import requests
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import streamlit as st
import unittest
from uuid import uuid4
import time
import sys

# Configure logging
logging.basicConfig(filename='rag_project.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class BaseScraper(abc.ABC):
    """Abstract base class for web scrapers."""
    @abc.abstractmethod
    def scrape(self, url: str) -> list:
        """Scrape data from a given URL and return cleaned text."""
        pass

class WikipediaScraper(BaseScraper):
    """Concrete scraper for Wikipedia pages."""
    def scrape(self, url: str) -> list:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            cleaned_text = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
            logging.info(f"Scraped {len(cleaned_text)} paragraphs from {url}")
            return cleaned_text
        except Exception as e:
            logging.error(f"Error scraping {url}: {str(e)}")
            return []

class BaseVectorStore(abc.ABC):
    """Abstract base class for vector stores."""
    @abc.abstractmethod
    def store(self, texts: list, embeddings: list):
        """Store texts and their embeddings."""
        pass
    
    @abc.abstractmethod
    def query(self, query_embedding: list, top_k: int) -> list:
        """Query the vector store for relevant texts."""
        pass

class ChromaVectorStore(BaseVectorStore):
    """Concrete vector store using ChromaDB."""
    def __init__(self, embedding_model: SentenceTransformer):
        logging.info("Initializing ChromaVectorStore")
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name="rag_collection")
        self.embedding_model = embedding_model
    
    def store(self, texts: list, embeddings: list):
        ids = [str(uuid4()) for _ in texts]
        self.collection.add(documents=texts, embeddings=embeddings, ids=ids)
        logging.info(f"Stored {len(texts)} documents in ChromaDB")
    
    def query(self, query_embedding: list, top_k: int) -> list:
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results['documents'][0]
    
    def count_documents(self) -> int:
        """Return the number of documents in the collection efficiently with debugging."""
        result = self.collection.get(include=[])  # Avoid loading documents
        logging.info(f"ChromaDB get result: {result}")  # Debug the raw response
        return len(result['ids']) if result['ids'] else 0

class BaseLLM(abc.ABC):
    """Abstract base class for LLMs."""
    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response based on the prompt."""
        pass

class HuggingFaceLLM(BaseLLM):
    """Concrete LLM using HuggingFace transformers."""
    def __init__(self):
        self.generator = None  # Lazy load the pipeline
    
    def generate(self, prompt: str) -> str:
        if self.generator is None:
            logging.info("Initializing HuggingFaceLLM pipeline")
            self.generator = pipeline('text-generation', model='distilgpt2')
        response = self.generator(
            prompt,
            max_new_tokens=50,
            truncation=True,
            num_return_sequences=1
        )
        return response[0]['generated_text']

class RAGPipeline:
    """RAG pipeline combining retrieval and generation."""
    def __init__(self, vector_store: BaseVectorStore, llm: BaseLLM, embedding_model: SentenceTransformer):
        self.vector_store = vector_store
        self.llm = llm
        self.embedding_model = embedding_model
    
    def process(self, question: str) -> str:
        query_embedding = self.embedding_model.encode(question, show_progress_bar=False).tolist()
        retrieved_docs = self.vector_store.query(query_embedding, top_k=3)
        context = " ".join(retrieved_docs)
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
        answer = self.llm.generate(prompt)
        logging.info(f"Question: {question}\nAnswer: {answer}")
        return answer

def scrape_and_store(scraper: BaseScraper, vector_store: BaseVectorStore, urls: list):
    """Scrape data and store in vector store only if collection is empty."""
    doc_count = vector_store.count_documents()
    logging.info(f"Documents in collection before processing: {doc_count}")
    if doc_count == 0:
        all_texts = []
        for url in urls:
            texts = scraper.scrape(url)
            all_texts.extend(texts)
            time.sleep(0.5)
        
        batch_size = 32
        embeddings = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            embeddings.extend(vector_store.embedding_model.encode(batch, show_progress_bar=False).tolist())
        
        vector_store.store(all_texts, embeddings)
    else:
        logging.info("Collection already contains data. Skipping scraping and storing.")

def run_ui(rag_pipeline: RAGPipeline):
    """Streamlit UI for Q&A."""
    st.title("RAG-LLM Q&A System")
    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if question:
            answer = rag_pipeline.process(question)
            st.write(f"**Answer**: {answer}")
        else:
            st.write("Please enter a question.")

class TestRAGSystem(unittest.TestCase):
    """Unit tests for RAG system components."""
    def setUp(self):
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.scraper = WikipediaScraper()
        self.vector_store = ChromaVectorStore(embedding_model)
        self.llm = HuggingFaceLLM()
        self.pipeline = RAGPipeline(self.vector_store, self.llm, embedding_model)
    
    def test_scraper(self):
        url = "https://en.wikipedia.org/wiki/Photosynthesis"
        texts = self.scraper.scrape(url)
        self.assertGreater(len(texts), 0, "Scraper failed to retrieve texts")
    
    def test_pipeline(self):
        question = "What is photosynthesis?"
        answer = self.pipeline.process(question)
        self.assertIsInstance(answer, str, "Pipeline failed to generate a string response")
    
    def test_vector_store(self):
        texts = ["Test document"]
        embeddings = self.vector_store.embedding_model.encode(texts).tolist()
        self.vector_store.store(texts, embeddings)
        query_embedding = embeddings[0]
        results = self.vector_store.query(query_embedding, top_k=1)
        self.assertEqual(results[0], "Test document", "Vector store query failed")

if __name__ == "__main__":
    logging.info("Starting script")
    scraper = WikipediaScraper()
    logging.info("Initializing embedding model")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("Embedding model initialized")
    vector_store = ChromaVectorStore(embedding_model)
    llm = HuggingFaceLLM()
    rag_pipeline = RAGPipeline(vector_store, llm, embedding_model)
    
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
    logging.info("Starting scrape_and_store")
    scrape_and_store(scraper, vector_store, urls)
    logging.info("Finished scrape_and_store")
    
    logging.info("Starting Streamlit UI")
    run_ui(rag_pipeline)
    
    if "test" in sys.argv:
        unittest.main(argv=[''])