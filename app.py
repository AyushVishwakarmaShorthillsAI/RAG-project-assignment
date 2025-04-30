import abc
import logging
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import streamlit as st
import unittest
from uuid import uuid4
import time
import sys
from functools import lru_cache
from typing import List, Dict
import torch  # Added missing import

# Configure logging with minimal overhead
logging.basicConfig(filename='rag_project.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

class BaseScraper(abc.ABC):
    """Abstract base class for web scrapers."""
    @abc.abstractmethod
    async def scrape(self, url: str) -> List[str]:
        """Scrape data from a given URL and return cleaned text."""
        pass

class WikipediaScraper(BaseScraper):
    """Concrete scraper for Wikipedia pages using async requests."""
    async def scrape(self, url: str) -> List[str]:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, timeout=10) as response:
                    response.raise_for_status()
                    text = await response.text()
                    soup = BeautifulSoup(text, 'html.parser')
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
    def store(self, texts: List[str], embeddings: List[List[float]]):
        """Store texts and their embeddings."""
        pass
    
    @abc.abstractmethod
    def query(self, query_embedding: List[float], top_k: int) -> List[str]:
        """Query the vector store for relevant texts."""
        pass

class FAISSVectorStore(BaseVectorStore):
    """Concrete vector store using FAISS for faster retrieval."""
    def __init__(self, embedding_model: SentenceTransformer, dimension: int = 384):
        logging.info("Initializing FAISSVectorStore")
        self.embedding_model = embedding_model
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance index
        self.texts = []
    
    def store(self, texts: List[str], embeddings: List[List[float]]):
        embeddings_np = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings_np)
        self.texts.extend(texts)
        logging.info(f"Stored {len(texts)} documents in FAISS")
    
    def query(self, query_embedding: List[float], top_k: int) -> List[str]:
        query_np = np.array([query_embedding], dtype=np.float32)
        _, indices = self.index.search(query_np, top_k)
        return [self.texts[i] for i in indices[0]]

class BaseLLM(abc.ABC):
    """Abstract base class for LLMs."""
    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response based on the prompt."""
        pass

class HuggingFaceLLM(BaseLLM):
    """Concrete LLM using HuggingFace transformers."""
    def __init__(self):
        logging.info("Initializing HuggingFaceLLM")
        model_name = "distilgpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.generator(
                prompt,
                max_new_tokens=50,
                truncation=True,
                num_return_sequences=1
            )
            return response[0]['generated_text']
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "Error generating response."

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
            retrieved_docs = self.vector_store.query(query_embedding, top_k=3)
            context = " ".join(retrieved_docs)
            prompt = f"Question: {question}\nContext: {context}\nAnswer:"
            answer = self.llm.generate(prompt)
            self.query_cache[question] = answer
            logging.info(f"Question: {question}\nAnswer: {answer}")
            return answer
        except Exception as e:
            logging.error(f"Error processing question: {str(e)}")
            return "Error processing question."

async def scrape_and_store(scraper: BaseScraper, vector_store: BaseVectorStore, urls: List[str]):
    """Scrape data asynchronously and store in vector store."""
    doc_count = len(vector_store.texts) if hasattr(vector_store, 'texts') else 0
    logging.info(f"Documents in collection before processing: {doc_count}")
    if doc_count == 0:
        tasks = [scraper.scrape(url) for url in urls]
        all_texts = []
        for future in asyncio.as_completed(tasks):
            texts = await future
            all_texts.extend(texts)
        
        batch_size = 64
        embeddings = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            embeddings.extend(
                vector_store.embedding_model.encode(
                    batch,
                    show_progress_bar=False,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                ).tolist()
            )
        
        vector_store.store(all_texts, embeddings)
    else:
        logging.info("Collection already contains data. Skipping scraping and storing.")

def run_ui(rag_pipeline: RAGPipeline):
    """Streamlit UI for Q&A with caching."""
    st.title("RAG-LLM Q&A System")
    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if question:
            start_time = time.time()
            answer = rag_pipeline.process(question)
            st.write(f"**Answer**: {answer}")
            st.write(f"**Response Time**: {time.time() - start_time:.2f} seconds")
        else:
            st.write("Please enter a question.")

class TestRAGSystem(unittest.TestCase):
    """Unit tests for RAG system components."""
    def setUp(self):
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.scraper = WikipediaScraper()
        self.vector_store = FAISSVectorStore(embedding_model)
        self.llm = HuggingFaceLLM()
        self.pipeline = RAGPipeline(self.vector_store, self.llm, embedding_model)
    
    def test_scraper(self):
        async def test():
            url = "https://en.wikipedia.org/wiki/Photosynthesis"
            texts = await self.scraper.scrape(url)
            self.assertGreater(len(texts), 0, "Scraper failed to retrieve texts")
        asyncio.run(test())
    
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
    try:
        scraper = WikipediaScraper()
        embedding_model = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        vector_store = FAISSVectorStore(embedding_model)
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
        asyncio.run(scrape_and_store(scraper, vector_store, urls))
        logging.info("Finished scrape_and_store")
        
        logging.info("Starting Streamlit UI")
        run_ui(rag_pipeline)
        
        if "test" in sys.argv:
            unittest.main(argv=[''])
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        raise