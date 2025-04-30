import unittest
import asyncio
import logging
import os
from sentence_transformers import SentenceTransformer
from scraper import WikipediaScraper
from vector_store import FAISSVectorStore
from llm import OllamaLLM
from rag_pipeline import RAGPipeline

# Configure logging for tests
logging.basicConfig(
    filename='test_rag_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TestRAGSystem(unittest.TestCase):
    """Unit tests for RAG system components."""
    def setUp(self):
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.scraper = WikipediaScraper()
        self.vector_store = FAISSVectorStore(embedding_model)
        self.llm = OllamaLLM()
        self.pipeline = RAGPipeline(self.vector_store, self.llm, embedding_model)
        # Define a list of URLs for testing
        self.urls = [
            "https://en.wikipedia.org/wiki/Photosynthesis",
            "https://en.wikipedia.org/wiki/Renaissance",
            "https://en.wikipedia.org/wiki/DNA",
            "https://en.wikipedia.org/wiki/Quantum_mechanics",
            "https://en.wikipedia.org/wiki/World_War_II",
            "https://en.wikipedia.org/wiki/Calculus",
            "https://en.wikipedia.org/wiki/Black_hole",
            "https://en.wikipedia.org/wiki/Climate_change",
            "https://en.wikipedia.org/wiki/Industrial_Revolution",
            "https://en.wikipedia.org/wiki/William_Shakespeare",
            "https://en.wikipedia.org/wiki/Ancient_Egypt"
        ]
    
    def test_scraper_multiple_urls(self):
        """Test the scraper with multiple URLs."""
        async def test():
            for url in self.urls:
                texts = await self.scraper.scrape(url)
                self.assertGreater(len(texts), 0, f"Scraper failed to retrieve texts from {url}")
                logging.info(f"Successfully scraped {url}")
        asyncio.run(test())
    
    def test_pipeline_factual_question(self):
        """Test the RAG pipeline with a factual question."""
        # First, scrape and store data
        async def prepare_data():
            for url in self.urls:
                texts = await self.scraper.scrape(url)
                embeddings = self.vector_store.embedding_model.encode(texts).tolist()
                self.vector_store.store(texts, embeddings)
        asyncio.run(prepare_data())
        
        question = "What is photosynthesis?"
        answer = self.pipeline.process(question)
        self.assertIsInstance(answer, str, "Pipeline failed to generate a string response")
        self.assertIn("chlorophyll", answer.lower(), "Answer does not contain expected keyword 'chlorophyll'")
        logging.info(f"Question: {question}\nAnswer: {answer}")
    
    def test_pipeline_list_question(self):
        """Test the RAG pipeline with a question requiring a list (e.g., major figures)."""
        question = "Who were the major figures in the Renaissance?"
        answer = self.pipeline.process(question)
        self.assertIsInstance(answer, str, "Pipeline failed to generate a string response")
        expected_figures = ["leonardo da vinci", "michelangelo", "petrarch"]
        for figure in expected_figures:
            self.assertIn(figure.lower(), answer.lower(), f"Answer does not contain expected figure '{figure}'")
        logging.info(f"Question: {question}\nAnswer: {answer}")
    
    def test_pipeline_descriptive_question(self):
        """Test the RAG pipeline with a descriptive question."""
        question = "What caused World War II?"
        answer = self.pipeline.process(question)
        self.assertIsInstance(answer, str, "Pipeline failed to generate a string response")
        self.assertIn("germany", answer.lower(), "Answer does not contain expected keyword 'Germany'")
        logging.info(f"Question: {question}\nAnswer: {answer}")
    
    def test_vector_store(self):
        """Test the vector store's ability to store and query data."""
        texts = ["Test document about the Renaissance"]
        embeddings = self.vector_store.embedding_model.encode(texts).tolist()
        self.vector_store.store(texts, embeddings)
        query_embedding = embeddings[0]
        results = self.vector_store.query(query_embedding, top_k=1)
        self.assertEqual(results[0], "Test document about the Renaissance", "Vector store query failed")
    
    def test_stress_1000_interactions(self):
        """Stress test the pipeline with 1000 Q&A interactions."""
        questions = [
            "What is photosynthesis?",
            "Who were the major figures in the Renaissance?",
            "What caused World War II?",
            "What is a black hole?",
            "What is calculus?"
        ]
        # Repeat questions to reach 1000 interactions
        for i in range(200):  # 200 iterations * 5 questions = 1000 interactions
            for question in questions:
                answer = self.pipeline.process(question)
                self.assertIsInstance(answer, str, f"Pipeline failed at interaction {i+1} for question: {question}")
                logging.info(f"Interaction {i*5 + questions.index(question) + 1}: Question: {question}\nAnswer: {answer}")
    
    def test_edge_case_empty_question(self):
        """Test the pipeline with an empty question."""
        question = ""
        answer = self.pipeline.process(question)
        self.assertIn("error", answer.lower(), "Pipeline did not handle empty question gracefully")
        logging.info(f"Question: {question}\nAnswer: {answer}")
    
    def test_edge_case_invalid_question(self):
        """Test the pipeline with an invalid question."""
        question = "12345!@#$%"
        answer = self.pipeline.process(question)
        self.assertIsInstance(answer, str, "Pipeline failed to handle invalid question")
        logging.info(f"Question: {question}\nAnswer: {answer}")
    
    def test_logging(self):
        """Test if Q&A interactions are logged correctly."""
        log_file = 'rag_project.log'
        # Clear the log file before the test
        if os.path.exists(log_file):
            os.remove(log_file)
        
        question = "What is DNA?"
        answer = self.pipeline.process(question)
        
        # Check if the log file exists and contains the interaction
        self.assertTrue(os.path.exists(log_file), "Log file was not created")
        with open(log_file, 'r') as f:
            log_content = f.read()
            self.assertIn(question, log_content, "Question was not logged")
            self.assertIn(answer, log_content, "Answer was not logged")

if __name__ == "__main__":
    unittest.main()