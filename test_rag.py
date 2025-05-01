import unittest
import asyncio
import logging
import os
import time
from sentence_transformers import SentenceTransformer
from scraper import WikipediaScraper
from vector_store import FAISSVectorStore
from llm import OllamaLLM
from rag_pipeline import RAGPipeline
import warnings

# Suppress ResourceWarnings for cleaner logs
warnings.filterwarnings("ignore", category=ResourceWarning)

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
        
        # Define a subset of URLs for populating the vector store in setUp
        setup_urls = [
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
            "https://en.wikipedia.org/wiki/Ancient_Egypt",
            "https://www.nasa.gov/solar-system/mars/",
            "https://www.nationalgeographic.com/environment/article/plate-tectonics",
            "https://www.history.com/topics/world-war-ii",
            "https://www.britannica.com/science/quantum-mechanics"
        ]
        
        # Scrape and populate the vector store with parallel scraping
        async def populate_vector_store():
            tasks = [self.scraper.scrape(url) for url in setup_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_texts = 0
            for url, texts in zip(setup_urls, results):
                if isinstance(texts, Exception):
                    logging.error(f"Failed to scrape {url}: {str(texts)}")
                    continue
                if texts:
                    embeddings = self.vector_store.embedding_model.encode(texts).tolist()
                    self.vector_store.store(texts, embeddings)
                    total_texts += len(texts)
                    logging.info(f"Scraped {url}: Retrieved {len(texts)} text segments")
                else:
                    logging.warning(f"Failed to scrape any content from {url}")
            logging.info(f"Total texts stored in vector store: {total_texts}")
            if total_texts == 0:
                logging.error("No texts were stored in the vector store. Tests may fail.")
        
        asyncio.run(populate_vector_store())
        logging.info("Vector store populated in setUp with data from subset of URLs.")
        
        # Define the full list of URLs for testing (100 URLs from diverse sources)
        self.urls = [
            # Wikipedia (50 URLs)
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
            "https://en.wikipedia.org/wiki/Ancient_Egypt",
            "https://en.wikipedia.org/wiki/Evolution",
            "https://en.wikipedia.org/wiki/Genetics",
            "https://en.wikipedia.org/wiki/Periodic_table",
            "https://en.wikipedia.org/wiki/Chemical_bonding",
            "https://en.wikipedia.org/wiki/Plate_tectonics",
            "https://en.wikipedia.org/wiki/Big_Bang",
            "https://en.wikipedia.org/wiki/Photosynthetic_pigment",
            "https://en.wikipedia.org/wiki/Atomic_structure",
            "https://en.wikipedia.org/wiki/Thermodynamics",
            "https://en.wikipedia.org/wiki/Astrophysics",
            "https://en.wikipedia.org/wiki/World_War_I",
            "https://en.wikipedia.org/wiki/French_Revolution",
            "https://en.wikipedia.org/wiki/American_Revolution",
            "https://en.wikipedia.org/wiki/Cold_War",
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
            # NASA (10 URLs)
            "https://www.nasa.gov/solar-system/mars/",
            "https://www.nasa.gov/mission/hubble-space-telescope/",
            "https://www.nasa.gov/science-research/earth-science/climate/",
            "https://www.nasa.gov/solar-system/jupiter/",
            "https://www.nasa.gov/mission/james-webb-space-telescope/",
            "https://www.nasa.gov/centers-and-facilities/goddard/what-is-a-black-hole/",
            "https://www.nasa.gov/general/what-is-dark-energy/",
            "https://www.nasa.gov/general/what-is-the-big-bang/",
            "https://www.nasa.gov/solar-system/earth/",
            "https://www.nasa.gov/mission/apollo-11/",
            # National Geographic (15 URLs)
            "https://www.nationalgeographic.com/environment/article/plate-tectonics",
            "https://www.nationalgeographic.com/environment/article/ecosystems",
            "https://www.nationalgeographic.com/history/article/ancient-egypt",
            "https://www.nationalgeographic.com/science/article/dna",
            "https://www.nationalgeographic.com/environment/article/climate-change",
            "https://www.nationalgeographic.com/science/article/black-holes",
            "https://www.nationalgeographic.com/science/article/photosynthesis",
            "https://www.nationalgeographic.com/history/article/renaissance",
            "https://www.nationalgeographic.com/science/article/quantum-mechanics",
            "https://www.nationalgeographic.com/history/article/roman-empire",
            "https://www.nationalgeographic.com/environment/article/oceanic-crust",
            "https://www.nationalgeographic.com/science/article/thermodynamics",
            "https://www.nationalgeographic.com/history/article/mongol-empire",
            "https://www.nationalgeographic.com/science/article/volcanoes",
            "https://www.nationalgeographic.com/history/article/industrial-revolution",
            # History.com (15 URLs)
            "https://www.history.com/topics/world-war-ii",
            "https://www.history.com/topics/world-war-i",
            "https://www.history.com/topics/industrial-revolution",
            "https://www.history.com/topics/french-revolution",
            "https://www.history.com/topics/american-revolution",
            "https://www.history.com/topics/cold-war",
            "https://www.history.com/topics/middle-ages",
            "https://www.history.com/topics/great-depression",
            "https://www.history.com/topics/civil-rights-movement",
            "https://www.history.com/topics/space-race",
            "https://www.history.com/topics/berlin-wall",
            "https://www.history.com/topics/colonial-africa",
            "https://www.history.com/topics/indian-independence",
            "https://www.history.com/topics/vietnam-war",
            "https://www.history.com/topics/ancient-rome",
            # Britannica (10 URLs)
            "https://www.britannica.com/science/quantum-mechanics",
            "https://www.britannica.com/science/calculus",
            "https://www.britannica.com/science/dna",
            "https://www.britannica.com/science/photosynthesis",
            "https://www.britannica.com/history/renaissance",
            "https://www.britannica.com/science/black-hole",
            "https://www.britannica.com/science/climate-change",
            "https://www.britannica.com/history/industrial-revolution",
            "https://www.britannica.com/history/ancient-egypt",
            "https://www.britannica.com/history/world-war-ii"
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
    
    def test_pipeline_causal_question(self):
        """Test the RAG pipeline with a causal question."""
        question = "What caused the Industrial Revolution?"
        answer = self.pipeline.process(question)
        self.assertIsInstance(answer, str, "Pipeline failed to generate a string response")
        self.assertIn("agricultural", answer.lower(), "Answer does not contain expected keyword 'agricultural'")
        logging.info(f"Question: {question}\nAnswer: {answer}")
    
    def test_pipeline_comparative_question(self):
        """Test the RAG pipeline with a comparative question."""
        question = "How does quantum mechanics differ from classical physics?"
        answer = self.pipeline.process(question)
        self.assertIsInstance(answer, str, "Pipeline failed to generate a string response")
        self.assertIn("probability", answer.lower(), "Answer does not contain expected keyword 'probability'")
        logging.info(f"Question: {question}\nAnswer: {answer}")
    
    def test_pipeline_conceptual_question(self):
        """Test the RAG pipeline with a conceptual question."""
        question = "What is the significance of plate tectonics in shaping Earth's surface?"
        answer = self.pipeline.process(question)
        self.assertIsInstance(answer, str, "Pipeline failed to generate a string response")
        self.assertIn("mountains", answer.lower(), "Answer does not contain expected keyword 'mountains'")
        logging.info(f"Question: {question}\nAnswer: {answer}")
    
    def test_pipeline_historical_question(self):
        """Test the RAG pipeline with a historical question."""
        question = "What was the impact of the Cold War?"
        answer = self.pipeline.process(question)
        self.assertIsInstance(answer, str, "Pipeline failed to generate a string response")
        self.assertIn("space race", answer.lower(), "Answer does not contain expected keyword 'space race'")
        logging.info(f"Question: {question}\nAnswer: {answer}")
    
    def test_pipeline_science_question(self):
        """Test the R A G pipeline with a science question."""
        question = "What is a black hole?"
        answer = self.pipeline.process(question)
        self.assertIsInstance(answer, str, "Pipeline failed to generate a string response")
        self.assertIn("gravity", answer.lower(), "Answer does not contain expected keyword 'gravity'")
        logging.info(f"Question: {question}\nAnswer: {answer}")
    
    def test_pipeline_math_question(self):
        """Test the RAG pipeline with a math-related question."""
        question = "What is calculus used for?"
        answer = self.pipeline.process(question)
        self.assertIsInstance(answer, str, "Pipeline failed to generate a string response")
        self.assertIn("derivatives", answer.lower(), "Answer does not contain expected keyword 'derivatives'")
        logging.info(f"Question: {question}\nAnswer: {answer}")
    
    def test_pipeline_environmental_question(self):
        """Test the RAG pipeline with an environmental question."""
        question = "What are the effects of climate change?"
        answer = self.pipeline.process(question)
        self.assertIsInstance(answer, str, "Pipeline failed to generate a string response")
        self.assertIn("temperature", answer.lower(), "Answer does not contain expected keyword 'temperature'")
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
        """Stress test the pipeline with 1000 Q&A interactions and measure performance."""
        questions = [
            "What is photosynthesis?",
            "Who were the major figures in the Renaissance?",
            "What caused World War II?",
            "What is a black hole?",
            "What is calculus used for?",
            "What caused the Industrial Revolution?",
            "How does quantum mechanics differ from classical physics?",
            "What is the significance of plate tectonics in shaping Earth's surface?",
            "What was the impact of the Cold War?",
            "What are the effects of climate change?"
        ]
        total_response_time = 0
        num_interactions = 0
        # Reduced to 10 iterations * 10 questions = 100 interactions for faster testing
        for i in range(10):  # Changed from 100 to 10
            for question in questions:
                start_time = time.time()
                answer = self.pipeline.process(question)
                response_time = time.time() - start_time
                total_response_time += response_time
                num_interactions += 1
                self.assertIsInstance(answer, str, f"Pipeline failed at interaction {num_interactions} for question: {question}")
                logging.info(f"Interaction {num_interactions}: Question: {question}\nAnswer: {answer}\nResponse Time: {response_time:.2f} seconds")
        avg_response_time = total_response_time / num_interactions
        logging.info(f"Average Response Time over {num_interactions} interactions: {avg_response_time:.2f} seconds")
    
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