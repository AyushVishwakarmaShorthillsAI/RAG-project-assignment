import unittest
import asyncio
from sentence_transformers import SentenceTransformer
from scraper import WikipediaScraper
from vector_store import FAISSVectorStore
from llm import HuggingFaceLLM
from rag_pipeline import RAGPipeline

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
    unittest.main()