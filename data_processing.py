import logging
import asyncio
from typing import List
from scraper import WikipediaScraper
from vector_store import FAISSVectorStore

async def scrape_and_store(scraper: WikipediaScraper, vector_store: FAISSVectorStore, urls: List[str], index_path: str, texts_path: str):
    """Scrape content from URLs and store in the vector store."""
    tasks = []
    for url in urls:
        tasks.append(scraper.scrape(url))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            logging.error(f"Failed to scrape {url}: {str(result)}")
            continue
        if not result:
            logging.warning(f"No content scraped from {url}")
            continue
        
        texts = result
        embeddings = vector_store.embedding_model.encode(texts).tolist()
        vector_store.store(texts, embeddings)
        logging.info(f"Stored content from {url}")
    
    vector_store.save(index_path, texts_path)