import logging
import numpy as np
from typing import List

def scrape_and_store(scraper, vector_store, urls: List[str], index_path: str, texts_path: str):
    """Scrape content from URLs, embed it, and store in a vector store."""
    texts = []
    for url in urls:
        try:
            paragraphs = scraper.scrape(url)
            texts.extend(paragraphs)
        except Exception as e:
            logging.error(f"Error processing URL {url}: {e}")
    
    # Clean and filter texts
    texts = [text.strip() for text in texts if text.strip()]
    if not texts:
        logging.warning("No texts extracted from the provided URLs.")
        return
    
    # Generate embeddings
    logging.info(f"Generating embeddings for {len(texts)} texts...")
    embeddings = scraper.embedding_model.encode(texts, show_progress_bar=True).tolist()
    
    # Store in vector store
    logging.info("Storing texts and embeddings in vector store...")
    vector_store.store(texts, embeddings)
    vector_store.save(index_path, texts_path)
    logging.info("Scraping and indexing completed.")