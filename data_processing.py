import asyncio
import logging
from typing import List
import torch
import os
try:
    from .scraper import BaseScraper
    from .vector_store import BaseVectorStore
except ImportError:
    from scraper import BaseScraper
    from vector_store import BaseVectorStore

# Configure logging
logging.basicConfig(filename='rag_project.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

async def scrape_and_store(scraper: BaseScraper, vector_store: BaseVectorStore, urls: List[str], 
                         index_path: str = "faiss_index.bin", texts_path: str = "texts.pkl"):
    """Scrape data asynchronously and store in vector store, or load existing data."""
    # Check if FAISS index and texts exist
    if os.path.exists(index_path) and os.path.exists(texts_path):
        logging.info("Existing FAISS index and texts found. Skipping scraping.")
        return
    
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
        vector_store.save(index_path, texts_path)
    else:
        logging.info("Collection already contains data. Skipping scraping and storing.")