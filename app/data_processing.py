import logging
import os
from app.scraper import WikipediaScraper
from app.vector_store import BaseVectorStore

# Configure logging
logging.basicConfig(
    filename='rag_project.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def scrape_and_store(scraper: WikipediaScraper, vector_store: BaseVectorStore, urls: list, 
                     index_path: str = "faiss_index.bin", texts_path: str = "texts.pkl"):
    """Scrape data and store in vector store, or load existing data."""
    if os.path.exists(index_path) and os.path.exists(texts_path):
        logging.info("Existing FAISS index and texts found. Skipping scraping.")
        return
    
    logging.info("Scraping and storing new data...")
    all_texts = []
    for url in urls:
        texts = scraper.scrape(url)
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
    logging.info("Data scraping and storing completed.")