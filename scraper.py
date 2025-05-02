import logging
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer
from typing import List

class WebScraper:
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        logging.info("WebScraper initialized.")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def scrape(self, url: str) -> List[str]:
        logging.info(f"Starting scrape for URL: {url}")
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            logging.info(f"Successfully fetched URL: {url}")
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = self.extract_text(soup)
            
            logging.info(f"Extracted {len(paragraphs)} paragraphs from {url}")
            return paragraphs
        except Exception as e:
            logging.error(f"Scraping failed for {url}: {str(e)}")
            raise
    
    def extract_text(self, soup: BeautifulSoup) -> List[str]:
        # Try different tags to extract text
        paragraphs = []
        
        # Try <p> tags
        p_tags = soup.find_all('p')
        paragraphs.extend([p.get_text(strip=True) for p in p_tags if p.get_text(strip=True)])
        
        # If no paragraphs found, try <div> tags with common classes
        if not paragraphs:
            div_tags = soup.find_all('div', class_=['content', 'article-body', 'text'])
            for div in div_tags:
                div_ps = div.find_all('p')
                paragraphs.extend([p.get_text(strip=True) for p in div_ps if p.get_text(strip=True)])
        
        # If still no paragraphs, try <article> tags
        if not paragraphs:
            article_tags = soup.find_all('article')
            for article in article_tags:
                article_ps = article.find_all('p')
                paragraphs.extend([p.get_text(strip=True) for p in article_ps if p.get_text(strip=True)])
        
        # Remove empty or very short paragraphs
        paragraphs = [p for p in paragraphs if len(p) > 50]
        return paragraphs