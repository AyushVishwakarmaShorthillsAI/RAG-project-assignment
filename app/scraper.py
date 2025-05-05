import requests
from bs4 import BeautifulSoup
import logging
import tenacity
from app.all_Urls import URLS
import os
# Ensure the logs directory exists
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, 'rag_project.log')

# print(log_file)

# Configure logging (use the same logger as other files)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WikipediaScraper:
    def scrape(self, url: str) -> list:
        logger.info(f"Starting scrape for URL: {url}")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            logger.info(f"Successfully fetched URL: {url}")
            soup = BeautifulSoup(response.text, 'html.parser')
            if "nasa.gov" in url.lower():
                paragraphs = soup.select('div.content, div.main-content p')
            else:
                paragraphs = soup.select('#mw-content-text p, .mw-parser-output p')
            texts = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
            filtered_texts = [text for text in texts if len(text) > 50]
            logger.info(f"Extracted {len(filtered_texts)} paragraphs from {url}")
            return filtered_texts
        except Exception as e:
            logger.error(f"Scraping failed for {url}: {str(e)}")
            return []

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(2))
    def scrape_with_retry(self, url: str) -> list:
        logger.info(f"Attempting scrape with retry for {url}")
        return self.scrape(url)

def save_scraped_data(urls: list, output_file: str = "scraped_data.txt"):
    scraper = WikipediaScraper()
    logger.info("Starting scraping process for all URLs")
    try:
        all_texts = []
        for i, url in enumerate(urls, 1):
            logger.info(f"Processing URL {i}/{len(urls)}: {url}")
            texts = scraper.scrape_with_retry(url)
            all_texts.extend(texts)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(all_texts))
        logger.info(f"Scraping completed. Data saved to {output_file}")
    except KeyboardInterrupt:
        logger.error("Process interrupted by user (KeyboardInterrupt)")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during scraping: {str(e)}")
        exit(1)

if __name__ == "__main__":
    save_scraped_data(URLS)