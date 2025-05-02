import requests
from bs4 import BeautifulSoup
import logging
import tenacity
import sys
from all_Urls import URLS

# Configure execution log (for execution flow details)
exec_logging = logging.getLogger('execution')
exec_logging.setLevel(logging.INFO)
exec_handler = logging.FileHandler('execution_log.log')
exec_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
exec_logging.addHandler(exec_handler)

# Configure Q&A log (for Q&A-related logs)
qa_logging = logging.getLogger('qa')
qa_logging.setLevel(logging.INFO)
qa_handler = logging.FileHandler('rag_project.log')
qa_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
qa_logging.addHandler(qa_handler)

class BaseScraper:
    def __init__(self):
        self.qa_logging = qa_logging
        self.exec_logging = exec_logging

    def scrape(self, url: str) -> list:
        raise NotImplementedError("Subclasses must implement the scrape method")

class WikipediaScraper(BaseScraper):
    def scrape(self, url: str) -> list:
        self.exec_logging.info(f"Starting scrape for URL: {url}")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            self.exec_logging.info(f"Successfully fetched URL: {url}")
            soup = BeautifulSoup(response.text, 'html.parser')
            if "nasa.gov" in url.lower():
                paragraphs = soup.select('div.content, div.main-content p')
            else:
                paragraphs = soup.select('#mw-content-text p, .mw-parser-output p')
            texts = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
            filtered_texts = [text for text in texts if text and len(text) > 50]
            self.exec_logging.info(f"Extracted {len(filtered_texts)} paragraphs from {url}")
            return filtered_texts
        except Exception as e:
            self.exec_logging.error(f"Scraping failed for {url}: {str(e)}")
            self.qa_logging.error(f"Scraping failed for {url}: {str(e)}")
            return []

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(2))
    def scrape_with_retry(self, url: str) -> list:
        self.exec_logging.info(f"Attempting scrape with retry for {url}")
        return self.scrape(url)

def scrape_all_urls(urls: list) -> str:
    scraper = WikipediaScraper()
    all_texts = []
    for i, url in enumerate(urls, 1):
        scraper.exec_logging.info(f"Processing URL {i}/{len(urls)}: {url}")
        texts = scraper.scrape_with_retry(url)
        all_texts.extend(texts)
    return "\n".join(all_texts)

def save_scraped_data(urls: list, output_file: str = "scraped_data.txt"):
    scraper = WikipediaScraper()
    scraper.exec_logging.info("Starting scraping process for all URLs")
    try:
        all_data = scrape_all_urls(urls)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(all_data)
        scraper.exec_logging.info(f"Scraping completed. Data saved to {output_file}")
        scraper.qa_logging.info(f"Scraped data saved to {output_file}")
    except KeyboardInterrupt:
        scraper.exec_logging.error("Process interrupted by user (KeyboardInterrupt)")
        scraper.qa_logging.error("Process interrupted by user (KeyboardInterrupt)")
        sys.exit(1)
    except Exception as e:
        scraper.exec_logging.error(f"Unexpected error during scraping: {str(e)}")
        scraper.qa_logging.error(f"Unexpected error during scraping: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Example URLs (replace with your list)
    urls = URLS
    save_scraped_data(urls)