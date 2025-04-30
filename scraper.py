import abc
import logging
import aiohttp
from bs4 import BeautifulSoup
from typing import List
import asyncio

# Configure logging
logging.basicConfig(filename='rag_project.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class BaseScraper(abc.ABC):
    """Abstract base class for web scrapers."""
    @abc.abstractmethod
    async def scrape(self, url: str) -> List[str]:
        """Scrape data from a given URL and return cleaned text."""
        pass

class WikipediaScraper(BaseScraper):
    """Concrete scraper for Wikipedia pages using async requests."""
    async def scrape(self, url: str, max_retries: int = 3) -> List[str]:
        for attempt in range(max_retries):
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(url, timeout=10) as response:
                        response.raise_for_status()
                        text = await response.text()
                        soup = BeautifulSoup(text, 'html.parser')
                        paragraphs = soup.find_all('p')
                        cleaned_text = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
                        logging.info(f"Scraped {len(cleaned_text)} paragraphs from {url}")
                        return cleaned_text
                except aiohttp.ClientResponseError as e:
                    if e.status in (429, 503) and attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logging.warning(f"Retry {attempt + 1}/{max_retries} for {url} after {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                        continue
                    logging.error(f"Error scraping {url}: {str(e)}")
                    return []
                except Exception as e:
                    logging.error(f"Error scraping {url}: {str(e)}")
                    return []
        logging.error(f"Failed to scrape {url} after {max_retries} attempts")
        return []