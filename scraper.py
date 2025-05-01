import aiohttp
import asyncio
from bs4 import BeautifulSoup
import logging

class WikipediaScraper:
    """A scraper for extracting text content from web pages."""
    
    def __init__(self):
        logging.info("WikipediaScraper initialized.")
    
    async def scrape(self, url: str) -> list:
        """Scrape text content from a given URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status != 200:
                        logging.error(f"Failed to fetch {url}: Status {response.status}")
                        return []
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Extract text from paragraphs, headings, and other relevant tags
                    texts = []
                    # Target different tags based on the website
                    if "wikipedia.org" in url:
                        # Wikipedia-specific scraping
                        content = soup.find('div', {'id': 'mw-content-text'})
                        if content:
                            paragraphs = content.find_all('p')
                            texts.extend([para.get_text(strip=True) for para in paragraphs if para.get_text(strip=True)])
                    elif "nasa.gov" in url:
                        # NASA-specific scraping
                        content = soup.find('div', class_='article-body') or soup.find('div', class_='content')
                        if content:
                            paragraphs = content.find_all('p')
                            texts.extend([para.get_text(strip=True) for para in paragraphs if para.get_text(strip=True)])
                    elif "nationalgeographic.com" in url:
                        # National Geographic-specific scraping (updated for new structure)
                        content = soup.find('div', class_='Article__Content') or soup.find('section', class_='article__body')
                        if content:
                            paragraphs = content.find_all('p')
                            texts.extend([para.get_text(strip=True) for para in paragraphs if para.get_text(strip=True)])
                    elif "history.com" in url:
                        # History.com-specific scraping (updated for new structure)
                        content = soup.find('div', class_='article-content') or soup.find('div', class_='content-wrapper')
                        if content:
                            paragraphs = content.find_all('p')
                            texts.extend([para.get_text(strip=True) for para in paragraphs if para.get_text(strip=True)])
                    elif "britannica.com" in url:
                        # Britannica-specific scraping (updated for new structure)
                        content = soup.find('div', class_='article-content') or soup.find('section', class_='md-content')
                        if content:
                            paragraphs = content.find_all('p')
                            texts.extend([para.get_text(strip=True) for para in paragraphs if para.get_text(strip=True)])
                    else:
                        # Generic scraping for other websites
                        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3'])
                        texts.extend([elem.get_text(strip=True) for elem in paragraphs if elem.get_text(strip=True)])
                    
                    # Filter out empty or very short texts
                    texts = [text for text in texts if len(text) > 50]
                    
                    if not texts:
                        logging.warning(f"No valid content extracted from {url}")
                    else:
                        logging.info(f"Extracted {len(texts)} text segments from {url}")
                    
                    return texts
        
        except Exception as e:
            logging.error(f"Error scraping {url}: {str(e)}")
            return []