from bs4 import BeautifulSoup
import requests
import os
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentationScraper:
    """Base class for scraping CDP documentation"""
    
    def __init__(self, base_url: str, name: str):
        self.base_url = base_url
        self.name = name
        self.visited_urls = set()
        
    def fetch_page(self, url: str) -> BeautifulSoup:
        """Fetch and parse a page"""
        try:
            logger.info(f"Fetching {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return BeautifulSoup("", 'html.parser')
    
    def extract_text(self, soup: BeautifulSoup) -> str:
        """Extract relevant text from the page"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text(separator=" ", strip=True)
        return text
    
    def crawl(self, max_pages: int = 50) -> List[Dict[str, str]]:
        """Crawl documentation and extract content"""
        documents = []
        urls_to_visit = [self.base_url]
        
        while urls_to_visit and len(self.visited_urls) < max_pages:
            current_url = urls_to_visit.pop(0)
            if current_url in self.visited_urls:
                continue
                
            self.visited_urls.add(current_url)
            soup = self.fetch_page(current_url)
            if not soup:
                continue
                
            # Extract content
            content = self.extract_text(soup)
            if content:
                documents.append({
                    "source": current_url,
                    "content": content,
                    "platform": self.name
                })
            
            # Find additional links within the same domain
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/') or self.base_url in href:
                    if href.startswith('/'):
                        full_url = self.base_url.rstrip('/') + href
                    else:
                        full_url = href
                    
                    if full_url not in self.visited_urls and self.base_url in full_url:
                        urls_to_visit.append(full_url)
        
        logger.info(f"Scraped {len(documents)} documents from {self.name}")
        return documents


class SegmentScraper(DocumentationScraper):
    def __init__(self):
        super().__init__("https://segment.com/docs/", "Segment")
        
    def extract_text(self, soup: BeautifulSoup) -> str:
        # Focus on main content areas
        main_content = soup.select('main, .article-content, .documentation-content')
        if main_content:
            return " ".join(section.get_text(separator=" ", strip=True) for section in main_content)
        return super().extract_text(soup)


class MParticleScraper(DocumentationScraper):
    def __init__(self):
        super().__init__("https://docs.mparticle.com/", "mParticle")


class LyticsScraper(DocumentationScraper):
    def __init__(self):
        super().__init__("https://docs.lytics.com/", "Lytics")


class ZeotapScraper(DocumentationScraper):
    def __init__(self):
        super().__init__("https://docs.zeotap.com/home/en-us/", "Zeotap")


def load_all_documentation(max_pages_per_cdp: int = 50) -> List[Dict[str, Any]]:
    """Load documentation from all CDPs"""
    scrapers = [
        SegmentScraper(),
        MParticleScraper(),
        LyticsScraper(),
        ZeotapScraper()
    ]
    
    all_docs = []
    for scraper in scrapers:
        docs = scraper.crawl(max_pages=max_pages_per_cdp)
        all_docs.extend(docs)
    
    return all_docs