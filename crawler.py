"""
crawler.py
----------
Crawls a university website starting from a root URL, following only
internal links (same domain) up to a configurable depth, and extracts
clean main-content text from each page (stripping navbars, footers,
scripts, and sidebars).
"""

import time
from collections import deque
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Tags that are almost never part of the "real" page content.
BOILERPLATE_TAGS = [
    "nav", "footer", "header", "aside", "script", "style",
    "noscript", "form", "iframe", "svg", "button",
]


@dataclass
class ScrapedPage:
    url: str
    title: str
    text: str


class UniversityCrawler:
    def __init__(
        self,
        max_depth: int = 2,
        max_pages: int = 50,
        request_timeout: int = 10,
        delay_seconds: float = 0.5,
        user_agent: str = "ExplainableFactCheckerBot/1.0",
    ):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.request_timeout = request_timeout
        self.delay_seconds = delay_seconds
        self.headers = {"User-Agent": user_agent}

    def crawl(self, root_url: str) -> list[ScrapedPage]:
        """
        Breadth-first crawl starting from root_url, restricted to the
        same domain, up to self.max_depth levels deep and self.max_pages
        total pages.
        """
        root_domain = urlparse(root_url).netloc

        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(root_url, 0)])
        pages: list[ScrapedPage] = []

        while queue and len(pages) < self.max_pages:
            url, depth = queue.popleft()

            if url in visited:
                continue
            visited.add(url)

            html = self._fetch(url)
            if html is None:
                continue

            soup = BeautifulSoup(html, "html.parser")
            title, clean_text = self._extract_clean_text(soup)

            if clean_text.strip():
                pages.append(ScrapedPage(url=url, title=title, text=clean_text))

            if depth < self.max_depth:
                for link in self._extract_internal_links(soup, url, root_domain):
                    if link not in visited:
                        queue.append((link, depth + 1))

            time.sleep(self.delay_seconds)  # be polite to the server

        return pages

    def _fetch(self, url: str) -> str | None:
        try:
            response = requests.get(url, headers=self.headers, timeout=self.request_timeout)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return None
            return response.text
        except requests.RequestException as e:
            print(f"[crawler] Skipping {url}: {e}")
            return None

    def _extract_clean_text(self, soup: BeautifulSoup) -> tuple[str, str]:
        """Remove boilerplate tags and return (title, main_text)."""
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else "Untitled Page"

        # Work on a copy so we don't mutate anything the caller might reuse.
        for tag_name in BOILERPLATE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Prefer <main> or <article> content if present; fall back to <body>.
        main_content = soup.find("main") or soup.find("article") or soup.find("body")
        if main_content is None:
            return title, ""

        text = main_content.get_text(separator=" ", strip=True)
        # Collapse excessive whitespace.
        text = " ".join(text.split())
        return title, text

    def _extract_internal_links(self, soup: BeautifulSoup, base_url: str, root_domain: str) -> list[str]:
        links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)

            # Only follow same-domain links; drop fragments and query noise.
            if parsed.netloc == root_domain and parsed.scheme in ("http", "https"):
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                links.append(clean_url)

        return links
