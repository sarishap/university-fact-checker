"""
chunker.py
----------
Splits scraped page text into semantic chunks using LangChain's
RecursiveCharacterTextSplitter, and attaches source metadata
(URL, title) to every chunk so answers can always be traced back
to the page they came from.
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from crawler import ScrapedPage


@dataclass
class Chunk:
    text: str
    source_url: str
    source_title: str


class DocumentChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_pages(self, pages: list[ScrapedPage]) -> list[Chunk]:
        """Split every scraped page into overlapping text chunks."""
        chunks: list[Chunk] = []

        for page in pages:
            for piece in self.splitter.split_text(page.text):
                if piece.strip():
                    chunks.append(
                        Chunk(text=piece.strip(), source_url=page.url, source_title=page.title)
                    )

        return chunks
