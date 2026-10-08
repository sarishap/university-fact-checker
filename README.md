# University RAG

A simple Python-based RAG system that crawls a university website, indexes the content in ChromaDB, and answers questions strictly from the scraped pages with a minimal Streamlit interface.

## How it works

1. We crawl a university website using a breadth-first scraper that stays within the same domain.
2. For each page, we remove navigation, footer, and script content, then extract clean text.
3. The crawled text is split into chunks and embedded with `all-MiniLM-L6-v2`.
4. The embeddings are stored in ChromaDB for semantic retrieval.
5. When a user asks a question, the system finds the most relevant chunks and checks their similarity score.
6. If the similarity is below a threshold, the system refuses to guess and returns a fallback answer instead of calling the LLM.
7. If the similarity is strong enough, the relevant context is passed to a lightweight text-generation model, and the answer is grounded only in the retrieved pages.

## Setup

```bash
pip install -r requirements.txt
```

## Run the UI

```bash
streamlit run streamlit_app.py
```

Then open the local URL Streamlit prints in your browser, enter a university root URL, and click the button to crawl and index the site.

## Project structure

```bash
crawler.py            Crawl pages from a website using a same-domain BFS strategy
chunker.py            Split crawled pages into text chunks for retrieval
vectorstore.py        Embed chunks and store them in ChromaDB
rag_chain.py          Retrieve relevant context and generate grounded answers
streamlit_app.py      Minimal Streamlit interface for indexing and querying
README.md             Project overview and setup instructions
requirements.txt      Python dependencies
```

## Guardrails against hallucination

1. Similarity thresholding in `rag_chain.py`
   - If the best matching chunk is below the configured threshold, the LLM is not called.
   - This prevents the system from answering questions with no reliable evidence.

2. Strict system prompt
   - Even when relevant context is found, the model is instructed to answer only from the provided source content.
   - If the context does not support the answer, it should say so.

## Known limitations

- No JavaScript rendering, so pages that require JS may not be captured.
- ChromaDB is reset when re-indexing, so indexing a new university overwrites the previous collection.
- The default model can be swapped depending on available hardware and quality needs.
