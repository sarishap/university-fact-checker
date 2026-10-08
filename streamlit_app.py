"""
streamlit_app.py
----------------
Basic Streamlit web app for the University RAG system.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st

from chunker import DocumentChunker
from crawler import UniversityCrawler
from rag_chain import UniversityRAGChain
from vectorstore import ChromaVectorStore

DB_PATH = "university_vectorstore"

st.set_page_config(page_title="University Fact Checker", page_icon="🎓")
st.title("🎓 University Fact Checker (RAG)")
st.caption("Ask questions grounded strictly in a university's own website content.")

if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chain" not in st.session_state:
    st.session_state.chain = None

with st.sidebar:
    st.header("1. Index a University Website")
    url = st.text_input("University root URL", placeholder="https://example.edu")
    depth = st.slider("Crawl depth", min_value=1, max_value=4, value=2)
    max_pages = st.slider("Max pages to crawl", min_value=5, max_value=100, value=30)
    threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.35, step=0.05)

    if st.button("Crawl & Index", type="primary"):
        if not url:
            st.error("Please enter a URL first.")
        else:
            with st.spinner("Crawling website..."):
                crawler = UniversityCrawler(max_depth=depth, max_pages=max_pages)
                pages = crawler.crawl(url)

            if not pages:
                st.error("No pages could be crawled from that URL.")
            else:
                with st.spinner(f"Chunking {len(pages)} pages..."):
                    chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)
                    chunks = chunker.chunk_pages(pages)

                with st.spinner(f"Embedding {len(chunks)} chunks into ChromaDB..."):
                    store = ChromaVectorStore(persist_directory=DB_PATH)
                    store.reset()
                    store.add_chunks(chunks)

                st.session_state.chain = UniversityRAGChain(
                    vectorstore=store, similarity_threshold=threshold
                )
                st.session_state.indexed = True
                st.success(f"Indexed {len(pages)} pages ({len(chunks)} chunks).")

st.header("2. Ask a Question")

if not st.session_state.indexed:
    st.info("Index a university website from the sidebar to get started.")
else:
    question = st.text_input("Your question")
    if st.button("Ask") and question:
        with st.spinner("Searching and generating answer..."):
            result = st.session_state.chain.ask(question)

        st.subheader("Answer")
        st.write(result.answer)

        if result.source_urls:
            st.subheader("Sources")
            for src in result.source_urls:
                st.markdown(f"- [{src}]({src})")

        if result.similarity is not None:
            st.caption(f"Top match similarity: {result.similarity:.2f}")
