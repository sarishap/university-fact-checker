"""
rag_chain.py
------------
The guardrailed RAG chain. This is the most important file in the
whole system: it enforces the "zero hallucination" requirement.

Two layers of protection:
1. Similarity Score Thresholding — if the best matching chunk isn't
   similar enough to the question, we NEVER call the LLM at all.
   We short-circuit straight to the fallback message.
2. Strict System Prompt — even when we do call the LLM, we instruct
   it (in no uncertain terms) to answer ONLY from the provided
   context and to say it doesn't know otherwise.
"""

from dataclasses import dataclass

from transformers import pipeline

from vectorstore import ChromaVectorStore

FALLBACK_MESSAGE = "Information not found on the provided website."

SYSTEM_INSTRUCTIONS = (
    "You are a factual university Q&A assistant. Answer the user's question "
    "using ONLY the provided context snippets below. If the answer cannot be "
    "explicitly derived from the context, respond ONLY with: "
    f"'{FALLBACK_MESSAGE}' Do not use any internal knowledge."
)


@dataclass
class RAGAnswer:
    answer: str
    source_urls: list[str]
    similarity: float | None


class UniversityRAGChain:
    def __init__( 
        self,
        vectorstore: ChromaVectorStore,
        similarity_threshold: float = 0.35,
        top_k: int = 3,
        llm_model_name: str = "google/flan-t5-base",
    ):
        """
        similarity_threshold: cosine similarity cutoff in [0, 1].
        Anything below this is treated as "not relevant enough" and
        triggers the fallback response WITHOUT calling the LLM.
        Tune this value based on your embedding model and data.
        """
        self.vectorstore = vectorstore
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.llm = pipeline("text2text-generation", model=llm_model_name)

    def ask(self, question: str) -> RAGAnswer:
        results = self.vectorstore.query(question, top_k=self.top_k)

        if not results:
            return RAGAnswer(answer=FALLBACK_MESSAGE, source_urls=[], similarity=None)

        top_similarity = results[0]["similarity"]

        # --- Guardrail #1: Similarity Score Thresholding ---
        if top_similarity < self.similarity_threshold:
            return RAGAnswer(answer=FALLBACK_MESSAGE, source_urls=[], similarity=top_similarity)

        # Only keep chunks that pass the threshold as context.
        relevant_results = [r for r in results if r["similarity"] >= self.similarity_threshold]
        context = "\n\n".join(r["text"] for r in relevant_results)
        source_urls = list(dict.fromkeys(r["source_url"] for r in relevant_results))  # dedupe, keep order

        prompt = (
            f"{SYSTEM_INSTRUCTIONS}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        # --- Guardrail #2: Strict system prompt enforced at generation time ---
        try:
            generated = self.llm(prompt, max_length=256, do_sample=False)
            answer_text = generated[0]["generated_text"].strip()
        except Exception as e:
            print(f"[rag_chain] LLM generation failed: {e}")
            return RAGAnswer(answer=FALLBACK_MESSAGE, source_urls=[], similarity=top_similarity)

        if not answer_text:
            answer_text = FALLBACK_MESSAGE
            source_urls = []

        return RAGAnswer(answer=answer_text, source_urls=source_urls, similarity=top_similarity)
