"""Advanced RAG pipeline with streaming, citations, history, and summary."""

from typing import List, Dict, Any, Optional
from src.vector_store import FaissVectorStore

class AdvancedRag:

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history: List[Dict[str, Any]] = []

    def _deduplicate(self, docs: List[Dict]) -> List[Dict]:
        """Remove duplicate documents based on (source, page) identity."""
        seen = set()
        unique = []
        for doc in docs:
            key = (
                doc["metadata"].get("source_file", ""),
                doc["metadata"].get("page", ""),
            )
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique

    def query(self,question: str,top_k: int = 3,min_score: float = 0.2,stream: bool = False,summarize: bool = False,) -> Dict[str, Any]:
        """
        Advanced RAG query with streaming, citations, history, and summary.

        Args:
            question:   The user's question.
            top_k:      Maximum number of documents to retrieve.
            min_score:  Minimum similarity score threshold.
            stream:     If True, stream the LLM answer token-by-token to stdout.
            summarize:  If True, append a 2-sentence summary to the result.

        Returns:
            A dict with keys: question, answer, sources, summary, history.
        """
        #Retrieve
        retrieved_docs = self.retriever.retrieve(
            question, top_k=top_k, score_threshold=min_score
        )
        retrieved_docs = self._deduplicate(retrieved_docs)

        # Build context & sources 
        if not retrieved_docs:
            answer = "No relevant information found."
            sources: List[Dict] = []
        else:
            context = "\n\n".join(doc["content"] for doc in retrieved_docs)

            sources = [
                {
                    "source": doc["metadata"].get(
                        "source_file",
                        doc["metadata"].get("score", "unknown"),
                    ),
                    "page": doc["metadata"].get("page", "unknown"),
                    "score": doc["similarity_score"],
                    "preview": doc["content"][:120] + "...",
                }
                for doc in retrieved_docs
            ]

            prompt = (
                "Use the following context to answer the question concisely.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )

            #Streaming
            if stream:
                # Stream the *answer* token-by-token via the LLM's streaming API.
                print("Generating answer (streaming):")
                answer_chunks: List[str] = []
                for chunk in self.llm.stream([prompt]):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    print(token, end="", flush=True)
                    answer_chunks.append(token)
                print() 
                answer = "".join(answer_chunks)
            else:
                llm_response = self.llm.invoke([prompt])
                answer = llm_response.content

        #citations
        citations = [
            f"[{i + 1}] {src['source']} (page {src['page']})"
            for i, src in enumerate(sources)
        ]
        answer_with_citations = (
            answer + "\n\nCitations:\n" + "\n".join(citations)
            if citations
            else answer
        )

        #summary 
        summary: Optional[str] = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_response = self.llm.invoke([summary_prompt])
            summary = summary_response.content

        #stores history 
        self.history.append(
            {
                "question": question,
                "answer": answer,
                "sources": sources,
                "summary": summary,
            }
        )

        return {
            "question": question,
            "answer": answer_with_citations,
            "sources": sources,
            "summary": summary,
            "history": self.history,
        }
