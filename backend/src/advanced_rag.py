import hashlib
from typing import List, Dict, Any, Optional, Generator
from langchain_core.messages import HumanMessage
from src.cache_manager import RAGCache

class AdvancedRag:
    def __init__(self, retriever, llm, cache: RAGCache = None):
        print("[RAG] Initializing AdvancedRag...")
        self.retriever = retriever
        self.llm = llm
        self.cache = cache
        self.history: List[Dict[str, Any]] = []

    def _extract(self, doc):
        if isinstance(doc, dict):
            return doc.get("content", ""), doc.get("metadata", {})
        content = getattr(doc, "page_content", "")
        metadata = getattr(doc, "metadata", {})
        return content, metadata

    def _deduplicate(self, docs: List[Any]) -> List[Any]:
        """Remove only truly identical content using content hash."""
        seen = set()
        unique = []
        for doc in docs:
            content, metadata = self._extract(doc)
            sig = hashlib.md5(content[:150].encode()).hexdigest()[:12]
            source = metadata.get("source", metadata.get("source_file", "unknown"))
            key = (source, sig)
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        print(f"[RAG] Deduplication: {len(docs)} -> {len(unique)} unique")
        return unique

    def _build_context(self, docs: List[Any]) -> tuple:
        """Build context string and sources list from documents."""
        context_parts = []
        sources = []
        for idx, doc in enumerate(docs):
            content, metadata = self._extract(doc)
            header = metadata.get("header_context", "")
            if header:
                part = f"[Source {idx+1}: {metadata.get('source','?')} | {header}]\n{content}"
            else:
                part = f"[Source {idx+1}: {metadata.get('source','?')}]\n{content}"
            context_parts.append(part)
            sources.append({
                "source": metadata.get("source", metadata.get("source_file", "unknown")),
                "page": metadata.get("page", "N/A"),
                "score": metadata.get("relevance_score", metadata.get("pinecone_score", 0)),
                "preview": content[:200] + "...",
            })
        return "\n\n---\n\n".join(context_parts), sources

    def _detect_empty_context(self, context: str, sources: List[dict]) -> dict:

        """Diagnose why context might be empty or useless."""
        issues = []
        if not context.strip():
            issues.append("Context string is completely empty")
        if len(sources) == 0:
            issues.append("No source documents retrieved")
        if len(context.strip()) < 200:
            issues.append(f"Context is very short ({len(context)} chars)")
        alpha_count = sum(1 for c in context if c.isalpha())
        if alpha_count < 100:
            issues.append("Context contains almost no readable text")
        return {
            "is_empty": len(issues) > 0,
            "issues": issues,
            "context_length": len(context),
            "source_count": len(sources),
        }

    # Use invoke() for all retrievers
    def _retrieve(self, question: str) -> List[Any]:
        """
        Universal retrieval method.
        Works with BaseRetriever, ContextualCompressionRetriever, and any LangChain retriever.
        """
        # Modern LangChain (v0.1+): use invoke()
        if hasattr(self.retriever, "invoke"):
            return self.retriever.invoke(question)
        # Legacy fallback
        if hasattr(self.retriever, "get_relevant_documents"):
            return self.retriever.invoke(question)
        raise AttributeError("Retriever has no invoke() or get_relevant_documents() method")

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.0,
        summarize: bool = False,
    ) -> Dict[str, Any]:
        print(f"[QUERY] {question}")

        # Check cache
        if self.cache:
            cached = self.cache.get_query_result(question, top_k)
            if cached:
                print("[QUERY] Cache HIT")
                return {
                    "question": question,
                    "answer": cached["answer"],
                    "sources": cached["sources"],
                    "retrieved_count": cached["retrieved_count"],
                    "cached": True,
                }

        try:
            # FIX: Use _retrieve() instead of get_relevant_documents()
            print("[STEP 1] Hybrid retrieval + reranking...")
            docs = self._retrieve(question)
            print(f"[STEP 1] Retrieved {len(docs)} documents")

            for i, doc in enumerate(docs[:3]):
                preview = doc.page_content[:200].replace("\n", " ")
                score = doc.metadata.get('relevance_score', 'N/A')
                print(f"  [{i+1}] Score: {score} | {preview}...")

            docs = self._deduplicate(docs)
            docs = [d for d in docs if (d.metadata.get("relevance_score", 0.75) >= min_score)]
            docs = docs[:top_k]

            final_docs = docs
            context, sources = self._build_context(final_docs)
            debug_info = self._detect_empty_context(context, sources)

            if debug_info["is_empty"]:
                print(f"[QUERY] WARNING: Empty context detected!")
                for issue in debug_info["issues"]:
                    print(f"         - {issue}")

            if not final_docs:
                answer = "I don't have enough information in the uploaded documents to answer this."
                result = {
                    "question": question,
                    "answer": answer,
                    "sources": [],
                    "retrieved_count": 0,
                    "debug": debug_info,
                }
                if self.cache:
                    self.cache.set_query_result(question, result, top_k)
                return result

            prompt = f"""You are a precise AI assistant. Answer using ONLY the provided context.
If the answer is not found in the context, respond exactly with: "I don't have sufficient information to answer this."
Cite sources using [Source N] format when referencing information.
Be concise but complete.

Context:
{context}

Question: {question}

Answer:"""

            messages = [HumanMessage(content=prompt)]
            print(f"[STEP 2] Sending {len(context)} chars context to LLM...")

            response = self.llm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)

            citations = [
                f"[{i+1}] {s['source']} (Page {s['page']}) — Score: {s['score']:.3f}"
                for i, s in enumerate(sources)
            ]
            final_answer = answer.strip() + "\n\n**Sources:**\n" + "\n".join(citations)

            summary = None
            if summarize:
                s_resp = self.llm.invoke([HumanMessage(content=f"Summarize in 2 sentences:\n{answer}")])
                summary = s_resp.content if hasattr(s_resp, "content") else str(s_resp)

            result = {
                "question": question,
                "answer": final_answer,
                "raw_answer": answer,
                "sources": sources,
                "summary": summary,
                "retrieved_count": len(final_docs),
                "debug": debug_info,
            }

            if self.cache:
                self.cache.set_query_result(question, result, top_k)

            self.history.append({"question": question, "answer": answer, "sources": sources})
            return result

        except Exception as e:
            print(f"\n[ERROR] Query failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "question": question,
                "answer": "An internal error occurred during retrieval.",
                "sources": [],
                "retrieved_count": 0,
            }

    def query_stream(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> Generator[str, None, None]:
        print("\n" + "="*50)
        print(f"[STREAM] {question}")
        print("="*50)

        try:
            yield "[STATUS] Retrieving documents...\n"

            # FIX: Use _retrieve() instead of get_relevant_documents()
            docs = self._retrieve(question)
            yield f"[STATUS] Found {len(docs)} candidates\n"

            docs = self._deduplicate(docs)
            docs = [d for d in docs if (d.metadata.get("relevance_score", 0.75) >= min_score)]
            docs = docs[:top_k]

            context, sources = self._build_context(docs)

            if not docs:
                yield "[STATUS] No relevant documents found\n"
                yield "I don't have enough information in the uploaded documents to answer this."
                return

            yield f"[STATUS] Generating answer from {len(docs)} sources...\n"
            yield "[BEGIN]\n"

            prompt = f"""You are a precise AI assistant. Answer using ONLY the provided context.
If the answer is not in the context, say exactly: "I don't have sufficient information to answer this."

Context:
{context}

Question: {question}

Answer:"""

            messages = [HumanMessage(content=prompt)]
            buffer = ""
            for chunk in self.llm.stream(messages):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                buffer += token
                yield token

            yield "\n\n**Sources:**\n"
            for i, s in enumerate(sources):
                yield f"[{i+1}] {s['source']} (Page {s['page']}) — Score: {s['score']:.3f}\n"

            self.history.append({"question": question, "answer": buffer, "sources": sources})

        except Exception as e:
            print(f"\n[ERROR] Stream failed: {e}")
            import traceback
            traceback.print_exc()
            yield f"\n[ERROR] {str(e)}"