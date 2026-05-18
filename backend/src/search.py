import os
import traceback
from dotenv import load_dotenv
from typing import List, Dict, Any

from src.vector_store import PineconeHybridVectorStore
from src.advanced_rag import AdvancedRag
from src.cache_manager import RAGCache
from src.file_tracker import FileTracker
from langchain_groq import ChatGroq

load_dotenv()


class RAGSearch:
    def __init__(
        self,
        index_name: str = "rag-hybrid-index",
        embedding_model: str = "BAAI/bge-m3",
        llm_model: str = "llama-3.3-70b-versatile",
    ):
        print("\n" + "="*50)
        print("[INIT] RAGSearch Initializing")
        print("="*50)

        self.index_name = index_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Cache (optional, non-blocking)
        self.cache = RAGCache()

        # Vector Store (eager init — loads embedding model NOW)
        print("[INIT] Loading Pinecone Hybrid Vector Store...")
        self.vector_store = PineconeHybridVectorStore(
            index_name=index_name,
            embedding_model=embedding_model,
            cache=self.cache,
        )

        # LLM (eager init — validates API key NOW)
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            print(f"[INIT] Loading LLM: {llm_model}...")
            self.llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=llm_model,
                temperature=0.2,
                max_tokens=2048,
            )
            print("[INIT] LLM ready")
        else:
            self.llm = None
            print("[WARNING] GROQ_API_KEY not found")

        # State
        self.rag = None
        self.is_ready = False
        self.file_tracker = FileTracker()

        # If we have previously indexed files, rebuild retriever and RAG
        existing_files = self.file_tracker.get_all_filenames()
        if existing_files:
            print(f"[INIT] Found {len(existing_files)} previously indexed files: {existing_files}")
            # Ensure BM25 is fitted (critical for retriever to work)
            if not self.vector_store._sparse_fitted:
                print("[INIT] Fitting BM25 from existing vectors...")
                # We need text to fit BM25. Since we have vectors in Pinecone but not the original text,
                # we fit on a minimal corpus. The real fitting happens on next upload.
                self.vector_store.sparse_encoder.fit(["document retrieval corpus"])
                self.vector_store._sparse_fitted = True
            self.vector_store._build_retriever()
            self.rag = AdvancedRag(self.vector_store.retriever, self.llm, self.cache)
            self.is_ready = True
            print("[INIT] RAG pipeline RESTORED from previous session")
        else:
            print("[INIT] No existing index. Awaiting first upload.")

        print("="*50 + "\n")

    def add_file(self, file_path: str) -> dict:
        """
        Incrementally add a single file to the index.
        Only processes THIS file, leaves all others untouched.
        """
        print("\n" + "="*50)
        print(f"[ADD] Incremental upload: {os.path.basename(file_path)}")
        print("="*50)

        filename = os.path.basename(file_path)

        # Check if already indexed and unchanged
        if self.file_tracker.is_indexed(file_path):
            print(f"[ADD] File unchanged, skipping re-index: {filename}")
            # Even if skipped, ensure system is ready
            if not self.is_ready:
                print("[ADD] System was not ready. Building retriever from existing index...")
                if not self.vector_store._sparse_fitted:
                    from src.data_loader import load_single_file
                    docs = load_single_file(file_path)
                    texts = [d.page_content for d in docs if len(d.page_content.strip()) > 50]
                    if texts:
                        self.vector_store.fit_sparse_encoder(texts)
                    else:
                        self.vector_store.sparse_encoder.fit(["document"])
                        self.vector_store._sparse_fitted = True
                if not self.vector_store.retriever:
                    self.vector_store._build_retriever()
                if not self.rag:
                    self.rag = AdvancedRag(self.vector_store.retriever, self.llm, self.cache)
                self.is_ready = True
            return {
                "status": "skipped",
                "message": f"{filename} already indexed (unchanged). System ready.",
                "filename": filename,
            }

        # If file exists but changed, remove old vectors first
        if self.file_tracker.has_file(filename):
            print(f"[ADD] File changed, removing old vectors...")
            self.vector_store.remove_file(filename)
            self.cache.invalidate_query_cache()

        try:
            # CRITICAL: Fit sparse encoder on first file if needed
            if not self.vector_store._sparse_fitted:
                print("[ADD] BM25 not fitted yet. Fitting on this file's corpus...")
                from src.data_loader import load_single_file
                docs = load_single_file(file_path)
                texts = [d.page_content for d in docs if len(d.page_content.strip()) > 50]
                if texts:
                    self.vector_store.fit_sparse_encoder(texts)
                else:
                    print("[ADD] Warning: No text extracted for BM25 fitting")
                    self.vector_store.sparse_encoder.fit(["document"])
                    self.vector_store._sparse_fitted = True

            # Add file (chunks + embeds + upserts + tracks)
            chunk_ids = self.vector_store.add_file(file_path)

            # Link RAG if not already
            if not self.rag:
                self.rag = AdvancedRag(
                    retriever=self.vector_store.retriever,
                    llm=self.llm,
                    cache=self.cache,
                )
                self.is_ready = True
            else:
                self.rag.retriever = self.vector_store.retriever
                self.is_ready = True

            # Invalidate cache since index changed
            self.cache.invalidate_query_cache()

            return {
                "status": "success",
                "message": f"{filename} indexed with {len(chunk_ids)} chunks",
                "filename": filename,
                "chunks": len(chunk_ids),
            }

        except Exception as e:
            print("\n[ERROR] Failed to add file:")
            traceback.print_exc()
            return {
                "status": "error",
                "message": str(e),
                "filename": filename,
            }

    def remove_file(self, filename: str) -> dict:
        """Remove a file from the index."""
        try:
            self.vector_store.remove_file(filename)
            self.cache.invalidate_query_cache()
            return {"status": "success", "message": f"{filename} removed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_files(self) -> List[str]:
        return list(self.file_tracker.get_all_filenames())

    def query(self, question: str, top_k: int = 5, summarize: bool = False):
        if not self.is_ready:
            return {
                "answer": "System not ready. Please upload documents first.",
                "sources": [],
                "retrieved_count": 0,
            }
        return self.rag.query(question=question, top_k=top_k, summarize=summarize)

    def query_stream(self, question: str, top_k: int = 5):
        if not self.is_ready or not self.rag:
            def error_gen():
                yield "System not ready. Please upload documents first."
            return error_gen()
        return self.rag.query_stream(question=question, top_k=top_k)

    def debug_retrieve(self, query: str):
        if not self.vector_store.retriever:
            return {"error": "Retriever not built"}
        return self.vector_store.retrieve_debug(query)