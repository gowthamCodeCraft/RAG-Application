import os
import traceback
from dotenv import load_dotenv
from typing import List, Dict, Any

from src.vector_store import PineconeHybridVectorStore
from src.advanced_rag import AdvancedRag
from src.cache_manager import RAGCache
from src.file_tracker import FileTracker
from src.multimodel_data_loader import MultimodelDocumentProcessor
from langchain_groq import ChatGroq

load_dotenv()


class RAGSearch:
    def __init__(
        self,
        index_name: str = "rag-hybrid-index",
        embedding_model: str = "BAAI/bge-m3",
        llm_model: str = "llama-3.3-70b-versatile",
        vision_llm=None,
    ):
        print("\n" + "="*50)
        print("[INIT] RAGSearch Initializing")
        print("="*50)

        self.index_name = index_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Cache
        self.cache = RAGCache()

        # Multimodal processor (handles PDFs, images, OCR, vision)
        self.multimodal_processor = MultimodelDocumentProcessor(vision_llm=vision_llm)

        # Vector Store
        print("[INIT] Loading Pinecone Hybrid Vector Store...")
        self.vector_store = PineconeHybridVectorStore(
            index_name=index_name,
            embedding_model=embedding_model,
            cache=self.cache,
        )

        # LLM
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

        # Restore from previous session
        existing_files = self.file_tracker.get_all_filenames()
        if existing_files:
            print(f"[INIT] Found {len(existing_files)} previously indexed files")
            if not self.vector_store._sparse_fitted:
                self.vector_store.sparse_encoder.fit(["document retrieval corpus"])
                self.vector_store._sparse_fitted = True
            self.vector_store._build_retriever()
            self.rag = AdvancedRag(self.vector_store.retriever, self.llm, self.cache)
            self.is_ready = True
            print("[INIT] RAG pipeline RESTORED")
        else:
            print("[INIT] No existing index. Awaiting first upload.")

        print("="*50 + "\n")

    def add_file(self, file_path: str) -> dict:
        """Incrementally add a single file. Only processes THIS file."""
        print("\n" + "="*50)
        print(f"[ADD] Incremental upload: {os.path.basename(file_path)}")
        print("="*50)

        filename = os.path.basename(file_path)

        # Check if already indexed
        if self.file_tracker.is_indexed(file_path):
            print(f"[ADD] File unchanged, skipping: {filename}")
            if not self.is_ready:
                if not self.vector_store._sparse_fitted:
                    docs = self.multimodal_processor.load_file(file_path)
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
                "message": f"{filename} already indexed. System ready.",
                "filename": filename,
            }

        # Remove old vectors if file changed
        if self.file_tracker.has_file(filename):
            print(f"[ADD] File changed, removing old vectors...")
            self.vector_store.remove_file(filename)
            self.cache.invalidate_query_cache()

        try:
            # Load documents using multimodal processor
            print(f"[ADD] Loading documents from {filename}...")
            docs = self.multimodal_processor.load_file(file_path)

            if not docs:
                return {"status": "error", "message": "No content extracted", "filename": filename}

            # Fit BM25 if needed
            if not self.vector_store._sparse_fitted:
                texts = [d.page_content for d in docs if len(d.page_content.strip()) > 50]
                if texts:
                    self.vector_store.fit_sparse_encoder(texts)
                else:
                    self.vector_store.sparse_encoder.fit(["document"])
                    self.vector_store._sparse_fitted = True

            # Index the documents
            chunk_ids = self.vector_store.add_file(file_path, docs)

            # Link RAG
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
            return {"status": "error", "message": str(e), "filename": filename}

    def remove_file(self, filename: str) -> dict:
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