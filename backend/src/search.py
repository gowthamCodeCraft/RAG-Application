import os
from dotenv import load_dotenv
from src.vector_store import FaissVectorStore
from src.advanced_rag import AdvancedRag
from langchain_groq import ChatGroq
from src.data_loader import load_all_docs

load_dotenv()


class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "openai/gpt-oss-120b"):
        
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        self.vector_store = FaissVectorStore(persist_dir, embedding_model)
        self.llm = None
        self.rag = None
        self.is_ready = False

        # Load LLM (this is fine to do at startup)
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            self.llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=llm_model
            )
            print(f"[INFO] LLM initialized: {llm_model}")
        else:
            print("[WARNING] GROQ_API_KEY not found!")

        print("[INFO] RAGSearch initialized. Vector store will load on first use.")

    def initialize_vector_store(self):
        """Build or load vector store - Call this when needed"""
        if self.is_ready:
            return True

        try:
            faiss_path = os.path.join(self.persist_dir, "faiss.index")
            meta_path = os.path.join(self.persist_dir, "metas.pkl")

            if os.path.exists(faiss_path) and os.path.exists(meta_path):
                self.vector_store.load()
                print(f"[INFO] Loaded existing vector store from {self.persist_dir}")
            else:
                print(f"[INFO] No existing vector store found. Loading documents from data folder...")
                docs = load_all_docs("data")
                
                if not docs or len(docs) == 0:
                    print("[WARNING] No documents found in data folder. Upload some documents first.")
                    return False
                
                self.vector_store.build_from_documents(docs)
                print(f"[SUCCESS] Vector store built with {len(docs)} documents")

            self.rag = AdvancedRag(self.vector_store, self.llm)
            self.is_ready = True
            return True

        except Exception as e:
            print(f"[ERROR] Failed to initialize vector store: {e}")
            return False

    def query(self, question: str, top_k: int = 3, min_score: float = 0.2, stream: bool = False, summarize: bool = False):
        """Query method with lazy initialization"""
        if not self.is_ready:
            success = self.initialize_vector_store()
            if not success:
                return "No documents have been uploaded yet. Please upload some PDFs first."

        if not self.rag:
            return "RAG system is not ready. Please try again."

        return self.rag.query(
            question=question,
            top_k=top_k,
            min_score=min_score,
            stream=stream,
            summarize=summarize,
        )

    def rebuild_vector_store(self):
        """Call this after uploading new documents"""
        self.is_ready = False  # Force rebuild on next query
        return self.initialize_vector_store()