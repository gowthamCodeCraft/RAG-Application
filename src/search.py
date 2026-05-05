import os
from dotenv import load_dotenv
from src.vector_store import FaissVectorStore
from src.advanced_rag import AdvancedRag
from langchain_groq import ChatGroq

load_dotenv()


class RAGSearch:

    def __init__(self,persist_dir: str = "faiss_store",embedding_model: str = "all-MiniLM-L6-v2",llm_model: str = "openai/gpt-oss-120b"):
        
        self.vector_store = FaissVectorStore(persist_dir, embedding_model)

        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metas.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vector_store.build_from_documents(docs)
        else:
            self.vector_store.load()

        groq_api_key = os.getenv("GROQ_API_KEY")

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model
        )

        print(f"[INFO] LLM initialized: {llm_model}")
   
        self.rag = AdvancedRag(self.vector_store, self.llm)

    def query(self,question: str,top_k: int = 3,min_score: float = 0.2,stream: bool = False,summarize: bool = False):
        return self.rag.query(
            question=question,
            top_k=top_k,
            min_score=min_score,
            stream=stream,
            summarize=summarize,
        )
