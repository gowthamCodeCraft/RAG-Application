from src.data_loader import load_all_docs
from src.vector_store import FaissVectorStore
from src.search import RAGSearch

if __name__ == "__main__":
    # docs = load_all_docs("data")

    rag_search = RAGSearch()
   
    vector_store = FaissVectorStore("faiss_store")
    # vector_store.build_from_documents(docs)
    # vector_store.load()
    
    response = rag_search.query("What is Business Intelligence?",top_k=3,min_score=0.1,stream=True,summarize=True)

    print("\nFinal Answer:\n", response["answer"])
    print("\nSummary:\n", response["summary"])
    print("\nHistory:\n", response["history"][-1])

    