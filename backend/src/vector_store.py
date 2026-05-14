import os 
import faiss
import numpy as np
import pickle
from typing import List, Any
from pinecone_text.sparse import BM25Encoder
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from src.embedding import EmbeddingPipeline

class PineconeHybridVectorStore:
    """
    Advanced Hybrid Search Vector Store (Dense + Sparse + Reranking)
    """
    def __init__(
        self,
        index_name: str = "rag-advanced-hybrid",
        embedding_model: str = "BAAI/bge-m3",
        chunk_size: int = 700,
        chunk_overlap: int = 120,
    ):
        self.index_name = index_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Embedding Pipeline (Advanced chunking + BGE-M3)
        self.embedding_pipeline = EmbeddingPipeline(
            model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.sparse_encoder = BM25Encoder()
        self.vector_store = None
        self.retriever = None

        print(f"[INFO] Pinecone Hybrid Vector Store initialized with {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        """Main method - Build Hybrid Index (Dense + Sparse + Reranker)"""
        print(f"[INFO] Building Advanced Hybrid Index from {len(documents)} documents...")

        # 1. Advanced Chunking
        chunks = self.embedding_pipeline.chunk_documents(documents)
        if not chunks:
            raise ValueError("No valid chunks created from documents")

        texts = [chunk.page_content for chunk in chunks]

        # 2. Fit BM25 Sparse Encoder
        self.sparse_encoder.fit(texts)
        print(f"[INFO] BM25 Sparse Encoder fitted on {len(texts)} chunks")

        # 3. Convert to LangChain Documents
        langchain_docs = [
            Document(
                page_content=chunk.page_content,
                metadata={
                    **chunk.metadata,
                    "source_file": chunk.metadata.get("source", "unknown"),
                    "page": chunk.metadata.get("page", 0)
                }
            )
            for chunk in chunks
        ]

        # 4. Create Pinecone Vector Store (Dense Embeddings)
        self.vector_store = PineconeVectorStore.from_documents(
            documents=langchain_docs,
            embedding=self.embedding_pipeline.model,   # SentenceTransformer works here
            index_name=self.index_name,
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        )

        # 5. Hybrid Retriever (Dense + Sparse)
        hybrid_retriever = PineconeHybridSearchRetriever(
            embeddings=self.embedding_pipeline.model,
            sparse_encoder=self.sparse_encoder,
            index_name=self.index_name,
            top_k=10,
            alpha=0.65,          # 0.5 = balanced, higher = more semantic
        )

        # 6. Add Cross-Encoder Reranker (Major quality boost)
        reranker = CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large"),
            top_n=5
        )

        self.retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=hybrid_retriever
        )

        print(f"✅ SUCCESS: Hybrid Index '{self.index_name}' built with reranking!")
        return self.retriever

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict]:
        """Same interface as your old FAISS retrieve method"""
        if not self.retriever:
            print("[WARNING] Retriever not initialized. Building now...")
            self.build_from_documents([])

        print(f"[INFO] Hybrid + Reranker retrieval for: '{query}'")

        compressed_docs = self.retriever.invoke(query)

        retrieved_docs = []
        for i, doc in enumerate(compressed_docs[:top_k]):
            score = doc.metadata.get("relevance_score", 0.75)
            if score < score_threshold:
                continue

            retrieved_docs.append({
                "id": i,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score),
                "rank": i + 1
            })

        print(f"[INFO] Retrieved {len(retrieved_docs)} high-quality documents")
        return retrieved_docs
