import os
import hashlib
from typing import List, Any, Dict, Optional, ClassVar

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings

from src.embedding import EmbeddingPipeline
from src.file_tracker import FileTracker
from src.cache_manager import RAGCache


class DirectHybridRetriever(BaseRetriever):
    """
    Production hybrid retriever using official Pinecone client.
    Dense (BGE-M3) + Sparse (BM25) simultaneous query.
    """

    # FIX: Pydantic v2 model_config allows arbitrary types (Pinecone Index is not a Pydantic model)
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        index,
        embeddings,
        sparse_encoder,
        top_k: int = 20,
        namespace: str = "",
        alpha: float = 0.5,
        **kwargs
    ):
        # FIX: Store as private attributes to bypass Pydantic validation
        super().__init__(**kwargs)
        self._index = index
        self._embeddings = embeddings
        self._sparse_encoder = sparse_encoder
        self._top_k = top_k
        self._namespace = namespace
        self._alpha = alpha

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Dense vector
        dense_vector = self._embeddings.embed_query(query)

        # Sparse vector (BM25)
        sparse_vector = self._sparse_encoder.encode_queries(query)
        if hasattr(sparse_vector, "to_dict"):
            sparse_vector = sparse_vector.to_dict()
        elif not isinstance(sparse_vector, dict):
            sparse_vector = {"indices": [], "values": []}

        # Hybrid query with alpha blending
        results = self._index.query(
            namespace=self._namespace,
            vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=self._top_k,
            include_metadata=True,
        )

        documents = []
        if not results.matches:
            print(f"[RETRIEVER] WARNING: 0 matches for query")
            return documents

        for match in results.matches:
            metadata = match.metadata or {}
            content = metadata.pop("text", "") if "text" in metadata else ""
            metadata["pinecone_score"] = float(match.score)
            metadata["relevance_score"] = float(match.score)
            documents.append(Document(page_content=content, metadata=metadata))

        print(f"[RETRIEVER] Pinecone returned {len(documents)} matches (alpha={self._alpha})")
        return documents


class PineconeHybridVectorStore:
    """
    Incremental Hybrid Vector Store.
    - Only adds vectors for new/changed files
    - Tuned BM25 + Dense + CrossEncoder reranking
    """

    def __init__(
        self,
        index_name: str = "rag-hybrid-index",
        embedding_model: str = "BAAI/bge-m3",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        namespace: str = "",
        cache: RAGCache = None,
    ):
        self.index_name = index_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.namespace = namespace
        self.cache = cache

        pc_api_key = os.getenv("PINECONE_API_KEY")
        if not pc_api_key:
            raise ValueError("PINECONE_API_KEY not set")

        self.pc = Pinecone(api_key=pc_api_key)
        self.index = self.pc.Index(self.index_name)

        # Stats
        stats = self.index.describe_index_stats()
        print(f"[STORE] Pinecone index stats: {stats}")

        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        # Dense embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"token": hf_token},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Chunking
        self.embedding_pipeline = EmbeddingPipeline(
            model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            cache=cache,
        )

        # Sparse encoder
        self.sparse_encoder = BM25Encoder()
        self._sparse_fitted = False

        self.retriever = None
        self.file_tracker = FileTracker()

        print(f"[STORE] Initialized: {index_name} | model={embedding_model}")

    def add_file(self, file_path: str) -> List[str]:
        """Process a SINGLE file and upsert its vectors."""
        from src.data_loader import load_single_file

        filename = os.path.basename(file_path)
        print(f"\n{'='*50}")
        print(f"[ADD] Processing file: {filename}")
        print(f"{'='*50}")

        docs = load_single_file(file_path)
        if not docs:
            print(f"[ADD] No content extracted from {filename}")
            return []

        chunks = self.embedding_pipeline.chunk_documents(docs, source_file=filename)
        if not chunks:
            print(f"[ADD] No valid chunks from {filename}")
            return []

        texts = [c.page_content for c in chunks]
        dense_embeddings = self.embedding_pipeline.embed_chunks(chunks)

        vectors = []
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            dense = dense_embeddings[i].tolist()
            sparse = self.sparse_encoder.encode_documents([chunk.page_content])[0]
            if hasattr(sparse, "to_dict"):
                sparse = sparse.to_dict()

            content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:12]
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in filename)[:40]
            chunk_id = f"{safe_name}_{i}_{content_hash}"

            chunk_ids.append(chunk_id)
            vectors.append({
                "id": chunk_id,
                "values": dense,
                "sparse_values": sparse,
                "metadata": {
                    **(chunk.metadata or {}),
                    "text": chunk.page_content,
                    "chunk_index": i,
                    "source_file": filename,
                }
            })

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)
            print(f"[ADD] Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")

        self.file_tracker.register(file_path, chunk_ids)
        self._build_retriever()

        print(f"[ADD] Done: {filename} -> {len(chunk_ids)} vectors")
        print(f"{'='*50}\n")
        return chunk_ids

    def remove_file(self, filename: str):
        """Remove all vectors belonging to a specific file."""
        chunk_ids = self.file_tracker.get_chunk_ids(filename)
        if chunk_ids:
            self.index.delete(ids=chunk_ids, namespace=self.namespace)
            print(f"[REMOVE] Deleted {len(chunk_ids)} vectors for {filename}")
        self.file_tracker.unregister(filename)
        self._build_retriever()

    def _build_retriever(self):
        """Build or rebuild the hybrid + rerank retrieval pipeline."""
        print("[STORE] Building retrieval pipeline...")

        # CRITICAL FIX: Ensure BM25 is fitted before building retriever
        if not self._sparse_fitted:
            print("[STORE] WARNING: BM25 not fitted. Fitting on available corpus...")
            # Try to get texts from existing vectors in Pinecone
            try:
                # Fetch a sample of vectors to fit BM25
                stats = self.index.describe_index_stats()
                vector_count = stats.total_vector_count
                if vector_count > 0:
                    # We can't easily fetch all texts back from Pinecone metadata
                    # So we fit on a dummy corpus and warn user to re-upload
                    print("[STORE] Cannot auto-fit BM25 from Pinecone vectors alone.")
                    print("[STORE] Please re-upload your documents to fit BM25 properly.")
                    # Fit on minimal corpus to prevent crash
                    self.sparse_encoder.fit(["placeholder document for bm25 fitting"])
                    self._sparse_fitted = True
                else:
                    self.sparse_encoder.fit(["empty corpus"])
                    self._sparse_fitted = True
            except Exception as e:
                print(f"[STORE] BM25 auto-fit failed: {e}")
                # Last resort: fit on dummy data so retriever doesn't crash
                self.sparse_encoder.fit(["document"])
                self._sparse_fitted = True

        base_retriever = DirectHybridRetriever(
            index=self.index,
            embeddings=self.embeddings,
            sparse_encoder=self.sparse_encoder,
            top_k=20,
            namespace=self.namespace,
            alpha=0.5,
        )

        reranker_model = HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-large",
            model_kwargs={"token": os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")},
        )
        reranker = CrossEncoderReranker(model=reranker_model, top_n=8)

        self.retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever,
        )
        print("[STORE] Pipeline ready: Hybrid (top-20) -> Reranker (top-8)")

    def fit_sparse_encoder(self, sample_texts: List[str]):
        """Fit BM25 on a sample of texts."""
        if not self._sparse_fitted and sample_texts:
            print("[STORE] Fitting BM25 sparse encoder...")
            self.sparse_encoder.fit(sample_texts)
            self._sparse_fitted = True
            print("[STORE] BM25 fit complete")

    def retrieve_debug(self, query: str, top_k: int = 5) -> Dict:
        """Debug retrieval: show raw hybrid results vs reranked results."""
        if not self.retriever:
            return {"error": "Retriever not built"}

        base = DirectHybridRetriever(
            index=self.index,
            embeddings=self.embeddings,
            sparse_encoder=self.sparse_encoder,
            top_k=20,
            namespace=self.namespace,
            alpha=0.5,
        )
        raw_docs = base.get_relevant_documents(query)
        final_docs = self.retriever.invoke(query)

        return {
            "query": query,
            "raw_count": len(raw_docs),
            "raw_results": [
                {"score": d.metadata.get("pinecone_score", 0), "source": d.metadata.get("source", "?"), "preview": d.page_content[:200]}
                for d in raw_docs[:5]
            ],
            "reranked_count": len(final_docs),
            "reranked_results": [
                {"score": d.metadata.get("relevance_score", 0), "source": d.metadata.get("source", "?"), "preview": d.page_content[:200]}
                for d in final_docs[:top_k]
            ],
        }