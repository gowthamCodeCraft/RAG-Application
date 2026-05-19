import os
import hashlib
from typing import List, Any, Dict, Optional

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
        super().__init__(**kwargs)
        self._index = index
        self._embeddings = embeddings
        self._sparse_encoder = sparse_encoder
        self._top_k = top_k
        self._namespace = namespace
        self._alpha = alpha

    def _get_relevant_documents(self, query: str) -> List[Document]:
        dense_vector = self._embeddings.embed_query(query)
        sparse_vector = self._sparse_encoder.encode_queries(query)

        if hasattr(sparse_vector, "to_dict"):
            sparse_vector = sparse_vector.to_dict()
        elif not isinstance(sparse_vector, dict):
            sparse_vector = {"indices": [], "values": []}

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

        print(f"[RETRIEVER] Pinecone returned {len(documents)} matches")
        return documents


class PineconeHybridVectorStore:
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

        stats = self.index.describe_index_stats()
        print(f"[STORE] Pinecone index stats: {stats}")

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"token": hf_token},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.embedding_pipeline = EmbeddingPipeline(
            model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            cache=cache,
        )

        self.sparse_encoder = BM25Encoder()
        self._sparse_fitted = False
        self.retriever = None
        self.file_tracker = FileTracker()

        print(f"[STORE] Initialized: {index_name}")

    def add_file(self, file_path: str, docs: List[Document]) -> List[str]:
        """
        Process pre-loaded documents and upsert vectors.
        The caller (search.py) is responsible for loading documents.
        """
        filename = os.path.basename(file_path)
        print(f"\n{'='*50}")
        print(f"[ADD] Indexing file: {filename}")
        print(f"{'='*50}")

        if not docs:
            print(f"[ADD] No documents provided")
            return []

        chunks = self.embedding_pipeline.chunk_documents(docs, source_file=filename)
        if not chunks:
            print(f"[ADD] No valid chunks")
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
            print(f"[ADD] Upserted batch {i//batch_size + 1}")

        self.file_tracker.register(file_path, chunk_ids)
        self._build_retriever()

        print(f"[ADD] Done: {len(chunk_ids)} vectors")
        print(f"{'='*50}\n")
        return chunk_ids

    def remove_file(self, filename: str):
        chunk_ids = self.file_tracker.get_chunk_ids(filename)
        if chunk_ids:
            self.index.delete(ids=chunk_ids, namespace=self.namespace)
            print(f"[REMOVE] Deleted {len(chunk_ids)} vectors")
        self.file_tracker.unregister(filename)
        self._build_retriever()

    def _build_retriever(self):
        print("[STORE] Building retrieval pipeline...")

        if not self._sparse_fitted:
            print("[STORE] WARNING: BM25 not fitted. Fitting placeholder...")
            self.sparse_encoder.fit(["document retrieval system"])
            self._sparse_fitted = True

        base_retriever = DirectHybridRetriever(
            index=self.index,
            embeddings=self.embeddings,
            sparse_encoder=self.sparse_encoder,
            top_k=20,
            namespace=self.namespace,
            alpha=0.5,
        )

        # Memory-efficient reranker loading
        # BGE-Reranker-Large is ~1.2GB. On Windows with limited RAM, use smaller model.
        import torch

        # Detect available memory and choose appropriate model
        try:
            if torch.cuda.is_available():
                mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            else:
                import psutil
                mem_gb = psutil.virtual_memory().available / (1024**3)
        except:
            mem_gb = 8  # Default assumption

        # Use smaller model if memory is tight (< 6GB available)
        if mem_gb < 6:
            reranker_model_name = "BAAI/bge-reranker-base"  # ~400MB vs 1.2GB
            print(f"[STORE] Low memory detected ({mem_gb:.1f}GB). Using smaller reranker: {reranker_model_name}")
        else:
            reranker_model_name = "BAAI/bge-reranker-large"
            print(f"[STORE] Using reranker: {reranker_model_name}")

        print(f"[STORE] Step 3/4: Loading reranker model {reranker_model_name}...")
        print("[STORE] This may take 30-60 seconds on first run (model download)...")

        try:
            reranker_model = HuggingFaceCrossEncoder(
                model_name=reranker_model_name,
                model_kwargs={
                    "token": os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
                    "device": "cpu",
                },
            )
            print(f"[STORE] Reranker model loaded successfully ✓")
        except Exception as e:
            print(f"[STORE] Failed to load reranker: {e}")
            print("[STORE] Retrying without memory limits...")
            reranker_model = HuggingFaceCrossEncoder(
                model_name=reranker_model_name,
                model_kwargs={
                    "token": os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
                },
            )

        print("[STORE] Step 4/4: Building compression retriever...")
        reranker = CrossEncoderReranker(model=reranker_model, top_n=8)

        self.retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever,
        )
        print("[STORE] Pipeline ready")

    def fit_sparse_encoder(self, sample_texts: List[str]):
        if not self._sparse_fitted and sample_texts:
            print("[STORE] Fitting BM25...")
            self.sparse_encoder.fit(sample_texts)
            self._sparse_fitted = True
            print("[STORE] BM25 fit complete")

    def retrieve_debug(self, query: str, top_k: int = 5) -> Dict:
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
            "raw_results": [{"score": d.metadata.get("pinecone_score", 0), "source": d.metadata.get("source", "?"), "preview": d.page_content[:200]} for d in raw_docs[:5]],
            "reranked_count": len(final_docs),
            "reranked_results": [{"score": d.metadata.get("relevance_score", 0), "source": d.metadata.get("source", "?"), "preview": d.page_content[:200]} for d in final_docs[:top_k]],
        }