# src/embedding.py
from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        chunk_size: int = 700,
        chunk_overlap: int = 120,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        
        print(f"[INFO] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"[SUCCESS] Loaded {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """Advanced multi-strategy chunking"""
        all_chunks = []

        for doc in documents:
            # 1. Semantic Chunking (Best for context preservation)
            try:
                semantic_splitter = SemanticChunker(
                    self.model,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=85,
                )
                semantic_chunks = semantic_splitter.split_documents([doc])
            except:
                semantic_chunks = []

            # 2. Recursive Character Splitter (Reliable fallback)
            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", "?", "!", " ", ""],
                keep_separator=True,
                add_start_index=True,
            )
            recursive_chunks = recursive_splitter.split_documents([doc])

            # Prefer semantic chunks when they make sense
            final_chunks = semantic_chunks if len(semantic_chunks) >= 3 else recursive_chunks
            all_chunks.extend(final_chunks)

        # Filter chunks
        all_chunks = [c for c in all_chunks if len(c.page_content.strip()) > 70]
        
        print(f"[INFO] Chunking complete: {len(all_chunks)} high-quality chunks")
        return all_chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks using {self.model_name}...")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,   #for better similarity
            batch_size=32
        )
        print(f"[INFO] Embeddings generated: {embeddings.shape}")
        return embeddings