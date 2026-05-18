import os
import re
from dotenv import load_dotenv
from typing import List, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np

from src.cache_manager import RAGCache

load_dotenv()


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        cache: RAGCache = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.cache = cache

        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        if hf_token:
            os.environ["HUGGINGFACE_TOKEN"] = hf_token

        print(f"[EMBED] Loading model: {model_name}")

        # Raw model for direct embedding queries
        self.model = SentenceTransformer(
            model_name,
            token=hf_token,
            trust_remote_code=True,
        )

        # LangChain wrapper for SemanticChunker (it needs embed_documents method)
        self.langchain_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"token": hf_token},
            encode_kwargs={"normalize_embeddings": True},
        )

        print(f"[EMBED] Model ready")

    def _is_boilerplate(self, text: str) -> bool:
        """Detect page numbers, headers, footers, and other noise."""
        t = text.strip()
        if len(t) < 80:
            return True
        if re.match(r"^\s*[0-9]+\s*$", t):
            return True
        if re.match(r"^(Page|©|Copyright|All rights reserved|Confidential)", t, re.IGNORECASE):
            return True
        return False

    def _extract_headers(self, text: str) -> str:
        """Extract likely section headers to prepend as context."""
        lines = text.split("\n")
        headers = []
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) < 120 and not line.endswith("."):
                if line.isupper() or line.startswith(("#", "##", "###", "Chapter", "Section", "Table", "Figure")):
                    headers.append(line)
        return " | ".join(headers[:3])

    def chunk_documents(self, documents: List[Any], source_file: str = "unknown") -> List[Any]:
        """
        High-quality chunking pipeline:
        1. Try semantic chunking for natural boundaries (uses HuggingFaceEmbeddings wrapper)
        2. Fallback to recursive with smart separators
        3. Enrich metadata with headers and position
        4. Filter boilerplate and tiny fragments
        """
        all_chunks = []

        for doc in documents:
            semantic_chunks = []

            # Attempt 1: Semantic chunking (uses LangChain wrapper which has embed_documents)
            try:
                semantic_splitter = SemanticChunker(
                    self.langchain_embeddings,  # <-- FIX: Use wrapper, not raw SentenceTransformer
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=80,
                )
                semantic_chunks = semantic_splitter.split_documents([doc])
            except Exception as e:
                print(f"[EMBED] Semantic chunking failed: {e}")

            # Attempt 2: Recursive character splitting (always reliable)
            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
                keep_separator=True,
                add_start_index=True,
            )
            recursive_chunks = recursive_splitter.split_documents([doc])

            # Choose best: semantic if it produced meaningful segmentation
            candidates = semantic_chunks if len(semantic_chunks) >= 2 else recursive_chunks

            # Enrich and filter
            for chunk in candidates:
                content = chunk.page_content.strip()

                # Skip boilerplate
                if self._is_boilerplate(content):
                    continue

                # Extract headers for context
                header_context = self._extract_headers(content)
                if chunk.metadata is None:
                    chunk.metadata = {}

                chunk.metadata["source"] = source_file
                chunk.metadata["header_context"] = header_context
                chunk.metadata["chunk_length"] = len(content)

                # Prepend header context to content if available
                if header_context:
                    chunk.page_content = f"[Context: {header_context}]\n{content}"

                all_chunks.append(chunk)

        print(f"[EMBED] Chunking complete: {len(all_chunks)} quality chunks from {len(documents)} docs")
        return all_chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[EMBED] Encoding {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=32,
            convert_to_numpy=True,
        )
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed query with caching support."""
        if self.cache:
            cached = self.cache.get_embedding(query)
            if cached:
                print("[EMBED] Query embedding cache HIT")
                return cached

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        result = embedding.tolist()

        if self.cache:
            self.cache.set_embedding(query, result)
        return result