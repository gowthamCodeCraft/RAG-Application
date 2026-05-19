import os
import re
from dotenv import load_dotenv
from typing import List, Any
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

        # FIX: Use HuggingFaceEmbeddings (LangChain wrapper) which handles
        # SentenceTransformers v4+ compatibility internally.
        # This avoids the AutoProcessor error with BAAI/bge-m3.
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "token": hf_token,
                "trust_remote_code": True,
                # FIX for v4+: disable the problematic processor loading
                "local_files_only": False,
            },
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32,
            },
        )

        # Keep a raw model reference for direct encoding (query cache, etc.)
        # But load it safely via the embeddings wrapper's underlying model
        try:
            self.model = self.embeddings.client
            print(f"[EMBED] Model loaded via HuggingFaceEmbeddings wrapper")
        except Exception as e:
            print(f"[EMBED] Warning: Could not get underlying model: {e}")
            self.model = None

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

    def _clean_ocr_text(self, text: str) -> str:
        """Clean up common OCR artifacts in handwritten/scanned text."""
        text = text.replace('|', 'I')
        text = text.replace('0', 'O')
        text = ' '.join(text.split())
        text = text.replace('- ', '')
        return text

    def chunk_documents(self, documents: List[Any], source_file: str = "unknown") -> List[Any]:
        """
        High-quality chunking pipeline.
        Uses HuggingFaceEmbeddings for semantic chunking (v4+ compatible).
        """
        all_chunks = []

        for doc in documents:
            semantic_chunks = []

            # Attempt 1: Semantic chunking using LangChain wrapper
            try:
                semantic_splitter = SemanticChunker(
                    self.embeddings,  # HuggingFaceEmbeddings handles v4+ internally
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=80,
                )
                semantic_chunks = semantic_splitter.split_documents([doc])
            except Exception as e:
                print(f"[EMBED] Semantic chunking failed: {e}")

            # Attempt 2: Recursive character splitting
            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
                keep_separator=True,
                add_start_index=True,
            )
            recursive_chunks = recursive_splitter.split_documents([doc])

            candidates = semantic_chunks if len(semantic_chunks) >= 2 else recursive_chunks

            for chunk in candidates:
                content = chunk.page_content.strip()

                # OCR cleaning
                if chunk.metadata and chunk.metadata.get("ocr"):
                    content = self._clean_ocr_text(content)
                    chunk.page_content = content

                if self._is_boilerplate(content):
                    continue

                header_context = self._extract_headers(content)
                if chunk.metadata is None:
                    chunk.metadata = {}

                chunk.metadata["source"] = source_file
                chunk.metadata["header_context"] = header_context
                chunk.metadata["chunk_length"] = len(content)

                if header_context:
                    chunk.page_content = f"[Context: {header_context}]\n{content}"

                all_chunks.append(chunk)

        print(f"[EMBED] Chunking complete: {len(all_chunks)} quality chunks")
        return all_chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        """Generate embeddings using HuggingFaceEmbeddings wrapper."""
        texts = [chunk.page_content for chunk in chunks]
        print(f"[EMBED] Encoding {len(texts)} chunks...")

        # Use the wrapper's embed_documents method (v4+ compatible)
        embeddings_list = self.embeddings.embed_documents(texts)
        embeddings = np.array(embeddings_list)
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed query with caching support."""
        if self.cache:
            cached = self.cache.get_embedding(query)
            if cached:
                print("[EMBED] Query embedding cache HIT")
                return cached

        # Use wrapper's embed_query (v4+ compatible)
        result = self.embeddings.embed_query(query)

        if self.cache:
            self.cache.set_embedding(query, result)
        return result