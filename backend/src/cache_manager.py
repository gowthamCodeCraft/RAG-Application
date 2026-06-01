import json
import hashlib
import redis as redis_lib
from typing import Optional, Any, List
import os


class RAGCache:
    """
    Redis-backed cache for RAG operations.
    Works with: Local Redis (Windows/Linux/Mac), Docker Redis, Redis Cloud, AWS ElastiCache, Upstash.

    Caches ONLY expensive computed data:
    - Query results (answer + sources)
    - Text embeddings (BGE-M3 vectors)
    - BM25 sparse encoder parameters

    Falls back to no-cache if Redis is unavailable.
    """

    def __init__(
        self,
        redis_url: str = None,
        ttl_query: int = 3600,
        ttl_embedding: int = 86400,
    ):
        self.ttl_query = ttl_query
        self.ttl_embedding = ttl_embedding
        self.enabled = False
        self.client = None
        self.redis_url = redis_url

        # Try to connect
        self._connect()

    def _connect(self):
        """Attempt Redis connection with multiple fallback strategies."""
        urls_to_try = []

        # Explicit URL from constructor
        if self.redis_url:
            urls_to_try.append(self.redis_url)

        # Environment variables (cloud deployments)
        env_urls = [
            os.getenv("REDIS_URL"),
            os.getenv("REDISCLOUD_URL"),
            os.getenv("UPSTASH_REDIS_REST_URL"),
            os.getenv("REDIS_TLS_URL"),
        ]
        urls_to_try.extend([u for u in env_urls if u])

        # Local defaults
        urls_to_try.extend([
            "redis://localhost:6379/0", 
            "redis://127.0.0.1:6379/0",
        ])

        for url in urls_to_try:
            try:

                # Handle Redis Cloud
                if url.startswith("redis://") or ":6380" in url or "ssl" in url.lower():
                    self.client = redis_lib.from_url(
                        url,
                        decode_responses=True,
                        ssl_cert_reqs=None,  # Allow self-signed certs
                        socket_connect_timeout=5,
                        socket_timeout=5,
                    )
                else:
                    self.client = redis_lib.from_url(
                        url,
                        decode_responses=True,
                        socket_connect_timeout=3,
                        socket_timeout=3,
                    )

                self.client.ping()
                self.enabled = True
                print(f"[CACHE] Redis connected: {url.split("@")[-1] if "@" in url else url}")
                return

            except Exception as e:
                print(f"[CACHE] Failed: {url.split("@")[-1] if "@" in url else url} — {str(e)[:60]}")
                continue

        print("[CACHE] No Redis available. Running WITHOUT cache.")
        self.enabled = False

    def _key(self, prefix: str, data: str) -> str:
        h = hashlib.sha256(data.encode()).hexdigest()[:16]
        return f"rag:{prefix}:{h}"

    # Query Result Cache
    def get_query_result(self, query: str, top_k: int = 5) -> Optional[dict]:
        if not self.enabled:
            return None
        try:
            key = self._key("query", f"{query}:{top_k}")
            val = self.client.get(key)
            if val:
                print(f"[CACHE] Query cache HIT")
                return json.loads(val)
        except Exception as e:
            print(f"[CACHE] Read error (ignoring): {e}")
        return None

    def set_query_result(self, query: str, result: dict, top_k: int = 5):
        if not self.enabled or not result.get("answer"):
            return
        try:
            key = self._key("query", f"{query}:{top_k}")
            cache_payload = {
                "answer": result.get("answer"),
                "sources": result.get("sources", []),
                "retrieved_count": result.get("retrieved_count", 0),
            }
            self.client.setex(key, self.ttl_query, json.dumps(cache_payload))
            print(f"[CACHE] Query cached (TTL {self.ttl_query}s)")
        except Exception as e:
            print(f"[CACHE] Write error (ignoring): {e}")

    def invalidate_query_cache(self):
        if not self.enabled:
            return
        try:
            count = 0
            for key in self.client.scan_iter(match="rag:query:*"):
                self.client.delete(key)
                count += 1
            print(f"[CACHE] Invalidated {count} query cache entries")
        except Exception as e:
            print(f"[CACHE] Invalidation error: {e}")

    # Embedding Cache 
    def get_embedding(self, text: str) -> Optional[List[float]]:
        if not self.enabled:
            return None
        try:
            key = self._key("emb", text)
            val = self.client.get(key)
            if val:
                return json.loads(val)
        except Exception:
            pass
        return None

    def set_embedding(self, text: str, embedding: List[float]):
        if not self.enabled:
            return
        try:
            key = self._key("emb", text)
            self.client.setex(key, self.ttl_embedding, json.dumps(embedding))
        except Exception:
            pass

    # BM25 Cache
    def get_sparse_params(self) -> Optional[dict]:
        if not self.enabled:
            return None
        try:
            val = self.client.get("rag:sparse:params")
            if val:
                return json.loads(val)
        except Exception:
            pass
        return None

    def set_sparse_params(self, params: dict):
        if not self.enabled:
            return
        try:
            self.client.setex("rag:sparse:params", self.ttl_embedding, json.dumps(params))
        except Exception:
            pass

    def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self.enabled:
            return {"enabled": False}
        try:
            info = self.client.info()
            return {
                "enabled": True,
                "url_masked": self.redis_url.split("@")[-1] if self.redis_url and "@" in self.redis_url else "configured",
                "used_memory_human": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
            }
        except Exception as e:
            return {"enabled": True, "error": str(e)}