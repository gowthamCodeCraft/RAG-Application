import os
import shutil

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from src.search import RAGSearch

app = FastAPI(title="Advanced Hybrid RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_rag = RAGSearch()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class FileDeleteRequest(BaseModel):
    filename: str


@app.get("/")
async def root():
    return {
        "message": "Advanced Hybrid RAG API (Incremental + Redis + Hybrid + Rerank)",
        "status": "healthy",
        "rag_ready": chat_rag.is_ready,
        "indexed_files": chat_rag.list_files(),
    }


@app.get("/health")
async def health():
    # Lazy properties: accessing them triggers loading if not already done
    # We use _is_ready to avoid triggering full model load on health checks
    return {
        "status": "ok",
        "rag_ready": chat_rag.is_ready,
        "llm_loaded": chat_rag.llm is not None,
        "indexed_files": chat_rag.list_files(),
        "redis_enabled": chat_rag.cache.enabled if chat_rag.cache else False,
        "lazy_mode": True,
    }


# ================= INCREMENTAL UPLOAD =================
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".json"}
    if not any(file.filename.lower().endswith(ext) for ext in allowed):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    os.makedirs("data/uploads", exist_ok=True)
    path = f"data/uploads/{file.filename}"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = chat_rag.add_file(path)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    return {
        "message": result["message"],
        "filename": result["filename"],
        "chunks_indexed": result.get("chunks", 0),
        "status": result["status"],
        "total_files": len(chat_rag.list_files()),
    }


# ================= FILE MANAGEMENT =================
@app.get("/api/files")
async def list_files():
    return {
        "files": chat_rag.list_files(),
        "count": len(chat_rag.list_files()),
    }


@app.delete("/api/files")
async def delete_file(request: FileDeleteRequest):
    result = chat_rag.remove_file(request.filename)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result


# ================= DEBUG RETRIEVAL =================
@app.post("/api/debug-retrieve")
async def debug_retrieve(request: QueryRequest):
    if not chat_rag.is_ready:
        raise HTTPException(status_code=400, detail="System not ready. Upload documents first.")

    try:
        debug_data = chat_rag.debug_retrieve(request.query)
        return debug_data
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")


# ================= DEBUG FULL PIPELINE =================
@app.post("/api/debug")
async def debug_pipeline(request: QueryRequest):
    """
    Full debug endpoint showing every stage:
    - Raw retrieval scores
    - Reranked scores
    - Context built
    - Empty-context diagnosis
    """
    if not chat_rag.is_ready:
        raise HTTPException(status_code=400, detail="System not ready")

    try:
        # Stage 1: Raw retrieval
        raw = chat_rag.debug_retrieve(request.query)

        # Stage 2: What RAG would send to LLM
        rag_result = chat_rag.query(request.query, top_k=request.top_k)

        return {
            "query": request.query,
            "retrieval": raw,
            "rag_result": {
                "retrieved_count": rag_result.get("retrieved_count"),
                "context_length": len(rag_result.get("answer", "")),
                "sources": [
                    {"source": s["source"], "page": s["page"], "score": s["score"], "preview": s["preview"][:100]}
                    for s in rag_result.get("sources", [])
                ],
                "debug": rag_result.get("debug", {}),
            },
            "cached": rag_result.get("cached", False),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ================= STANDARD QUERY =================
@app.post("/api/query")
async def query_document(request: QueryRequest):
    if not chat_rag.is_ready:
        return {
            "answer": "System not ready. Please upload documents first.",
            "sources": [],
            "retrieved_count": 0,
        }
    return chat_rag.query(request.query, top_k=request.top_k)


# ================= STREAMING QUERY (SSE) =================
@app.post("/api/query-stream")
async def query_stream(request: QueryRequest):
    query = request.query
    print("\n" + "="*50)
    print(f"[STREAM API] {query}")
    print("="*50)

    if not query:
        raise HTTPException(400, "Query required")

    if not chat_rag.is_ready:
        async def not_ready():
            yield "data: System not ready. Please upload documents first.\n\n"
        return StreamingResponse(not_ready(), media_type="text/event-stream")

    def generate():
        try:
            for token in chat_rag.query_stream(question=query, top_k=request.top_k):
                # SSE format: data: <text>\n\n
                # Escape newlines in token for SSE
                safe_token = token.replace("\n", "\ndata: ")
                yield f"data: {safe_token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering if behind proxy
        },
    )


# ================= CACHE STATS =================
@app.get("/api/cache-stats")
async def cache_stats():
    """Show Redis cache statistics and connection info."""
    if not chat_rag.cache:
        return {"enabled": False, "message": "Cache not initialized"}
    return chat_rag.cache.get_stats()


# ================= VOICE QUERY =================
@app.post("/api/voice-query")
async def voice_query(file: UploadFile = File(...)):
    allowed_audio = (".wav", ".mp3", ".m4a", ".ogg", ".webm")
    if not file.filename.lower().endswith(allowed_audio):
        raise HTTPException(status_code=400, detail="Only audio files supported")

    os.makedirs("temp", exist_ok=True)
    audio_path = f"temp/{file.filename}"

    with open(audio_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        with open(audio_path, "rb") as audio:
            transcription = client.audio.transcriptions.create(
                file=audio,
                model="whisper-large-v3",
                response_format="json"
            )

        text = transcription.text.strip()
        print(f"[VOICE] Transcribed: '{text}'")

        if not chat_rag.is_ready:
            return {
                "transcribed_text": text,
                "answer": "System not ready. Please upload documents first.",
                "sources": [],
            }

        result = chat_rag.query(text)
        return {
            "transcribed_text": text,
            **result
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")