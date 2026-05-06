import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.search import RAGSearch
from groq import Groq
from src.data_loader import load_all_docs

app = FastAPI(title="Search Document RAG API")
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
    top_k: int = 3

@app.get("/")
async def root():
    return {
        "message": "🚀 RAG Document Chat API is Running!",
        "docs_url": "http://127.0.0.1:8000/docs",
        "status": "healthy"
    }


# ====================== UPLOAD ======================
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):

    if not file.filename.endswith((".pdf", ".docx", ".txt",".csv",".xlsx")):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    os.makedirs("data/uploads", exist_ok=True)
    save_path = f"data/uploads/{file.filename}"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    load_documents = load_all_docs("data/uploads")
    chat_rag.vector_store.build_from_documents(load_documents)

    return {"message": f"File '{file.filename}' uploaded and processed successfully."}

# ====================== QUERY ======================
@app.post("/api/query")
async def query_document(request: QueryRequest):
    result = chat_rag.query(request.query, request.top_k)
    return {"result": result}

#====================== VOICE QUERY (Whisper) ======================
@app.post("/api/voice-query")
async def voice_query(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".ogg")):
        raise HTTPException(400, "Only audio files supported")

    os.makedirs("temp", exist_ok=True)
    audio_path = f"temp/{file.filename}"

    with open(audio_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Groq Whisper
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    with open(audio_path, "rb") as audio:
        transcription = client.audio.transcriptions.create(
            file=audio,
            model="whisper-large-v3",
            response_format="json"
        )

    text = transcription.text
    result = chat_rag.query(text)

    return {
        "transcribed_text": text,
        **result
    }

# ===================== Streaming ======================

@app.post("/api/query-stream")
async def query_stream(request: dict):

    query = request.get("query")

    def generate():
        try:
            for chunk in chat_rag.llm.stream([query]):   # adjust to your pipeline
                if hasattr(chunk, "content"):
                    yield chunk.content
                else:
                    yield str(chunk)

        except Exception as e:
            print("STREAM ERROR:", e)
            yield "\n[Error generating response]"

    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)