# 🚀 Advanced RAG Pipeline (Retrieval-Augmented Generation)

This project implements an **Advanced RAG (Retrieval-Augmented Generation) system** with modern features like:

- 🔍 Semantic Retrieval
- ⚡ Streaming Responses
- 📌 Citations (Source Attribution)
- 🧠 Conversation History
- ✂️ Answer Summarization

---

## 📌 What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that:

1. Retrieves relevant documents from a knowledge base
2. Uses an LLM to generate answers based on that data

👉 This helps avoid hallucinations and improves accuracy.

---

## 🧠 Features

### ✅ 1. Retrieval
- Uses vector similarity search
- Fetches top-k relevant documents

### ⚡ 2. Streaming
- Streams LLM responses in real-time (if supported)

### 📌 3. Citations
- Displays source documents used for answering
- Improves trust and explainability

### 🧠 4. History
- Stores previous questions and answers
- Enables conversational memory

### ✂️ 5. Summarization
- Generates short summaries of answers
- Helps reduce verbosity

---

## 🏗️ Project Structure

project/
│
├── data/
│ └── text_files/
│ └── gen_ai.txt
│
├── rag_retriever.py
├── advanced_rag.py
├── embedding_manager.py
├── vector_store.py
├── main.py
└── README.md


---

---

## 🛠️ Installation (Using uv)

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd agentic_eda_analysis

### 2️⃣ Create virtual environment

uv venv

# Windows
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
uv pip install -r requirements.txt
```

### Use environment variables for security:

TYPESENSE_API_KEY=your_key_here
GROQ_API_KEY = your_key_here
---

## ⚙️ How It Works

### 🔄 Pipeline Flow

1. User asks a question
2. Retriever finds relevant documents
3. Context is created
4. LLM generates answer
5. Citations are added
6. Summary is generated (optional)
7. History is stored

---

## 🧪 Example Usage
<!-- 
```python
advanced_rag = AdvancedRag(rag_retriever, llm)

response = advanced_rag.query(
    "What is Business Intelligence?",
    top_k=3,
    min_score=0.1,
    stream=True,
    summarize=True
)

print(response['answer'])
print(response['summary'])  -->

## 🚀 Future Enhancements (Planned)
🔹 FastAPI backend integration
🔹 React frontend UI
🔹 Improved streaming (token-level)
🔹 Re-ranking for better retrieval accuracy
🔹 Memory-aware conversations
🔹 Deployment (Cloud / Docker)
🔹 Multi-document support (PDF, DOCX, etc.)