from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    JSONLoader,
)


def load_single_file(file_path: str) -> List[Any]:
    """
    Load a SINGLE file and return LangChain Document objects.
    Enriches metadata with source filename and page info.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    loader = None

    if ext == ".pdf":
        loader = PyMuPDFLoader(str(path))
    elif ext == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(str(path))
    elif ext == ".csv":
        loader = CSVLoader(str(path))
    elif ext == ".xlsx":
        loader = UnstructuredExcelLoader(str(path), mode="elements")
    elif ext == ".json":
        loader = JSONLoader(str(path), jq_schema=".[]", text_content=False)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()

    # Enrich metadata
    for i, doc in enumerate(docs):
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["source"] = path.name
        # Normalize page metadata across loaders
        if "page" not in doc.metadata:
            doc.metadata["page"] = doc.metadata.get("page_number", i + 1)
        doc.metadata["doc_index"] = i

    print(f"[LOADER] {path.name}: {len(docs)} pages/records loaded")
    return docs


def load_all_docs(data_dir: str = "data/uploads") -> List[Any]:
    """
    Load ALL supported documents from directory (used for full rebuilds only).
    """
    data_path = Path(data_dir).resolve()
    print(f"[LOADER] Scanning directory: {data_path}")

    all_docs = []
    supported = {".pdf", ".txt", ".docx", ".xlsx", ".csv", ".json"}

    for file_path in data_path.glob("**/*"):
        if file_path.is_file() and file_path.suffix.lower() in supported:
            try:
                docs = load_single_file(str(file_path))
                all_docs.extend(docs)
            except Exception as e:
                print(f"[LOADER] Failed {file_path.name}: {e}")

    print(f"[LOADER] Total loaded: {len(all_docs)} documents from all files")
    return all_docs