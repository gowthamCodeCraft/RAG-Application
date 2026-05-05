from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_community. document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader

def load_all_docs(data_dir:str) -> list[Any]:
    """
    Load all documents from the specified directory and its subdirectories.
    Supported file types include .txt, .pdf, .docx, .xlsx, and .json.

    """

    data_path = Path(data_dir).resolve()
    print(f"Loading documents from: {data_path}")
    all_docs = []
    
    #load pdf files
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"[Debug] found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"[Debug] loading PDF file: {pdf_file}")
        try:
            pdf_loader = PyPDFLoader(str(pdf_file))
            loaded_docs = pdf_loader.load()
            print(f"[Debug] loaded {len(loaded_docs)} documents from {pdf_file}")
            all_docs.extend(loaded_docs)
        except Exception as e:
            print(f"[Error] Failed to load PDF file: {pdf_file}, Error: {e}")

    # #load text files
    # text_files = list(data_path.glob("**/*.txt"))
    # print(f"[Debug] found {len(text_files)} text_files: {[str(f) for f in text_files]}")
    # for text_file in text_files:
    #     print(f"[Debug] loading text file: {text_file}")
    #     try:
    #         text_loader = TextLoader(str(text_file))
    #         loaded_docs = text_loader.load()
    #         print(f"[Debug] loaded {len(loaded_docs)} documents from {text_file}")
    #         all_docs.extend(loaded_docs)
    #     except Exception as e:
    #         print(f"[Error] Failed to load text file: {text_file}, Error: {e}")

    # load csv files
    csv_files = list(data_path.glob("**/*.csv"))
    print(f"[Debug] Found {len(csv_files)} csv files: {[str(f) for f in csv_files]}")
    for csv_file in csv_files:
        print(f"[Debug] loading csv file: {csv_file}")
        try:
            csv_loader = CSVLoader(str(csv_file))
            loaded_docs = csv_loader.load()
            print(f"[Debug] loaded {len(loaded_docs)} from {csv_file}")
            all_docs.extend(loaded_docs)
        except Exception as e:
            print(f"[Error] Failed to load csv file: {csv_file}, Error: {e}")

    return all_docs