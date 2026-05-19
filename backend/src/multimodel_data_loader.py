from pathlib import Path
from typing import List, Any, Optional, Dict
import io
import base64
import hashlib

from langchain_core.documents import Document

from langchain_community.document_loaders import (
    PyMuPDFLoader, TextLoader, CSVLoader, Docx2txtLoader,
    UnstructuredExcelLoader, JSONLoader
)

# OCR
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


class MultimodelDocumentProcessor:
    
    def __init__(self, vision_llm=None):
        """
        A multimodal LLM instance (ChatOpenAI with gpt-4o, or ChatGroq with vision model)
        """
        self.vision_llm = vision_llm
        print(f"[INFO]] Processor initialized")
        print(f"[INFO]] OCR available: {OCR_AVAILABLE}")
        print(f"[INFO]] PDF2Image available: {PDF2IMAGE_AVAILABLE}")
        print(f"[INFO]] PDFPlumber available: {PDFPLUMBER_AVAILABLE}")
        print(f"[INFO]] Vision LLM: {'Yes' if vision_llm else 'No (text-only mode)'}")

    #IMAGE ENCODING
    def _encode_image_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for LLM vision APIs."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _encode_image_path(self, image_path: str) -> str:
        """Convert image file to base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # VISION LLM
    def _describe_image(self, image: Image.Image, context: str = "") -> str:
        """
        Use multimodal LLM to describe an image/chart/diagram.
        Returns structured description as text.
        """
        if not self.vision_llm:
            return ""

        try:
            base64_image = self._encode_image_base64(image)

            # Build multimodal message
            from langchain_core.messages import HumanMessage

            prompt = f"""Analyze this image from a document and provide a detailed text description.

Context: This image is from page {context} of a document.

If this is a chart or graph:
- Describe the chart type (bar, line, pie, etc.)
- List all axis labels, legends, and data series
- Describe trends, peaks, comparisons, and key insights
- Include specific numbers if visible

If this is a diagram:
- Describe the components and their relationships
- Explain the flow or process shown
- List any labels, arrows, or annotations

If this is a table image:
- Extract all rows and columns as markdown table
- Include headers and all data cells

If this is a photo or screenshot:
- Describe what is shown in detail
- Transcribe any visible text
- Note any UI elements, buttons, or interface components

Format your response as structured text that can be used for semantic search."""

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                ]
            )

            response = self.vision_llm.invoke([message])
            description = response.content if hasattr(response, "content") else str(response)

            return f"[IMAGE DESCRIPTION]\n{description}\n[END IMAGE DESCRIPTION]"

        except Exception as e:
            print(f"[VISION] Failed to describe image: {e}")
            return ""

    # PDF ANALYSIS
    def _detect_page_content_type(self, page_image: Image.Image, text_content: str) -> str:
        """
        Classify what type of content dominates a PDF page.
        Returns: 'text', 'table', 'chart', 'handwritten', 'mixed'
        """
        text_len = len(text_content.strip())

        # Simple heuristics
        if text_len < 50:
            return "chart_or_image" 
        elif text_len > 500:
            return "text"
        else:
            return "mixed"

    def _extract_tables_pdfplumber(self, page) -> str:
        """Extract tables from a pdfplumber page as markdown."""
        if not PDFPLUMBER_AVAILABLE:
            return ""

        tables = page.extract_tables()
        if not tables:
            return ""

        result = "\n\n[TABLES]\n"
        for idx, table in enumerate(tables):
            if not table:
                continue
            result += f"\nTable {idx + 1}:\n"
            for row in table:
                result += "| " + " | ".join(str(cell or "") for cell in row) + " |\n"
        result += "[END TABLES]\n"
        return result

    # MAIN PROCESSING
    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Full multimodal PDF processing pipeline:
        1. Extract native text (PyMuPDF / pdfplumber)
        2. Detect content type per page
        3. For text-heavy pages: use extracted text
        4. For image/chart pages: use vision LLM description
        5. For scanned pages: use OCR
        6. Combine all into structured documents
        """
        filename = Path(file_path).name
        print(f"\n[INFO]] Processing PDF: {filename}")

        docs = []

        # Step 1: Try native text extraction first
        native_docs = self._extract_native_text(file_path)
        has_native_text = len(native_docs) > 0 and sum(len(d.page_content.strip()) for d in native_docs) > 100

        if has_native_text:
            print(f"[INFO]] Native text detected ({sum(len(d.page_content.strip()) for d in native_docs)} chars)")

            # Check each page for images/charts that need vision description
            if PDF2IMAGE_AVAILABLE and self.vision_llm:
                images = pdf2image.convert_from_path(file_path, dpi=200)

                for i, (doc, img) in enumerate(zip(native_docs, images)):
                    content_type = self._detect_page_content_type(img, doc.page_content)

                    if content_type in ["chart_or_image", "mixed"]:
                        print(f"[INFO]] Page {i+1} has visual content, using vision LLM...")
                        vision_desc = self._describe_image(img, f"page {i+1} of {filename}")
                        if vision_desc:
                            doc.page_content = vision_desc + "\n\n[EXTRACTED TEXT]\n" + doc.page_content

                    # Also extract tables if pdfplumber available
                    if PDFPLUMBER_AVAILABLE:
                        with pdfplumber.open(file_path) as pdf:
                            if i < len(pdf.pages):
                                table_text = self._extract_tables_pdfplumber(pdf.pages[i])
                                if table_text:
                                    doc.page_content += table_text

                    doc.metadata["source"] = filename
                    doc.metadata["page"] = i + 1
                    doc.metadata["multimodal"] = True
                    docs.append(doc)
            else:
                # No vision available, just use native text
                for doc in native_docs:
                    doc.metadata["source"] = filename
                    docs.append(doc)

        else:
            # Step 2: No native text — scanned or image PDF
            print(f"[INFO]] No native text found. Using OCR + Vision pipeline...")

            if not PDF2IMAGE_AVAILABLE:
                print("[ERROR] pdf2image not available. Cannot process scanned PDF.")
                return []

            images = pdf2image.convert_from_path(file_path, dpi=300)
            print(f"[INFO]] Converted {len(images)} pages to images")

            for i, image in enumerate(images):
                print(f"[INFO]] Processing page {i+1}/{len(images)}...")

                page_content = ""

                # 2a: OCR for any readable text
                if OCR_AVAILABLE:
                    try:
                        ocr_text = pytesseract.image_to_string(image, config=r"--oem 3 --psm 6 -l eng")
                        if ocr_text.strip():
                            page_content += f"[OCR TEXT]\n{ocr_text.strip()}\n\n"
                    except Exception as e:
                        print(f"[OCR] Failed on page {i+1}: {e}")

                # 2b: Vision LLM for charts, diagrams, context
                if self.vision_llm:
                    vision_desc = self._describe_image(image, f"page {i+1} of {filename}")
                    if vision_desc:
                        page_content += vision_desc + "\n\n"

                if page_content.strip():
                    docs.append(Document(
                        page_content=page_content.strip(),
                        metadata={
                            "source": filename,
                            "page": i + 1,
                            "multimodal": True,
                            "ocr": OCR_AVAILABLE and bool(ocr_text.strip()),
                            "vision": bool(self.vision_llm),
                        }
                    ))

        print(f"[INFO]] Extracted {len(docs)} rich documents from {filename}")
        return docs

    def _extract_native_text(self, file_path: str) -> List[Document]:
        """Extract native text layer from PDF."""
        docs = []

        # Try pdfplumber first (better tables)
        if PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(file_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        if text.strip():
                            docs.append(Document(
                                page_content=text.strip(),
                                metadata={"page": i + 1, "extractor": "pdfplumber"}
                            ))
                if docs:
                    return docs
            except Exception as e:
                print(f"[INFO]] pdfplumber failed: {e}")

        # Fallback to PyMuPDF
        try:
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            for d in docs:
                if d.metadata is None:
                    d.metadata = {}
                d.metadata["extractor"] = "pymupdf"
            return docs
        except Exception as e:
            print(f"[INFO]] PyMuPDF failed: {e}")
            return []

    def process_image_file(self, file_path: str) -> List[Document]:
        """Process a standalone image file (PNG, JPG, etc.)."""
        filename = Path(file_path).name
        ext = Path(file_path).suffix.lower()

        if ext not in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"]:
            raise ValueError(f"Unsupported image type: {ext}")

        print(f"[INFO]] Processing image: {filename}")

        # OCR for text
        ocr_text = ""
        if OCR_AVAILABLE:
            try:
                image = Image.open(file_path)
                ocr_text = pytesseract.image_to_string(image, config=r"--oem 3 --psm 6 -l eng")
            except Exception as e:
                print(f"[OCR] Failed: {e}")

        # Vision LLM for description
        vision_desc = ""
        if self.vision_llm:
            try:
                image = Image.open(file_path)
                vision_desc = self._describe_image(image, f"image file {filename}")
            except Exception as e:
                print(f"[VISION] Failed: {e}")

        content = ""
        if vision_desc:
            content += vision_desc + "\n\n"
        if ocr_text.strip():
            content += f"[OCR TEXT]\n{ocr_text.strip()}\n"

        if not content.strip():
            content = "[No extractable content from image]"

        return [Document(
            page_content=content.strip(),
            metadata={
                "source": filename,
                "page": 1,
                "multimodal": True,
                "ocr": bool(ocr_text.strip()),
                "vision": bool(vision_desc),
            }
        )]

    def load_file(self, file_path: str) -> List[Document]:
        """Universal file loader — detects type and routes to correct processor."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        filename = path.name

        if ext == ".pdf":
            return self.process_pdf(str(path))
        elif ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"]:
            return self.process_image_file(str(path))
        elif ext == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()
            for d in docs:
                d.metadata = d.metadata or {}
                d.metadata["source"] = filename
            return docs
        elif ext == ".docx":
            loader = Docx2txtLoader(str(path))
            docs = loader.load()
            for d in docs:
                d.metadata = d.metadata or {}
                d.metadata["source"] = filename
            return docs
        elif ext == ".csv":
            loader = CSVLoader(str(path))
            docs = loader.load()
            for d in docs:
                d.metadata = d.metadata or {}
                d.metadata["source"] = filename
            return docs
        elif ext == ".xlsx":
            loader = UnstructuredExcelLoader(str(path), mode="elements")
            docs = loader.load()
            for d in docs:
                d.metadata = d.metadata or {}
                d.metadata["source"] = filename
            return docs
        elif ext == ".json":
            loader = JSONLoader(str(path), jq_schema=".[]", text_content=False)
            docs = loader.load()
            for d in docs:
                d.metadata = d.metadata or {}
                d.metadata["source"] = filename
            return docs
        else:
            raise ValueError(f"Unsupported file type: {ext}")