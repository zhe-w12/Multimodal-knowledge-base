"""Unstructured + PaddleOCR + Qwen-VL multimodal ETL orchestration."""

from __future__ import annotations

import base64
import hashlib
import re
import tempfile
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urlsplit

from manufacturing_rag.adapters.ocr import PaddleOCRAdapter
from manufacturing_rag.adapters.vision import QwenVLAdapter
from manufacturing_rag.chunking import ETLValidationReport, prepare_chunks
from manufacturing_rag.config import DEFAULT_CONFIG, RAGConfig
from manufacturing_rag.debug_log import write_debug_log
from manufacturing_rag.pdf_quality import chunks_have_unusable_pdf_text, suspicious_pdf_text
from manufacturing_rag.schemas import DocumentChunk


SUPPORTED_DOCUMENT_SUFFIXES = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".txt",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}
IMAGE_SUFFIXES = PaddleOCRAdapter.image_suffixes


@dataclass(frozen=True)
class ETLResult:
    chunks: List[DocumentChunk]
    validation: ETLValidationReport
    parser_chain: List[str]


class MultimodalETLPipeline:
    """Extract structured elements, enrich visuals and emit validated child chunks."""

    def __init__(
        self,
        config: RAGConfig = DEFAULT_CONFIG,
        ocr: Optional[PaddleOCRAdapter] = None,
        vision: Optional[QwenVLAdapter] = None,
    ) -> None:
        self.config = config
        self.ocr = ocr or PaddleOCRAdapter(config)
        self.vision = vision or QwenVLAdapter(config)

    def parse_file(self, path: str, max_pages: Optional[int] = None, render_dpi: Optional[int] = None) -> ETLResult:
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
            raise ValueError(f"unsupported document type: {suffix or '<none>'}")
        if suffix in IMAGE_SUFFIXES:
            raw_chunks = self._parse_image(file_path)
            parser_chain = ["PaddleOCR", "Qwen3-VL"]
        elif suffix in {".txt", ".md"}:
            raw_chunks = [self._plain_text_chunk(file_path)]
            parser_chain = ["plain-text"]
        else:
            parser_chain = ["Unstructured"]
            try:
                raw_chunks = self._parse_unstructured(file_path)
            except Exception:  # noqa: BLE001 - PDF text parsing falls back to PyMuPDF/PaddleOCR.
                if suffix != ".pdf":
                    raise
                raw_chunks = self._parse_pdf_fallback(file_path, max_pages=max_pages)
                parser_chain.append("PyMuPDF")
            if not raw_chunks and suffix == ".pdf":
                raw_chunks = self._parse_pdf_fallback(file_path, max_pages=max_pages)
                parser_chain.append("PyMuPDF")
            if not raw_chunks and suffix == ".pdf":
                raw_chunks = self.ocr.parse_file(str(file_path))
                parser_chain.append("PaddleOCR")
            if suffix == ".pdf" and raw_chunks and chunks_have_unusable_pdf_text(raw_chunks):
                raw_chunks = self._repair_low_quality_pdf(
                    file_path, raw_chunks, max_pages=max_pages, render_dpi=render_dpi
                )
                if any(str(chunk.metadata.get("ocr_provider") or "").lower() == "paddleocr" for chunk in raw_chunks):
                    parser_chain.append("PaddleOCR")
            raw_chunks.extend(self._enrich_unstructured_images(raw_chunks))
            if any(chunk.modality == "visual_description" for chunk in raw_chunks):
                parser_chain.append("Qwen3-VL")
        if suffix == ".pdf" and max_pages and max_pages > 0:
            raw_chunks = [chunk for chunk in raw_chunks if chunk.page is None or chunk.page <= max_pages]
        prepared, report = prepare_chunks(raw_chunks, self.config)
        if not prepared:
            raise ValueError("document did not contain usable text, OCR content or visual descriptions")
        return ETLResult(prepared, report, parser_chain)

    def _repair_low_quality_pdf(
        self,
        path: Path,
        chunks: List[DocumentChunk],
        max_pages: Optional[int] = None,
        render_dpi: Optional[int] = None,
    ) -> List[DocumentChunk]:
        """Prefer OCR for garbled font extraction and mark fallback text for review."""

        try:
            ocr_chunks = self.ocr.parse_file(str(path))
        except Exception:  # noqa: BLE001 - preserve source text when optional OCR is unavailable.
            ocr_chunks = []
        if ocr_chunks:
            return [chunk for chunk in ocr_chunks if not max_pages or max_pages <= 0 or chunk.page is None or chunk.page <= max_pages]
        reviewed: List[DocumentChunk] = []
        for chunk in chunks:
            if not suspicious_pdf_text(chunk.text):
                reviewed.append(chunk)
                continue
            metadata = dict(chunk.metadata)
            metadata.update({"pdf_text_quality": "needs_review", "quality_reason": "PDF 文本提取疑似乱码，OCR 不可用"})
            reviewed.append(
                replace(
                    chunk,
                    metadata=metadata,
                    confidence=min(float(chunk.confidence), 0.2),
                    needs_review=True,
                )
            )
        return reviewed

    def parse_bytes(
        self,
        content: bytes,
        filename: str,
        max_pages: Optional[int] = None,
        render_dpi: Optional[int] = None,
    ) -> ETLResult:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
            raise ValueError(f"unsupported document type: {suffix or '<none>'}")
        if suffix in IMAGE_SUFFIXES:
            stable_path = persist_image_bytes(content, suffix, self.config)
            result = self.parse_file(str(stable_path), max_pages=max_pages, render_dpi=render_dpi)
            chunks = [replace_source(chunk, Path(filename).name or filename, self.config) for chunk in result.chunks]
            return ETLResult(chunks, result.validation, result.parser_chain)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(content)
            temporary_path = Path(handle.name)
        try:
            result = self.parse_file(str(temporary_path), max_pages=max_pages, render_dpi=render_dpi)
            chunks = [replace_source(chunk, Path(filename).name or filename, self.config) for chunk in result.chunks]
            return ETLResult(chunks, result.validation, result.parser_chain)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _parse_image(self, path: Path) -> List[DocumentChunk]:
        ocr_chunks: List[DocumentChunk] = []
        ocr_error: Optional[Exception] = None
        try:
            ocr_chunks = self.ocr.parse_file(str(path))
        except Exception as exc:  # noqa: BLE001 - combine OCR and vision failures below.
            ocr_error = exc
        description = self.vision.describe_file(str(path))
        chunks = list(ocr_chunks)
        if description:
            chunks.append(
                DocumentChunk(
                    id=f"{path.name}:qwen-vl:1",
                    text=description,
                    source=str(path),
                    page=1,
                    block_id="qwen-vl-1",
                    modality="visual_description",
                    image_path=str(path),
                    metadata={"doc_type": "图片资料", "parser": "qwen-vl", "ingest_format": "image"},
                    confidence=0.9,
                )
            )
        elif ocr_chunks:
            chunks = [replace(chunk, image_path=str(path)) for chunk in ocr_chunks]
        else:
            chunks.append(
                DocumentChunk(
                    id=f"{path.name}:image:1",
                    text="图片资料",
                    source=str(path),
                    page=1,
                    block_id="image-1",
                    modality="image",
                    image_path=str(path),
                    metadata={
                        "doc_type": "图片资料",
                        "parser": "image-fallback",
                        "ingest_format": "image",
                        "needs_review": bool(ocr_error),
                        "quality_reason": "OCR 与视觉描述均不可用，已保留原图供多模态检索",
                    },
                    confidence=0.4 if ocr_error else 0.6,
                    needs_review=bool(ocr_error),
                )
            )
        return chunks

    def _parse_unstructured(self, path: Path) -> List[DocumentChunk]:
        try:
            from unstructured.partition.auto import partition  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Unstructured is not installed; install the production requirements") from exc
        try:
            try:
                elements = partition(
                    filename=str(path),
                    strategy="auto",
                    extract_image_block_types=["Image", "Table"],
                    extract_image_block_to_payload=True,
                )
            except TypeError:
                elements = partition(filename=str(path), strategy="auto")
        except Exception as exc:  # noqa: BLE001 - log the parser stage without document content.
            self._log_etl_failure(path, "unstructured_partition", exc)
            raise
        chunks: List[DocumentChunk] = []
        for index, element in enumerate(elements, start=1):
            text = str(element).strip()
            if not text:
                continue
            metadata_object = getattr(element, "metadata", None)
            metadata = metadata_object.to_dict() if hasattr(metadata_object, "to_dict") else {}
            page = positive_int(metadata.get("page_number"))
            category = str(getattr(element, "category", element.__class__.__name__))
            image_payload = str(metadata.get("image_base64") or "").strip()
            image_path = metadata.get("image_path")
            if image_payload and category.lower() in {"image", "table"}:
                try:
                    image_bytes = base64.b64decode(image_payload)
                    mime_type = str(metadata.get("image_mime_type") or "image/png")
                    image_suffix = {"image/jpeg": ".jpg", "image/webp": ".webp", "image/bmp": ".bmp", "image/tiff": ".tiff"}.get(mime_type, ".png")
                    image_path = str(persist_image_bytes(image_bytes, image_suffix, self.config))
                except Exception as exc:  # noqa: BLE001 - image persistence failures must remain visible.
                    self._log_etl_failure(path, "embedded_image_asset", exc)
                    raise
                try:
                    visual_text = self.vision.describe_bytes(
                        image_bytes,
                        mime_type=mime_type,
                    )
                except Exception as exc:  # noqa: BLE001 - image semantics are optional for text ingestion.
                    self._log_etl_failure(path, "embedded_image_description", exc)
                    visual_text = ""
                if visual_text:
                    text = f"{text}\n图像语义补全：{visual_text}".strip()
            chunks.append(
                DocumentChunk(
                    id=f"{path.name}:unstructured:{index}",
                    text=text,
                    source=str(path),
                    page=page,
                    block_id=f"unstructured-{index}",
                    modality="document_text",
                    image_path=image_path,
                    metadata={
                        "doc_type": suffix_doc_type(path.suffix),
                        "ingest_format": path.suffix.lower().lstrip("."),
                        "parser": "unstructured",
                        "element_category": category,
                        **safe_metadata(metadata),
                    },
                )
            )
        return chunks

    def _enrich_unstructured_images(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        enriched: List[DocumentChunk] = []
        seen: set[str] = set()
        for chunk in chunks:
            image_path = str(chunk.image_path or "").strip()
            if not image_path or image_path in seen or not Path(image_path).is_file():
                continue
            seen.add(image_path)
            try:
                description = self.vision.describe_file(image_path)
            except Exception as exc:  # noqa: BLE001 - preserve extracted document text if vision is unavailable.
                self._log_etl_failure(Path(chunk.source), "extracted_image_description", exc)
                continue
            if not description:
                continue
            enriched.append(
                DocumentChunk(
                    id=f"{chunk.id}:vision",
                    text=description,
                    source=chunk.source,
                    page=chunk.page,
                    block_id=f"{chunk.block_id or chunk.id}-vision",
                    modality="visual_description",
                    image_path=image_path,
                    metadata={"doc_type": chunk.metadata.get("doc_type", "教育文档"), "parser": "qwen-vl"},
                    confidence=0.9,
                )
            )
        return enriched

    def _log_etl_failure(self, path: Path, stage: str, error: Exception) -> None:
        status = getattr(error, "code", None)
        if status is None:
            match = re.search(r"\bHTTP Error (\d{3})\b", str(error))
            status = int(match.group(1)) if match else None
        source_url = getattr(error, "url", None)
        upstream_host = urlsplit(str(source_url)).hostname if source_url else None
        frames = traceback.extract_tb(error.__traceback__)
        write_debug_log(
            self.config,
            "document_etl_stage_failed",
            filename=path.name,
            stage=stage,
            error_type=type(error).__name__,
            upstream_status=status,
            upstream_host=upstream_host,
            traceback_frames=[
                {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                for frame in frames[-10:]
            ],
        )

    def _parse_pdf_fallback(self, path: Path, max_pages: Optional[int] = None) -> List[DocumentChunk]:
        try:
            import fitz  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("PDF fallback requires PyMuPDF") from exc
        document = fitz.open(str(path))
        try:
            chunks: List[DocumentChunk] = []
            page_limit = document.page_count if not max_pages or max_pages <= 0 else min(document.page_count, max_pages)
            for page_index in range(page_limit):
                text = document.load_page(page_index).get_text("text").strip()
                if not text:
                    continue
                chunks.append(DocumentChunk(
                    id=f"{path.name}:p{page_index + 1}",
                    text=text,
                    source=str(path),
                    page=page_index + 1,
                    block_id=f"p{page_index + 1}",
                    modality="pdf_text",
                    metadata={"doc_type": "PDF文档", "parser": "pymupdf"},
                ))
            return chunks
        finally:
            document.close()

    @staticmethod
    def _plain_text_chunk(path: Path) -> DocumentChunk:
        return DocumentChunk(
            id=f"{path.name}:text:1",
            text=path.read_text(encoding="utf-8-sig"),
            source=str(path),
            page=1,
            block_id="text-1",
            modality="document_text",
            metadata={"doc_type": "文本资料", "parser": "plain-text"},
        )


def replace_source(chunk: DocumentChunk, source: str, config: RAGConfig = DEFAULT_CONFIG) -> DocumentChunk:
    metadata = dict(chunk.metadata)
    parent_id = chunk.parent_id.replace(Path(chunk.source).name, source) if chunk.parent_id else None
    if parent_id:
        metadata["parent_id"] = parent_id
    image_path = persist_image_file(chunk.image_path, config) if chunk.image_path else None
    return DocumentChunk(
        id=chunk.id.replace(Path(chunk.source).name, source),
        text=chunk.text,
        source=source,
        page=chunk.page,
        block_id=chunk.block_id,
        modality=chunk.modality,
        coordinates=chunk.coordinates,
        metadata=metadata,
        image_path=str(image_path) if image_path else None,
        parent_id=parent_id,
        chunk_level=chunk.chunk_level,
        confidence=chunk.confidence,
        needs_review=chunk.needs_review,
    )


def persist_image_bytes(content: bytes, suffix: str, config: RAGConfig) -> Path:
    """Keep an uploaded image available for multimodal vector indexing after restart."""

    asset_dir = image_asset_directory(config)
    asset_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content).hexdigest()
    safe_suffix = suffix.lower() if suffix.lower() in IMAGE_SUFFIXES else ".png"
    destination = asset_dir / f"{digest}{safe_suffix}"
    if not destination.exists():
        destination.write_bytes(content)
    return destination


def persist_image_file(path: Optional[str], config: RAGConfig) -> Optional[Path]:
    if not path:
        return None
    source_path = Path(path)
    try:
        if not source_path.is_file():
            return None
        content = source_path.read_bytes()
        return persist_image_bytes(content, source_path.suffix, config)
    except OSError:
        return None


def image_asset_directory(config: RAGConfig) -> Path:
    runtime_path = config.kb_runtime_path.strip() or "tmp/runtime_knowledge_base.jsonl"
    return Path(runtime_path).expanduser().parent / "assets"


def positive_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = {"filename", "file_directory", "section", "category_depth", "page_name"}
    return {key: value for key, value in metadata.items() if key in allowed and value is not None}


def suffix_doc_type(suffix: str) -> str:
    return {".pdf": "PDF文档", ".doc": "Word文档", ".docx": "Word文档", ".ppt": "PPT课件", ".pptx": "PPT课件"}.get(
        suffix.lower(), "教育文档"
    )
