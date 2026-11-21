import base64
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import fitz
from docx import Document
from docx.shared import Inches
from pptx import Presentation
from pptx.util import Inches as PptxInches

from manufacturing_rag.config import RAGConfig
from manufacturing_rag.etl import MultimodalETLPipeline
from manufacturing_rag.schemas import DocumentChunk


class FakeOCR:
    def __init__(self, text="图中文字 OCR：节点 A、节点 B"):
        self.text = text

    def parse_image_bytes(self, image_bytes, source, page=1):
        return [DocumentChunk(id="ocr", text=self.text, source=source, page=page)]


class FakeVision:
    def __init__(self, description="箭头由节点 A 指向节点 B"):
        self.description = description
        self.prompts = []

    def describe_bytes(self, image_bytes, mime_type="image/png", prompt=None):
        self.prompts.append(prompt or "")
        return self.description


def make_pipeline(tmp_path, ocr_text=None, vision_text=None, max_pages=0):
    return MultimodalETLPipeline(
        RAGConfig(
            kb_runtime_path=str(tmp_path / "runtime.jsonl"),
            debug_log_enabled=False,
            baidu_ocr_pdf_max_pages=max_pages,
        ),
        ocr=FakeOCR(ocr_text or "图中文字 OCR：节点 A、节点 B"),
        vision=FakeVision(vision_text or "箭头由节点 A 指向节点 B"),
    )


def make_png(tmp_path):
    from PIL import Image, ImageDraw

    path = tmp_path / "embedded.png"
    image = Image.new("RGB", (80, 60), "white")
    ImageDraw.Draw(image).rectangle((8, 8, 70, 50), outline="black", width=2)
    image.save(path)
    return path


def test_standalone_image_fuses_ocr_visual_description_and_original_image(tmp_path):
    image_path = make_png(tmp_path)
    pipeline = make_pipeline(tmp_path)

    [chunk] = pipeline._parse_image(image_path)

    assert "图中文字 OCR" in chunk.text
    assert "箭头由节点 A 指向节点 B" in chunk.text
    assert chunk.modality == "multimodal_visual"
    assert chunk.image_path and Path(chunk.image_path).is_file()


def test_empty_unstructured_image_element_keeps_payload_and_nearby_caption(tmp_path, monkeypatch):
    image_path = make_png(tmp_path)

    class Caption:
        category = "Caption"
        metadata = SimpleNamespace(to_dict=lambda: {"page_number": 1})

        def __str__(self):
            return "图 1：学习曲线示意图"

    caption = Caption()

    class Element:
        category = "Image"
        metadata = SimpleNamespace(
            to_dict=lambda: {
                "page_number": 1,
                "image_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
                "image_mime_type": "image/png",
            }
        )

        def __str__(self):
            return ""

    class TextElement:
        category = "NarrativeText"
        metadata = SimpleNamespace(to_dict=lambda: {"page_number": 1})

        def __str__(self):
            return "横轴表示学习时间，纵轴表示正确率。"

    auto_module = ModuleType("unstructured.partition.auto")
    auto_module.partition = lambda **kwargs: [caption, Element(), TextElement()]
    unstructured_module = ModuleType("unstructured")
    unstructured_module.__path__ = []
    partition_module = ModuleType("unstructured.partition")
    partition_module.__path__ = []
    monkeypatch.setitem(sys.modules, "unstructured", unstructured_module)
    monkeypatch.setitem(sys.modules, "unstructured.partition", partition_module)
    monkeypatch.setitem(sys.modules, "unstructured.partition.auto", auto_module)

    document_path = tmp_path / "lesson.docx"
    document_path.write_bytes(b"test placeholder")
    pipeline = make_pipeline(tmp_path)

    chunks = pipeline._parse_unstructured(document_path)
    visual_chunks = [chunk for chunk in chunks if chunk.modality == "multimodal_visual"]

    assert len(visual_chunks) == 1
    chunk = visual_chunks[0]
    assert "图 1：学习曲线示意图" in chunk.text
    assert "横轴表示学习时间" in chunk.text
    assert "图中文字 OCR" in chunk.text
    assert "箭头由节点 A 指向节点 B" in chunk.text
    assert chunk.image_path and Path(chunk.image_path).is_file()


def test_docx_embedded_image_is_bound_to_nearby_paragraph_text(tmp_path):
    image_path = make_png(tmp_path)
    docx_path = tmp_path / "lesson.docx"
    document = Document()
    document.add_paragraph("图 2 展示了实验装置的连接关系。")
    document.add_picture(str(image_path), width=Inches(1))
    document.add_paragraph("电源输出连接到测量仪器，再接入被测元件。")
    document.save(docx_path)

    chunks = make_pipeline(tmp_path)._extract_docx_visuals(docx_path, [])

    assert len(chunks) == 1
    assert "图 2 展示了实验装置" in chunks[0].text
    assert "电源输出连接到测量仪器" in chunks[0].text
    assert "图中文字 OCR" in chunks[0].text
    assert chunks[0].image_path and Path(chunks[0].image_path).is_file()


def test_docx_parse_file_uses_local_fallback_if_unstructured_fails(tmp_path, monkeypatch):
    image_path = make_png(tmp_path)
    docx_path = tmp_path / "lesson.docx"
    document = Document()
    document.add_paragraph("实验装置图")
    document.add_picture(str(image_path), width=Inches(1))
    document.add_paragraph("电源连接测量仪器后再接入被测元件。")
    document.save(docx_path)
    pipeline = make_pipeline(tmp_path)

    def fail_unstructured(_path):
        raise RuntimeError("Unstructured unavailable")

    monkeypatch.setattr(pipeline, "_parse_unstructured", fail_unstructured)
    result = pipeline.parse_file(str(docx_path))

    assert "python-docx" in result.parser_chain
    assert "Qwen3-VL" in result.parser_chain
    assert any("实验装置图" in chunk.text for chunk in result.chunks)
    assert any(chunk.image_path for chunk in result.chunks)


def test_pptx_picture_keeps_slide_text_and_position_context(tmp_path, monkeypatch):
    image_path = make_png(tmp_path)
    pptx_path = tmp_path / "lesson.pptx"
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    title = slide.shapes.add_textbox(PptxInches(0.5), PptxInches(0.3), PptxInches(5), PptxInches(0.6))
    title.text = "实验流程图"
    label = slide.shapes.add_textbox(PptxInches(0.5), PptxInches(1.2), PptxInches(3), PptxInches(0.5))
    label.text = "先采集数据，再计算结果"
    slide.shapes.add_picture(str(image_path), PptxInches(4), PptxInches(1), width=PptxInches(2))
    presentation.save(pptx_path)

    pipeline = make_pipeline(tmp_path)
    chunks = pipeline._extract_pptx_visuals(pptx_path, [])
    visual_chunks = [chunk for chunk in chunks if chunk.modality == "multimodal_visual"]

    assert len(visual_chunks) == 1
    assert "实验流程图" in visual_chunks[0].text
    assert "先采集数据，再计算结果" in visual_chunks[0].text
    assert "图形、图表与版面关系" in visual_chunks[0].text
    assert visual_chunks[0].image_path and Path(visual_chunks[0].image_path).is_file()

    def fail_unstructured(_path):
        raise RuntimeError("Unstructured unavailable")

    monkeypatch.setattr(pipeline, "_parse_unstructured", fail_unstructured)
    result = pipeline.parse_file(str(pptx_path))
    assert "python-pptx" in result.parser_chain
    assert any("实验流程图" in chunk.text for chunk in result.chunks)
    assert any(chunk.image_path for chunk in result.chunks)


def test_pdf_vector_diagram_page_is_rendered_with_extracted_text(tmp_path):
    pdf_path = tmp_path / "flow.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((80, 70), "流程图说明：先准备，再执行，最后复核")
    boxes = [(60, 180, 180, 250), (230, 180, 350, 250), (400, 180, 520, 250), (230, 340, 350, 410)]
    for rect in boxes:
        page.draw_rect(fitz.Rect(*rect), color=(0, 0, 0), width=2)
    page.draw_line((180, 215), (230, 215), color=(0, 0, 0), width=2)
    page.draw_line((350, 215), (400, 215), color=(0, 0, 0), width=2)
    page.draw_line((290, 250), (290, 340), color=(0, 0, 0), width=2)
    second_page = document.new_page(width=600, height=800)
    second_page.insert_text((80, 70), "第二页流程图")
    for rect in boxes:
        second_page.draw_rect(fitz.Rect(*rect), color=(0, 0, 0), width=2)
    second_page.draw_line((180, 215), (230, 215), color=(0, 0, 0), width=2)
    second_page.draw_line((350, 215), (400, 215), color=(0, 0, 0), width=2)
    second_page.draw_line((290, 250), (290, 340), color=(0, 0, 0), width=2)
    document.save(pdf_path)
    document.close()
    page_text = DocumentChunk(
        id="flow:p1",
        text="流程图说明：先准备，再执行，最后复核",
        source=str(pdf_path),
        page=1,
        modality="pdf_text",
    )

    chunks = make_pipeline(tmp_path, max_pages=1)._extract_pdf_visual_pages(
        pdf_path, [page_text], max_pages=None, render_dpi=120
    )

    assert len(chunks) == 1
    assert chunks[0].page == 1
    assert "流程图说明" in chunks[0].text
    assert "箭头由节点 A 指向节点 B" in chunks[0].text
    assert chunks[0].image_path and Path(chunks[0].image_path).is_file()
