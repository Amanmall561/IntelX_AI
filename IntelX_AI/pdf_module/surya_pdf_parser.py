from __future__ import annotations

from typing import Any, Dict, List, Tuple
import os
import time
from datetime import datetime
import gc
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
from pdf2image import convert_from_path
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
# Late imports for models in subprocess
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings
from surya.table_rec import TableRecPredictor
import json
import torch
from collections import defaultdict


def process_single_fast(pdf_path: str, dpi: int, use_gpu: bool, skip_tables_text: bool) -> Dict[str, Any]:
    """Standalone fast worker. Batched OCR per page, skip table cell OCR."""
   

    Image.MAX_IMAGE_PIXELS = None

    pdf_name = Path(pdf_path).name
    t0 = time.time()

    use_gpu = use_gpu and torch.cuda.is_available()
    if use_gpu:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if free_bytes < 2 * 1024**3:  # < 2 GB free → fallback to CPU
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU
            use_gpu = False

    try:
        # Stream pages, do not load all at once if many pages: process in chunks
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = Path(pdf_name).stem
        pages = convert_from_path(pdf_path, dpi=dpi, use_pdftocairo=True)
        total_pages = len(pages)

        # Initialize models once
        

        json_output: List[Dict[str, Any]] = []
        processed = 0

        for idx, img in enumerate(pages):
            page_json = fast_process_page(img, idx)
            json_output.append(page_json)
            processed += 1

            # cleanup per page
            del img
            gc.collect()

        simple_pages = build_simple_page_records(json_output)
        # with open(simple_json_path, 'w', encoding='utf-8') as f:
        #     json.dump(simple_pages, f, ensure_ascii=False, indent=2)

        return {
            "success": True,
            "pages_processed": processed,
            "processing_time": time.time() - t0,
            "simple_json": simple_pages,
            "error_message": None,
        }

    except Exception as e:
        return {
            "success": False,
            "pages_processed": 0,
            "processing_time": time.time()-t0,
            "simple_json": [],
            "error_message": str(e)
        }


def fast_process_page(img: Image.Image, page_idx: int):
    foundation = FoundationPredictor()
    recognition = RecognitionPredictor(foundation)
    detection = DetectionPredictor()
    layout = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
    table_rec = TableRecPredictor()

    layout_pred = layout([img])[0]

    text_lines: List[Dict[str, Any]] = []
    tables = []

    # Collect crops for batched OCR
    crops: List[Image.Image] = []
    crop_meta: List[Tuple[str, List[int]]] = []  # (label, bbox)

    if hasattr(layout_pred, "bboxes"):
        for block in layout_pred.bboxes:
            bbox = list(map(int, block.bbox))
            x0, y0, x1, y1 = bbox
            ox1, oy1, _, _ = bbox
            label = block.label

            if label == 'Table':
                table_img = img.crop((x0, y0, x1, y1))
                try:
                    table_pred = table_rec([table_img])[0]
                except Exception:
                    table_pred = None


                if hasattr(table_pred, "cells") and table_pred.cells:
                    for cell in table_pred.cells:
                        bbox_tb = list(map(int, cell.bbox))
                        crop_tb = table_img.crop((bbox_tb[0], bbox_tb[1], bbox_tb[2], bbox_tb[3]))
                        rec_preds_tb = recognition([crop_tb], det_predictor=detection)
                        page_pred_tb = rec_preds_tb[0]
                        
                        textt = ''
                        for tl_tb in page_pred_tb.text_lines:
                            textt = tl_tb.text.strip()
                        cell.text_lines = textt
                        cell.bbox[0] += ox1
                        cell.bbox[1] += oy1
                        cell.bbox[2] += ox1
                        cell.bbox[3] += oy1
                
                rows = {}
                for cell in getattr(table_pred, "cells", []):
                    row_id = getattr(cell, "row_id", 0)
                    rows.setdefault(row_id, []).append(cell)
                
                for row_id, row_cells in sorted(rows.items()):
                    
                    for cell in sorted(row_cells, key=lambda c: getattr(c, "col_id", 0)):
                        cell_attrs = {
                            "id": str(getattr(cell, "cell_id", "")),
                            "col_id": str(getattr(cell, "col_id", "")),
                            "rowspan": str(getattr(cell, "rowspan", 1)),
                            "colspan": str(getattr(cell, "colspan", 1)),
                            "header": str(getattr(cell, "is_header", False)).lower()
                        }
                        # cell_elem = ET.SubElement(row_elem, "cell", **cell_attrs)
                        
                        # text_elem = ET.SubElement(cell_elem, "text")
                        # text_elem.text = getattr(cell, "text_lines", "").strip()
                        
                        # bbox_elem = ET.SubElement(cell_elem, "bbox")
                        # bbox_elem.text = ",".join(map(str, getattr(cell, "bbox", [])))

            

            # For text-like blocks, add to batch list
            lower = label.lower()
            if any(k in lower for k in ["text", "paragraph", "title", "heading", "caption", "list", "section"]):
                crops.append(img.crop((x0, y0, x1, y1)))
                crop_meta.append((lower, [x0, y0, x1, y1]))

    # Batched OCR for all text crops on the page
    if crops:
        rec_preds_list = recognition(crops, det_predictor=detection)
        for (lower, bbox), page_pred in zip(crop_meta, rec_preds_list):
            x0, y0, x1, y1 = bbox
            full_text = []
            for tl in page_pred.text_lines:
                ox = [tl.bbox[0] + x0, tl.bbox[1] + y0, tl.bbox[2] + x0, tl.bbox[3] + y0]
                text_lines.append({"text": tl.text.strip(), "bbox": ox, "confidence": tl.confidence})
                if tl.text:
                    full_text.append(tl.text.strip())
            text = " ".join(full_text)
            bbox_str = ",".join(map(str, bbox))
            conf = "1.0"
            
    # Merge lines into paragraphs (simple)
    paragraphs = merge_lines_to_paragraphs(text_lines)

    page_json = {
        "page": page_idx + 1,
        "paragraphs": paragraphs,
        "tables": tables    }

    return page_json


def merge_lines_to_paragraphs(lines: List[Dict[str, Any]], y_threshold: int = 18) -> List[Dict[str, Any]]:
    if not lines:
        return []
    lines = sorted(lines, key=lambda l: l["bbox"][1])
    paragraphs, current = [], [lines[0]]
    for line in lines[1:]:
        prev_y = current[-1]["bbox"][3]
        curr_y = line["bbox"][1]
        if curr_y - prev_y <= y_threshold:
            current.append(line)
        else:
            paragraphs.append(_lines_to_paragraph(current))
            current = [line]
    if current:
        paragraphs.append(_lines_to_paragraph(current))
    return paragraphs


def _lines_to_paragraph(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    text = " ".join([sanitize_text(l["text"]) for l in lines if l["text"]])
    return {
        "type": "paragraph",
        "text": text,
        "bbox": [
            min(l["bbox"][0] for l in lines),
            min(l["bbox"][1] for l in lines),
            max(l["bbox"][2] for l in lines),
            max(l["bbox"][3] for l in lines),
        ],
        "lines": lines
    }


def sanitize_text(text: str) -> str:
    return " ".join(text.strip().split())


def build_simple_page_records(pages: Any) -> List[Dict[str, Any]]:
    """
    Convert rich page structures into a simple JSON format:
      {"page_no": 1, "extracted_data": [{"text": "...", "text_type": "paragraph"}, ...]}.
    Works with both the scanned JSON produced by this module and generic
    parser outputs that contain "text" and "type" fields.
    """

    simple: List[Dict[str, Any]] = []

    # Normalize to list of page dicts
    if isinstance(pages, dict):
        pages_iter: List[Any] = [pages]
    else:
        pages_iter = list(pages)  # type: ignore[arg-type]

    for page in pages_iter:
        page_no = page.get("page") or page.get("page_no") or page.get("page_num") or 1
        extracted_items: List[Dict[str, Any]] = []

        # Special handling for our own scanned JSON structure
        for para in page.get("paragraphs", []):
            text = para.get("text")
            if text:
                extracted_items.append(
                    {"text": sanitize_text(text), "text_type": para.get("type", "paragraph")}
                )

        for tbl in page.get("tables", []):
            text = tbl.get("text")
            if text:
                extracted_items.append(
                    {"text": sanitize_text(text), "text_type": tbl.get("type", "table")}
                )

        # Fallback: recursively search for generic {"text", "type"} patterns
        if not extracted_items:
            _collect_generic_items(page, page_no, extracted_items)

        if extracted_items:
            simple.append({"page_no": page_no, "extracted_data": extracted_items})

    return simple


def _collect_generic_items(
    node: Any, page_no: int, out: List[Dict[str, Any]]
) -> None:
    """Recursively collect items with 'text' and 'type' fields."""

    if isinstance(node, dict):
        text = node.get("text")
        text_type = node.get("type")
        if isinstance(text, str) and isinstance(text_type, str) and text.strip():
            out.append({"text": sanitize_text(text), "text_type": text_type})

        for value in node.values():
            _collect_generic_items(value, page_no, out)

    elif isinstance(node, list):
        for item in node:
            _collect_generic_items(item, page_no, out)


# res=process_single_fast(pdf_path='/home/ubuntu/Airline_identify/pdf_layout_parser/2510.02665v1.pdf',dpi=120,use_gpu=True,skip_tables_text=False)
# print(res['simple_json'])