"""
AdvancedPDFParser - improved and runnable .py
Features:
- Extracts text lines via pdfminer for born-digital PDFs
- Falls back to Tesseract OCR (pytesseract) for scanned/image PDFs
- Detects single or two-column layout using kmeans
- Segments paragraphs, headings, lists, and tables (heuristic + optional Camelot)
- Exports structured JSON

Requirements:
pip install pdfminer.six camelot-py[cv] pytesseract pillow numpy pandas scikit-learn
System: poppler (for pdf2image) and tesseract-ocr installed if using OCR fallback

Usage:
python advanced_pdf_parser.py /path/to/file.pdf

"""

import sys
import os
import re
import json
from dataclasses import dataclass
from statistics import median
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from sklearn.cluster import KMeans

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal, LTTextLineHorizontal, LTChar


# Optional libs
try:
    import camelot
    _HAS_CAMELOT = True
except Exception:
    _HAS_CAMELOT = False

try:
    from pdf2image import convert_from_path
    from PIL import Image
    import pytesseract
    Image.MAX_IMAGE_PIXELS = None  # optional

    _HAS_TESSERACT = True
except Exception:
    _HAS_TESSERACT = False


# -----------------------
# Configuration
# -----------------------
@dataclass
class Config:
    para_gap_factor: float = 1.6
    indent_threshold: float = 8.0
    heading_font_factor: float = 1.18
    column_overlap_threshold: float = 0.25
    table_min_rows: int = 3


# -----------------------
# Data classes
# -----------------------
@dataclass
class Line:
    text: str
    x0: float
    x1: float
    y0: float
    y1: float
    page: int
    font_size: float = 0.0
    font_name: str = ""
    is_bold: bool = False
    is_italic: bool = False

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def center_x(self):
        return (self.x0 + self.x1) / 2

    @property
    def height(self):
        return self.y1 - self.y0


@dataclass
class Paragraph:
    id: str
    page: int
    ptype: str
    text: str
    bbox: Tuple[float, float, float, float]
    lines: List[Line]
    meta: Dict[str, Any]


# -----------------------
# Helpers
# -----------------------
BULLET_RE = re.compile(r'^\s*([•\u2022\-\*\u2013\u2014]|\d+[\.\)]|[A-Z]\.)\s+')


def _is_bold_fontname(fontname: str) -> bool:
    fname = (fontname or "").lower()
    return any(tok in fname for tok in ("bold", "bd", "black"))


def _is_italic_fontname(fontname: str) -> bool:
    fname = (fontname or "").lower()
    return "italic" in fname or "oblique" in fname


def clean_text_for_paragraph(t: str) -> str:
    return re.sub(r'\s+', ' ', t).strip()


def median_or_default(values: List[float], default=12.0) -> float:
    if not values:
        return default
    return median(values)


# -----------------------
# Parser
# -----------------------
class AdvancedPDFParser:
    def __init__(self, config: Config = Config()):
        self.config = config

    def extract_lines_from_pdf(self, pdf_path: str) -> Dict[int, List[Line]]:
        """Use pdfminer to extract text lines (born-digital PDFs).
        If pdfminer returns no text for a page, fallback to OCR when available.
        Returns: page_num -> List[Line]
        """
        pages_lines: Dict[int, List[Line]] = {}

        for page_idx, page_layout in enumerate(extract_pages(pdf_path), start=1):
            page_lines: List[Line] = []
            for element in page_layout:
                if isinstance(element, LTTextBoxHorizontal):
                    for obj in element:
                        if not isinstance(obj, LTTextLineHorizontal):
                            continue
                        # accumulate text and pick font info from LTChar
                        raw_text = "".join(getattr(c, 'get_text', lambda: '')() if not isinstance(c, LTChar) else c.get_text() for c in obj)
                        line_text = raw_text.strip()
                        if not line_text:
                            continue

                        font_sizes = []
                        font_names = []
                        for ch in obj:
                            if isinstance(ch, LTChar):
                                font_sizes.append(round(getattr(ch, 'size', 0), 2))
                                font_names.append(getattr(ch, 'fontname', ''))
                        fsize = median_or_default(font_sizes, default=12.0)
                        fname = font_names[0] if font_names else ""
                        is_bold = _is_bold_fontname(fname)
                        is_italic = _is_italic_fontname(fname)
                        x0, y0, x1, y1 = obj.bbox
                        page_lines.append(Line(text=line_text, x0=x0, x1=x1, y0=y0, y1=y1, page=page_idx, font_size=fsize, font_name=fname, is_bold=is_bold, is_italic=is_italic))

            # sort top->bottom, left->right
            page_lines.sort(key=lambda L: (-L.y1, L.x0))
            pages_lines[page_idx] = page_lines

        # If pages_lines is empty (scanned PDF), try OCR fallback
        any_text = any(pages_lines.values())
        if not any_text and _HAS_TESSERACT:
            print("[INFO] No text extracted with pdfminer; using Tesseract OCR fallback.")
            return self._ocr_fallback(pdf_path)

        return pages_lines

    def _ocr_fallback(self, pdf_path: str) -> Dict[int, List[Line]]:
        """Convert pages to images and use pytesseract to get word boxes and form lines."""
        pages_lines: Dict[int, List[Line]] = {}
        images = convert_from_path(pdf_path)
        for i, pil_img in enumerate(images, start=1):
            data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
            n = len(data['text'])
            # group words into simple lines based on top coordinate
            rows: Dict[int, List[Dict[str, Any]]] = {}
            for idx in range(n):
                txt = data['text'][idx].strip()
                if not txt:
                    continue
                top = int(data['top'][idx])
                left = int(data['left'][idx])
                width = int(data['width'][idx])
                height = int(data['height'][idx])
                conf = int(data['conf'][idx]) if str(data['conf'][idx]).isdigit() else -1

                # bucket by top with tolerance
                bucket = int(top / 10)
                rows.setdefault(bucket, []).append({'text': txt, 'left': left, 'top': top, 'w': width, 'h': height})
            # convert rows to Line objects
            page_lines = []
            for bucket, items in sorted(rows.items(), key=lambda x: -min(it['top'] for it in x[1])):
                items.sort(key=lambda it: it['left'])
                text = ' '.join(it['text'] for it in items)
                x0 = min(it['left'] for it in items)
                y0 = min(it['top'] for it in items)
                x1 = max(it['left'] + it['w'] for it in items)
                y1 = max(it['top'] + it['h'] for it in items)
                # Note: Tesseract coords are image pixels; keep as-is but user may convert
                page_lines.append(Line(text=text, x0=float(x0), x1=float(x1), y0=float(y0), y1=float(y1), page=i, font_size=float(np.median([it['h'] for it in items]))))
            page_lines.sort(key=lambda L: (-L.y1, L.x0))
            pages_lines[i] = page_lines
        return pages_lines

    def cluster_columns(self, lines: List[Line], page_width: float) -> List[List[Line]]:
        if not lines:
            return []
        x_centers = np.array([(ln.x0 + ln.x1) / 2 for ln in lines]).reshape(-1, 1)
        if len(x_centers) > 10:
            try:
                kmeans = KMeans(n_clusters=2, n_init=5, random_state=42).fit(x_centers)
                labels = kmeans.labels_
                cluster_means = [(np.mean(x_centers[labels == i]), i) for i in range(2)]
                cluster_means = sorted(cluster_means, key=lambda x: x[0])
                gap = abs(cluster_means[1][0] - cluster_means[0][0])
                if gap > (page_width * 0.25):
                    left_lines = [ln for idx, ln in enumerate(lines) if labels[idx] == cluster_means[0][1]]
                    right_lines = [ln for idx, ln in enumerate(lines) if labels[idx] == cluster_means[1][1]]
                    left_lines.sort(key=lambda l: -l.y0)
                    right_lines.sort(key=lambda l: -l.y0)
                    return [left_lines, right_lines]
            except Exception:
                pass
        # fallback single column
        return [sorted(lines, key=lambda l: (-l.y0, l.x0))]

    def detect_table_block_heuristic(self, lines: List[Line]) -> Optional[Dict[str, Any]]:
        rows = []
        for ln in lines:
            parts = [p for p in re.split(r'\s{2,}', ln.text) if p.strip()]
            rows.append((ln, parts))
        if len(rows) >= self.config.table_min_rows:
            avg_parts = sum(len(parts) for _, parts in rows) / max(1, len(rows))
            if avg_parts >= 2.0:
                table_rows = [parts for _, parts in rows]
                bbox = (min(ln.x0 for ln, _ in rows), min(ln.y0 for ln, _ in rows), max(ln.x1 for ln, _ in rows), max(ln.y1 for ln, _ in rows))
                return {"rows": table_rows, "bbox": bbox}
        return None

    def segment_paragraphs_in_column(self, lines: List[Line], page_median_line_h: float, page_median_font: float, page_width: float, pdf_path: Optional[str]=None) -> List[Paragraph]:
        paras: List[Paragraph] = []
        pid_seq = 1

        def flush_block(block: List[Line], ptype: str = "paragraph"):
            nonlocal pid_seq
            if not block:
                return
            x0 = min(l.x0 for l in block)
            y0 = min(l.y0 for l in block)
            x1 = max(l.x1 for l in block)
            y1 = max(l.y1 for l in block)
            page_no = block[0].page
            text = clean_text_for_paragraph(" ".join(l.text for l in block))
            avg_font = median_or_default([l.font_size for l in block], default=page_median_font)
            pid = f"P_p{page_no}_{pid_seq}"
            paras.append(Paragraph(id=pid, page=page_no, ptype=ptype, text=text, bbox=(x0, y0, x1, y1), lines=list(block), meta={"lines": len(block), "avg_font": avg_font}))
            pid_seq += 1

        # Check for table-like block
        table_check = self.detect_table_block_heuristic(lines)
        if table_check:
            pid = f"T_p{lines[0].page}_1"
            paras.append(Paragraph(id=pid, page=lines[0].page, ptype="table", text=json.dumps(table_check['rows'], ensure_ascii=False), bbox=table_check['bbox'], lines=lines, meta={"rows": len(table_check['rows']), "source": "heuristic"}))
            return paras

        # Segment by vertical gaps and indentation
        current: List[Line] = []
        prev: Optional[Line] = None
        for ln in lines:
            if prev is None:
                current.append(ln)
                prev = ln
                continue
            dy = prev.y1 - ln.y1
            vertical_break = dy > (self.config.para_gap_factor * page_median_line_h)
            indent_break = abs(ln.x0 - prev.x0) > self.config.indent_threshold and ln.x0 > prev.x0 + (self.config.indent_threshold * 0.5)
            is_bullet = bool(BULLET_RE.match(ln.text))
            heading_candidate = (ln.is_bold or ln.font_size >= page_median_font * self.config.heading_font_factor) and abs(ln.center_x - page_width/2) < (page_width*0.08)

            if heading_candidate:
                if current:
                    flush_block(current)
                    current = []
                flush_block([ln], ptype="heading")
                prev = None
                continue

            if is_bullet:
                if current:
                    flush_block(current)
                    current = []
                flush_block([ln], ptype="list")
                prev = ln
                continue

            if vertical_break or indent_break:
                flush_block(current)
                current = [ln]
            else:
                # handle hyphenation
                if current and current[-1].text.rstrip().endswith('-'):
                    current[-1] = Line(text=current[-1].text.rstrip('-') + ln.text, x0=current[-1].x0, x1=ln.x1, y0=current[-1].y0, y1=ln.y1, page=ln.page, font_size=ln.font_size)
                else:
                    current.append(ln)

            prev = ln

        if current:
            flush_block(current)

        return paras

    # def parse(self, pdf_path: str) -> List[Paragraph]:
    #     pages_lines = self.extract_lines_from_pdf(pdf_path)
    #     all_paragraphs: List[Paragraph] = []
    #     for page_num, lines in pages_lines.items():
    #         if not lines:
    #             continue
    #         heights = [max(1.0, l.height) for l in lines]
    #         median_line_h = median_or_default(heights, default=12.0)
    #         font_sizes = [l.font_size for l in lines]
    #         median_font = median_or_default(font_sizes, default=12.0)
    #         page_width = max((l.x1 for l in lines), default=600.0)

    #         # Camelot detection (optional)
    #         camelot_tables = None
    #         if _HAS_CAMELOT:
    #             try:
    #                 tables = camelot.read_pdf(pdf_path, pages=str(page_num))
    #                 if tables and len(tables) > 0:
    #                     camelot_tables = [t.df.values.tolist() for t in tables]
    #             except Exception as e:
    #                 print(f"[Camelot error] page {page_num}: {e}")

    #         columns = self.cluster_columns(lines, page_width)
    #         for col in columns:
    #             paras = self.segment_paragraphs_in_column(col, median_line_h, median_font, page_width, pdf_path)
    #             all_paragraphs.extend(paras)

    #         if camelot_tables:
    #             for idx, tbl in enumerate(camelot_tables, start=1):
    #                 pid = f"Camelot_t{page_num}_{idx}"
    #                 bbox = (0, 0, page_width, 0)
    #                 all_paragraphs.append(Paragraph(id=pid, page=page_num, ptype='table', text=json.dumps(tbl, ensure_ascii=False), bbox=bbox, lines=[], meta={"rows": len(tbl), "cols": len(tbl[0]) if tbl else 0, "source": "camelot"}))

    #     all_paragraphs.sort(key=lambda p: (p.page, -p.bbox[3] if p.bbox else 0, p.bbox[0] if p.bbox else 0))
    #     return all_paragraphs


    def parse(self, pdf_path: str) -> str:
        pages_lines = self.extract_lines_from_pdf(pdf_path)
        all_paragraphs: List[Paragraph] = []

        for page_num, lines in pages_lines.items():
            if not lines:
                continue

            heights = [max(1.0, l.height) for l in lines]
            median_line_h = median_or_default(heights, default=12.0)
            font_sizes = [l.font_size for l in lines]
            median_font = median_or_default(font_sizes, default=12.0)
            page_width = max((l.x1 for l in lines), default=600.0)

            # Camelot detection (optional)
            camelot_tables = None
            if _HAS_CAMELOT:
                try:
                    tables = camelot.read_pdf(pdf_path, pages=str(page_num))
                    if tables and len(tables) > 0:
                        camelot_tables = [t.df.values.tolist() for t in tables]
                except Exception as e:
                    print(f"[Camelot error] page {page_num}: {e}")

            columns = self.cluster_columns(lines, page_width)
            for col in columns:
                paras = self.segment_paragraphs_in_column(col, median_line_h, median_font, page_width, pdf_path)
                all_paragraphs.extend(paras)

            if camelot_tables:
                for idx, tbl in enumerate(camelot_tables, start=1):
                    pid = f"Camelot_t{page_num}_{idx}"
                    bbox = (0, 0, page_width, 0)
                    all_paragraphs.append(
                        Paragraph(
                            id=pid,
                            page=page_num,
                            ptype="table",
                            text=json.dumps(tbl, ensure_ascii=False),
                            bbox=bbox,
                            lines=[],
                            meta={
                                "rows": len(tbl),
                                "cols": len(tbl[0]) if tbl else 0,
                                "source": "camelot",
                            },
                        )
                    )

        # Sort the paragraphs
        all_paragraphs.sort(
            key=lambda p: (p.page, -p.bbox[3] if p.bbox else 0, p.bbox[0] if p.bbox else 0)
        )

        # Extract only required fields
        all_text=''
        output = []
        for p in all_paragraphs:
            output.append({
                "page_num": p.page,
                "paragraph_id": p.id,
                "text": p.text.strip() if p.text else "",
            })
            all_text=all_text+'\n'+p.text.strip() if p.text else ""
           

        # Return as JSON string
        return {"Json_data":output, "All_text":all_text}


    # def to_json(self, paragraphs: List[Paragraph], out_path: str = 'parsed_paragraphs.json') -> str:
    #     payload = []
    #     pdf_json=[]
    #     for p in paragraphs:
    #         payload.append({
    #             'id': p.id,
    #             'page': p.page,
    #             'type': p.ptype,
    #             'text': p.text,
    #             'bbox': p.bbox,
    #             'lines_count': len(p.lines) if isinstance(p.lines, (list, tuple)) else 0,
    #             'meta': p.meta
    #         })
    #         pdf_json.append({'page_no':p.page,"paragraph":p.id,'text':p.text})

    #     s = json.dumps(payload, ensure_ascii=False, indent=2)
    #     with open(out_path, 'w', encoding='utf-8') as f:
    #         f.write(s)
    #     return s, pdf_json
