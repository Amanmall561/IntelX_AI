import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import tempfile
import numpy as np
from PIL import Image
import cv2
import os
import json


def detect_pdf_type(pdf_path, check_layout=True, layout_threshold=0.25):
    """
    Detect the type of PDF file:
    - Digital PDF: contains selectable/extractable text
    - Scanned PDF: only images, no extractable text
    - Hybrid PDF: contains both text and image layers
    - Structured Layout PDF: digital PDF with visible multi-column or table structure

    Args:
        pdf_path (str): path to the PDF file
        check_layout (bool): whether to detect structured layouts (columns/tables)
        layout_threshold (float): fraction of image area with detected lines to classify as structured

    Returns:
        dict: {
            'pdf_type': str,
            'has_text': bool,
            'has_images': bool,
            'structured': bool,
            'pages_analyzed': int
        }
    """
    doc = fitz.open(pdf_path)
    has_text, has_images, structured = False, False, False

    for page_idx, page in enumerate(doc):
        text = page.get_text("text")
        images = page.get_images(full=True)

        if text.strip():
            has_text = True
        if len(images) > 0:
            has_images = True

        # Optional layout structure check
        if check_layout and has_text:
            img = page.get_pixmap(dpi=150)
            img_np = np.frombuffer(img.samples, dtype=np.uint8).reshape(img.height, img.width, img.n)
            if img.n == 4:  # remove alpha
                img_np = img_np[:, :, :3]

            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120,
                                    minLineLength=50, maxLineGap=5)
            if lines is not None and len(lines) > 30:
                line_density = len(lines) / (img.width * img.height)
                if line_density > layout_threshold:
                    structured = True
                    break  # one page enough to confirm

    doc.close()
    # Determine type
    if has_text and not has_images:
        pdf_type = "Digital PDF"
    elif has_images and not has_text:
        pdf_type = "Scanned PDF"
    elif has_text and has_images:
        pdf_type = "Hybrid PDF"
    else:
        pdf_type = "Unknown / Corrupt PDF"

    # Structured layout override
    if structured and pdf_type == "Digital PDF":
        pdf_type = "Structured Layout PDF"

    return {
        "pdf_type": pdf_type,
        "has_text": has_text,
        "has_images": has_images,
        "structured": structured
    }

# if __name__ == "__main__":
#     pdf_path = "/home/ubuntu/Airline_identify/pdf_layout_parser/2510.02665v1.pdf" # Change this to your test file
#     result = detect_pdf_type(pdf_path)
#     print("\n=== PDF Type Detection Result ===")
#     for k, v in result.items():
#         print(f"{k}: {v}")




class HybridPDFParser:
    def __init__(self):
        self.detector = PDFTypeDetector()
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def parse_pdf(self, pdf_path: str, output_xml="parsed_output.xml"):
        pdf_type = self.detector.detect_pdf_type(pdf_path)
        print(f"[INFO] Detected PDF type: {pdf_type}")

        # Prepare XML structure
        root = etree.Element("Document")
        root.set("pdf_type", pdf_type)

        # Convert pages to images (for OCR / Hybrid)
        pages = convert_from_path(pdf_path, dpi=200)
        doc = fitz.open(pdf_path)

        for idx, page_img in enumerate(pages):
            page_node = etree.SubElement(root, "Page")
            page_node.set("number", str(idx + 1))

            # ============ Digital Text Extraction ============
            text_digital = ""
            if pdf_type in ["Digital PDF", "Hybrid PDF"]:
                text_digital = doc[idx].get_text("text") or ""
                digital_node = etree.SubElement(page_node, "DigitalText")
                digital_node.text = text_digital.strip()

            # ============ OCR Extraction ============
            if pdf_type in ["Scanned PDF", "Hybrid PDF"]:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    page_img.save(tmp.name, "PNG")
                    ocr_result = self.ocr.ocr(tmp.name)
                    os.unlink(tmp.name)

                text_ocr = " ".join([line[1][0] for block in ocr_result for line in block])
                ocr_node = etree.SubElement(page_node, "OCRText")
                ocr_node.text = text_ocr.strip()

            # ============ Merge Hybrid Content ============
            if pdf_type == "Hybrid PDF":
                merged_node = etree.SubElement(page_node, "MergedText")
                merged_text = (text_digital + "\n" + text_ocr).strip()
                merged_node.text = merged_text

        # ============ Save as XML ============
        xml_str = etree.tostring(root, pretty_print=True, encoding="utf-8").decode("utf-8")
        with open(output_xml, "w", encoding="utf-8") as f:
            f.write(xml_str)

        print(f"[✅] Parsing complete. XML saved as: {output_xml}")
        return output_xml

