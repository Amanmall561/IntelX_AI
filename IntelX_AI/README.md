# IntelX_AI

IntelX_AI is an advanced, multi-modal document and file parsing intelligence system. It handles automated ingestion, classification, text extraction, and parsing across a wide variety of file formats including PDFs, Word documents, Excel sheets, JSON, CSV files, and plain text. Furthermore, it supports scanned documents and image-based formats using state-of-the-art vision models (YOLOv12, Qwen-VL, Surya OCR, and OLM-OCR) to categorize visually and parse data cleanly.

## Key Features
- **Multi-Format Support:** Easily handles `.pdf` (digital, scanned, hybrid), `.docx`, `.xlsx`, `.csv`, `.json`, `.txt`, and images (`.jpg`, `.png`, etc.).
- **Smart Routing (`File_Loader_handler.py`):** Automatically detects the file type (based on extension and inner structure) and routes it to the specific processing pipeline.
- **Vision-based Classification (`document_classifier.py`):** Uses a fine-tuned YOLOv12 model to classify image documents into categories such as AADHAR, AIRTICKET, DL, PAN, Passport, and Other.
- **Advanced PDF Parsing:** 
  - Extracts raw text from native digital PDFs.
  - For scanned PDFs, leverages layout parsers, `olmocr`, and `surya-ocr` to pull visual structure.
- **LLM Integration:** Uses a Language Model (`mpt-7b-instruct` / `Qwen2-VL-7B-Instruct`) to extract structured information from chunked textual/visual data.
- **Custom Error Handling (`exceptions.py`):** Clear exceptions (`FileHandlerError`, `PDFParserError`, etc.) across the complete pipeline for robust application behavior.

## Repository Structure

```
IntelX_AI/
│
├── .env / env.template         # Environment variables configuration
├── config.py                   # Centralised configuration loader
├── exceptions.py               # Custom exceptions for all modules
├── File_Loader_handler.py      # Core module to detect and handle any file
├── moduler_call.py             # Main pipeline executor connecting parsing to LLM/vision models
├── test.py                     # Test script demonstrating visual-language inference (Qwen-VL & olmocr)
│
├── classifier/
│   └── document_classifier.py  # YOLOv12 model script to classify document images
│
├── models/                     # Directory to store fine-tuned model weights
│   └── yolo_classifire/weights/best.pt 
│
├── pdf_module/                 # Document parsing logic per format
│   ├── aadhar_extraction.py
│   ├── csv_parser.py
│   ├── detect_pdf.py
│   ├── digital_pdf_parser.py
│   ├── doc_parser.py
│   ├── json_parser.py
│   ├── Scanned_pdf_module.py
│   ├── surya_pdf_parser.py
│   ├── Ticket_parser_LLOCR.py
│   ├── DL_parser.py
│   ├── passport_Layout_parser.py
│   ├── other_LLOCR.py
│   └── xlsx_parser.py
│
└── temp_img/                   # Temporary storage for image conversions
```

## Setup Instructions

### 1. Prerequisites
- Python 3.9+ (Recommended)
- System capable of running PyTorch (CUDA supported GPU is highly recommended for faster inference).

### 2. Install Dependencies
While there is no explicit `requirements.txt`, you will need to install the following Python packages based on the module imports:

```bash
pip install python-dotenv pypdf torch torchvision torchaudio transformers ultralytics Pillow 
pip install surya-ocr olmocr
```

*(Note: Depending on your specific layout parsing libraries used inside `pdf_module`, additional dependencies such as `pandas`, `openpyxl`, or `python-docx` might be required).*

### 3. Environment Variables
Copy the `env.template` file to `.env` in the root directory:

```bash
cp env.template .env
```

Update the values in `.env`:
```env
DEFAULT_INPUT_FILE="/path/to/your/default_file.pdf"
YOLO_MODEL_PATH="/path/to/your/models/yolo_classifire/weights/best.pt"
LLM_MODEL_NAME="mosaicml/mpt-7b-instruct"
TICKET_MODEL_ID="allenai/olmOCR-7B-0725"
TEMP_IMAGE_DIR="temp_img"
```

### 4. Download Model Weights
Ensure you have the fine-tuned YOLO weights available at the `YOLO_MODEL_PATH` specified in your `.env`.

## How to Run

### Using the Main Pipeline (`moduler_call.py`)
You can process a file by calling the `main()` function in `moduler_call.py`. If no file is provided, it falls back to `DEFAULT_INPUT_FILE` defined in your `.env`.

```bash
python moduler_call.py
```
To run it programmatically in your own script:

```python
from moduler_call import main

try:
    result = main("path/to/your/document.pdf")
    print(result)
except Exception as e:
    print(f"Error processing document: {e}")
```

### Running the Image Classifier independently
If you only need to classify an image using YOLO:

```python
from classifier.document_classifier import yolo_classifier_model
from config import YOLO_MODEL_PATH

results = yolo_classifier_model(
    model_path=YOLO_MODEL_PATH,
    source="path/to/your/image.jpg",
    conf_threshold=0.55
)
print(results)
```

## Architecture Flow (`moduler_call.py`)
1. **Input Stage:** The system accepts a file path.
2. **Detection:** `File_Loader_handler.py` infers the file type.
3. **Routing:**
   - **PDFs:** Checks if Digital or Scanned/Hybrid. Processes with basic extraction or `olmocr/surya` based on page count.
   - **Text/Doc/CSV/XLSX/JSON:** Uses dedicated parsers to extract raw textual data.
   - **Images:** First routes through YOLO (`yolo_classifier_model`). Based on the predicted class (e.g., AADHAR, PAN, Passport), it utilizes a specialized OCR parser to pull strict JSON metrics.
4. **LLM Extraction:** For text-heavy chunks, the extracted text is streamed through the designated LLM (`extract_data_pdf`) to format the output efficiently into structured dictionaries.
5. **Output:** Returns parsed structured Python dictionaries/lists.
