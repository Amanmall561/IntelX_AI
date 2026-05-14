import json
import sys
from pathlib import Path
from File_Loader_handler import file_handler_module
from config import DEFAULT_INPUT_FILE, YOLO_MODEL_PATH
from pdf_module.detect_pdf import detect_pdf_type
from pdf_module.digital_pdf_parser import AdvancedPDFParser, Config
from classifier.document_classifier import yolo_classifier_model
from pdf_module.aadhar_extraction import extract_aadhaar_fields
from pdf_module.passport_Layout_parser import passport_layout_parser_model
from pdf_module.other_LLOCR import other_parser_model
from pdf_module.DL_parser import licence_parser_model, PAN_parser_model
from pdf_module.Ticket_parser_LLOCR import ticket_parser_model
from pdf_module.Scanned_pdf_module import extract_data_pdf, parse_pdf_minpage, extract_data_pdf_with_olm
from pdf_module.surya_pdf_parser import process_single_fast
from pdf_module.csv_parser import parse_csv_file, CSVParserError
from pdf_module.json_parser import parse_json_file, JSONParserError
from pdf_module.xlsx_parser import parse_xlsx_file, XLSXParserError
from pdf_module.doc_parser import parse_doc_file_generic, DOCParserError
from exceptions import (
    IntelXAIError, FileHandlerError, ParserError, ModelError, 
    LLMError, DependencyError, PDFParserError
)
from pypdf import PdfReader

def count_pdf_pages(pdf_path: str) -> int:
    """Counts the number of pages in a given PDF file.
    
    Raises:
        PDFParserError: If PDF reading fails
        FileHandlerError: If file operations fail
    """
    if not pdf_path:
        raise PDFParserError("PDF file path is required", pdf_path)
    
    path = Path(pdf_path)
    if not path.exists():
        raise FileHandlerError(f"PDF file not found: {pdf_path}", pdf_path)
    
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        return num_pages
    except FileNotFoundError as e:
        raise FileHandlerError(f"PDF file not found: {pdf_path}", pdf_path) from e
    except PermissionError as e:
        raise FileHandlerError(f"Permission denied reading PDF: {pdf_path}", pdf_path) from e
    except Exception as e:
        raise PDFParserError(f"Error counting PDF pages: {str(e)}", pdf_path) from e

def split_text(text, chunk_size=1800):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

# def split_text(text, chunk_size=2000):
#     chunks = []
#     start = 0
#     text_len = len(text)

#     while start < text_len:
#         end = min(start + chunk_size, text_len)
#         chunk = text[start:end]

#         # Prefer splitting at "},"
#         split_pos = chunk.rfind("},")
#         if split_pos != -1:
#             end = start + split_pos + 2
#         else:
#             # fallback: split at last "."
#             split_pos = chunk.rfind(".")
#             if split_pos != -1:
#                 end = start + split_pos + 1

#         # If no split point found, force hard cut
#         if end <= start:
#             end = min(start + chunk_size, text_len)

#         chunks.append(text[start:end].strip())
#         start = end

#     return chunks


def main(file_path: str = None) -> dict:
    """
    Main function to process files based on their detected type.
    
    Args:
        file_path: Path to the file to process. If None, uses default test file.
        
    Returns:
        Dictionary with extracted data or error information
        
    Raises:
        IntelXAIError: Base exception for all IntelX_AI errors
    """
    if file_path is None:
        file_path = DEFAULT_INPUT_FILE
    
    if not file_path:
        raise FileHandlerError("File path is required")
    
    try:
        result = file_handler_module(file_path)
    except Exception as e:
        raise FileHandlerError(f"Error detecting file type: {str(e)}", file_path) from e
    
    if 'error' in result:
        raise FileHandlerError(result.get('message', 'Unknown file handler error'), file_path)
    
    detected_type = result.get('detected_type')
    if not detected_type:
        raise FileHandlerError("Could not detect file type", file_path)

    try:
        if detected_type == 'pdf':
            pdf_layout_detect = detect_pdf_type(result['file'])
            
            if pdf_layout_detect.get('pdf_type') == 'Digital PDF':
                """call digital pdf parser and return data into json"""
                try:
                    parser = AdvancedPDFParser(Config())
                    paras = parser.parse(result['file'])
                    chunks = list(split_text(paras['All_text']))

                    print(f"Parsed {len(paras['Json_data'])} paragraph-like objects. Output Extracted.")

                    """Call a LLM model to extract relevent info within chunks."""
                    outputs = []
                    for c in chunks:
                        try:
                            out = extract_data_pdf(c)
                            outputs.append(out)
                        except LLMError as e:
                            raise LLMError(f"LLM extraction failed: {str(e)}", result['file']) from e
                        except Exception as e:
                            raise ModelError(f"Error in LLM extraction: {str(e)}", result['file']) from e
                    
                    return outputs
                except Exception as e:
                    raise PDFParserError(f"Digital PDF parsing failed: {str(e)}", result['file']) from e

            elif pdf_layout_detect.get('pdf_type') in ('Scanned PDF', 'Hybrid PDF'):
                try:
                    page_count = count_pdf_pages(result['file'])
                    print("total Pages:", page_count)
                    
                    if page_count > 0 and page_count <= 2:
                        try:
                            extracted_json = parse_pdf_minpage(result['file'])
                            return extracted_json
                        except Exception as e:
                            raise PDFParserError(f"Error parsing PDF with minpage: {str(e)}", result['file']) from e
                    else:
                        extracted_json=extract_data_pdf_with_olm(result['file'])
                        return extracted_json
                        # try:
                        #     pdf_ext_data = process_single_fast(
                        #         pdf_path=result['file'],
                        #         dpi=120,
                        #         use_gpu=True,
                        #         skip_tables_text=False
                        #     )
                            
                        #     if not pdf_ext_data.get('success'):
                        #         raise PDFParserError(
                        #             pdf_ext_data.get('error_message', 'Unknown error'),
                        #             result['file']
                        #         )
                            
                        #     chunks = list(split_text(str(pdf_ext_data['simple_json'])))
                        #     outputs = []
                        #     for c in chunks:
                        #         try:
                        #             out = extract_data_pdf(c)
                        #             outputs.append(out)
                        #         except LLMError as e:
                        #             raise LLMError(f"LLM extraction failed: {str(e)}", result['file']) from e
                        #         except Exception as e:
                        #             raise ModelError(f"Error in LLM extraction: {str(e)}", result['file']) from e
                            
                        #     return outputs
                        # except Exception as e:
                        #     raise PDFParserError(f"Error processing scanned PDF: {str(e)}", result['file']) from e
                except PDFParserError:
                    raise
                except Exception as e:
                    raise PDFParserError(f"Error processing PDF: {str(e)}", result['file']) from e
            else:
                return pdf_layout_detect

        elif detected_type == 'doc':
            """call doc classification model and surya OCR parser for extract data into json"""
            print('handle with doc parser module')
            try:
                content = parse_doc_file_generic(result['file'])
                chunks = list(split_text(content))
                outputs = []
                for c in chunks:
                    try:
                        out = extract_data_pdf(c)
                        outputs.append(out)
                    except LLMError as e:
                        raise LLMError(f"LLM extraction failed: {str(e)}", result['file']) from e
                    except Exception as e:
                        raise ModelError(f"Error in LLM extraction: {str(e)}", result['file']) from e
                return outputs
            except (DOCParserError, DependencyError, FileHandlerError):
                raise
            except Exception as e:
                raise DOCParserError(f"Unexpected error processing DOC file: {str(e)}", result['file']) from e

        elif detected_type == 'text':
            print('handle with text parser module')
            try:
                content = None
                with open(result['file'], 'r', encoding='utf-8') as file:
                    content = file.read()
                
                if content:
                    chunks = list(split_text(content))
                    outputs = []
                    for c in chunks:
                        try:
                            out = extract_data_pdf(c)
                            outputs.append(out)
                        except LLMError as e:
                            raise LLMError(f"LLM extraction failed: {str(e)}", result['file']) from e
                        except Exception as e:
                            raise ModelError(f"Error in LLM extraction: {str(e)}", result['file']) from e
                    return outputs
                else:
                    raise FileHandlerError("Text file is empty", result['file'])
            except FileNotFoundError as e:
                raise FileHandlerError(f"Text file not found: {result['file']}", result['file']) from e
            except PermissionError as e:
                raise FileHandlerError(f"Permission denied reading text file: {result['file']}", result['file']) from e
            except UnicodeDecodeError as e:
                raise FileHandlerError(f"Encoding error reading text file: {result['file']}", result['file']) from e
            except Exception as e:
                raise FileHandlerError(f"Error processing text file: {str(e)}", result['file']) from e

        elif detected_type == 'csv':
            print('handle with CSV parser module')
            try:
                content = parse_csv_file(result['file'])
                chunks = list(split_text(content))
                outputs = []
                for c in chunks:
                    try:
                        out = extract_data_pdf(c)
                        outputs.append(out)
                    except LLMError as e:
                        raise LLMError(f"LLM extraction failed: {str(e)}", result['file']) from e
                    except Exception as e:
                        raise ModelError(f"Error in LLM extraction: {str(e)}", result['file']) from e
                return outputs
            except (CSVParserError, FileHandlerError, DependencyError):
                raise
            except Exception as e:
                raise CSVParserError(f"Unexpected error processing CSV file: {str(e)}", result['file']) from e

        elif detected_type == 'json':
            print('handle with JSON parser module')
            try:
                content = parse_json_file(result['file'])
                chunks = list(split_text(content))
                outputs = []
                for c in chunks:
                    try:
                        out = extract_data_pdf(c)
                        outputs.append(out)
                    except LLMError as e:
                        raise LLMError(f"LLM extraction failed: {str(e)}", result['file']) from e
                    except Exception as e:
                        raise ModelError(f"Error in LLM extraction: {str(e)}", result['file']) from e
                return outputs
            except (JSONParserError, FileHandlerError):
                raise
            except Exception as e:
                raise JSONParserError(f"Unexpected error processing JSON file: {str(e)}", result['file']) from e

        elif detected_type == 'xlsx':
            print('handle with SHEET parser module')
            try:
                content = parse_xlsx_file(result['file'])
                chunks = list(split_text(content))
                outputs = []
                for c in chunks:
                    try:
                        out = extract_data_pdf(c)
                        outputs.append(out)
                    except LLMError as e:
                        raise LLMError(f"LLM extraction failed: {str(e)}", result['file']) from e
                    except Exception as e:
                        raise ModelError(f"Error in LLM extraction: {str(e)}", result['file']) from e
                return outputs
            except (XLSXParserError, FileHandlerError, DependencyError):
                raise
            except Exception as e:
                raise XLSXParserError(f"Unexpected error processing XLSX file: {str(e)}", result['file']) from e

        elif detected_type == 'image':
            try:
                classifier_model_result = yolo_classifier_model(
                    model_path=YOLO_MODEL_PATH,
                    source=result['file'],
                    conf_threshold=0.55,
                    iou_threshold=0.45,
                    save_results=False,
                    show_results=False
                )
                #model classes  ['AADHAR', 'AIRTICKET', 'DL', 'Other', 'PAN', 'Passport']
                
                if len(classifier_model_result) > 0:
                    for item in classifier_model_result:
                        class_name = item.get('class_name', 'Other')
                        try:
                            if class_name == 'AIRTICKET':
                                print('handle with AIRTICKET parser module')
                                extracted_json = ticket_parser_model(result['file'])
                                return extracted_json
                                
                            elif class_name == 'DL':
                                print('handle with DL parser module')
                                extracted_json = licence_parser_model(result['file'])
                                return extracted_json
                                
                            elif class_name == 'AADHAR':
                                print('handle with AADHAR parser module')
                                extracted_json = extract_aadhaar_fields(result['file'])
                                return extracted_json

                            elif class_name == 'PAN':
                                print('handle with PAN parser module')
                                extracted_json = PAN_parser_model(result['file'])
                                return extracted_json

                            elif class_name == 'Passport':
                                print('handle with Passport parser module')
                                extracted_json = passport_layout_parser_model(result['file'])
                                return extracted_json

                            elif class_name == 'Other':
                                print('handle with Other parser module')
                                extracted_json = other_parser_model(result['file'])
                                return extracted_json

                            else:
                                print('handle with OCR parser module')
                                extracted_json = other_parser_model(result['file'])
                                return extracted_json
                        except Exception as e:
                            raise ModelError(f"Error processing {class_name} image: {str(e)}", result['file']) from e
                else:
                    print('handle with OCR parser module')
                    try:
                        extracted_json = other_parser_model(result['file'])
                        return extracted_json
                    except Exception as e:
                        raise ModelError(f"Error in OCR processing: {str(e)}", result['file']) from e
            except Exception as e:
                raise ModelError(f"Error processing image: {str(e)}", result['file']) from e
        
        else:
            raise FileHandlerError(f"Unsupported file type: {detected_type}", result['file'])
    
    except IntelXAIError:
        raise
    except Exception as e:
        raise IntelXAIError(f"Unexpected error in main processing: {str(e)}", file_path) from e
           
          
if __name__ == "__main__":
    try:
        ress = main()
        print(json.dumps(ress, indent=2, default=str))
    except IntelXAIError as e:
        print(f"Error: {e.message}", file=sys.stderr)
        if e.file_path:
            print(f"File: {e.file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)
