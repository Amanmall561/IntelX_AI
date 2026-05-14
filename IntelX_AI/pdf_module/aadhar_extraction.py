import re
# from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
# Initialize OCR
# ocr = PaddleOCR(use_angle_cls=True, lang='en')  # supports multilingual too if needed

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.table_rec import TableRecPredictor

# === Load Predictors ===
foundation = FoundationPredictor()
recognition = RecognitionPredictor(foundation)
detection = DetectionPredictor()
layout = LayoutPredictor(foundation)
table_rec = TableRecPredictor()

def extract_aadhar_info(items):
    
    result = {"name": None, "dob": None, "gender": None, "id_number": None}
    
    
    # Common regex patterns
    dob_patterns = [
        r'(\d{2}[/-]\d{2}[/-]\d{4})',  # 01/01/2002 or 01-01-2002
        r'(\d{2}[/-]\d{2}[/-]\d{2})',  # 01/01/02
        r'(\d{4})'                     # 2002
    ]
    id_pattern = r'(\d{4}\s?\d{4}\s?\d{4})'  # Aadhaar-like number
    gender_keywords = {"male": "Male", "female": "Female", "other": "Other"}
    
    # Flatten list into a string for searching
    # all_text = " ".join(items)
    for all_text in items:
    
        # 1. Extract ID number
        match = re.search(id_pattern, all_text['text'])
        if match:
            result["id_number"] = match.group(1).replace(" ", "")
    
        # 2. Extract DOB
        for pat in dob_patterns:
            if all_text['text'].__contains__('DOB') or all_text['text'].__contains__('जन्म तिथि'):
                dob_text=all_text['text'].split(':')
                if len(dob_text)>=2:
                    match = re.search(pat, dob_text[1])
                    if match:
                        result["dob"] = match.group(1)
                        break
                else:
                    match = re.search(pat, dob_text[0])
                    if match:
                        result["dob"] = match.group(1)
                        break
        
    # 3. Extract Gender
    for word in items:
        lw = word['text'].strip().lower()
        for k, v in gender_keywords.items():
            if k in lw:
                result["gender"] = v
                break
        if result["gender"]:
            break
    
    # 4. Extract Name (heuristic → longest string with multiple words & not Govt or DOB/Gender/ID)
    possible_names = [
        w for w in items 
        if len(w['text'].split()) >= 2 
        and not re.search(id_pattern, w['text'])         # not Aadhaar number
        and not re.search(r'\d', w['text'])              # exclude if contains digits
    ]
    possible_names = [
        w['text'] for w in possible_names 
        if "gov" not in w['text'].lower() and "india" not in w['text'].lower()
    ]
    if possible_names:
        for possible_name in possible_names:
            if re.match(r'^[A-Za-z\s]+$', possible_name):
                # ✅ valid name
                # print(possible_name)
                result["name"] = possible_name

    print(result)
    return result

# # Example usage
# items = ['HIRE FRER', 'C', 'Government of India', ' ToS', 'Kartik Vinod Rathod', '/DOB01/01/2002', '/ Male', '6595 0962 2164', 'onen', '201T', 'et']
# print(extract_aadhar_info(items))

def extract_aadhaar_fields(image_path):
    # result = ocr.ocr(image_path)
    img = Image.open(image_path)
    rec_preds = recognition([img], det_predictor=detection)
    page_pred = rec_preds[0]

    # # Layout
    # layout_pred = layout([img])[0]

    # # Table recognition
    # table_preds = table_rec([img])[0]

    text_lines = []
    for tl in page_pred.text_lines:
        bbox = tl.bbox
        text = tl.text.strip()
        conf = tl.confidence
        text_lines.append({"text": text, "confidence": conf, "bbox": bbox})
    

    print(text_lines)
    result=extract_aadhar_info(text_lines)
    result['doc_type']='AADHAR'
    print(result)

    return result

    # for res in result:
    #     for line in res['rec_texts']:
    #         print(line)
    #         text_lines.append(line)

    # text = " ".join(text_lines)

    # fields = {
    #     "name": None,
    #     "dob": None,
    #     "father_name": None,
    #     "gender": None,
    #     "address": None,
    #     "aadhaar_number": None
    # }

    # # Aadhaar Number (12 digit)
    # aadhaar_match = re.search(r"\b\d{4}\s\d{4}\s\d{4}\b", text)
    # if aadhaar_match:
    #     fields["aadhaar_number"] = aadhaar_match.group()

    # # DOB
    # dob_match = re.search(r"(\d{2}[/-]\d{2}[/-]\d{4})", text)
    # if dob_match:
    #     fields["dob"] = dob_match.group()

    # # Gender
    # if re.search(r"Female", text, re.I):
    #     fields["gender"] = "Female"
    # elif re.search(r"Male", text, re.I):
    #     fields["gender"] = "Male"

    # # Name (heuristic: comes before DOB)
    # for line in text_lines:
    #     if "DOB" in line or "Date of Birth" in line:
    #         idx = text_lines.index(line)
    #         if idx > 0:
    #             fields["name"] = text_lines[idx - 1].strip()
    #         break

    # # Father name (not always present, but look for keywords)
    # for line in text_lines:
    #     if re.search(r"(S/O|D/O|W/O)", line, re.I):
    #         fields["father_name"] = line.strip()
    #         break

    # # Address (look for "Address" keyword or use heuristic)
    # address = []
    # for line in text_lines:
    #     if "Address" in line or re.search(r"House|Street|Road|Nagar|City|Village|District|State|Pincode", line, re.I):
    #         address.append(line)
    # if address:
    #     fields["address"] = " ".join(address)

    # return fields


# Run on your Aadhaar card image
# image_path = "/home/ubuntu/Airline_identify/new_data/train/images/60643f2e835245dfa9b0c9c988abb748_jpg.rf.c801fb10e8bc88ee0a91d0fb94e82c3b.jpg"
# extracted = extract_aadhaar_fields(image_path)



