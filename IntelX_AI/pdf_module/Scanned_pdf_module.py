import fitz
import os
import json
from File_Loader_handler import file_handler_module
from pdf_module.detect_pdf import detect_pdf_type
from pdf_module.digital_pdf_parser import AdvancedPDFParser, Config
from classifier.document_classifier import yolo_classifier_model
from pdf_module.aadhar_extraction import extract_aadhaar_fields
from pdf_module.passport_Layout_parser import passport_layout_parser_model
from pdf_module.other_LLOCR import other_parser_model
from pdf_module.DL_parser import licence_parser_model, PAN_parser_model
from pdf_module.Ticket_parser_LLOCR import ticket_parser_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from config import YOLO_MODEL_PATH, LLM_MODEL_NAME, TEMP_IMAGE_DIR

dpi = 300
zoom = dpi / 72
magnify = fitz.Matrix(zoom, zoom)
temp_dir = TEMP_IMAGE_DIR or "temp_img"
os.makedirs(temp_dir, exist_ok=True)


def parse_pdf_minpage(pdf_path):
    count = 0
    doc = fitz.open(pdf_path)
    final_json=[]
    for page in doc:
        count+=1
        pix = page.get_pixmap(matrix=magnify)
        page_image=f"{temp_dir}/TC_page_{count}.png"
        pix.save(page_image)
        classifier_model_result = yolo_classifier_model(
            model_path=YOLO_MODEL_PATH,
            source=page_image, # Can be a folder, image, video, or webcam index
            conf_threshold=0.55,
            iou_threshold=0.45,
            save_results=False,
            show_results=False
        )
        #model classes  ['AADHAR', 'AIRTICKET', 'DL', 'Other', 'PAN', 'Passport']nv
        if len(classifier_model_result)>0:
            for item in classifier_model_result:
                if item['class_name']=='AIRTICKET':
                    print('handle with AIRTICKET parser module')
                    extracted_json = ticket_parser_model(page_image)
                    # return extracted_json
                    
                elif item['class_name']=='DL':
                    print('handle with DL parser module')
                    extracted_json = licence_parser_model(page_image)
                    # return extracted_json
                    
                elif item['class_name']=='AADHAR':
                    print('handle with AADHAR parser module')
                    extracted_json = extract_aadhaar_fields(page_image)
                    # return extracted_json

                elif item['class_name']=='PAN':
                    print('handle with PAN parser module')
                    extracted_json = PAN_parser_model(page_image)
                    # return extracted_json


                elif item['class_name']=='Passport':
                    print('handle with Passport parser module')
                    extracted_json = passport_layout_parser_model(page_image)
                    # return extracted_json

                elif item['class_name']=='Other':
                    print('handle with Other parser module')
                    extracted_json = other_parser_model(page_image)
                    # return extracted_json

                else:
                    print('handle with OCR parser module')
                    extracted_json = other_parser_model(page_image)
                    # return extracted_json
                    
        else:
            print('handle with OCR parser module')
            extracted_json = other_parser_model(page_image)
            # return extracted_json
        final_json.append({"page_nu":count,"extracted_data":extracted_json})
        return final_json

def extract_data_pdf_with_olm(pdf_path):
    count = 0
    doc = fitz.open(pdf_path)
    final_json=[]
    for page in doc:
        count+=1
        pix = page.get_pixmap(matrix=magnify)
        page_image=f"{temp_dir}/TC_page_{count}.png"
        pix.save(page_image)
        extracted_json = other_parser_model(page_image)
        final_json.append({"page_nu":count,"extracted_data":extracted_json})
        print(extracted_json,"\nPage",count)   
    return final_json


   
def extract_data_pdf(text_chunk):
    # pip install transformers accelerate
    instruction="""
    You are an information extraction model. Analyze ONLY the provided text chunk and extract real entities that exist inside this chunk. Do NOT guess, assume, infer missing data, or hallucinate anything.

    Process the text strictly as-is. If the chunk does not contain information for a field, return an empty list or empty string for that field.

    Return a clean JSON object with these top-level keys:

    {
    "people": [],
    "locations": [],
    "identifiers": [],
    "dates": [],
    "organization_details": [],
    "document_type": "",
    "vehicle_details": [],
    "relationships": []
    }

    Extraction Rules:

    1. **people**
    For every person-related entity found in this chunk, return an object containing any of:
    "name", "date_of_birth", "gender",
    "father_name", "mother_name", "spouse_name",
    "address", "contact_number", "nationality",
    "signature_presence" (true/false)

    2. **locations**
    Extract any place-related information:
    "address", "city", "state", "country",
    "pin_code", "office_name", "location_name",
    "place_of_issue"

    3. **identifiers**
    Extract all identity numbers or alphanumeric codes:
    Aadhaar, PAN, Passport No., DL No., Voter ID,
    invoice_numbers, ticket_numbers, booking_IDs,
    membership_numbers, policy_numbers,
    serial_numbers, account_numbers,
    or any unique identifier.

    4. **dates**
    Extract any date-like information:
    "issue_date", "expiry_date", "validity_period",
    "registration_date", "event_date",
    "transaction_date" or any date visible in the chunk.

    5. **organization_details**
    For any organization:
    "organization_name", "department",
    "issuer", "authority",
    "logo_presence" (true/false)

    6. **document_type**
    Infer ONLY from this chunk what document type it resembles.
    If unclear, set "Unknown".

    7. **vehicle_details**
    Extract vehicle-related info:
    "vehicle_number", "vehicle_type",
    "vehicle_color", "other"

    8. **relationships**
    If any relationship between humans is explicitly mentioned
    (e.g., “brother of”, “employee of”, “friend of”):
    Return:
    {
        "relationship_between": ["A", "B"],
        "relationship_type": "brother/friend/employee/etc"
    }
    9. **bank_account_details**
    If the chunk resembles a bank document, extract:
    "account_holder_name"
    "account_number"
    "bank_name"
    "branch"
    "ifsc_code"
    "customer_id"
    "statement_period"

    If any field is missing, leave it empty.

    10. **transactions**
    If the chunk contains bank transactions, extract each entry as an object:
    {
        "date": "",
        "description": "",
        "transaction_type": "",   // credit, debit, transfer, deposit, withdrawal
        "amount": "",
        "balance": "",
        "reference_number": "",
        "mode": ""               // UPI, IMPS, NEFT, ATM, cash, cheque, etc.
    }

       Extract ONLY what is explicitly visible in this chunk.


    Hard Rules:
    - Use ONLY the text chunk provided.
    - No merging with previous chunks.
    - No assumptions about missing data.
    - No made-up values.
    - Output only the final JSON object. No explanations.
    - Do not provide any RAWTEXT in response. 
    _ Only provide json response, If no inforamation just simple return None.

    """


    model_name = LLM_MODEL_NAME  # instruction-tuned version (configurable)
    try:

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = transformers.AutoConfig.from_pretrained(model_name)
        config.max_position_embeddings = 128000   # enable full 128k context

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype="auto",
            device_map="auto"
        )

        # prompt = f"""
        # Follow the instruction strictly.

        # INSTRUCTION:
        # {instruction}

        # RAW_TEXT_START:
        # {text_chunk}
        # RAW_TEXT_END.
        # """
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": text_chunk}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            response_format={"type": "json_object"}   # << Forces JSON
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.1,
            top_p=0.95,
            do_sample=False
        )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)


        # response = model.chat(
        #     tokenizer,
        #     messages,
        #     response_format={"type": "json_object"},
        #     max_new_tokens=512,
        #     temperature=0.1,
        # )


        # generator = transformers.pipeline(
        #     "text-generation",
        #     model=model,
        #     tokenizer=tokenizer,
        #     device_map="auto",
        #     torch_dtype="auto"
        # )

        # output = generator(
        #     prompt,
        #     max_new_tokens=300,
        #     temperature=0.1,
        #     top_p=0.95,
        #     do_sample=False
        # )

        output_text = decoded
        print('========')
        print(output_text,'========-----')
        output_text=output_text.replace('\n','')
        output_text=output_text.replace('```json','')
        output_text=output_text.replace('```','')
        # print(output_text)
        result = json.loads(output_text)

        return result
    except Exception as e:
        print('Error in model ',e)



        