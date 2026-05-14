import torch 
from transformers import AutoModelForImageTextToText, AutoProcessor 
import json
from pdf_module.repair_json import fix_broken_json
def other_parser_model(image_path: str) -> dict:

    model_id = "allenai/olmOCR-7B-0725" 
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.float16 ).to("cuda").eval()

    # You are given an image of a flight booking ticket along with its extracted raw text and position information (the origin [0,0] is the lower-left corner of the image).

    # Your task is to return a clean JSON object with the following fields only:

    # "passenger_name"

    # "departure_from"

    # "departure_to"

    # "departure_date"

    # "departure_time"

    # "arrival_time"

    # "flight_details" (flight number, airline, class, seat if available)

    # "travel_loc" (booking source/platform, e.g., Easemytrip, Indigo, Goibibo, etc.)

    PROMPT = """
    You are given an image of an unknown document along with its extracted raw text and positional data (origin [0,0] is the lower-left corner of the image).

    Your task is to analyze the document content and return a clean JSON object containing the entities found in the document.
    The document type is unknown — so infer whatever entities are present without assuming the domain.

    Extract only real entities visible in the document, using these buckets:

    1. "people"

    List of person-related information. Each entry may include:

    "name"

    "date_of_birth"

    "gender"

    "father_name" / "mother_name" / "spouse_name"

    "address"

    "contact_number"

    "nationality"

    "signature_presence" (true/false)

    2. "locations"

    List of any place-related entities (if any available):

    "address"

    "city"

    "state"

    "country"

    "pin_code"

    "office/location_name"

    "place_of_issue"

    3. "identifiers"

    List all identity numbers or codes found:

    govt IDs (Aadhaar, PAN, Passport No., DL No., Voter ID, etc.)

    "ticket_numbers", "booking_IDs"

    "invoice_numbers"

    "membership_numbers"

    "policy_numbers"

    "account_numbers"

    "serial_numbers"

    any alphanumeric unique identifiers

    4. "dates"

    Any relevant dates:

    "issue_date"

    "expiry_date"

    "validity_period"

    "registration_date"

    "event_date"

    "transaction_date"

    5. "organization_details"

    Details of any organization:

    "organization_name"

    "department"

    "issuer/authority"

    "logo_presence" (true/false)

    6. "document_type" (inferred)

    A single field describing the most likely document category
    (e.g., “Aadhaar Card”, “Invoice”, “Finencial Report”, “Boarding Pass”, “Certificate”,“Unknown official document”).

    7. "Vehicle_details"
    Details of vehicle:

    "vehicle_number"

    "type of vehicle" (e.g., car, moterbike, bus etc)

    "vehicle color" (if avialable)

    "other" (if avialable)

    "

    8. **relationships**
    If any relationship between people is explicitly mentioned:
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
        "date": "",
        "description": "",
        "transaction_type": "",   // credit, debit, transfer, deposit, withdrawal
        "amount": "",
        "balance": "",
        "reference_number": "",
        "mode": ""               // UPI, IMPS, NEFT, ATM, cash, cheque, etc.

    Extract ONLY what is explicitly visible in this chunk.


    Rules

    If a field has multiple values, return them in a list.

    If something is not present, return an empty list or empty string accordingly.

    If any kind of human relationship (e.g., brother, friend, employee etc.) found in the doc mention them within the json by adding keys like "relationship_between", "retlationship_type" 

    No hallucinations — only return what exists in the document.

    Use both the raw extracted text and visual cues from the image.

    If you want, I can also build:
    🔥 a more strict schema
    🔥 a confidence-scored version
    🔥 a layout-aware version (bounding boxes linked to entities)
    🔥 or a NER-tagging format for training your model

    Do not output anything except the final JSON object.

    RAW_TICKET_TEXT_START
    { base_text }
    RAW_TICKET_TEXT_END
    """

    # PROMPT="""

    # Below is the image of one page of a PDF document , as well as some raw textual content that was previously extracted for it that includes position information for each image and block of text ( The origin [0 x0 ] of the coordinates is in the lower left corner of the image ). Just return the json text representation of this document as if you were reading it naturally . Turn equations into a LaTeX representation , and tables into markdown format . Remove the headers and footers , but keep references and footnotes . Read any natural handwriting . This is likely one page out of several in the document , so be sure to preserve any sentences that come from the previous page , or continue onto the next page , exactly as they are . If there is no text at all that you think you should read , you can output null . Do not hallucinate . RAW_TEXT_START { base_text } RAW_TEXT_END"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=1000)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    output_text=output_text.split('```json')[-1]
    output_text=output_text.replace('\n','')
    output_text=output_text.split('```')[0]
    output_text=output_text.replace('```','')
    # print(output_text,'=======')
    result = fix_broken_json(output_text)
    return result