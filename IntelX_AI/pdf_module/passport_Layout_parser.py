
import torch 
from transformers import AutoModelForImageTextToText, AutoProcessor 
import json


def passport_layout_parser_model(image_path: str) -> dict:

    model_id = "allenai/olmOCR-7B-0825" # allenai/olmOCR-7B-0825
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.float16 ).to("cuda").eval()

    PROMPT = """
    You are given an image of a Passport along with its extracted raw text and position information (the origin [0,0] is the lower-left corner of the image).

    Your task is to return a clean single JSON object with exactly the following fields:

    "person_name"

    "Nationality"

    "Date_of_birth"

    "Passport_number"

    "valid_date"(if available, otherwise null)

    "address" (if available, otherwise null)

    "other_info" (if available, otherwise null)


    Rules:

    Text filtering: Ignore headers, footers, ads, watermarks, and any unrelated text.

    Names: If multiple names are found, return them as an array under "person_name".

    Dates: Normalize all dates into the YYYY-MM-DD format. If time appears, normalize to HH:MM.

    Case sensitivity: Use the extracted text exactly as it appears (don’t generate or guess new words).

    Missing fields: For any field not present in the raw text, set its value as null.

    Other info: Put any relevant information (like Passport Number, Place of Birth, etc.) that doesn’t belong to the above fields into "other_info".

    Output format: Return only the JSON object, with no explanations, no extra text, and no markdown formatting.


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
                    "image": "/home/ubuntu/Airline_identify/new_data/valid/images/passUsa2_png_jpg.rf.a9361fce1f2087be1b35344af915418b.jpg",
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
    output_text=output_text.replace('\n','')
    output_text=output_text.replace('```json','')
    output_text=output_text.replace('```','')

    result = json.loads(output_text)
    return result
