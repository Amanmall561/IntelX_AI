import torch 
from transformers import AutoModelForImageTextToText, AutoProcessor 
import json
from config import TICKET_MODEL_ID

def ticket_parser_model(image_path: str) -> dict:
    """
    This function parses a flight booking ticket image using the LLOCR model.
    Args:
        image_path: The path to the image of the flight booking ticket.
    Returns:
        A dictionary containing the parsed data from the image.
    """

    model_id = TICKET_MODEL_ID 
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
    You are given an image of a travel ticket (it may be a flight, train, bus, ship, or any other transport mode) along with its extracted raw text and positional data (origin [0,0] is the lower-left corner of the image).

    Your task is to return a clean JSON object containing only the following fields:

    "passenger_name"

    "departure_from"

    "departure_to"

    "departure_date"

    "departure_time"

    "arrival_time"

    "transport_details"
    (transport mode, operator/carrier name, service/vehicle/train/flight number, class/type, coach/seat/berth/cabin if available)

    "booking_source"
    (platform or provider such as IRCTC, Redbus, Indigo, Goibibo, MakeMyTrip, etc.)

    Rules:

    Ignore headers, footers, ads, and unrelated text.

    If a field is missing, set it as null.

    Use the text exactly as it appears (no hallucinations).

    Normalize dates/times into YYYY-MM-DD and HH:MM formats where possible.

    If multiple passengers are listed, return an array under "passenger_name".

    Extract airline or booking platform branding/logos from text if available (travel_loc).

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
    output_text=output_text.replace('\n','')
    output_text=output_text.replace('```json','')
    output_text=output_text.replace('```','')
    # print(output_text)
    result = json.loads(output_text)
    return result