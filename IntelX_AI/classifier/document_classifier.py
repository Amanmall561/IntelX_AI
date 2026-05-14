import os
import torch
from ultralytics import YOLO

def yolo_classifier_model(
    model_path: str,
    source: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    save_results: bool = True,
    show_results: bool = True, 
):
    """
    Test a YOLOv12 fine-tuned model on images/videos.

    Args:
        model_path (str): Path to the fine-tuned YOLOv12 weights (.pt).
        source (str): Path to image, video, or folder to run inference on.
        conf_threshold (float): Minimum confidence threshold for detections.
        iou_threshold (float): IOU threshold for NMS filtering.
        save_results (bool): Whether to save prediction results.
        show_results (bool): Whether to display images/videos with predictions.
    """

    # Check device availability (GPU > CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Using device: {device}")

    # Load the YOLOv12 model
    print(f"🔄 Loading model: {model_path}")
    model = YOLO(model_path)

    # Run inference
    print(f"🔍 Running inference on: {source}")
    results = model.predict(
        source=source,
        conf=conf_threshold,
        iou=iou_threshold,
        save=save_results,
        show=show_results,
        device=device
    )

    # Print predictions summary
    print("\n📌 Detection Results:")
    result_dict = []
    for idx, result in enumerate(results):
        print(f"\nImage/Video {idx + 1}: {result.path}")
        for box in result.boxes:
            cls_id = int(box.cls)
            label = model.names[cls_id]
            conf = float(box.conf)
            xyxy = box.xyxy.cpu().numpy().tolist()[0]
            print(f" - {label}: {conf:.2f} | BBox: {xyxy}")
            result_dict.append({'class_name': label, 'confidence': conf, 'bbox': xyxy})
    return result_dict




# if __name__ == "__main__":
#     # Example usage
#     yolo_classifier_model(
#         model_path="/home/ubuntu/Airline_identify/IntelX_AI/models/yolo_classifire/weights/best.pt",  # Your fine-tuned weights path
#         source="/home/ubuntu/Airline_identify/Donut_module/images/ticket_1.jpg",                       # Can be a folder, image, video, or webcam index
#         conf_threshold=0.3,
#         iou_threshold=0.45,
#         save_results=True,
#         show_results=True
#     )
