import torch
from ultralytics import YOLO

model = None

def get_model():
    global model

    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = YOLO("app/api/yolo/yolo26n.pt").to(device)

    return model

def detect(image):

    yolo_model = get_model()

    results = yolo_model(image)

    detections = []

    for result in results:

        for box in result.boxes:

            detections.append({
                "class_name": int(box.cls[0]),
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist(),
            })

    return detections