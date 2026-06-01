from datetime import datetime
import os
from typing import Optional

import torch
from ultralytics import YOLO
import cv2
from app.api.core.config import Config

model = None

SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)


def get_model():
    global model

    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = YOLO(Config.YOLO_MODEL, verbose=False).to(device)

    return model

def get_yolo_classes():
    yolo_model = get_model()
    return yolo_model.names

def save_image(result):
    """
    Save image with bounding boxes
    """
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
    path = os.path.join(SAVE_DIR, filename)

    # result.plot() devolve a imagem anotada (numpy array)
    annotated_img = result.plot()

    success = cv2.imwrite(path, annotated_img)

    if not success:
        raise RuntimeError(f"Falha ao guardar imagem em {path}")


def detect(image, chosen_class_ids: Optional[list[int]] = None):

    yolo_model = get_model()

    if not chosen_class_ids:
        chosen_class_ids = None

    results = yolo_model(image, classes=chosen_class_ids)

    detections = []

    for result in results:

        for box in result.boxes:
            conf = float(box.conf[0])

            if conf <= Config.CONF_THRESHOLD:
                continue

            class_id = int(box.cls[0])

            detections.append({
                "class_id": class_id,
                "class_name": yolo_model.names[class_id],
                "confidence": conf,
                "bbox": box.xyxy[0].tolist(),
            })

    save_image(results[0])

    return detections

