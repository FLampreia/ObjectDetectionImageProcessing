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

def crop_image(image, area):
    x1, y1, x2, y2 = area
    return image[int(y1):int(y2), int(x1):int(x2)], (x1, y1)

def save_image(image, detections, area=None):
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
    path = os.path.join(SAVE_DIR, filename)

    annotated_img = image.copy()

    # desenhar ROI
    if area is not None:
        x1, y1, x2, y2 = area
        cv2.rectangle(
            annotated_img,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

    # desenhar bounding boxes
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])

        cv2.rectangle(
            annotated_img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_img,
            det["class_name"],
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    success = cv2.imwrite(path, annotated_img)
    if not success:
        raise RuntimeError(f"Falha ao guardar imagem em {path}")

def detect(
    image,
    flg_save_image,
    chosen_class_ids: Optional[list[int]] = None,
    area: tuple[int, int, int, int] | None = None
):
    yolo_model = get_model()
    original_image = image.copy()

    # Se houver ROI, cortar imagem
    if area is not None:
        roi_x1, roi_y1, roi_x2, roi_y2 = area
        crop = image[roi_y1:roi_y2, roi_x1:roi_x2]
        offset_x, offset_y = roi_x1, roi_y1
    else:
        crop = image
        offset_x, offset_y = 0, 0

    results = yolo_model(crop, classes=chosen_class_ids)

    detections = []

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf <= Config.CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            class_id = int(box.cls[0])

            # ajustar para coordenadas originais
            x1 += offset_x
            x2 += offset_x
            y1 += offset_y
            y2 += offset_y

            detections.append({
                "class_id": class_id,
                "class_name": yolo_model.names[class_id],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            })

    if flg_save_image:
        save_image(original_image, detections, area)

    return detections

