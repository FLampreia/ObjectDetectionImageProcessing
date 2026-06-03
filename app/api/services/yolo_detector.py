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

def save_image(result, area=None):
    """
    Save image with bounding boxes + optional ROI
    """

    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
    path = os.path.join(SAVE_DIR, filename)

    annotated_img = result.plot()

    # Draw Area (ROI)
    if area is not None:
        x1, y1, x2, y2 = area

        cv2.rectangle(
            annotated_img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            2
        )

        cv2.putText(
            annotated_img,
            "ROI",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
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

    results = yolo_model(image, classes=chosen_class_ids)

    detections = []

    # ROI Region of Interest
    roi_x1 = roi_y1 = roi_x2 = roi_y2 = None
    if area is not None:
        roi_x1, roi_y1, roi_x2, roi_y2 = area

    for result in results:

        for box in result.boxes:
            conf = float(box.conf[0])

            if conf <= Config.CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])

            if area is not None:
                # Box needs to be inside Area
                if x2 < roi_x1 or x1 > roi_x2 or y2 < roi_y1 or y1 > roi_y2:
                    continue

            class_id = int(box.cls[0])

            detections.append({
                "class_id": class_id,
                "class_name": yolo_model.names[class_id],
                "confidence": conf,
                "bbox": box.xyxy[0].tolist(),
            })

    if flg_save_image: save_image(results[0], area)

    return detections

