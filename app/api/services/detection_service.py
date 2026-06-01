from typing import Optional

import cv2
import numpy as np
from fastapi import UploadFile

from app.api.services.yolo_detector import detect, get_yolo_classes


async def process_image(file: UploadFile, chosen_class_id: Optional[list[int]] = None):

    contents = await file.read()

    np_array = np.frombuffer(contents, np.uint8)

    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    detections = detect(image, chosen_class_id)

    return {
        "filename": file.filename,
        "detections": detections
    }

def get_detection_classes():
    return  get_yolo_classes()