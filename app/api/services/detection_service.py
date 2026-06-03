from typing import Optional

import cv2
import numpy as np
from fastapi import UploadFile

from app.api.services.yolo_detector import detect, get_yolo_classes


async def process_image(
        file: UploadFile,
        flg_save_image: bool,
        chosen_class_id: list[int] | None = None,
        area: tuple[int, int, int, int] | None = None
):

    contents = await file.read()

    np_array = np.frombuffer(contents, np.uint8)

    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    detections = detect(image, flg_save_image, chosen_class_id, area)

    return {
        "filename": file.filename,
        "size_x": image.shape[1],
        "size_y": image.shape[0],
        "detections": detections
    }

def get_detection_classes():
    return  get_yolo_classes()