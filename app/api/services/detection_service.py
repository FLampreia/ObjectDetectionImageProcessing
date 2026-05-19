import cv2
import numpy as np
from fastapi import UploadFile

from app.api.services.yolo_detector import detect


async def process_image(file: UploadFile):

    contents = await file.read()

    np_array = np.frombuffer(contents, np.uint8)

    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    height, width = image.shape[:2]

    detections = detect(image)

    return {
        "filename": file.filename,
        "height": height,
        "width": width,
        "detections": detections,
    }

