from typing import Optional

import cv2
from fastapi import APIRouter, UploadFile, File, status, Query

from app.api.model.detection import DetectionResponse
from app.api.services.detection_service import process_image, get_detection_classes

detection_router = APIRouter()


@detection_router.post("/detect/image", response_model=DetectionResponse, status_code=status.HTTP_200_OK)
async def detect_image(
        file: UploadFile = File(...),
        chosen_class_id: Optional[list[int]] = Query(default=None)
):
    result = await process_image(file, chosen_class_id)
    return result


@detection_router.get("/detect/classes")
def detection_classes():
    return get_detection_classes()
# @detection_router.po

# @detection_router.post("/detect/video", response_model=DetectionResponse, status_code=status.HTTP_200_OK)
# async def detect_video():
#     video_capture = cv2.VideoCapture(0)
