from fastapi import APIRouter, UploadFile, File, status

from app.api.model.detection import DetectionResponse
from app.api.services.detection_service import process_image

detection_router = APIRouter()

@detection_router.post("/detect/image", response_model=DetectionResponse, status_code=status.HTTP_200_OK)
async def detect_image(file: UploadFile = File(...)):
    result = await process_image(file)
    return result