from typing import Optional

import cv2
from fastapi import APIRouter, UploadFile, File, status, Query, Depends, Form

from app.api.model.detection import DetectionResponse
from app.api.services.detection_service import DetectionService
from app.api.services.model_service import ModelService

detection_router = APIRouter()


@detection_router.post("/detect/image", status_code=status.HTTP_200_OK, response_model=DetectionResponse)
async def detect_image(
        file: UploadFile = File(...),
        algorithm: str = Form(...),
        conf_threshold: float = Form(...),
        chosen_class_id: list[str] | None = Form(None),
        detection_service: DetectionService = Depends(DetectionService)
):
    return await detection_service.process_image(
        raw_image=file,
        chosen_class_id=chosen_class_id,
        algorithm=algorithm,
        conf_threshold=conf_threshold,
    )


# @detection_router.post("/detect/image", status_code=status.HTTP_200_OK)
# async def detect_image(
#         file: UploadFile = File(...),
#         algorithm: str = Form(...),
#         conf_threshold: float = Form(...),
#         chosen_class_id: list[int] | None = Form(None),
#         detection_service: DetectionService = Depends(get_detection_service)
# ):
#     return await detection_service.process_image(
#         raw_image=file,
#         algorithm=algorithm,
#         conf_threshold=conf_threshold,
#         chosen_class_id=chosen_class_id,
#     )


@detection_router.get("/detect/models", status_code=status.HTTP_200_OK)
def get_models(
        model_service: ModelService = Depends(ModelService)
):
    return model_service.get_models_overview()


@detection_router.post("/detect/models/{model_name}")
async def download_model(
        model_name: str,
        model_service: ModelService = Depends(ModelService)
):
    return await model_service.download_model(model_name)

# @detection_router.post("/detect/video", response_model=DetectionResponse, status_code=status.HTTP_200_OK)
# async def detect_video(
#         file: UploadFile = File(...),
#         flg_save_video: bool = Query(False),
#         chosen_class_id: Optional[list[int]] = Query(default=None),
#         area: Optional[tuple[int, int, int, int]] = Query(default=None)
# ):
#     result = await process_video(
#         file,
#         flg_save_video,
#         chosen_class_id,
#         area
#     )
#     return result
#
# # @detection_router.get("/detect/file")
# # async def detect_file(
# #         file: UploadFile = File(...),
# #
# # )

# @detection_router.post("/detect/video", response_model=DetectionResponse, status_code=status.HTTP_200_OK)
# async def detect_video():
#     video_capture = cv2.VideoCapture(0)
