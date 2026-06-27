from typing import Optional

import cv2
from fastapi import APIRouter, UploadFile, File, status, Query, Depends, Form

from app.api.core.deps import get_detection_service
from app.api.model.detection import DetectionResponse
from app.api.services.detection_service import DetectionService

detection_router = APIRouter()


# @detection_router.post("/detect/image", response_model=DetectionResponse, status_code=status.HTTP_200_OK)
# async def detect_image(
#         file: UploadFile = File(...),
#         flg_save_image: bool = Query(False), #TODO retirar a flag
#         chosen_class_id: Optional[list[int]] = Query(default=None),
#         area: Optional[tuple[int, int, int, int]] = Query(default=None),
#
# ):
#     # result = await process_image(
#     #     file,
#     #     flg_save_image,
#     #     chosen_class_id,
#     #     area
#     # )
#     # return result
#
#     return "Cheguei"
@detection_router.post("/detect/image", status_code=status.HTTP_200_OK)
async def detect_image(
        file: UploadFile = File(...),
        algorithm: str = Form(...),
        conf_threshold: float = Form(...),
        chosen_class_id: list[str] | None = Form(None),
        detection_service: DetectionService = Depends(get_detection_service)
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
        detection_service: DetectionService = Depends(get_detection_service)
):
    return detection_service.get_models_overview()


@detection_router.get("/detect/classes")
def detection_classes(
        detection_service: DetectionService = Depends(get_detection_service)
):
    return detection_service.get_detection_classes()


@detection_router.post("/detect/models/{model_name}")
async def download_model(
        model_name: str,
        detection_service: DetectionService = Depends(get_detection_service)
):
    return await detection_service.download_model(model_name)

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
