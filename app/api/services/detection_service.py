from pathlib import Path
from typing import Optional, List
import json
import cv2
import numpy as np
from fastapi import UploadFile
import httpx

from app.api.model.detection import DetectionResponse
from app.api.services.yolo_service import YoloService


class DetectionService:

    def __init__(self):
        self._yolo_service = YoloService()

    async def process_image(
            self,
            raw_image: UploadFile,
            algorithm: str,
            conf_threshold: float,
            chosen_class_id: list[int] | None = None,
    ):
        # Read uploaded Image
        contents = await raw_image.read()
        np_array = np.frombuffer(contents, np.uint8)

        # Decode Image
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Transforms str list into int list
        if chosen_class_id:
            chosen_class_id = [int(x) for x in chosen_class_id]

        # Inference
        detections = self._yolo_service.detect_image(
            image=image,
            chosen_class_ids=chosen_class_id,
            conf_threshold=conf_threshold,
            model_name=algorithm
        )

        return DetectionResponse(
            filename=raw_image.filename,
            size_x=image.shape[1],
            size_y=image.shape[0],
            processing_time_ms=detections.processing_time,
            detections=detections.detections,
        )



# async def process_video(
#         raw_video: UploadFile,
#         flg_save_video: bool, #TODO retirar a flag
#         chosen_class_id: list[int] | None = None,
#         area: tuple[int, int, int, int] | None = None
# ):
#
#     # Save uploaded video as temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#         tmp.write(await raw_video.read())
#         video_path = tmp.name
#
#     detections = detect_video(video_path, flg_save_video, chosen_class_id, area)
#
#     return {
#         "filename": raw_video.filename,
#         "detections": detections
#     }


#
# async def process_video(
#     file: UploadFile,
#     flg_save_image: bool,
#     chosen_class_id: list[int] | None = None,
#     area: tuple[int, int, int, int] | None = None
# ):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#         tmp.write(await file.read())
#         video_path = tmp.name
#
#     cap = cv2.VideoCapture(video_path)
#
#     all_detections = []
#     frame_number = 0
#
#     while True:
#         ret, frame = cap.read()
#
#         if not ret:
#             break
#
#         detections = detect(
#             frame,
#             flg_save_image,
#             chosen_class_ids=chosen_class_id,
#             area=area
#         )
#
#         all_detections.append({
#             "frame": frame_number,
#             "detections": detections
#         })
#
#         frame_number += 1
#
#     cap.release()
#
#     return {
#         "filename": file.filename,
#         "frames": frame_number,
#         "results": all_detections
#     }
