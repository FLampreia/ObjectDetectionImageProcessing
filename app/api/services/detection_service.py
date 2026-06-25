from pathlib import Path
from typing import Optional, List
import json
import cv2
import numpy as np
from fastapi import UploadFile
import tempfile

from app.api.core.config import Settings
from app.api.services.yolo_detector import get_yolo_classes, detect_image, detect_video


class DetectionService:
    def __init__(self):
        self._allowed_models_file = Path("app/config/allowed_models.json")
        self._installed_models_dir = Path("app/api/yolo/")
        self._allowed_models_cache = None

    def get_detection_classes(self):
        return get_yolo_classes()

    async def process_image(
            self,
            raw_image: UploadFile,
            chosen_class_id: list[int] | None = None,
    ):
        contents = await raw_image.read()

        np_array = np.frombuffer(contents, np.uint8)

        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        detections = detect_image(image, chosen_class_id)

        return {
            "filename": raw_image.filename,
            "size_x": image.shape[1],
            "size_y": image.shape[0],
            "detections": detections
        }

    def get_allowed_models_list(self) -> List[str]:
        if self._allowed_models_cache is not None:
            return self._allowed_models_cache

        with self._allowed_models_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self._allowed_models_cache = data.get("models", [])
        return self._allowed_models_cache

    def get_installed_models_list(self) -> List[str]:
        if not self._installed_models_dir.exists():
            return []

        return [
            f.stem
            for f in self._installed_models_dir.iterdir()
            if f.is_file() and f.suffix == ".pt"
        ]

    def get_models_overview(self) -> List[dict]:
        installed = set(self.get_installed_models_list())
        allowed = self.get_allowed_models_list()

        return [
            {
                "name": model,
                "installed": model in installed
            }
            for model in allowed
        ]

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
