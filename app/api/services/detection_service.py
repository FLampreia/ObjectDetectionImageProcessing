import cv2
import numpy as np
from fastapi import UploadFile

from app.api.model.detection import FrameProcessingResponse, Area
from app.api.services.yolo_service import YoloService


class DetectionService:

    def __init__(self):
        self._yolo_service = YoloService()

    def get_model(self, algorithm: str):
        return self._yolo_service.get_model(algorithm)

    def create_tracking_model(self, algorithm: str):
        return self._yolo_service.create_tracking_model(algorithm)

    # async def process_image(
    #         self,
    #         raw_image: UploadFile,
    #         algorithm: str,
    #         conf_threshold: float,
    #         chosen_class_id: list[int] | None = None,
    # ):
    #     # Read uploaded Image
    #     contents = await raw_image.read()
    #     np_array = np.frombuffer(contents, np.uint8)
    #     # Decode Image
    #     image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    #
    #     # Transforms str list into int list
    #     if chosen_class_id:
    #         chosen_class_id = [int(x) for x in chosen_class_id]
    #
    #     # Load the model
    #     yolo_model = self._yolo_service.get_model(algorithm)
    #
    #     # Inference
    #     detections = self._yolo_service.detect_image(
    #         model=yolo_model,
    #         image=image,
    #         chosen_class_ids=chosen_class_id,
    #         conf_threshold=conf_threshold,
    #     )
    #
    #     return FrameProcessingResponse(
    #         processing_time_per_frame=detections.processing_time_per_frame,
    #         detections_per_frame=detections.detections_per_frame,
    #     )

    async def process_image_upload(
            self,
            raw_image: UploadFile,
            algorithm: str,
            conf_threshold: float,
            chosen_class_id: list[int] | None = None,
            area: Area | None = None
    ):
        # Read uploaded Image
        contents = await raw_image.read()

        np_array = np.frombuffer(contents, np.uint8)

        # Decode Image
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Transforms str list into int list
        if chosen_class_id:
            chosen_class_id = [int(x) for x in chosen_class_id]

        # Load the model
        yolo_model = self._yolo_service.get_model(algorithm)

        height, width = image.shape[:2]
        image_to_process = image

        if area is not None:
            # Calculate and order the coordinates of the selected area, from top left to bottom right
            left = max(0, min(width, min(area.x1, area.x2)))
            right = max(0, min(width, max(area.x1, area.x2)))
            top = max(0, min(height, min(area.y1, area.y2)))
            bottom = max(0, min(height, max(area.y1, area.y2)))

            if right <= left or bottom <= top:
                raise ValueError("Selected area is empty or invalid.")

            image_to_process = image[top:bottom, left:right]

        return self.process_frame(
            image=image_to_process,
            yolo_model=yolo_model,
            conf_threshold=conf_threshold,
            chosen_class_id=chosen_class_id,
            tracking=False,
        )

    def process_frame(
            self,
            image: np.ndarray,
            yolo_model,
            conf_threshold: float,
            chosen_class_id: list[int] | None = None,
            tracking: bool = False,
    ) -> FrameProcessingResponse:

        if tracking:
            return self._yolo_service.track(
                model=yolo_model,
                image=image,
                chosen_class_ids=chosen_class_id,
                conf_threshold=conf_threshold,
            )

        return self._yolo_service.detect(
            model=yolo_model,
            image=image,
            chosen_class_ids=chosen_class_id,
            conf_threshold=conf_threshold,
        )

    # return DetectionResponse(
    #     filename=raw_image.filename,
    #     size_x=image.shape[1],
    #     size_y=image.shape[0],
    #     processing_time_ms=detections.processing_time,
    #     detections=detections.detections,
    #
    # async def process_video(
    #         self,
    #         raw_video: UploadFile,
    #         algorithm: str,
    #         conf_threshold: float,
    #         chosen_class_id: list[int] | None = None,
    # ):
    #     # Transforms str list into int list
    #     if chosen_class_id:
    #         chosen_class_id = [int(x) for x in chosen_class_id]
    #
    #     # Create temporary file
    #     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
    #         shutil.copyfileobj(raw_video.file, tmp)
    #         temp_path = tmp.name
    #
    #     # Reads the file
    #     cap = cv2.VideoCapture(temp_path)
    #
    #     # Load the model
    #     yolo_model = self._yolo_service.get_model(algorithm)
    #
    #     # Have all detections from all frames
    #     all_detections = []
    #     full_time = 0
    #
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #
    #     output_path = tempfile.NamedTemporaryFile(
    #         delete=False,
    #         suffix=".mp4"
    #     ).name
    #
    #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #
    #     writer = cv2.VideoWriter(
    #         output_path,
    #         fourcc,
    #         fps,
    #         (width, height)
    #     )
    #
    #     while True:
    #         ret, frame = cap.read()
    #
    #         if not ret:
    #             break
    #
    #         detections = self._yolo_service.detect_image(
    #             model=yolo_model,
    #             image=frame,
    #             chosen_class_ids=chosen_class_id,
    #             conf_threshold=conf_threshold,
    #         )
    #
    #         all_detections.append(detections.detections_per_frame)
    #         full_time += detections.processing_time_per_frame
    #
    #         self._yolo_service.draw_bboxes(
    #             image=frame,
    #             detections=detections.detections_per_frame,
    #         )
    #
    #         writer.write(frame)
    #
    #
    #
    #     # desenhar bounding boxes
    #     # escrever frame no vídeo de saída
    #
    #     cap.release()
    #     writer.release()
    #
    #     return output_path, all_detections, full_time
# ________________________________________________________________________
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
