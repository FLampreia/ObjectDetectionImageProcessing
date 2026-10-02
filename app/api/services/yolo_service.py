from pathlib import Path
from typing import Optional

import cv2
import torch
from ultralytics import YOLO

from app.api.model.detection import FrameProcessingResponse, Detection
from app.config.constants import MODELS_DIR


class YoloService:
    def __init__(self):
        self._model_cache: dict[Path, YOLO] = {}

    def get_model(self, model_name: str) -> YOLO:
        model_path = MODELS_DIR / f"{model_name}.pt"

        if model_path not in self._model_cache:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model_cache[model_path] = YOLO(
                str(model_path),
                verbose=False,
            ).to(device)

        return self._model_cache[model_path]


    def detect(
            self,
            model: YOLO,
            image,
            conf_threshold: float = 0.8,
            chosen_class_ids: Optional[list[int]] = None,
    ) -> FrameProcessingResponse:

        results = model(
            image,
            classes=chosen_class_ids,
            conf=conf_threshold,

        )

        detections = []

        total_ms = sum(results[0].speed.values())

        for box in results[0].boxes:
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])

            detections.append(
                Detection(
                    class_id=class_id,
                    confidence=confidence,
                    class_name=model.names[class_id],
                    bbox=[
                        x1,
                        y1,
                        x2,
                        y2
                    ],
                )
            )

        return FrameProcessingResponse(
            processing_time_per_frame=total_ms,
            detections_per_frame=detections,
        )

    def track(
            self,
            model: YOLO,
            image,
            conf_threshold: float = 0.8,
            chosen_class_ids: Optional[list[int]] = None,
    ) -> FrameProcessingResponse:

        results = model.track(
            image,
            persist=True,
            classes=chosen_class_ids,
            verbose=False,
        )

        detections = []
        processing_time = {}

        for result in results:
            processing_time = result.speed

            if result.boxes is None:
                continue

            for box in result.boxes:
                confidence = float(box.conf[0])

                if confidence < conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])

                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0])

                detections.append(
                    Detection(
                        track_id=track_id,
                        class_id=class_id,
                        confidence=confidence,
                        class_name=model.names[class_id],
                        bbox=[x1, y1, x2, y2],
                    )
                )

        return FrameProcessingResponse(
            processing_time_per_frame=sum(processing_time.values()),
            detections_per_frame=detections,
        )

    # def draw_bboxes(
    #         self,
    #         image,
    #         detections: list[Detection],
    # ):
    #     for detection in detections:
    #         x1, y1, x2, y2 = map(int, detection.bbox)
    #
    #         label = f"{detection.class_name} {detection.confidence:.2f}"
    #
    #         # Bounding box
    #         cv2.rectangle(
    #             image,
    #             (x1, y1),
    #             (x2, y2),
    #             (0, 255, 0),
    #             2,
    #         )
    #
    #         # Text size
    #         (text_width, text_height), baseline = cv2.getTextSize(
    #             label,
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5,
    #             1,
    #         )
    #
    #         # Text background
    #         cv2.rectangle(
    #             image,
    #             (x1, y1 - text_height - baseline - 4),
    #             (x1 + text_width, y1),
    #             (0, 255, 0),
    #             -1,
    #         )
    #
    #         # Text
    #         cv2.putText(
    #             image,
    #             label,
    #             (x1, y1 - baseline - 2),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5,
    #             (0, 0, 0),
    #             1,
    #             cv2.LINE_AA,
    #         )

    # def detect_frame(
    #         self,
    #         model: YOLO,
    #         image,
    #         model_name: str,
    #         conf_threshold: float = 0.8,
    #         chosen_class_ids: Optional[list[int]] = None,
    # ) -> ProcessingResponse:
    #
    #     results = model(
    #         image,
    #         classes=chosen_class_ids,
    #     )
    #
    #

# def detect_video(
#         video_path: str,
#         flg_save_image,
#         chosen_class_ids: Optional[list[int]] = None,
#         area: tuple[int, int, int, int] | None = None
# ):
#     yolo_model = get_model()
#     cap = cv2.VideoCapture(video_path)
#     tracked_ids = set()
#     object_count = 0
#
#     while True:
#         if cap is not None:
#             ret, frame = cap.read()
#
#             if ret:
#
#                 annotated_frame = frame.copy()
#
#                 # Generate mask for defined area
#                 # mask = self.create_mask()
#                 # mask_array, _ = mask
#
#                 detections = []
#                 # flag_new_detections = False  # Track if any new sheep detected
#
#
#                 result = yolo_model.track(frame, persist=True, verbose=False)
#
#                 if result[0] is not None and result[0].boxes.id is not None:
#                     boxes = result[0].boxes
#                     for xyxy, track_id, class_id, conf in zip(
#                             boxes.xyxy.cpu().tolist(),
#                             boxes.id.cpu().tolist(),
#                             boxes.cls.int().cpu().tolist(),
#                             boxes.conf.cpu().tolist()
#                     ):
#
#                         # Filter by class and confidence
#                         if class_id != chosen_class_ids or conf <= Config.CONF_THRESHOLD:
#                             continue
#
#                         x1, y1, x2, y2 = map(int, xyxy)
#                         cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#
#                         # Check if detection is inside the defined area
#                         # inside_area = mask_array[cy, cx] > 0
#
#                         new_track = track_id not in tracked_ids #and inside_area
#
#                         if new_track:
#                             tracked_ids.add(track_id)
#                             object_count += 1
#                             # flag_new_detections = True
#                             # logging_service.info(
#                             #     f"[{sheep_count}] New Sheep ID {track_id} | Confidence: {conf:.2f}")
#
#                         detections.append({
#                             "bbox": (x1, y1, x2, y2),
#                             "id": track_id,
#                             "class_id": class_id,
#                             "class_name": yolo_model.names[class_id],
#                             "tracked": track_id in tracked_ids
#                         })

# Annotate frame with boxes and area
# annotated_frame = self.annotate_frame(detections, annotated_frame, mask)

# Encode frame to JPEG
# if self.counting_event.is_set():
#     _, buffer = cv2.imencode('.jpg', annotated_frame)
# else:
#     _, buffer = cv2.imencode('.jpg', frame)

# Optionally send text version if new detections
# if flag_new_detections:
#     jpg_as_text = base64.b64encode(buffer).decode('utf-8')
#     await websocket.send_text(jpg_as_text)

# Send raw bytes to WebSocket
# await websocket.send_bytes(buffer.tobytes())
