from datetime import datetime
import os
from typing import Optional

import torch
from ultralytics import YOLO
import cv2
from app.api.core.config import Config

model = None

SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)


def get_model():
    global model

    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = YOLO(Config.YOLO_MODEL, verbose=False).to(device)

    return model


def get_yolo_classes():
    yolo_model = get_model()
    return yolo_model.names


def crop_image(image, area):
    x1, y1, x2, y2 = area
    return image[int(y1):int(y2), int(x1):int(x2)], (x1, y1)


def save_image(image, detections, area=None):
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
    path = os.path.join(SAVE_DIR, filename)

    annotated_img = image.copy()

    # desenhar ROI
    if area is not None:
        x1, y1, x2, y2 = area
        cv2.rectangle(
            annotated_img,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

    # desenhar bounding boxes
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])

        cv2.rectangle(
            annotated_img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_img,
            det["class_name"],
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    success = cv2.imwrite(path, annotated_img)
    if not success:
        raise RuntimeError(f"Falha ao guardar imagem em {path}")


def detect_image(
        image,
        chosen_class_ids: Optional[list[int]] = None,
):
    yolo_model = get_model()

    results = yolo_model(image, classes=chosen_class_ids)

    detections = []

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf <= Config.CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            class_id = int(box.cls[0])


            detections.append({
                "class_id": class_id,
                "class_name": yolo_model.names[class_id],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            })

    return detections


def detect_video(
        video_path: str,
        flg_save_image,
        chosen_class_ids: Optional[list[int]] = None,
        area: tuple[int, int, int, int] | None = None
):
    yolo_model = get_model()
    cap = cv2.VideoCapture(video_path)
    tracked_ids = set()
    object_count = 0

    while True:
        if cap is not None:
            ret, frame = cap.read()

            if ret:

                annotated_frame = frame.copy()

                # Generate mask for defined area
                # mask = self.create_mask()
                # mask_array, _ = mask

                detections = []
                # flag_new_detections = False  # Track if any new sheep detected


                result = yolo_model.track(frame, persist=True, verbose=False)

                if result[0] is not None and result[0].boxes.id is not None:
                    boxes = result[0].boxes
                    for xyxy, track_id, class_id, conf in zip(
                            boxes.xyxy.cpu().tolist(),
                            boxes.id.cpu().tolist(),
                            boxes.cls.int().cpu().tolist(),
                            boxes.conf.cpu().tolist()
                    ):

                        # Filter by class and confidence
                        if class_id != chosen_class_ids or conf <= Config.CONF_THRESHOLD:
                            continue

                        x1, y1, x2, y2 = map(int, xyxy)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        # Check if detection is inside the defined area
                        # inside_area = mask_array[cy, cx] > 0

                        new_track = track_id not in tracked_ids #and inside_area

                        if new_track:
                            tracked_ids.add(track_id)
                            object_count += 1
                            # flag_new_detections = True
                            # logging_service.info(
                            #     f"[{sheep_count}] New Sheep ID {track_id} | Confidence: {conf:.2f}")

                        detections.append({
                            "bbox": (x1, y1, x2, y2),
                            "id": track_id,
                            "class_id": class_id,
                            "class_name": yolo_model.names[class_id],
                            "tracked": track_id in tracked_ids
                        })

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
