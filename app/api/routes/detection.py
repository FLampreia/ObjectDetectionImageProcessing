import json
import logging
import asyncio
from contextlib import suppress
from starlette.concurrency import run_in_threadpool

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, status, Query, Depends, Form, WebSocket, WebSocketDisconnect

from app.api.core.deps import get_detection_service, get_model_service, parse_area
from app.api.model.detection import FrameProcessingResponse, Area
from app.api.services.detection_service import DetectionService

detection_router = APIRouter()


@detection_router.post("/detect/image", status_code=status.HTTP_200_OK, response_model=FrameProcessingResponse)
async def detect_image(
        image: UploadFile = File(...),
        algorithm: str = Form(...),
        conf_threshold: float = Form(...),
        chosen_class_id: list[str] | None = Form(None),
        area: Area | None = Depends(parse_area),
        detection_service: DetectionService = Depends(get_detection_service)
):
    return await detection_service.process_image_upload(
        raw_image=image,
        chosen_class_id=chosen_class_id,
        algorithm=algorithm,
        conf_threshold=conf_threshold,
        area=area
    )


#
# @detection_router.post("/detect/video", status_code=status.HTTP_200_OK, response_model=VideoProcessingResponse)
# async def detect_video(
#         video: UploadFile = File(...),
#         algorithm: str = Form(...),
#         conf_threshold: float = Form(...),
#         chosen_class_id: list[str] | None = Form(None),
#         detection_service: DetectionService = Depends(get_detection_service)
# ):
#     output_path, all_detections, full_time = await detection_service.process_video(
#         raw_video=video,
#         chosen_class_id=chosen_class_id,
#         algorithm=algorithm,
#         conf_threshold=conf_threshold,
#     )
#
#     return VideoProcessingResponse(
#         processed_video=FileResponse(
#             output_path,
#             media_type="video/mp4",
#             filename="processed_video.mp4"
#         ),
#         processing_time_ms=full_time,
#         detections=all_detections,
#     )


#
@detection_router.websocket("/detect/video/socket/connect")
async def video_detect_websocket(
        websocket: WebSocket,
        detection_service: DetectionService = Depends(get_detection_service)
):
    # Internal endpoint: browser sessions are validated exclusively by the backend.
    await websocket.accept()

    try:
        parameters = await asyncio.wait_for(websocket.receive_json(), timeout=30)

        algorithm = parameters.get("algorithm")
        conf_threshold = float(parameters.get("conf_threshold"))
        if not 0 <= conf_threshold <= 1:
            raise ValueError("Invalid confidence threshold")
        chosen_class_id = parameters.get("chosen_class_id")
        area = parameters.get("area")

        # Load the model
        yolo_model = await run_in_threadpool(detection_service.create_tracking_model, algorithm)

        # Transforms str list into int list
        if chosen_class_id:
            chosen_class_id = [int(x) for x in chosen_class_id]

        while True:

            message = await asyncio.wait_for(websocket.receive_json(), timeout=120)

            # Client want to disconnect
            if message["type"] == "finish":
                break
            if message["type"] != "frame":
                raise ValueError("Expected frame or finish")

            image_bytes = await asyncio.wait_for(websocket.receive_bytes(), timeout=30)
            if len(image_bytes) > 8 * 1024 * 1024:
                raise ValueError("Frame exceeds 8 MiB")

            image = cv2.imdecode(
                np.frombuffer(image_bytes, np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                raise ValueError("Invalid image frame")
            if area is not None:
                left, right = sorted([int(area["x1"]), int(area["x2"])])
                top, bottom = sorted([int(area["y1"]), int(area["y2"])])
                height, width = image.shape[:2]
                if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                    raise ValueError("Selected area is outside the frame")
                image = image[top:bottom, left:right]

            response = await run_in_threadpool(
                detection_service.process_frame,
                image=image,
                yolo_model=yolo_model,
                conf_threshold=conf_threshold,
                chosen_class_id=chosen_class_id,
                tracking=True
            )

            await websocket.send_json({
                **response.model_dump(),
                "frame_number": message["frame_number"],
            })

        # Confirm to client that have finished
        await websocket.send_json({
            "status": "finished"
        })

        # Closes connection
        await websocket.close()

    except WebSocketDisconnect:
        logging.info("Client disconnected")
    except Exception:
        logging.exception("Frame processing failed")
        with suppress(Exception):
            await websocket.send_json({"status": "error", "detail": "Worker processing failed"})
            await websocket.close(code=1011)

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
