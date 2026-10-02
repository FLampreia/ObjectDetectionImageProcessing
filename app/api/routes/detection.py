import json
import logging

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, status, Query, Depends, Form, WebSocket, WebSocketDisconnect

from app.api.core.deps import get_detection_service, get_model_service, get_socket_service, parse_area
from app.api.model.detection import FrameProcessingResponse, Area
from app.api.services.detection_service import DetectionService
from app.api.services.socket_service import SocketService

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
@detection_router.websocket("/detect/video/socket/connect/{session_id}")
async def video_detect_websocket(
        websocket: WebSocket,
        session_id: str,
        socket_service: SocketService = Depends(get_socket_service),
        detection_service: DetectionService = Depends(get_detection_service)
):
    if session_id not in socket_service.sessions:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    socket_service.set_session_connected_true(session_id)

    try:
        parameters = await websocket.receive_json()

        algorithm = parameters.get("algorithm")
        chosen_team_id = parameters.get("chosen_team_id")
        conf_threshold = parameters.get("conf_threshold")
        chosen_class_id = parameters.get("chosen_class_id")

        print(algorithm, chosen_team_id, conf_threshold, chosen_class_id)

        # Load the model
        yolo_model = detection_service.get_model(algorithm)

        # Transforms str list into int list
        if chosen_class_id:
            chosen_class_id = [int(x) for x in chosen_class_id]

        video_detections = []
        frame = 0

        while True:

            message = await websocket.receive_json()

            # Client want to disconnect
            if message["type"] == "finish":
                break

            image_bytes = await websocket.receive_bytes()

            image = cv2.imdecode(
                np.frombuffer(image_bytes, np.uint8),
                cv2.IMREAD_COLOR,
            )

            response = detection_service.process_frame(
                image=image,
                yolo_model=yolo_model,
                conf_threshold=conf_threshold,
                chosen_class_id=chosen_class_id,
                tracking=True
            )

            video_detections.append(response)

            frame += 1

            print(
                f"Frame {frame} processado"
            )

            await websocket.send_json(response.model_dump())

        # # Guardar todas as deteções
        # await detection_service.save_video_detections(
        #     chosen_team_id=chosen_team_id,
        #     detections=video_detections,
        # )

        # Confirm to client that have finished
        await websocket.send_json({
            "status": "finished"
        })

        # Closes connection
        await websocket.close()

    except WebSocketDisconnect:
        logging.warn("Client disconnected")

    finally:
        # after closing the websocket, session is deleted
        socket_service.remove_session(session_id)

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
