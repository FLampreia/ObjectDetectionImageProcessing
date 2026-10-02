from pydantic import BaseModel


class Detection(BaseModel):
    track_id: int | None = None
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]

class Area(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

# class DetectionResponse(BaseModel):
#     processing_time_ms: float
#     detections: list[Detection]

# class ImageProcessingResponse(BaseModel):
#     processed_image: FileResponse
#     processing_time_ms: float
#     detections: list[Detection]
#
# class VideoProcessingResponse(BaseModel):
#     processed_video: FileResponse
#     processing_time_ms: float
#     detections: list[list[Detection]]

class FrameProcessingResponse(BaseModel):
    processing_time_per_frame: float
    detections_per_frame: list[Detection]

# class ProcessingResponse(BaseModel):
#     detections: list[Detection]
#     processing_time: float