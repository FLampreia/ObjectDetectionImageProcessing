from pydantic import BaseModel
from starlette.responses import FileResponse


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]


class DetectionResponse(BaseModel):
    filename: str
    size_x: int
    size_y: int
    processing_time_ms: float
    detections: list[Detection]

class ProcessingResponse(BaseModel):
    detections: list[Detection]
    processing_time: float

class ModelInfo(BaseModel):
    name: str
    installed: bool
