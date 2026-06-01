from pydantic import BaseModel
from starlette.responses import FileResponse


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]


class DetectionResponse(BaseModel):
    filename: str
    detections: list[Detection]
