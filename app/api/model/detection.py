from pydantic import BaseModel


class Detection(BaseModel):
    class_name: int
    confidence: float
    bbox: list[float]


class DetectionResponse(BaseModel):
    filename: str
    height: int
    width: int
    detections: list[Detection]