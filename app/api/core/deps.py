from app.api.services.detection_service import DetectionService


def get_detection_service() -> DetectionService:
    return DetectionService()