from fastapi import Form, HTTPException
from pydantic import ValidationError

from app.api.model.detection import Area
from app.api.services.detection_service import DetectionService
from app.api.services.model_service import ModelService

# todo Centralizar a criação de dependências para acada vez q é chamado um endpoint
#   nao ser criada uma instancia de cada serviço


model_service = ModelService()
detection_service = DetectionService()

def get_model_service() -> ModelService:
    return model_service

def get_detection_service() -> DetectionService:
    return detection_service

def parse_area(area: str | None = Form(None)) -> Area | None:
    if area is None:
        return None

    try:
        return Area.model_validate_json(area)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "loc": ["body", "area", *error["loc"]],
                    "msg": error["msg"],
                    "type": error["type"],
                }
                for error in exc.errors()
            ],
        ) from exc
