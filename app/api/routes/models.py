from fastapi import APIRouter, Depends
from starlette import status

from app.api.core.deps import get_model_service
from app.api.services.model_service import ModelService

models_router = APIRouter()

@models_router.get("/models", status_code=status.HTTP_200_OK)
def get_models(
        model_service: ModelService = Depends(get_model_service)
):
    return model_service.get_models_overview()


@models_router.post("/models/{model_name}")
async def download_model(
        model_name: str,
        model_service: ModelService = Depends(get_model_service)
):
    return await model_service.download_model(model_name)
