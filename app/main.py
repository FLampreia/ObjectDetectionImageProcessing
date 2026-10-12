from fastapi import FastAPI

from app.api.routes.detection import detection_router
from app.api.routes.health_check import health_router
from app.api.routes.models import models_router

app = FastAPI()

app.include_router(health_router, tags=["health"])
app.include_router(detection_router, tags=["detection"])
app.include_router(models_router, tags=["models"])
