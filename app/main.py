from fastapi import FastAPI

from app.api.routes.detection import detection_router

app = FastAPI()

app.include_router(detection_router, tags=["detection"])