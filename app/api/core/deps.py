from app.api.services.detection_service import DetectionService
from app.api.services.model_service import ModelService
from app.api.services.socket_service import SocketService

# Centralizar a criação de dependências para acada vez q é chamado um endpoint
# nao ser criada uma instancia de cada serviço


model_service = ModelService()
detection_service = DetectionService()
socket_service = SocketService()

def get_model_service() -> ModelService:
    return model_service

def get_detection_service() -> DetectionService:
    return detection_service

def get_socket_service() -> SocketService:
    return socket_service