import logging

from fastapi import APIRouter, Depends
from starlette import status

from app.api.core.deps import get_socket_service
from app.api.services.socket_service import SocketService

socket_router = APIRouter()

@socket_router.post("/socket/session/create")
def create_socket_session(
        socket_service: SocketService = Depends(get_socket_service),
):
    # todo para isto funcionar o worker está exposto

    # todo
    #  se eu fizer 2 connexões websocket, o worker deixa de estar
    #  exposto e o backend funciona como intermediário
    #  e ja consigo guardar os dados na base de dados
    logging.info(f"Trying to create socket session")
    return socket_service.create_socket_session()


@socket_router.get("/socket/session/list")
def get_socket_session(
        socket_service: SocketService = Depends(get_socket_service),
):
    return socket_service.get_all_sessions()


@socket_router.delete("/socket/session/{session_id}", status_code=status.HTTP_200_OK)
def remove_socket_session(
        session_id: str,
        socket_service: SocketService = Depends(get_socket_service),
):
    socket_service.remove_session(session_id)

