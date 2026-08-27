from datetime import datetime
from uuid import uuid4


class SocketService:
    def __init__(self):
        self.sessions: dict[str, dict] = {}

    def create_socket_session(self):
        session_id = str(uuid4())

        self.sessions[session_id] = {
            "connected": False,
            "websocket": None,
            "created_at": datetime.utcnow(),
        }

        return session_id

    def get_all_sessions(self):
        return self.sessions

    def remove_session(self, session_id: str):
        self.sessions.pop(session_id, None)

    def set_session_connected_true(self, session_id: str):
        self.sessions[session_id]["connected"] = True