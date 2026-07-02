
import json
from typing import List

import httpx

from app.config.constants import ALLOWED_MODELS_FILE, MODELS_DIR


class ModelService:
    def __init__(self):
        self._allowed_models_cache: List[str] | None = None


    def get_allowed_models_list(self) -> List[str]:
        """Returns the list of models allowed by the application."""

        if self._allowed_models_cache is None:
            with ALLOWED_MODELS_FILE.open("r", encoding="utf-8") as file:
                data = json.load(file)

            self._allowed_models_cache = data.get("models", [])

        return self._allowed_models_cache



    def get_installed_models_list(self) -> List[str]:
        """Returns the list of installed YOLO models."""

        if not MODELS_DIR.exists():
            return []

        return sorted(
            file.stem
            for file in MODELS_DIR.iterdir()
            if file.is_file() and file.suffix == ".pt"
        )

    def get_models_overview(self) -> List[dict]:
        """Returns all allowed models together with their installation status."""

        installed = set(self.get_installed_models_list())

        return [
            {
                "name": model,
                "installed": model in installed,
            }
            for model in self.get_allowed_models_list()
        ]


    async def download_model(self, model_name: str) -> dict:
        """Downloads a YOLO model from the official Ultralytics release."""

        model_path = MODELS_DIR / f"{model_name}.pt"

        if model_path.exists():
            return {
                "status": "already_exists",
                "model": model_name,
            }

        url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}.pt"

        async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()

                with open(model_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)

        return {
            "status": "downloaded",
            "model": model_name,
            "path": str(model_path)
        }