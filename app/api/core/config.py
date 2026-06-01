from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    CONF_THRESHOLD: float
    YOLO_MODEL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

Config = Settings()

