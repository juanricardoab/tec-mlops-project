from functools import lru_cache
from typing import List, Union
from pydantic_settings import BaseSettings

from pydantic import AnyHttpUrl, Field, validator


class Settings(BaseSettings):
    """
    Settings for the application.
    """

    testing: bool = Field(0, env="TESTING")
    ENVIRONMENT: str = Field(...)
    APP_NAME: str = Field(...)
    APP_DESCRIPTION: str = Field(...)
    APP_VERSION: str = Field(...)
    API_V1_STR: str = "/api/v1"
    DEFAULT_EXPIRE_TIME: int = Field(...)
    DEBUG: bool = Field(...)
    MODEL_PATH: str = Field(...)


@lru_cache()
def get_settings() -> BaseSettings:
    """Get the settings for the application."""
    return Settings()


settings: Settings = Settings()
