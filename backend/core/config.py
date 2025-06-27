from functools import lru_cache
from decouple import config, UndefinedValueError
from urllib.parse import quote_plus

class Settings:
    PROJECT_NAME: str = "Sewer Flow API"
    API_V1_STR: str = "/api/v1"
    try:
        # database
        POSTGRES_USER: str = str(config("POSTGRES_USER"))
        POSTGRES_PASSWORD: str = str(config("POSTGRES_PASSWORD"))
        POSTGRES_HOST: str = str(config("POSTGRES_HOST"))
        POSTGRES_PORT: str = str(config("POSTGRES_PORT"))
        POSTGRES_DB: str = str(config("POSTGRES_DB"))


        safe_password = quote_plus(POSTGRES_PASSWORD)
        DATABASE_URL: str = (
            f"postgresql+asyncpg://{POSTGRES_USER}:{safe_password}"
            f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
        )

        # object storage
        MINIO_ENDPOINT: str = str(config("MINIO_ENDPOINT"))
        MINIO_ACCESS_KEY: str = str(config("MINIO_ACCESS_KEY"))
        MINIO_SECRET_KEY: str = str(config("MINIO_SECRET_KEY"))

        # JWT
        SECRET_KEY: str = str(config("SECRET_KEY"))
        ACCESS_TOKEN_EXPIRE_MINUTES: int = 60*24

        UPLOAD_DIR: str = str(config("UPLOAD_DIR"))
    except UndefinedValueError as e:
        raise RuntimeError("Environment variable is required in .env for production: " + str(e))


@lru_cache
def get_settings():
    return Settings()
