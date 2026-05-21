from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    All environment variables are defined here.

    Pydantic reads them automatically from the .env file.
    If a required variable is missing from .env
    the application will NOT start and will tell you
    exactly which variable is missing.

    This is better than the app starting and then
    crashing later when it tries to use a missing value.
    """

    # ── MongoDB ───────────────────────────────────────
    # Where MongoDB is running
    # Local dev: mongodb://localhost:27017
    # Production: your MongoDB Atlas or EC2 URL
    MONGODB_URL: str

    # Which database inside MongoDB to use
    # We use: thumbnail_generator
    MONGODB_DB_NAME: str

    # ── Redis ─────────────────────────────────────────
    # Where Redis is running
    # Local dev: redis://localhost:6379/0
    # Production: your ElastiCache URL
    REDIS_URL: str

    # ── AWS ───────────────────────────────────────────
    # Your AWS credentials
    # These are created in AWS IAM console
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str

    # Which AWS region your S3 bucket is in
    # For India: ap-south-1
    AWS_REGION: str

    # The name of your S3 bucket
    S3_BUCKET_NAME: str

    # ── Auth ──────────────────────────────────────────
    # The JWT secret key
    # We use this only to decode the token
    # We do NOT use it to sign or create tokens
    # Frontend team owns token creation
    JWT_SECRET: str

    # ── App ───────────────────────────────────────────
    # development or production
    APP_ENV: str = "development"

    # How many Celery workers run in parallel
    # 4 means 4 videos can process simultaneously
    MAX_WORKERS: int = 4

    class Config:
        # Tell pydantic to read from .env file
        env_file = ".env"
        # Make variable names case insensitive
        case_sensitive = False


# Create one instance of Settings
# This is imported by every other file that needs config
# Everyone imports this same object
# .env is read only once
settings = Settings()