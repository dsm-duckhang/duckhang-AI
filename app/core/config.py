from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # AI / model settings
    GEMINI_API_KEY: str | None = None
    GEMINI_VISION_MODEL: str | None = "gemini-2.5-flash"

    CLAUDE_API_KEY: str | None = None
    CLAUDE_VISION_MODEL: str | None = "claude-haiku-4-5-20251001"

    OCR_LANGUAGE: str | None = "korean"

    OCR_STRONG_MATCH_SCORE: float = 0.85
    OCR_MEDIUM_MATCH_SCORE: float = 0.6
    VISION_STRONG_CONFIDENCE: float = 0.85
    VISION_MEDIUM_CONFIDENCE: float = 0.6

    MIN_IMAGE_SHORT_SIDE: int = 400
    BLUR_THRESHOLD: float = 100.0
    DARKNESS_THRESHOLD: float = 40.0

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
