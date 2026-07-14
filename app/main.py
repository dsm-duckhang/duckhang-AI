import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.verification_routes import router
from app.providers.paddle_ocr import warmup as warmup_paddle_ocr
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await asyncio.to_thread(warmup_paddle_ocr, getattr(settings, "OCR_LANGUAGE", "korean"))
    except Exception:
        logger.exception("PaddleOCR warmup failed; OCR will lazy-init on first request")
    yield


app = FastAPI(title="DuckHang Verification API", lifespan=lifespan)
app.include_router(router)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
