import io
import json
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main_module, "warmup_paddle_ocr", lambda *a, **kw: None)
    with TestClient(main_module.app) as c:
        yield c


@pytest.fixture
def mock_providers(monkeypatch):
    """Stub out the AI providers the service imports by reference, so tests
    never make real network calls and can dictate OCR/vision results."""
    paddle_mock = AsyncMock(return_value={"text": ""})
    claude_mock = AsyncMock(return_value={"confidence": 0.0})
    gemini_mock = AsyncMock(return_value={"confidence": 0.0})
    monkeypatch.setattr("app.services.verification_service.paddle_analyze", paddle_mock)
    monkeypatch.setattr("app.services.verification_service.claude_analyze", claude_mock)
    monkeypatch.setattr("app.services.verification_service.gemini_analyze", gemini_mock)
    return {"paddle": paddle_mock, "claude": claude_mock, "gemini": gemini_mock}


def _jpeg_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def sharp_image_bytes() -> bytes:
    """High-contrast checkerboard: well above the blur threshold and mid
    brightness, so it clears the quality gate cleanly."""
    size, block = 256, 8
    n = size // block
    tile = (np.indices((n, n)).sum(axis=0) % 2).astype(np.uint8)
    board = np.kron(tile, np.ones((block, block), dtype=np.uint8))
    gray = np.where(board.astype(bool), 240, 10).astype(np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    return _jpeg_bytes(rgb)


@pytest.fixture
def blurry_image_bytes() -> bytes:
    """Uniform mid-gray fill: PIL's edge filter still reports non-zero
    variance from image-border padding, and that border contribution shrinks
    as the image grows, so this must be large enough (and bright enough) to
    land below BLUR_THRESHOLD without tripping the darkness check too."""
    rgb = np.full((1200, 1200, 3), 150, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def corrupt_image_bytes() -> bytes:
    return b"this is not a real image file"


@pytest.fixture
def post_verify(client):
    def _post(event, image_bytes: bytes, filename: str = "photo.jpg", content_type: str = "image/jpeg"):
        event_data = event if isinstance(event, str) else json.dumps(event)
        return client.post(
            "/verify",
            data={"event": event_data},
            files={"image": (filename, image_bytes, content_type)},
        )

    return _post
