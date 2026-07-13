import base64
import json
import logging
import re
from typing import Dict, Any

import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

_PROMPT_TEMPLATE = (
    "You are verifying whether a photo was actually taken at a specific event/venue.\n"
    "Event title: {title}\n"
    "Venue: {venue}\n"
    "Keywords: {keywords}\n\n"
    "Look at the photo and judge how strongly it matches this event (signage, screens, "
    "banners, venue interior, crowd/stage setup, etc). "
    'Respond with ONLY a compact JSON object: {{"confidence": <0.0-1.0>, "event_relation": '
    '"STRONGLY_RELATED"|"RELATED"|"UNRELATED", "reasoning": "<short reason>"}}'
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_model_json(text: str) -> Dict[str, Any]:
    match = _JSON_RE.search(text or "")
    if not match:
        return {"confidence": 0.0, "event_relation": None}
    try:
        data = json.loads(match.group(0))
    except Exception:
        logger.warning("gemini_vision: could not parse JSON from model reply: %r", text[:300])
        return {"confidence": 0.0, "event_relation": None}
    try:
        confidence = float(data.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    return {"confidence": confidence, "event_relation": data.get("event_relation")}


async def analyze(image_bytes: bytes, event_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Call the Gemini generateContent API (vision) to judge event relevance of a photo.
    Returns {'confidence': float, 'event_relation': str|None, 'raw': ...}.
    Uses `GEMINI_API_KEY` and `GEMINI_VISION_MODEL` from settings. If not configured, returns confidence 0.0.
    """
    api_key = settings.GEMINI_API_KEY
    model = settings.GEMINI_VISION_MODEL
    if not api_key or not model:
        logger.info("gemini_vision: skipped, GEMINI_API_KEY/GEMINI_VISION_MODEL not configured")
        return {"confidence": 0.0}

    model_path = model if model.startswith("models/") else f"models/{model}"

    event_config = event_config or {}
    prompt = _PROMPT_TEMPLATE.format(
        title=event_config.get("title") or "",
        venue=event_config.get("venue_name") or "",
        keywords=", ".join(event_config.get("keywords") or []),
    )

    base_url = getattr(settings, "GEMINI_API_URL", None) or GEMINI_API_BASE
    url = f"{base_url}/{model_path}:generateContent"
    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    },
                ]
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, params={"key": api_key}, json=body)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates") or []
            reply_text = ""
            if candidates:
                parts = (candidates[0].get("content") or {}).get("parts") or []
                reply_text = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict))
            parsed = _parse_model_json(reply_text)
            logger.info(
                "gemini_vision: confidence=%.3f event_relation=%s",
                parsed["confidence"],
                parsed.get("event_relation"),
            )
            return {**parsed, "raw": data}
    except Exception:
        logger.exception("gemini_vision: request failed")
        return {"confidence": 0.0}
