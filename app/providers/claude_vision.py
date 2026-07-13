import base64
import json
import logging
import re
from typing import Dict, Any

import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

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
        logger.warning("claude_vision: could not parse JSON from model reply: %r", text[:300])
        return {"confidence": 0.0, "event_relation": None}
    try:
        confidence = float(data.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    return {"confidence": confidence, "event_relation": data.get("event_relation")}


async def analyze(image_bytes: bytes, event_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Call the Anthropic Messages API (vision) to judge event relevance of a photo.
    Returns {'confidence': float, 'event_relation': str|None, 'raw': ...}.
    Uses `CLAUDE_API_KEY` and `CLAUDE_VISION_MODEL` from settings. If not configured, returns confidence 0.
    """
    api_key = settings.CLAUDE_API_KEY
    model = settings.CLAUDE_VISION_MODEL
    if not api_key or not model:
        logger.info("claude_vision: skipped, CLAUDE_API_KEY/CLAUDE_VISION_MODEL not configured")
        return {"confidence": 0.0}

    event_config = event_config or {}
    prompt = _PROMPT_TEMPLATE.format(
        title=event_config.get("title") or "",
        venue=event_config.get("venue_name") or "",
        keywords=", ".join(event_config.get("keywords") or []),
    )

    url = getattr(settings, "CLAUDE_API_URL", None) or ANTHROPIC_API_URL
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": 300,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            text_parts = [
                block.get("text", "")
                for block in data.get("content", [])
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            reply_text = "\n".join(text_parts)
            parsed = _parse_model_json(reply_text)
            logger.info(
                "claude_vision: confidence=%.3f event_relation=%s",
                parsed["confidence"],
                parsed.get("event_relation"),
            )
            return {**parsed, "raw": data}
    except Exception:
        logger.exception("claude_vision: request failed")
        return {"confidence": 0.0}
