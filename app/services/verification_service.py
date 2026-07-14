import logging
from typing import Dict, Any
from app.services.image_processor import validate_image_bytes, preprocess_image
from app.rule_engine import decide
from app.core.config import settings
from app.domain.enums import VerificationStatus
import math

# AI providers
from app.providers.paddle_ocr import analyze as paddle_analyze
from app.providers.gemini_vision import analyze as gemini_analyze
from app.providers.claude_vision import analyze as claude_analyze
import asyncio

logger = logging.getLogger(__name__)


class AIProvidersUnavailableError(Exception):
    """Raised when OCR and Vision both fail outright (timeout/exception), as
    opposed to legitimately running and finding no match. Callers should
    surface this as a retryable server error rather than a verification
    rejection, since we have no real signal to judge the photo on."""


class VerificationService:
    async def verify_event_image(self, event_payload: Dict[str, Any], image_bytes: bytes) -> Dict[str, Any]:
        file_valid, file_error = validate_image_bytes(image_bytes)
        img_signals = {}
        if file_valid:
            sanitized, img_signals = preprocess_image(image_bytes)
        else:
            sanitized = None

        # AI analysis: OCR (Paddle) and Vision (Claude primary, Gemini fallback)
        ocr_task = None
        vision_task = None
        if file_valid:
            ocr_task = asyncio.create_task(
                paddle_analyze(sanitized or image_bytes, event_payload, language=getattr(settings, 'OCR_LANGUAGE', 'korean'))
            )
            vision_task = asyncio.create_task(claude_analyze(sanitized or image_bytes, event_payload))

        ocr_res = {"text": ""}
        vision_res = {"confidence": 0.0}
        ocr_failed = False
        vision_failed = False
        if ocr_task:
            try:
                ocr_res = await asyncio.wait_for(ocr_task, timeout=45.0)
            except Exception:
                logger.exception("OCR task failed or timed out")
                ocr_res = {"text": ""}
                ocr_failed = True
        logger.info("signal[ocr]: text_len=%d text=%r", len(ocr_res.get("text") or ""), (ocr_res.get("text") or "")[:200])

        if vision_task:
            try:
                vision_res = await asyncio.wait_for(vision_task, timeout=12.0)
                if (vision_res.get("confidence", 0.0) or 0.0) < settings.VISION_MEDIUM_CONFIDENCE:
                    try:
                        g_res = await asyncio.wait_for(gemini_analyze(sanitized or image_bytes, event_payload), timeout=12.0)
                        if (g_res.get("confidence", 0.0) or 0.0) > (vision_res.get("confidence", 0.0) or 0.0):
                            vision_res = g_res
                    except Exception:
                        # Claude already produced a real (if low-confidence) result, so
                        # this is not a total vision outage - just no fallback available.
                        logger.exception("Gemini fallback vision task failed or timed out")
            except Exception:
                logger.exception("Claude vision task failed or timed out, falling back to Gemini")
                try:
                    vision_res = await asyncio.wait_for(gemini_analyze(sanitized or image_bytes, event_payload), timeout=20.0)
                except Exception:
                    logger.exception("Gemini vision task failed or timed out")
                    vision_res = {"confidence": 0.0}
                    vision_failed = True
        logger.info("signal[vision]: confidence=%.3f event_relation=%s", vision_res.get("confidence", 0.0) or 0.0, vision_res.get("event_relation"))

        # If the image itself is readable but every AI provider (OCR + both vision
        # fallbacks) failed outright, we have no real signal to judge on - that's an
        # infrastructure outage, not "insufficient evidence" the user caused.
        if file_valid and ocr_failed and vision_failed:
            raise AIProvidersUnavailableError("OCR and Vision providers all failed or timed out")

        quality_signals = {"ok": True}
        if file_valid:
            quality_signals["too_blurry"] = img_signals.get("blur_score", 0.0) < settings.BLUR_THRESHOLD
            quality_signals["too_dark"] = img_signals.get("mean_brightness", 0.0) < settings.DARKNESS_THRESHOLD
            quality_signals["ok"] = not (quality_signals["too_blurry"] or quality_signals["too_dark"])
        else:
            quality_signals["ok"] = False
        logger.info("signal[file]: valid=%s error=%r", file_valid, file_error)
        logger.info("signal[quality]: %s", quality_signals)

        signals = {
            "file": {"valid": file_valid, "error": file_error},
            "quality": quality_signals,
            "ocr": ocr_res,
            "vision": vision_res,
        }

        # Location/distance signals: if event contains venue/user coords, compute distance
        try:
            vlat = float(event_payload.get("venue_lat")) if event_payload.get("venue_lat") is not None else None
            vlng = float(event_payload.get("venue_lng")) if event_payload.get("venue_lng") is not None else None
            ulat = float(event_payload.get("user_lat")) if event_payload.get("user_lat") is not None else None
            ulng = float(event_payload.get("user_lng")) if event_payload.get("user_lng") is not None else None
            radius = float(event_payload.get("radius_m")) if event_payload.get("radius_m") is not None else None
        except Exception:
            vlat = vlng = ulat = ulng = radius = None

        if vlat is not None and vlng is not None and ulat is not None and ulng is not None:
            # haversine distance in meters
            def haversine(lat1, lon1, lat2, lon2):
                R = 6371000.0
                phi1 = math.radians(lat1)
                phi2 = math.radians(lat2)
                dphi = math.radians(lat2 - lat1)
                dlambda = math.radians(lon2 - lon1)
                a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                return R * c

            dist_m = haversine(vlat, vlng, ulat, ulng)
            within = True if (radius is None) else (dist_m <= radius)
            signals["location"] = {
                "venue": {"lat": vlat, "lng": vlng},
                "user": {"lat": ulat, "lng": ulng},
                "distance_m": dist_m,
                "within_radius": within,
                "radius_m": radius,
            }

        if "location" in signals:
            logger.info("signal[location]: %s", signals["location"])

        decision = decide(signals, config=event_payload)
        logger.info("decision: status=%s confidence=%.3f reasons=%s",
                     decision["status"], decision.get("confidence", 0.0), decision.get("reason_codes"))

        confidence = decision.get("confidence", 0.0)

        status = decision["status"]
        if status == VerificationStatus.VERIFIED.value:
            user_message = "방문 인증이 완료되었습니다!"
        elif status == VerificationStatus.ADDITIONAL_CAPTURE_REQUIRED.value:
            user_message = "사진을 다시 촬영해주세요."
        else:
            user_message = "인증에 실패했습니다."

        return {
            "status": status,
            "confidence": confidence,
            "reasons": decision.get("reason_codes", []),
            "user_message": user_message,
        }
