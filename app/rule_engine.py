import logging
import re
from typing import Dict, Any, List
from app.domain.enums import VerificationStatus, ReasonCode
from app.core.config import settings

logger = logging.getLogger(__name__)

# Word/number tokens (Korean + Latin + digits). Splitting on this means
# hyphens, punctuation, and inconsistent whitespace never break a match.
_TOKEN_RE = re.compile(r"[0-9a-zA-Z가-힣]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _keyword_match_score(keyword: str, ocr_text: str) -> float:
    """Fraction of a keyword phrase's tokens found in the OCR text.

    Matching the *entire* phrase as one exact substring is too brittle: OCR
    line-segmentation can put different whitespace between words than the
    original phrase, and a single misread character used to zero out an
    otherwise-correct long phrase. Scoring per-token instead means a phrase
    only loses credit for the words that actually failed to read.
    """
    tokens = _tokenize(keyword)
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in ocr_text)
    return hits / len(tokens)


def decide(signals: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    reason_codes: List[str] = []

    # An unreadable file outranks every other signal: if we can't even decode
    # the image, distance/quality/AI signals about it are meaningless.
    if not signals.get("file", {}).get("valid", True):
        reason_codes.append(ReasonCode.INVALID_FILE.value)
        return {"status": VerificationStatus.REJECTED.value, "reason_codes": reason_codes, "confidence": 0.0}

    # If location signals are provided, reject immediately when out of radius
    loc = signals.get("location")
    if loc is not None:
        if not loc.get("within_radius", True):
            reason_codes.append(ReasonCode.OUT_OF_RADIUS.value)
            return {"status": VerificationStatus.REJECTED.value, "reason_codes": reason_codes, "confidence": 0.0}

    quality = signals.get("quality", {})
    if not quality.get("ok", True):
        if quality.get("too_blurry"):
            reason_codes.append(ReasonCode.IMAGE_TOO_BLURRY.value)
        if quality.get("too_dark"):
            reason_codes.append(ReasonCode.IMAGE_TOO_DARK.value)
        return {"status": VerificationStatus.ADDITIONAL_CAPTURE_REQUIRED.value, "reason_codes": reason_codes, "confidence": 0.0}

    ocr = signals.get("ocr", {})
    vision = signals.get("vision", {})

    ocr_text = (ocr.get("text") or "").lower()
    event = config or {}
    keywords = [k for k in (event.get("keywords") or []) if k]
    title = event.get("title") or ""
    if title:
        keywords.append(title)

    ocr_score = 0.0
    keyword_scores: Dict[str, float] = {}
    if keywords:
        keyword_scores = {kw: _keyword_match_score(kw, ocr_text) for kw in keywords}
        ocr_score = sum(keyword_scores.values()) / len(keyword_scores)

    vision_conf = vision.get("confidence") or 0.0
    vision_relation = vision.get("event_relation")

    # vision_conf comes straight from an external model's JSON reply and isn't
    # guaranteed to stay within [0, 1], so clamp before it reaches the API response.
    confidence = max(0.0, min(1.0, max(ocr_score, vision_conf)))
    logger.info(
        "rule_engine: ocr_score=%.3f keyword_scores=%s vision_conf=%.3f vision_relation=%s",
        ocr_score, {k: round(v, 2) for k, v in keyword_scores.items()}, vision_conf, vision_relation,
    )

    if ocr_score >= settings.OCR_STRONG_MATCH_SCORE:
        reason_codes.append(ReasonCode.OCR_STRONG_MATCH.value)
        return {"status": VerificationStatus.VERIFIED.value, "reason_codes": reason_codes, "confidence": confidence}

    if vision_relation == "STRONGLY_RELATED" or vision_conf >= settings.VISION_STRONG_CONFIDENCE:
        reason_codes.append(ReasonCode.VISION_STRONG_MATCH.value)
        return {"status": VerificationStatus.VERIFIED.value, "reason_codes": reason_codes, "confidence": confidence}

    if ocr_score >= settings.OCR_MEDIUM_MATCH_SCORE and vision_conf >= settings.VISION_MEDIUM_CONFIDENCE:
        reason_codes.append(ReasonCode.OCR_MEDIUM_MATCH.value)
        reason_codes.append(ReasonCode.VISION_MEDIUM_MATCH.value)
        return {"status": VerificationStatus.VERIFIED.value, "reason_codes": reason_codes, "confidence": confidence}

    if vision_conf >= settings.VISION_MEDIUM_CONFIDENCE:
        reason_codes.append(ReasonCode.VISION_MEDIUM_MATCH.value)
        return {"status": VerificationStatus.VERIFIED.value, "reason_codes": reason_codes, "confidence": confidence}

    if ocr_score >= settings.OCR_MEDIUM_MATCH_SCORE:
        reason_codes.append(ReasonCode.OCR_MEDIUM_MATCH.value)
        return {"status": VerificationStatus.VERIFIED.value, "reason_codes": reason_codes, "confidence": confidence}

    reason_codes.append(ReasonCode.INSUFFICIENT_EVIDENCE.value)
    return {"status": VerificationStatus.ADDITIONAL_CAPTURE_REQUIRED.value, "reason_codes": reason_codes, "confidence": confidence}
