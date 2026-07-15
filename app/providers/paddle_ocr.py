import io
import asyncio
import logging
from typing import Dict, Any
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

_paddle_ocr = None
_paddle_lock = asyncio.Lock()


def _init_paddle(language: str = "korean"):
    global _paddle_ocr
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR
        except Exception as e:
            raise RuntimeError("PaddleOCR is not installed. Install paddleocr and paddlepaddle first.") from e
        
        _paddle_ocr = PaddleOCR(
            lang=language,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            enable_mkldnn=False,
        )
    return _paddle_ocr


def warmup(language: str = "korean") -> None:
    """Load PaddleOCR models and run one throwaway inference so the (slow, blocking)
    cold-start cost happens at process startup instead of during the first request.
    """
    ocr = _init_paddle(language=language)
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    dummy.fill(255)
    list(ocr.predict(dummy))
    logger.info("paddle_ocr: warmup complete")


async def analyze(image_bytes: bytes, event_config: Dict[str, Any] = None, language: str = "korean") -> Dict[str, Any]:
    """Run PaddleOCR on image bytes and return {'text': combined_text, 'raw': result}.
    Runs OCR in a thread via asyncio.to_thread because PaddleOCR is synchronous.
    """
    try:
        async with _paddle_lock:
            ocr = _init_paddle(language=language)
    except Exception:
        logger.exception("PaddleOCR initialization failed")
        return {"text": "", "raw": None}

    def _run():
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            results = ocr.predict(np.array(img))
            texts = []
            scores = []
            for page in results:
                rec_texts = page.get("rec_texts") or []
                rec_scores = page.get("rec_scores") or []
                texts.extend(t for t in rec_texts if t)
                scores.extend(rec_scores)
            combined = " ".join(t.strip() for t in texts if t and t.strip())
            logger.info(
                "paddle_ocr: extracted %d text line(s), %d chars, avg_score=%.3f",
                len(texts),
                len(combined),
                (sum(scores) / len(scores)) if scores else 0.0,
            )
            return {"text": combined, "raw": {"rec_texts": texts, "rec_scores": scores}}
        except Exception:
            logger.exception("PaddleOCR predict() failed")
            return {"text": "", "raw": None}

    return await asyncio.to_thread(_run)
