from PIL import Image, ImageOps, ImageStat, ImageFilter
import io
from app.core.config import settings


def validate_image_bytes(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
    except Exception as e:
        return False, f"invalid image: {e}"
    return True, ""


def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    max_pixels = settings.MIN_IMAGE_SHORT_SIDE * 4
    if max(img.size) > max_pixels:
        ratio = max_pixels / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    mean = stat.mean[0]

    edges = gray.filter(ImageFilter.FIND_EDGES)
    fm = float(ImageStat.Stat(edges).var[0])

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    sanitized = out.getvalue()

    signals = {
        "mean_brightness": mean,
        "blur_score": fm,
    }

    return sanitized, signals
