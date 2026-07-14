from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.schemas import EventPayload, VerifyResponse
from app.services.verification_service import VerificationService, AIProvidersUnavailableError

router = APIRouter()


@router.post("/verify", response_model=VerifyResponse)
async def verify(event: str = Form(...), image: UploadFile = File(...)):
    try:
        event_payload = EventPayload.model_validate_json(event)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid event payload: {exc}")

    try:
        image_bytes = await image.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"cannot read image file: {exc}")

    svc = VerificationService()
    try:
        return await svc.verify_event_image(event_payload.model_dump(), image_bytes)
    except AIProvidersUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
