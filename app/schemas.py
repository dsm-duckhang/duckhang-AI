from pydantic import BaseModel
from typing import List, Optional


class EventPayload(BaseModel):
    category: str
    title: str
    artist_name: Optional[str] = None
    venue_name: Optional[str] = None
    keywords: Optional[List[str]] = None
    # Optional venue location and user location fields. These are expected
    # to be provided inside the `event` JSON when the caller wants FastAPI
    # to compute distance and validate proximity.
    venue_lat: Optional[float] = None
    venue_lng: Optional[float] = None
    radius_m: Optional[float] = None

    # user location reported by client/front-end (optional)
    user_lat: Optional[float] = None
    user_lng: Optional[float] = None
    accuracy: Optional[float] = None


class VerifyResponse(BaseModel):
    status: str
    confidence: float
    reasons: List[str]
    user_message: Optional[str] = None
