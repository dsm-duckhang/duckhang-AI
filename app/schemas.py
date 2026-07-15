from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from typing import List, Optional

from app.domain.enums import ReasonCode, VerificationStatus


class EventPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    category: str
    title: str
    category_label: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("categoryLabel", "category_label"),
    )
    description: Optional[str] = None
    artist_name: Optional[str] = None
    venue_name: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("venueName", "venue_name"),
    )
    address: Optional[str] = None
    related_link: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("relatedLink", "related_link"),
    )
    keywords: Optional[List[str]] = None
    venue_lat: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("latitude", "venue_lat"),
    )
    venue_lng: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("longitude", "venue_lng"),
    )
    radius_m: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("radiusM", "radius_m"),
    )
    user_lat: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("userLatitude", "user_lat"),
    )
    user_lng: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("userLongitude", "user_lng"),
    )
    accuracy: Optional[float] = None
    start_at: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("startAt", "start_at"),
    )
    end_at: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("endAt", "end_at"),
    )

    @model_validator(mode="after")
    def populate_default_keywords(self):
        if not self.keywords:
            self.keywords = [self.venue_name] if self.venue_name else None
        return self


class VerifyResponse(BaseModel):
    status: VerificationStatus
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: List[ReasonCode]
    user_message: Optional[str] = None
