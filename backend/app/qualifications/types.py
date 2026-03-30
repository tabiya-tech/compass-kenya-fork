import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class QualificationType(str, Enum):
    CERTIFICATE = "CERTIFICATE"
    DIPLOMA = "DIPLOMA"
    DEGREE = "DEGREE"
    TRADE_LICENSE = "TRADE_LICENSE"
    PROFESSIONAL_LICENSE = "PROFESSIONAL_LICENSE"
    TRAINING_COMPLETION = "TRAINING_COMPLETION"
    OTHER = "OTHER"


class QualificationEntity(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    qualification_type: QualificationType
    name: str
    institution: Optional[str] = None
    date_obtained: Optional[str] = None
    expiry_date: Optional[str] = None
    level: Optional[str] = None
    field_of_study: Optional[str] = None
    source: str = "conversation"  # "cv" | "conversation"
