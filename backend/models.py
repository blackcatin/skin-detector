from sqlalchemy import (
    Column,
    Integer,
    Numeric,
    String,
    Float,
    DateTime,
    Boolean
)
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)

    # Image Paths
    image_path = Column(String, nullable=False)
    processed_image_path = Column(String, nullable=True)

    # Prediction Result
    predicted_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)

    # Image Quality Metadata
    blur_score = Column(Numeric(10, 2))
    brightness_score = Column(Numeric(10, 2))
    quality_score = Column(Numeric(10, 2))

    # Dataset Eligibility
    is_valid = Column(Boolean, default=True)

    # Retraining Tracking
    used_for_retraining = Column(Boolean, default=False)

    # Model Version Tracking
    model_version = Column(String, nullable=False)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)