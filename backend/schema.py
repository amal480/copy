from datetime import datetime, UTC
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class AnomalyBase(BaseModel):
    """Base schema for anomaly detection records"""
    session_id: str = Field(..., description="Unique identifier for the proctoring session")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="When the anomaly was detected")
    type: str = Field(..., description="Type of anomaly detected")

class CellPhoneAnomaly(AnomalyBase):
    """Schema for cell phone detection anomalies"""
    type: str = "cell_phone"
    count: int = Field(..., description="Number of cell phones detected")

class MultiplePeopleAnomaly(AnomalyBase):
    """Schema for multiple people detection anomalies"""
    type: str = "multiple_people"
    count: int = Field(..., description="Number of people detected")

class HeadDirectionAnomaly(AnomalyBase):
    """Schema for head direction anomalies"""
    type: str = "head_direction"
    direction: str = Field(..., description="Direction the head is facing")

class GazeDirectionAnomaly(AnomalyBase):
    """Schema for gaze direction anomalies"""
    type: str = "gaze_direction"
    direction: str = Field(..., description="Direction of gaze")

class SpeechAnomaly(AnomalyBase):
    """Schema for speech detection anomalies"""
    type: str = "speech"

# Union type for all possible anomaly types
Anomaly = CellPhoneAnomaly | MultiplePeopleAnomaly | HeadDirectionAnomaly | GazeDirectionAnomaly | SpeechAnomaly

# MongoDB collection schemas
class AnomalyCollection(BaseModel):
    """Schema for the anomalies collection"""
    collection_name: str = "anomalies"
    indexes: List[Dict[str, Any]] = [
        {"keys": {"session_id": 1}, "name": "session_id_idx"},
        {"keys": {"timestamp": 1}, "name": "timestamp_idx"},
        {"keys": {"type": 1}, "name": "type_idx"}
    ] 