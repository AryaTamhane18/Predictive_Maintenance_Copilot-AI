from pydantic import BaseModel
from typing import Optional
from enum import Enum


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SensorReading(BaseModel):
    machine_id: str
    timestamp: str
    temperature: float
    vibration: float
    pressure: float
    rotation_speed: float
    tool_wear: float


class AnomalyResult(BaseModel):
    machine_id: str
    file_index: int
    total_files: int
    is_anomalous: bool
    channel: str
    max_z_score: float
    anomaly_count: int
    total_readings: int
    severity: str
    timestamp_name: str
    message: str


class ManualSearchResult(BaseModel):
    relevant_section: str
    page_number: Optional[int] = None
    confidence: float


class FaultReport(BaseModel):
    machine_id: str
    timestamp: str
    fault_type: str
    severity: SeverityLevel
    recommended_action: str
    cited_manual_section: Optional[str] = None
    sensor_readings: Optional[SensorReading] = None
    anomaly_details: Optional[AnomalyResult] = None
    approved: bool = False