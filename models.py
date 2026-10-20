from pydantic import BaseModel
from typing import Optional
from enum import Enum


class SeverityLevel(str, Enum):
    NONE     = "none"
    MINOR    = "minor"
    MAJOR    = "major"
    CRITICAL = "critical"
    # LOW = "LOW"
    # MEDIUM = "MEDIUM"
    # HIGH = "HIGH"
    # CRITICAL = "CRITICAL"


# class SensorReading(BaseModel):
#     machine_id: str
#     timestamp: str
#     temperature: float
#     vibration: float
#     pressure: float
#     rotation_speed: float
#     tool_wear: float


class AnomalyResult(BaseModel):
    machine_id: str
    file_index: int
    total_files: int
    is_anomalous: bool
    channel: str
    deviation_pct: float
    anomaly_count: int
    total_readings: int
    severity: SeverityLevel
    timestamp_name: str
    message: str


class ManualSearchResult(BaseModel):
    relevant_section: str
    page_number: Optional[int] = None
    confidence: float


class FaultReport(BaseModel):
    asset_id: str
    machine_id: str
    file_index: int
    timestamp_name: str
    severity: SeverityLevel
    fault_type: str
    root_cause: str
    recommended_action: str
    # cited_manual_section: Optional[str] = None
    # sensor_readings: Optional[SensorReading] = None
    # anomaly_details: Optional[AnomalyResult] = None
    # approved: bool = False
    manual_reference: Optional[str] = None
    sensor_summary: Optional[str] = None
    analysis_mode: str = "sensor_only"
    confidence: Optional[str] = None
    caveat: Optional[str] = None
    approved: Optional[bool] = None
