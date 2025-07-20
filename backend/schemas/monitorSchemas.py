from pydantic import BaseModel, ConfigDict, field_validator
from typing import Optional, List, Dict
from datetime import datetime, date

class MonitorSchema(BaseModel):
    id: int
    monitor_id: str
    monitor_name: str
    type: str
    install_date: Optional[datetime] = None
    w3w: Optional[str] = None
    area: Optional[str] = None
    location: Optional[str] = None
    mh_reference: Optional[str] = None
    pipe: Optional[str] = None
    height_mm: Optional[int] = None
    width_mm: Optional[int] = None
    shape: Optional[str] = None
    depth_mm: Optional[int] = None
    status: str
    status_reason: Optional[str] = None
    action: Optional[str] = ''

    model_config = ConfigDict(from_attributes=True)

class DashboardSummary(BaseModel):
    removal_count: int
    retain_count: int
    category_counts: Dict[str, int]
    category_removal_counts: Dict[str, int]

class ActionResponsibilityBase(BaseModel):
    requester: Optional[str] = None
    removal_checker: Optional[str] = None
    removal_reviewer: Optional[str] = None
    removal_date: Optional[datetime] = None
    action: Optional[str] = None

class ActionResponsibilityCreate(ActionResponsibilityBase):
    monitor_id: int

class ActionResponsibilityUpdate(ActionResponsibilityCreate):
    pass

class ActionResponsibilitySchema(ActionResponsibilityBase):
    id: Optional[int] = None
    monitor_id: int

    model_config = ConfigDict(from_attributes=True)

class ActionResponsibilityBulkUpdateItem(BaseModel):
    id: int
    monitor_id: int
    requester: Optional[str] = None
    removal_checker: Optional[str] = None
    removal_reviewer: Optional[str] = None
    removal_date: Optional[date] = None
    action: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

    @field_validator('removal_date', mode='before')
    @classmethod
    def parse_removal_date(cls, v):
        if isinstance(v, str) and v.strip():
            try:
                # Handle full datetime strings (e.g., from toISOString)
                return datetime.fromisoformat(v.replace('Z', '+00:00')).date()
            except ValueError:
                # Handle 'YYYY-MM-DD' strings
                return date.fromisoformat(v)
        if isinstance(v, datetime):
            return v.date()
        return None

class MonitorDetailSchema(MonitorSchema):
    responsibilities: List[ActionResponsibilitySchema] = []

    model_config = ConfigDict(from_attributes=True)

class MonitorPage(BaseModel):
    total: int
    page: int
    limit: int
    data: List[MonitorSchema] 

class WeatherEventSchema(BaseModel):
    id: int
    event_type: str
    start_time: datetime
    end_time: datetime
    area: str
    interim: str
    storm_type: Optional[str]
    coverage: Optional[str]
    event_comment: Optional[Dict]
    rain_gauge_monitor_id: int

    model_config = ConfigDict(from_attributes=True) 