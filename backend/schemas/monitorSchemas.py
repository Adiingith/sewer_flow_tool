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

    class Config:
        orm_mode = True

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
    action_type: Optional[str] = None

class ActionResponsibilityCreate(ActionResponsibilityBase):
    monitor_id: int

class ActionResponsibilityUpdate(ActionResponsibilityCreate):
    pass

class ActionResponsibilitySchema(ActionResponsibilityBase):
    id: Optional[int] = None
    monitor_id: int

    class Config:
        orm_mode = True

class ActionResponsibilityBulkUpdateItem(BaseModel):
    id: int
    action_type: Optional[str] = None
    requester: Optional[str] = None
    removal_checker: Optional[str] = None
    removal_reviewer: Optional[str] = None
    removal_date: Optional[date] = None

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

class MonitorPage(BaseModel):
    total: int
    page: int
    limit: int
    data: List[MonitorSchema] 