from pydantic import BaseModel, validator, ConfigDict
from typing import Optional, Any
from datetime import datetime, date
import json

# Presite Install Check Schemas
class PresiteInstallCheckBase(BaseModel):
    monitor_id: int
    mh_reference: Optional[str] = None
    pipe: Optional[str] = None
    position: Optional[str] = None
    correct_location: Optional[bool] = True
    correct_install_pipe: Optional[bool] = True
    correct_pipe_size: Optional[bool] = True
    correct_pipe_shape: Optional[bool] = True
    comments: Optional[Any] = None

class PresiteInstallCheckCreate(PresiteInstallCheckBase):
    pass

class PresiteInstallCheckRead(PresiteInstallCheckBase):
    id: int
    checked_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

# Weekly Quality Check Schemas
class WeeklyQualityCheckBase(BaseModel):
    monitor_id: int
    check_date: Optional[date] = None
    silt_mm: Optional[int] = None
    comments: Optional[Any] = None
    actions: Optional[Any] = None
    interim: Optional[str] = None
    data_quality_check: Optional[str] = None
    device_status: Optional[str] = None
    device_status_reason: Optional[str] = None

class WeeklyQualityCheckCreate(WeeklyQualityCheckBase):
    pass

class WeeklyQualityCheckUpdate(BaseModel):
    check_date: Optional[date] = None
    silt_mm: Optional[int] = None
    comments: Optional[Any] = None
    actions: Optional[Any] = None
    interim: Optional[str] = None
    data_quality_check: Optional[str] = None
    device_status: Optional[str] = None
    device_status_reason: Optional[str] = None

class WeeklyQualityCheckRead(WeeklyQualityCheckBase):
    id: int
    check_date: Optional[date] = None
    comments: Optional[str] = None
    actions: Optional[str] = None
    data_quality_check: Optional[str] = None
    device_status: Optional[str] = None
    device_status_reason: Optional[str] = None

    @validator('comments', 'actions', pre=True)
    def json_to_string(cls, v):
        if v is None:
            return ""
        if isinstance(v, dict):
            try:
                first_value = next(iter(v.values()))
                return str(first_value) if first_value is not None else ""
            except StopIteration:
                return ""
        if isinstance(v, list):
            return json.dumps(v)
        return str(v) if v is not None else ""

    model_config = ConfigDict(from_attributes=True) 