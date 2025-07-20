"""
database model module

contains all SQLAlchemy model classes, used to define database table structure and relationships.
"""

from .base import Base
from .measurement import Measurement
from .weatherEvent import WeatherEvent
from .monitor import Monitor
from .WeeklyQualityCheck import WeeklyQualityCheck
from .presiteInstallCheck import PresiteInstallCheck
from .ActionResponsibility import ActionResponsibility
from .rain_gauge import RainGauge

__all__ = [
    "Base",
    "Measurement",
    "WeatherEvent",
    "Monitor",
    "WeeklyQualityCheck",
    "PresiteInstallCheck",
    "ActionResponsibility",
    "RainGauge"
]

# ensure all models are imported, so Alembic can detect them
# this import order is important, because there are foreign key relationships