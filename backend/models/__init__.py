"""
数据库模型模块

包含所有的 SQLAlchemy 模型类，用于定义数据库表结构和关系。
"""

from .base import Base
from .measurement import Measurement
from .StormEvent import StormEvent
from .DryDayEvent import DryDayEvent
from .monitor import Monitor
from .WeeklyQualityCheck import WeeklyQualityCheck
from .presiteInstallCheck import PresiteInstallCheck
from .ActionResponsibility import ActionResponsibility
# 导出所有模型类
__all__ = [
    "Base",
    "Measurement",
    "StormEvent",
    "DryDayEvent",
    "Monitor",
    "WeeklyQualityCheck",
    "PresiteInstallCheck",
    "ActionResponsibility"
]

# 确保所有模型都被导入，这样 Alembic 才能检测到它们
# 这个导入顺序很重要，因为存在外键关系