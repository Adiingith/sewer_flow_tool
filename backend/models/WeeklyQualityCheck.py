from sqlalchemy import Column, Identity, Integer, String, Text, TIMESTAMP, Boolean, ForeignKey, JSON, Date
from sqlalchemy.orm import declarative_base, relationship
from .base import Base


class WeeklyQualityCheck(Base):
    __tablename__ = 'weekly_quality_check'

    id = Column(Integer, primary_key=True, server_default=Identity())
    monitor_id = Column(Integer, ForeignKey('monitor.id'))
    check_date = Column(Date)
    silt_mm = Column(Integer)
    comments = Column(JSON)
    actions = Column(JSON)
    interim = Column(Text)

    monitor = relationship("Monitor", back_populates="weekly_checks")