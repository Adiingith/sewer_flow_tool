from pydoc import text
from prometheus_client import Enum
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, Boolean, ForeignKey, JSON,Identity
from sqlalchemy.orm import declarative_base, relationship
from .base import Base

class Monitor(Base):
    __tablename__ = 'monitor'
    id = Column(Integer, primary_key=True, server_default=Identity())
    monitor_id = Column(Text, nullable=False, unique=True)
    monitor_name = Column(Text, nullable=False)
    type = Column(Text, nullable=False) 
    install_date = Column(TIMESTAMP(timezone=True))
    w3w = Column(Text)
    area = Column(Text , nullable=True)
    location = Column(Text)
    mh_reference = Column(Text)
    pipe = Column(Text)
    height_mm = Column(Integer)
    width_mm = Column(Integer)
    shape = Column(Text)
    depth_mm = Column(Integer)
    status = Column(String, nullable=False)

    # relationships
    install_checks = relationship("PresiteInstallCheck", back_populates="monitor")
    weekly_checks = relationship("WeeklyQualityCheck", back_populates="monitor")
    storm_events = relationship("StormEvent", back_populates="monitor")
    dry_days = relationship("DryDayEvent", back_populates="monitor")
    responsibilities = relationship("ActionResponsibility", back_populates="monitor")
    measurements = relationship("Measurement", back_populates="monitor")