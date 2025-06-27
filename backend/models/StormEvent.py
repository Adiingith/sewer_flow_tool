from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, Boolean, ForeignKey, JSON,Identity
from sqlalchemy.orm import declarative_base, relationship
from .base import Base

class StormEvent(Base):
    __tablename__ = 'storm_event'

    id = Column(Integer, primary_key=True, server_default=Identity())
    monitor_id = Column(Integer, ForeignKey('monitor.id'))
    storm_type = Column(String, nullable=False)  # should be 'A', 'B', 'C'
    event_date = Column(TIMESTAMP(timezone=True))
    event_comment = Column(JSON)
    coverage = Column(Text)

    monitor = relationship("Monitor", back_populates="storm_events")