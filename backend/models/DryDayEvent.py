from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, Boolean, ForeignKey, JSON,Identity
from sqlalchemy.orm import declarative_base, relationship
from .base import Base


class DryDayEvent(Base):
    __tablename__ = 'dry_day_event'

    id = Column(Integer, primary_key=True, server_default=Identity())
    monitor_id = Column(Integer, ForeignKey('monitor.id'))
    dry_day_number = Column(Integer)
    event_date = Column(TIMESTAMP(timezone=True))
    event_comment = Column(JSON)

    monitor = relationship("Monitor", back_populates="dry_days")