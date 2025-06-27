from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, Boolean, ForeignKey, JSON,Identity
from sqlalchemy.orm import declarative_base, relationship
from .base import Base



class ActionResponsibility(Base):
    __tablename__ = 'action_responsibility'

    id = Column(Integer, primary_key=True, server_default=Identity())
    monitor_id = Column(Integer, ForeignKey('monitor.id'))
    requester = Column(Text)
    removal_checker = Column(Text)
    removal_reviewer = Column(Text)
    removal_date = Column(TIMESTAMP(timezone=True))
    action_type = Column(Text)

    monitor = relationship("Monitor", back_populates="responsibilities")