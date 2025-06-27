from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, Boolean, ForeignKey, JSON,Identity
from sqlalchemy.orm import declarative_base, relationship
from .base import Base

class PresiteInstallCheck(Base):
    __tablename__ = 'presite_install_check'

    id = Column(Integer, primary_key=True, server_default=Identity())
    monitor_id = Column(Integer, ForeignKey('monitor.id'))
    mh_reference = Column(Text)
    pipe = Column(Text)
    position = Column(Text)
    correct_location = Column(Boolean)
    correct_install_pipe = Column(Boolean)
    correct_pipe_size = Column(Boolean)
    correct_pipe_shape = Column(Boolean)
    comments = Column(JSON)
    checked_at = Column(TIMESTAMP(timezone=True))

    monitor = relationship("Monitor", back_populates="install_checks")