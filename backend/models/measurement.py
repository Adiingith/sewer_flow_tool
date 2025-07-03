from sqlalchemy import Column, Integer, Numeric, PrimaryKeyConstraint, Text, DateTime, ForeignKey, Identity
from sqlalchemy.orm import relationship
from .base import Base

class Measurement(Base):
    __tablename__ = 'measurement'
    
    
    time = Column(DateTime(timezone=True), nullable=False)  # timestamp with time zone
    monitor_id = Column(Integer, ForeignKey('monitor.id'), nullable=False)    
    depth = Column(Numeric, nullable=True)  # numeric type
    flow = Column(Numeric, nullable=True)   # numeric type
    quality_flags = Column(Text, nullable=True)
    interim = Column(Text, nullable=True)  # new interim field
    velocity = Column(Numeric, nullable=True)  # new velocity field

    __table_args__ = (
        PrimaryKeyConstraint("time", "monitor_id"),
    )   
    # relationship definition - multiple measurements belong to one device
    monitor = relationship("Monitor", back_populates="measurements")
    
    def __repr__(self):
        return f"<Measurement(id={self.id}, monitor_id={self.monitor_id}, depth={self.depth}, flow={self.flow}, time='{self.time}')>"
    
    def to_dict(self):
        """convert model instance to dictionary"""
        return {
            'id': self.id,
            'depth': float(self.depth) if self.depth else None,
            'flow': float(self.flow) if self.flow else None,
            'velocity': float(self.velocity) if self.velocity else None,
            'quality_flags': self.quality_flags,
            'time': self.time.isoformat() if self.time else None,
            'monitor_id': self.monitor_id,
            'interim': self.interim
        }