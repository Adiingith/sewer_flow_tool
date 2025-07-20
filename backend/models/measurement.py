from sqlalchemy import Column, Integer, Numeric, PrimaryKeyConstraint, Text, DateTime, ForeignKey, Identity
from sqlalchemy.orm import relationship
from .base import Base
from decimal import Decimal

class Measurement(Base):
    __tablename__ = 'measurement'
    
    
    time = Column(DateTime(timezone=True), nullable=False)  # timestamp with time zone
    monitor_id = Column(Integer, ForeignKey('monitor.id'), nullable=False)    
    depth = Column(Numeric, nullable=True)  # numeric type
    flow = Column(Numeric, nullable=True)   # numeric type
    quality_flags = Column(Text, nullable=True)
    interim = Column(Text, nullable=False)  # new interim field
    velocity = Column(Numeric, nullable=True)  # new velocity field

    __table_args__ = (
        PrimaryKeyConstraint("time", "monitor_id", "interim"),
    )   
    # relationship definition - multiple measurements belong to one device
    monitor = relationship("Monitor", back_populates="measurements")
    
    def __repr__(self):
        return f"<Measurement(id={self.id}, monitor_id={self.monitor_id}, depth={self.depth}, flow={self.flow}, time='{self.time}')>"
    
    def to_dict(self):
        """convert model instance to dictionary"""
        def safe_float(value):
            """Safely convert value to float, handling None, empty strings, and Decimal objects"""
            if value is None:
                return None
            if isinstance(value, str) and value.strip() == '':
                return None
            if isinstance(value, Decimal):
                return float(value)
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
        
        return {
            # 'id': self.id,  # Removed because Measurement has no id field
            'depth': safe_float(self.depth),
            'flow': safe_float(self.flow),
            'velocity': safe_float(self.velocity),
            'quality_flags': self.quality_flags,
            'time': self.time.isoformat() if self.time else None,
            'monitor_id': self.monitor_id,
            'interim': self.interim
        }