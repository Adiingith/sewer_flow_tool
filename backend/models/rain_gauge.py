from sqlalchemy import Column, Integer, Numeric, DateTime, Text, ForeignKey, PrimaryKeyConstraint, func, Identity
from sqlalchemy.orm import relationship
from .base import Base

class RainGauge(Base):
    __tablename__ = 'rain_gauge'

    monitor_id = Column(Integer, ForeignKey('monitor.id'), nullable=False)  
    timestamp = Column(DateTime(timezone=True), nullable=False)
    intensity_mm_per_hr = Column(Numeric, nullable=True)
    interim = Column(Text, nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('monitor_id', 'timestamp', 'interim'),
    )

    # relationship: many rain_gauge data belong to one monitor
    monitor = relationship("Monitor", back_populates="rain_gauges")

    def __repr__(self):
        return f"<RainGauge(id={self.id}, monitor_id={self.monitor_id}, timestamp='{self.timestamp}', intensity_mm_per_hr={self.intensity_mm_per_hr})>"

    def to_dict(self):
        return {
            # 'id': self.id,  # Removed because RainGauge has no id field
            'monitor_id': self.monitor_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'intensity_mm_per_hr': float(self.intensity_mm_per_hr) if self.intensity_mm_per_hr is not None else None,
            'interim': self.interim
        } 