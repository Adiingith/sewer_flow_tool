from sqlalchemy import Column, Integer, Numeric, PrimaryKeyConstraint, Text, DateTime, ForeignKey, Identity
from sqlalchemy.orm import relationship
from .base import Base

class Measurement(Base):
    __tablename__ = 'measurement'
    
    # 添加主键
    time = Column(DateTime(timezone=True), nullable=False)  # timestamp with time zone
    monitor_id = Column(Integer, ForeignKey('monitor.id'), nullable=False)    
    depth = Column(Numeric, nullable=True)  # numeric 类型
    flow = Column(Numeric, nullable=True)   # numeric 类型
    quality_flags = Column(Text, nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("time", "monitor_id"),
    )   
    # 关系定义 - 多个测量数据属于一个设备
    monitor = relationship("Monitor", back_populates="measurements")
    
    def __repr__(self):
        return f"<Measurement(id={self.id}, monitor_id={self.monitor_id}, depth={self.depth}, flow={self.flow}, time='{self.time}')>"
    
    def to_dict(self):
        """将模型实例转换为字典"""
        return {
            'id': self.id,
            'depth': float(self.depth) if self.depth else None,
            'flow': float(self.flow) if self.flow else None,
            'quality_flags': self.quality_flags,
            'time': self.time.isoformat() if self.time else None,
            'monitor_id': self.monitor_id
        }