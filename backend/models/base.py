from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

Base = declarative_base()

# 如果需要，可以在这里添加通用的模型方法
class BaseModel:
    """基础模型类，提供通用方法"""
    
    def to_dict(self):
        """将模型实例转换为字典"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
    
    def __repr__(self):
        """字符串表示"""
        return f"<{self.__class__.__name__}(id={getattr(self, 'id', 'N/A')})>"