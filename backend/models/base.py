from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

Base = declarative_base()

# if needed, add common model methods here
class BaseModel:
    """Base model class, providing common methods"""
    
    def to_dict(self):
        """Convert model instance to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
    
    def __repr__(self):
        """String representation"""
        return f"<{self.__class__.__name__}(id={getattr(self, 'id', 'N/A')})>"