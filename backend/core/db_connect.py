from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from .config import get_settings
from typing import AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

settings = get_settings()

print(f"Using database URL: {settings.DATABASE_URL}")

# Async engine and session
async_engine = create_async_engine(settings.DATABASE_URL, echo=False, pool_size=20, max_overflow=0)
AsyncSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

# Synchronous engine and session for data analysis/model training

def get_sync_engine():
    """
    Create a synchronous SQLAlchemy engine for use in data analysis or model training scripts.
    Make sure to use a sync database URL, e.g., postgresql+psycopg2://
    """
    from .config import get_settings
    settings = get_settings()
    # Replace async driver with sync driver if needed
    sync_db_url = settings.DATABASE_URL.replace('postgresql+asyncpg', 'postgresql+psycopg2')
    return create_engine(sync_db_url)

def get_sync_session():
    """
    Create a synchronous SQLAlchemy session for use in data analysis or model training scripts.
    """
    engine = get_sync_engine()
    Session = sessionmaker(bind=engine)
    return Session()
