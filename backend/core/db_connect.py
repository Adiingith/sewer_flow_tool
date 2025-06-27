from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from .config import get_settings
from typing import AsyncGenerator

settings = get_settings()

print(f"Using database URL: {settings.DATABASE_URL}")

# 异步引擎和会话
async_engine = create_async_engine(settings.DATABASE_URL, echo=False, pool_size=20, max_overflow=0)
AsyncSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
