import asyncio
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import text
from backend.core.db_connect import AsyncSessionLocal

async def refresh_product_view():
    async with AsyncSessionLocal() as session:
        try:
            await session.execute(text("REFRESH MATERIALIZED VIEW product_view"))
            await session.commit()
            print(" product_view refreshed")
        except Exception as e:
            print(" Failed to refresh view:", e)
            await session.rollback()

def refresh_product_view_sync():
    """同步包装器，用于调度器调用"""
    asyncio.run(refresh_product_view())

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(refresh_product_view_sync, 'interval', hours=1)
    scheduler.start()
