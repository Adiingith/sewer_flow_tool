import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import aliased
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case
from backend.core.db_connect import get_session
from backend.models.monitor import Monitor
from backend.models.ActionResponsibility import ActionResponsibility
from backend.schemas.monitorSchemas import MonitorSchema, DashboardSummary
from typing import List
from datetime import datetime, timedelta

# Add logger configuration
logger = logging.getLogger(__name__)

router = APIRouter()

# 固定路径接口优先
@router.get("/visualization/dashboard_summary", response_model=DashboardSummary, tags=["Monitors"])
async def get_dashboard_summary(db: AsyncSession = Depends(get_session)):
    logger.info("Request received for dashboard summary.")
    latest_resp_subquery = (
        select(
            ActionResponsibility.monitor_id,
            func.max(ActionResponsibility.id).label("latest_id"),
        )
        .group_by(ActionResponsibility.monitor_id)
        .subquery()
    )
    ar_alias = aliased(ActionResponsibility)
    query = (
        select(
            func.count(Monitor.id).label("total_monitors"),
            func.sum(
                case(
                    (func.lower(ar_alias.action_type) == 'removal', 1),
                    else_=0
                )
            ).label("removal_count"),
            func.sum(
                case(
                    (func.lower(ar_alias.action_type) == 'removal', 0),
                    else_=1 # If not 'removal', it's 'retain' (includes NULL)
                )
            ).label("retain_count"),
            func.sum(case((Monitor.monitor_name.startswith('FM'), 1), else_=0)).label("fm_count"),
            func.sum(case((Monitor.monitor_name.startswith('DM'), 1), else_=0)).label("dm_count"),
            func.sum(case((Monitor.monitor_name.startswith('PL'), 1), else_=0)).label("pl_count"),
            func.sum(case((Monitor.monitor_name.startswith('RG'), 1), else_=0)).label("rg_count"),
        )
        .select_from(Monitor)
        .outerjoin(latest_resp_subquery, Monitor.id == latest_resp_subquery.c.monitor_id)
        .outerjoin(ar_alias, (Monitor.id == ar_alias.monitor_id) & (latest_resp_subquery.c.latest_id == ar_alias.id))
        .where(Monitor.status != 'scrapped')
    )
    result = await db.execute(query)
    summary = result.first()
    logger.info(f"Raw summary from DB: {dict(summary._mapping) if summary else 'None'}")
    if not summary:
        logger.warning("No summary data found, returning zeros.")
        return DashboardSummary(
            removal_count=0,
            retain_count=0,
            category_counts={"FM": 0, "DM": 0, "PL": 0, "RG": 0}
        )
    total_monitors = int(summary.total_monitors or 0)
    fm_count = int(summary.fm_count or 0)
    dm_count = int(summary.dm_count or 0)
    pl_count = int(summary.pl_count or 0)
    rg_count = int(summary.rg_count or 0)
    removal_count = int(summary.removal_count or 0)
    retain_count = int(summary.retain_count or 0)
    total_categorized = fm_count + dm_count + pl_count + rg_count
    other_count = total_monitors - total_categorized
    # 统计各类别移除数
    fm_removal_count = await db.scalar(
        select(func.count(Monitor.id)).select_from(Monitor)
        .join(ActionResponsibility, Monitor.id == ActionResponsibility.monitor_id)
        .where(
            Monitor.status != 'scrapped',
            Monitor.monitor_name.startswith('FM'),
            func.lower(ActionResponsibility.action_type) == 'removal'
        )
    )
    dm_removal_count = await db.scalar(
        select(func.count(Monitor.id)).select_from(Monitor)
        .join(ActionResponsibility, Monitor.id == ActionResponsibility.monitor_id)
        .where(
            Monitor.status != 'scrapped',
            Monitor.monitor_name.startswith('DM'),
            func.lower(ActionResponsibility.action_type) == 'removal'
        )
    )
    pl_removal_count = await db.scalar(
        select(func.count(Monitor.id)).select_from(Monitor)
        .join(ActionResponsibility, Monitor.id == ActionResponsibility.monitor_id)
        .where(
            Monitor.status != 'scrapped',
            Monitor.monitor_name.startswith('PL'),
            func.lower(ActionResponsibility.action_type) == 'removal'
        )
    )
    rg_removal_count = await db.scalar(
        select(func.count(Monitor.id)).select_from(Monitor)
        .join(ActionResponsibility, Monitor.id == ActionResponsibility.monitor_id)
        .where(
            Monitor.status != 'scrapped',
            Monitor.monitor_name.startswith('RG'),
            func.lower(ActionResponsibility.action_type) == 'removal'
        )
    )
    response_data = {
        "removal_count": removal_count,
        "retain_count": retain_count,
        "category_counts": {
            "FM": fm_count,
            "DM": dm_count,
            "PL": pl_count,
            "RG": rg_count
        },
        "category_removal_counts": {
            "FM": fm_removal_count or 0,
            "DM": dm_removal_count or 0,
            "PL": pl_removal_count or 0,
            "RG": rg_removal_count or 0
        }
    }
    logger.info(f"Processed data for response: {response_data}")
    logger.info("Successfully processed dashboard summary request.")
    return response_data

@router.get("/visualization/daily_removals", response_model=List[dict], tags=["Monitors"])
async def get_daily_removals(db: AsyncSession = Depends(get_session)):
    logger.info("Request received for daily removals.")
    latest_resp_subquery = (
        select(
            ActionResponsibility.monitor_id,
            func.max(ActionResponsibility.id).label("latest_id")
        )
        .group_by(ActionResponsibility.monitor_id)
        .subquery()
    )
    ar_alias = aliased(ActionResponsibility)
    query = (
        select(
            func.count(Monitor.id).label("count"),
            func.date(ar_alias.removal_date).label("date")
        )
        .select_from(Monitor)
        .join(latest_resp_subquery, Monitor.id == latest_resp_subquery.c.monitor_id)
        .join(ar_alias, (Monitor.id == ar_alias.monitor_id) & (latest_resp_subquery.c.latest_id == ar_alias.id))
        .where(
            (Monitor.status != 'scrapped') &
            (func.lower(ar_alias.action_type) == 'removal') &
            (ar_alias.removal_date != None)
        )
        .group_by(func.date(ar_alias.removal_date))
        .order_by(func.date(ar_alias.removal_date))
    )
    result = await db.execute(query)
    raw_results = result.fetchall()
    logger.info(f"Raw daily removals from DB: {[dict(row._mapping) for row in raw_results]}")
    response_data = [
        {
            "date": row.date.strftime('%Y-%m-%d'),
            "count": row.count
        }
        for row in raw_results
    ]
    logger.info(f"Processed daily removals for response: {response_data}")
    logger.info("Successfully processed daily removals request.")
    return response_data

@router.get("/monitors/ids", response_model=List[int])
async def get_all_monitor_ids(db: AsyncSession = Depends(get_session)):
    logger.info("Request received to get all monitor IDs.")
    query = select(Monitor.id).order_by(Monitor.id)
    result = await db.execute(query)
    ids = result.scalars().all()
    logger.info(f"Found and returning {len(ids)} monitor IDs.")
    return ids

# 参数路径接口最后，且参数名统一为 id
@router.get("/monitors/{id}", response_model=MonitorSchema)
async def get_monitor_details(id: int, db: AsyncSession = Depends(get_session)):
    logger.info(f"Request received for details of monitor ID: {id}")
    query = select(Monitor).where(Monitor.id == id)
    result = await db.execute(query)
    monitor = result.scalars().first()
    if not monitor:
        logger.warning(f"Monitor with ID {id} not found.")
        raise HTTPException(status_code=404, detail="Monitor not found")
    logger.info(f"Successfully found and returning details for monitor ID: {id}.")
    return monitor 