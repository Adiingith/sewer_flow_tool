from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, update
from sqlalchemy.orm import aliased
from typing import List, Optional, Dict
from datetime import datetime
from backend.models.presiteInstallCheck import PresiteInstallCheck
from backend.models.WeeklyQualityCheck import WeeklyQualityCheck
from backend.models.monitor import Monitor
from backend.schemas.checkSchemas import PresiteInstallCheckCreate, PresiteInstallCheckRead, WeeklyQualityCheckCreate, WeeklyQualityCheckRead, WeeklyQualityCheckUpdate
from backend.core.db_connect import get_session


router = APIRouter()

@router.post("/presite_install_check/", response_model=PresiteInstallCheckRead)
async def create_presite_install_check(check: PresiteInstallCheckCreate, db: AsyncSession = Depends(get_session)):
    query = select(Monitor).where(Monitor.id == check.monitor_id)
    result = await db.execute(query)
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail=f"Monitor with id {check.monitor_id} not found")

    check_dict = check.dict()
    # If comments is a non-empty string, wrap it in a JSON object.
    if isinstance(check_dict.get('comments'), str) and check_dict['comments'].strip():
        check_dict['comments'] = {"notes": check_dict['comments']}
    else:
        check_dict['comments'] = None

    db_check = PresiteInstallCheck(**check_dict, checked_at=datetime.utcnow())
    db.add(db_check)
    await db.commit()
    await db.refresh(db_check)
    return db_check

@router.post("/presite_install_check/batch", response_model=List[PresiteInstallCheckRead])
async def create_batch_presite_install_checks(checks: List[PresiteInstallCheckCreate], db: AsyncSession = Depends(get_session)):
    created_checks = []
    for check_data in checks:
        query = select(Monitor).where(Monitor.id == check_data.monitor_id)
        result = await db.execute(query)
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail=f"Monitor with id {check_data.monitor_id} not found")

        check_dict = check_data.dict()
        # If comments is a non-empty string, wrap it in a JSON object.
        if isinstance(check_dict.get('comments'), str) and check_dict['comments'].strip():
            check_dict['comments'] = {"notes": check_dict['comments']}
        else:
            check_dict['comments'] = None

        db_check = PresiteInstallCheck(**check_dict, checked_at=datetime.utcnow())
        db.add(db_check)
        created_checks.append(db_check)
    
    await db.commit()
    for check in created_checks:
        await db.refresh(check)
    return created_checks

@router.get("/presite_install_check/{monitor_id}", response_model=List[PresiteInstallCheckRead])
async def get_presite_install_checks_for_monitor(monitor_id: int, db: AsyncSession = Depends(get_session)):
    query = select(PresiteInstallCheck).where(PresiteInstallCheck.monitor_id == monitor_id)
    result = await db.execute(query)
    return result.scalars().all()

@router.post("/presite_install_check/latest_batch", response_model=List[PresiteInstallCheckRead])
async def get_latest_presite_checks_for_monitors(monitor_ids: List[int], db: AsyncSession = Depends(get_session)):
    subquery = (
        select(
            PresiteInstallCheck.monitor_id,
            func.max(PresiteInstallCheck.checked_at).label("latest_checked_at"),
        )
        .where(PresiteInstallCheck.monitor_id.in_(monitor_ids))
        .group_by(PresiteInstallCheck.monitor_id)
        .subquery()
    )

    p_check_alias = aliased(PresiteInstallCheck)
    query = (
        select(
            Monitor.id.label("monitor_id"),
            p_check_alias
        )
        .select_from(Monitor)
        .outerjoin(subquery, Monitor.id == subquery.c.monitor_id)
        .outerjoin(
            p_check_alias,
            (Monitor.id == p_check_alias.monitor_id) & (subquery.c.latest_checked_at == p_check_alias.checked_at)
        )
        .where(Monitor.id.in_(monitor_ids))
    )
    
    result = await db.execute(query)
    
    final_checks = []
    for row in result.fetchall():
        monitor_id, check_obj = row
        if check_obj:
            final_checks.append(check_obj)
        else:
            final_checks.append(
                PresiteInstallCheckRead(
                    id=-1,
                    monitor_id=monitor_id,
                    mh_reference=None,
                    pipe=None,
                    position=None,
                    correct_location=True,
                    correct_install_pipe=True,
                    correct_pipe_size=True,
                    correct_pipe_shape=True,
                    comments=None,
                    checked_at=None
                )
            )
            
    return final_checks

@router.post("/weekly_quality_check/", response_model=WeeklyQualityCheckRead)
async def create_weekly_quality_check(check: WeeklyQualityCheckCreate, db: AsyncSession = Depends(get_session)):
    query = select(Monitor).where(Monitor.id == check.monitor_id)
    result = await db.execute(query)
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Monitor not found")
        
    check_dict = check.dict()
    if isinstance(check_dict.get('comments'), str) and check_dict['comments'].strip():
        check_dict['comments'] = {"notes": check_dict['comments']}
    else:
        check_dict['comments'] = None
    
    if isinstance(check_dict.get('actions'), str) and check_dict['actions'].strip():
        check_dict['actions'] = {"actions": check_dict['actions']}
    else:
        check_dict['actions'] = None

    db_check = WeeklyQualityCheck(**check_dict)
    db.add(db_check)
    await db.commit()
    await db.refresh(db_check)
    return db_check

@router.post("/weekly_quality_check/batch", response_model=List[WeeklyQualityCheckRead])
async def create_batch_weekly_checks(checks: List[WeeklyQualityCheckCreate], db: AsyncSession = Depends(get_session)):
    created_checks = []
    for check_data in checks:
        query = select(Monitor).where(Monitor.id == check_data.monitor_id)
        result = await db.execute(query)
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail=f"Monitor with id {check_data.monitor_id} not found")

        check_dict = check_data.dict()
        
        # Standardize comments
        if isinstance(check_dict.get('comments'), str) and check_dict['comments'].strip():
            check_dict['comments'] = {"notes": check_dict['comments']}
        else:
            check_dict['comments'] = None
        
        # Standardize actions
        if isinstance(check_dict.get('actions'), str) and check_dict['actions'].strip():
            check_dict['actions'] = {"actions": check_dict['actions']}
        else:
            check_dict['actions'] = None

        db_check = WeeklyQualityCheck(**check_dict)
        db.add(db_check)
        created_checks.append(db_check)
    
    await db.commit()
    for check in created_checks:
        await db.refresh(check)
    return created_checks

@router.get("/weekly_quality_check/{monitor_id}", response_model=List[WeeklyQualityCheckRead])
async def get_weekly_quality_checks_for_monitor(monitor_id: int, db: AsyncSession = Depends(get_session)):
    query = select(WeeklyQualityCheck).where(WeeklyQualityCheck.monitor_id == monitor_id).order_by(WeeklyQualityCheck.interim)
    result = await db.execute(query)
    checks = result.scalars().all()
    return checks

@router.post("/weekly_quality_check/by_monitors", response_model=Dict[int, List[WeeklyQualityCheckRead]])
async def get_weekly_checks_by_monitors(monitor_ids: List[int], db: AsyncSession = Depends(get_session)):
    """
    For a given list of monitor_ids, retrieve all associated weekly quality checks,
    grouped by monitor_id.
    """
    if not monitor_ids:
        return {}

    query = (
        select(WeeklyQualityCheck)
        .where(WeeklyQualityCheck.monitor_id.in_(monitor_ids))
        .order_by(WeeklyQualityCheck.monitor_id, WeeklyQualityCheck.interim)
    )
    result = await db.execute(query)
    checks = result.scalars().all()

    # Group checks by monitor_id
    grouped_checks = {monitor_id: [] for monitor_id in monitor_ids}
    for check in checks:
        grouped_checks[check.monitor_id].append(check)

    return grouped_checks

@router.post("/weekly_quality_check/latest_batch", response_model=List[WeeklyQualityCheckRead])
async def get_latest_weekly_checks_for_monitors(monitor_ids: List[int], db: AsyncSession = Depends(get_session)):
    subquery = (
        select(
            WeeklyQualityCheck.monitor_id,
            func.max(WeeklyQualityCheck.id).label("latest_id")
        )
        .where(WeeklyQualityCheck.monitor_id.in_(monitor_ids))
        .group_by(WeeklyQualityCheck.monitor_id)
        .subquery()
    )
    
    w_check_alias = aliased(WeeklyQualityCheck)

    query = (
        select(
            Monitor.id.label("monitor_id"),
            w_check_alias
        )
        .select_from(Monitor)
        .outerjoin(subquery, Monitor.id == subquery.c.monitor_id)
        .outerjoin(w_check_alias, w_check_alias.id == subquery.c.latest_id)
        .where(Monitor.id.in_(monitor_ids))
    )

    result = await db.execute(query)
    
    final_checks = []
    for row in result.fetchall():
        monitor_id, check_obj = row
        if check_obj:
            final_checks.append(check_obj)
        else:
            final_checks.append(
                WeeklyQualityCheckRead(
                    id=-1,
                    monitor_id=monitor_id,
                    silt_mm=None,
                    comments=None,
                    actions=None,
                    interim=None
                )
            )
    return final_checks

@router.get("/monitors/{monitor_id}/presite-check/latest", response_model=PresiteInstallCheckRead)
async def get_latest_presite_check_for_monitor(monitor_id: int, db: AsyncSession = Depends(get_session)):
    query = (
        select(PresiteInstallCheck)
        .where(PresiteInstallCheck.monitor_id == monitor_id)
        .order_by(desc(PresiteInstallCheck.checked_at))
        .limit(1)
    )
    result = await db.execute(query)
    latest_check = result.scalars().first()
    if not latest_check:
        raise HTTPException(status_code=404, detail="No presite install check found for this monitor")
    return latest_check

@router.get("/monitors/{monitor_id}/weekly-quality-checks", response_model=List[WeeklyQualityCheckRead])
async def get_weekly_quality_checks_for_monitor(monitor_id: int, db: AsyncSession = Depends(get_session)):
    query = (
        select(WeeklyQualityCheck)
        .where(WeeklyQualityCheck.monitor_id == monitor_id)
        .order_by(WeeklyQualityCheck.check_date)
    )
    result = await db.execute(query)
    checks = result.scalars().all()
    return checks

@router.get("/presite_install_check/{check_id}", response_model=PresiteInstallCheckRead)
async def read_presite_install_check(check_id: int, db: AsyncSession = Depends(get_session)):
    query = select(PresiteInstallCheck).where(PresiteInstallCheck.id == check_id)
    result = await db.execute(query)
    check = result.scalars().first()
    if not check:
        raise HTTPException(status_code=404, detail="Presite install check not found")
    return check

@router.put("/weekly_quality_check/{check_id}", response_model=WeeklyQualityCheckRead)
async def update_weekly_quality_check(check_id: int, check_update: WeeklyQualityCheckUpdate, db: AsyncSession = Depends(get_session)):
    query = select(WeeklyQualityCheck).where(WeeklyQualityCheck.id == check_id)
    result = await db.execute(query)
    db_check = result.scalars().first()

    if not db_check:
        raise HTTPException(status_code=404, detail="Weekly quality check not found")

    update_data = check_update.model_dump(exclude_unset=True)
    
    for key, value in update_data.items():
        setattr(db_check, key, value)

    await db.commit()
    await db.refresh(db_check)
    return db_check 