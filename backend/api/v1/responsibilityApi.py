from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, aliased
from sqlalchemy import select, update, func, desc
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from backend.core.db_connect import get_session
from backend.models.ActionResponsibility import ActionResponsibility
from backend.models.monitor import Monitor
from backend.schemas.monitorSchemas import ActionResponsibilityCreate, ActionResponsibilityUpdate, ActionResponsibilitySchema, ActionResponsibilityBulkUpdateItem
from backend.models.WeeklyQualityCheck import WeeklyQualityCheck
from backend.services.data_processor import ensure_actions_is_object

router = APIRouter()

# Helper: always extract action as string
def extract_action_str(action_val):
    if not action_val:
        return ''
    if isinstance(action_val, dict):
        return action_val.get('actions', '')
    return str(action_val)

@router.put("/responsibilities/bulk_update", tags=["ActionResponsibility"])
async def update_responsibilities_in_bulk(updates: List[ActionResponsibilityBulkUpdateItem], db: AsyncSession = Depends(get_session)):
    """
    Update multiple action responsibilities in a single transaction.
    """
    for item in updates:
        update_data = item.model_dump(exclude_unset=True)
        update_data.pop('action', None)  
        stmt = (
            update(ActionResponsibility)
            .where(ActionResponsibility.id == item.id)
            .values(**update_data)
        )
        await db.execute(stmt)
        # Sync write the latest action
        await update_latest_weekly_action(db, item.monitor_id, extract_action_str(getattr(item, 'action', None)))
    await db.commit()
    return {"message": "Update successful"}

@router.post("/responsibilities/", response_model=ActionResponsibilitySchema)
async def create_action_responsibility(resp: ActionResponsibilityCreate, db: AsyncSession = Depends(get_session)):
    create_data = resp.model_dump()
    create_data.pop('action', None)  
    db_resp = ActionResponsibility(**create_data)
    db.add(db_resp)
    await db.commit()
    await db.refresh(db_resp)
    await update_latest_weekly_action(db, db_resp.monitor_id, extract_action_str(getattr(resp, 'action', None)))
    return db_resp

@router.put("/responsibilities/{resp_id}", response_model=ActionResponsibilitySchema)
async def update_action_responsibility(resp_id: int, resp_update: ActionResponsibilityUpdate, db: AsyncSession = Depends(get_session)):
    query = select(ActionResponsibility).where(ActionResponsibility.id == resp_id)
    result = await db.execute(query)
    db_resp = result.scalars().first()

    if not db_resp:
        raise HTTPException(status_code=404, detail="ActionResponsibility not found")

    update_data = resp_update.model_dump(exclude_unset=True)
    update_data.pop('action', None)  
    for key, value in update_data.items():
        setattr(db_resp, key, value)
    await db.commit()
    await db.refresh(db_resp)
    await update_latest_weekly_action(db, db_resp.monitor_id, extract_action_str(getattr(resp_update, 'action', None)))
    return db_resp

@router.get("/responsibilities/monitor/{monitor_id}", response_model=List[ActionResponsibilitySchema])
async def get_responsibilities_for_monitor(monitor_id: int, db: AsyncSession = Depends(get_session)):
    """
    Get all action responsibilities for a specific monitor.
    """
    query = select(ActionResponsibility).where(ActionResponsibility.monitor_id == monitor_id)
    result = await db.execute(query)
    responsibilities = result.scalars().all()
    latest_action = await get_latest_weekly_action(db, monitor_id)
    action_str = extract_action_str(latest_action)
    for resp in responsibilities:
        resp.action = action_str
    return responsibilities

@router.post("/responsibilities/bulk_get", response_model=List[ActionResponsibilitySchema])
async def get_responsibilities_in_bulk(monitor_ids: List[int], db: AsyncSession = Depends(get_session)):
    """
    For a given list of monitor_ids, retrieve the latest action responsibility for each.
    If a monitor has no responsibility record, a default object is returned.
    This uses a LEFT OUTER JOIN to ensure all requested monitors are represented.
    """
    if not monitor_ids:
        return []

    subquery = (
        select(
            ActionResponsibility.monitor_id,
            func.max(ActionResponsibility.id).label("latest_id"),
        )
        .where(ActionResponsibility.monitor_id.in_(monitor_ids))
        .group_by(ActionResponsibility.monitor_id)
        .subquery()
    )

    resp_alias = aliased(ActionResponsibility)

    query = (
        select(
            Monitor.id.label("monitor_id"),
            resp_alias
        )
        .select_from(Monitor)
        .outerjoin(subquery, Monitor.id == subquery.c.monitor_id)
        .outerjoin(
            resp_alias,
            (Monitor.id == resp_alias.monitor_id) & (subquery.c.latest_id == resp_alias.id)
        )
        .where(Monitor.id.in_(monitor_ids))
    )

    result = await db.execute(query)
    final_resps = []
    for row in result.fetchall():
        monitor_id, resp_obj = row
        latest_action = await get_latest_weekly_action(db, monitor_id)
        action_str = extract_action_str(latest_action)
        if resp_obj:
            resp_obj.action = action_str
            final_resps.append(resp_obj)
        else:
            final_resps.append(
                ActionResponsibilitySchema(
                    id=None,
                    monitor_id=monitor_id,
                    action=action_str,
                    requester=None,
                    removal_checker=None,
                    removal_reviewer=None,
                    removal_date=None
                )
            )
    return final_resps

@router.post("/responsibilities/bulk_create", status_code=201, tags=["ActionResponsibility"])
async def create_responsibilities_in_bulk(creations: List[ActionResponsibilityCreate], db: AsyncSession = Depends(get_session)):
    """
    Create multiple action responsibilities in a single transaction.
    """
    db_items = []
    for item in creations:
        create_data = item.model_dump()
        create_data.pop('action', None)  
        db_items.append(ActionResponsibility(**create_data))
    db.add_all(db_items)
    await db.commit()
    # Sync write the latest action
    for item in creations:
        await update_latest_weekly_action(db, item.monitor_id, extract_action_str(getattr(item, 'action', None)))
    return {"message": f"{len(db_items)} items created successfully."}

# Generic function: Get the latest WeeklyQualityCheck.actions with fallback logic
async def get_latest_weekly_action(db, monitor_id):
    # First, get the largest interim for this monitor
    result = await db.execute(
        select(func.max(func.lower(WeeklyQualityCheck.interim)))
        .where(WeeklyQualityCheck.monitor_id == monitor_id)
    )
    largest_interim = result.scalar()
    
    if not largest_interim:
        return None
    
    # Get action from largest interim
    result = await db.execute(
        select(WeeklyQualityCheck.actions)
        .where(
            WeeklyQualityCheck.monitor_id == monitor_id,
            func.lower(WeeklyQualityCheck.interim) == largest_interim
        )
    )
    largest_action = result.scalar()
    
    # If largest interim has action, return it
    if largest_action and extract_action_str(largest_action):
        return largest_action
    
    # Otherwise, get the second largest interim
    result = await db.execute(
        select(func.max(func.lower(WeeklyQualityCheck.interim)))
        .where(
            WeeklyQualityCheck.monitor_id == monitor_id,
            func.lower(WeeklyQualityCheck.interim) < largest_interim
        )
    )
    second_largest_interim = result.scalar()
    
    if not second_largest_interim:
        return largest_action  # Return the empty action from largest interim
    
    # Get action from second largest interim
    result = await db.execute(
        select(WeeklyQualityCheck.actions)
        .where(
            WeeklyQualityCheck.monitor_id == monitor_id,
            func.lower(WeeklyQualityCheck.interim) == second_largest_interim
        )
    )
    second_action = result.scalar()
    
    return second_action if second_action else largest_action

# Generic function: Update the latest WeeklyQualityCheck.actions
async def update_latest_weekly_action(db, monitor_id, new_action):
    result = await db.execute(
        select(WeeklyQualityCheck).where(WeeklyQualityCheck.monitor_id == monitor_id).order_by(desc(WeeklyQualityCheck.check_date), desc(WeeklyQualityCheck.id)).limit(1)
    )
    latest = result.scalars().first()
    if latest:
        old_actions = latest.actions
        latest.actions = ensure_actions_is_object(new_action, old_actions)
        await db.commit()
        await db.refresh(latest)
        return True
    return False 