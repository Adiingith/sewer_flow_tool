from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, aliased
from sqlalchemy import select, update, func
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from backend.core.db_connect import get_session
from backend.models.ActionResponsibility import ActionResponsibility
from backend.models.monitor import Monitor
from backend.schemas.monitorSchemas import ActionResponsibilityCreate, ActionResponsibilityUpdate, ActionResponsibilitySchema, ActionResponsibilityBulkUpdateItem

router = APIRouter()

@router.put("/responsibilities/bulk_update", tags=["ActionResponsibility"])
async def update_responsibilities_in_bulk(updates: List[ActionResponsibilityBulkUpdateItem], db: AsyncSession = Depends(get_session)):
    """
    Update multiple action responsibilities in a single transaction.
    """
    for item in updates:
        stmt = (
            update(ActionResponsibility)
            .where(ActionResponsibility.id == item.id)
            .values(**item.model_dump(exclude_unset=True))
        )
        await db.execute(stmt)
    
    await db.commit()
    return {"message": "Update successful"}

@router.post("/responsibilities/", response_model=ActionResponsibilitySchema)
async def create_action_responsibility(resp: ActionResponsibilityCreate, db: AsyncSession = Depends(get_session)):
    db_resp = ActionResponsibility(**resp.model_dump())
    db.add(db_resp)
    await db.commit()
    await db.refresh(db_resp)
    return db_resp

@router.put("/responsibilities/{resp_id}", response_model=ActionResponsibilitySchema)
async def update_action_responsibility(resp_id: int, resp_update: ActionResponsibilityUpdate, db: AsyncSession = Depends(get_session)):
    query = select(ActionResponsibility).where(ActionResponsibility.id == resp_id)
    result = await db.execute(query)
    db_resp = result.scalars().first()

    if not db_resp:
        raise HTTPException(status_code=404, detail="ActionResponsibility not found")

    update_data = resp_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_resp, key, value)
    
    await db.commit()
    await db.refresh(db_resp)
    return db_resp

@router.get("/responsibilities/monitor/{monitor_id}", response_model=List[ActionResponsibilitySchema])
async def get_responsibilities_for_monitor(monitor_id: int, db: AsyncSession = Depends(get_session)):
    """
    Get all action responsibilities for a specific monitor.
    """
    query = select(ActionResponsibility).where(ActionResponsibility.monitor_id == monitor_id)
    result = await db.execute(query)
    responsibilities = result.scalars().all()
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

    # There might be multiple responsibility records per monitor. We are getting the one with the highest ID.
    # This assumes higher ID means "later" or "more current".
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
        if resp_obj:
            final_resps.append(resp_obj)
        else:
            # Create a default ActionResponsibility-like object for monitors without one
            final_resps.append(
                ActionResponsibilitySchema(
                    id=None,
                    monitor_id=monitor_id,
                    actioned=None,
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
    db_items = [ActionResponsibility(**item.model_dump()) for item in creations]
    db.add_all(db_items)
    await db.commit()
    return {"message": f"{len(db_items)} items created successfully."} 