from doctest import debug
import pandas as pd
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form, Query
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from backend.core.db_connect import get_session
from backend.models.monitor import Monitor
import backend.schemas.monitorSchemas as monitorSchemas
from backend.services.data_processor import DataProcessor
# Placeholder for future service
# from backend.services.time_series_processor import TimeSeriesProcessor


router = APIRouter()

async def process_time_series_data_placeholder(file: UploadFile, area: str):
    """
    This is a placeholder for processing time series data.
    It will be replaced with the actual implementation logic.
    """
    print(f"Received Time Series data for area '{area}' with filename '{file.filename}'.")
    # In the future, this will call a dedicated service like TimeSeriesProcessor
    # and perform data validation, cleaning, and database insertion.
    return {"message": f"Time series file '{file.filename}' received but not processed (placeholder)."}


@router.post("/upload_file")
async def upload_file(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    area: str = Form(...),
    db: AsyncSession = Depends(get_session)
):
    """
    A unified endpoint for file uploads.
    It dispatches the file to the appropriate processor based on model_type.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded or filename is missing.")
    
    try:
        if model_type in ["Flow Survey Sheet", "Rainfall Assessment Sheet"]:
            # Use the existing DataProcessor for monitor-related sheets
            processor = DataProcessor()
            processed_data = await processor.process_file(file, model_type=model_type, area=area)
            
            if not processed_data:
                raise HTTPException(status_code=400, detail="No valid data found or matched.")

            monitors_to_add = []
            for data in processed_data:
                # Defensively remove the 'measurements' key if it exists.
                # This key corresponds to a relationship and cannot be set directly during creation.
                data.pop('measurements', None)
                monitors_to_add.append(Monitor(**data))

            db.add_all(monitors_to_add)
            await db.commit()
            
            return {"message": f"Successfully imported {len(monitors_to_add)} monitors."}

        elif model_type == "Monitor Time Series Data":
            # Call the placeholder function for time series data
            result = await process_time_series_data_placeholder(file, area)
            return result

        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: '{model_type}'. Please select a valid type."
            )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        await db.rollback()
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during file processing.")

@router.get("/monitors", response_model=monitorSchemas.MonitorPage)
async def get_monitors(
    db: AsyncSession = Depends(get_session),
    page: int = 1,
    limit: int = 10,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = 'asc',
    search: Optional[str] = None,
    status: Optional[str] = None
):
    query = select(Monitor)
    
    if search:
        query = query.where(Monitor.monitor_name.ilike(f"%{search}%"))
    
    if status:
        query = query.where(Monitor.status == status)

    if sort_by and hasattr(Monitor, sort_by):
        column = getattr(Monitor, sort_by)
        if sort_order == 'desc':
            query = query.order_by(column.desc())
        else:
            query = query.order_by(column.asc())

    total = (await db.execute(select(func.count()).select_from(query.subquery()))).scalar()
    
    query = query.offset((page - 1) * limit).limit(limit)
    
    result = await db.execute(query)
    monitors = result.scalars().all()
    
    return {"data": monitors, "total": total, "page": page, "limit": limit}

@router.post("/monitors/by_ids", response_model=List[monitorSchemas.MonitorSchema])
async def get_monitors_by_ids(ids: List[int], db: AsyncSession = Depends(get_session)):
    """
    Fetch multiple monitors by their primary key IDs.
    """
    if not ids:
        return []
    query = select(Monitor).where(Monitor.id.in_(ids))
    result = await db.execute(query)
    monitors = result.scalars().all()
    return monitors

@router.get("/monitors/ids", response_model=List[int])
async def get_non_scrapped_monitor_ids(db: AsyncSession = Depends(get_session)):
    """
    Returns a list of all monitor IDs that are not in 'scrapped' status.
    """
    query = select(Monitor.id).where(Monitor.status != 'scrapped')
    result = await db.execute(query)
    ids = result.scalars().all()
    return ids

@router.get("/monitors/{monitor_id}", response_model=monitorSchemas.MonitorSchema)
async def get_monitor_by_id(monitor_id: int, db: AsyncSession = Depends(get_session)):
    query = select(Monitor).where(Monitor.id == monitor_id)
    result = await db.execute(query)
    monitor = result.scalar_one_or_none()
    if not monitor:
        raise HTTPException(status_code=404, detail="Monitor not found")
    return monitor