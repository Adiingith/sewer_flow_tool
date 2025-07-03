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
from backend.services.time_series_processor import TimeSeriesProcessor
import yaml


router = APIRouter()

# Load device ID mapping configuration
with open('backend/schemas/device_id_mappings.yaml', 'r', encoding='utf-8') as f:
    DEVICE_ID_MAP = yaml.safe_load(f)['id_map']

async def process_time_series_data(file: UploadFile, area: str, db: AsyncSession):
    """
    Process time series data: save file, parse, and write to database.
    """
    processor = TimeSeriesProcessor()
    # Save file
    file_path = await processor.save_file(file, area, 'Monitor Time Series Data')
    # Parse fdv file
    measurements = processor.parse_fdv(file_path)
    if not measurements:
        return {"message": f"File parsing failed or no valid data: {file.filename}"}
    # Write to database
    await processor.insert_measurements(db, measurements)
    return {"message": f"Successfully imported {len(measurements)} time series data."}


@router.post("/upload_file")
async def upload_file(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    area: str = Form(...),
    interim: Optional[str] = Form(None),
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
            # interim parameter validation
            if not interim or not interim.isdigit():
                raise HTTPException(status_code=400, detail="The 'interim' field is required and must be a number!")
            interim_str = f"interim{int(interim)}"
            processor = TimeSeriesProcessor()
            file_url, object_name = await processor.save_file(file, area, 'Monitor Time Series Data', interim=interim_str)
            # Parse IDENTIFIER and START_END_INTERVAL
            # Get content from minio
            obj = processor.s3_client.get_object(Bucket=processor.bucket_name, Key=object_name)
            content = obj['Body'].read().decode('utf-8', errors='ignore')
            lines = content.splitlines()
            monitor_id_str = None
            start_time_str = None
            cstart_found = False
            for idx, line in enumerate(lines):
                if line.strip().startswith('**IDENTIFIER'):
                    # Get the content after the colon, then after the last comma, remove spaces
                    after_colon = line.strip().split(':', 1)[-1] if ':' in line else line.strip()
                    monitor_id_str = after_colon.split(',')[-1].strip()
                if line.strip().startswith('*CSTART'):
                    cstart_found = True
                    cstart_idx = idx
                if monitor_id_str and cstart_found:
                    break
            # If *CSTART found, get the next line's first field as start_time_str
            if cstart_found and cstart_idx + 2 < len(lines):
                start_time_str = lines[cstart_idx + 2].strip().split()[0]
            if not monitor_id_str or not start_time_str:
                raise HTTPException(status_code=400, detail="FDV file is missing IDENTIFIER or start time information.")
            # device id mapping
            if monitor_id_str and len(monitor_id_str) >= 2:
                prefix = monitor_id_str[0]
                if prefix in DEVICE_ID_MAP:
                    mapped_prefix = DEVICE_ID_MAP[prefix]
                    # ensure zero padding format consistent, e.g. F01->FM01
                    monitor_id_str = f"{mapped_prefix}{monitor_id_str[1:]}"
            # Find monitor.id
            monitor_obj = (await db.execute(select(Monitor).where(Monitor.monitor_id == monitor_id_str))).scalar_one_or_none()
            if not monitor_obj:
                raise HTTPException(status_code=400, detail=f"Device with monitor_id={monitor_id_str} not found.")
            # Parse FDV data
            measurements = processor.parse_fdv(object_name, monitor_id=monitor_obj.id, interim=interim_str, start_time_str=start_time_str)
            if not measurements:
                return {"message": f"File parsing failed or no valid data: {file.filename}"}
            await processor.insert_measurements(db, measurements)
            return {"message": f"Successfully imported {len(measurements)} time series data."}

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