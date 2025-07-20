import pandas as pd
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form, Query
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from backend.core.db_connect import get_session
from backend.models.monitor import Monitor
from backend.models.ActionResponsibility import ActionResponsibility
from backend.models.WeeklyQualityCheck import WeeklyQualityCheck
import backend.schemas.monitorSchemas as monitorSchemas
from backend.services.data_processor import DataProcessor
from backend.services.time_series_processor import TimeSeriesProcessor
import yaml
import re


router = APIRouter()

# Load device ID mapping configuration
with open('backend/schemas/device_id_mappings.yaml', 'r', encoding='utf-8') as f:
    DEVICE_ID_MAP = yaml.safe_load(f)['id_map']

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
                raise HTTPException(status_code=400, detail="The 'Interim' field is required and must be a number!")
            interim_str = f"Interim{int(interim)}"
            
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

            if cstart_found:
                # unified logic: skip all lines whose first field is not 10 digits (e.g. 0.0, -1.0, UNKNOWN, etc.), find the line whose first field is 10 digits as start_time_str
                for i in range(cstart_idx + 1, len(lines)):
                    line_strip = lines[i].strip()
                    if not line_strip:
                        continue
                    first_field = line_strip.split()[0]
                    # only accept 10 digits timestamp (e.g. 2408221652)
                    if first_field.isdigit() and len(first_field) == 10:
                        start_time_str = first_field
                        break
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
            is_rain_gauge = monitor_id_str.startswith('RG')
            
            categorized_measurements = await processor.parse_fdv(db, object_name, monitor_id=monitor_obj.id, interim=interim_str, start_time_str=start_time_str, monitor_id_str=monitor_obj.monitor_id, is_rain_gauge=is_rain_gauge)
            
            if not categorized_measurements:
                return {"message": f"File parsing failed or no valid data: {file.filename}"}
            
            # Count total data points across all interims
            total_data_points = sum(len(data) for data in categorized_measurements.values())
            if total_data_points == 0:
                return {"message": f"No valid data found in file: {file.filename}"}
            
            # Insert parsed time series data into the correct table (full replacement mode)
            await processor.insert_time_series_data(db, categorized_measurements, is_rain_gauge=is_rain_gauge)
            
            # Generate summary message
            interim_summary = []
            for interim, data in categorized_measurements.items():
                if data:
                    start_time = min(d['timestamp' if is_rain_gauge else 'time'] for d in data)
                    end_time = max(d['timestamp' if is_rain_gauge else 'time'] for d in data)
                    interim_summary.append(f"{interim}: {len(data)} records ({start_time.strftime('%m-%d %H:%M')} to {end_time.strftime('%m-%d %H:%M')})")
            
            data_type = "rain gauge" if is_rain_gauge else "time series"
            summary = f"Successfully imported {total_data_points} {data_type} data points across {len(categorized_measurements)} interim(s): " + "; ".join(interim_summary)
            
            return {"message": summary}

        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: '{model_type}'. Please select a valid type."
            )

    except ValueError as e:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during file processing: {str(e)}")

@router.get("/monitors", response_model=monitorSchemas.MonitorPage)
async def get_monitors(
    db: AsyncSession = Depends(get_session),
    page: int = 1,
    limit: int = 10,
    search: Optional[str] = None,
    status: Optional[str] = None
):
    query = select(Monitor)
    if search:
        query = query.where(Monitor.monitor_name.ilike(f"%{search}%"))
    if status:
        query = query.where(Monitor.status == status)
    # first get all, then sort by natural order, then paginate
    result = await db.execute(query)
    monitors = result.scalars().all()
    def natural_key(s):
        m = re.match(r'([A-Za-z]+)(\d+)', s)
        if m:
            return (m.group(1), int(m.group(2)))
        return (s, 0)
    monitors.sort(key=lambda m: natural_key(m.monitor_name))
    total = len(monitors)
    start = (page - 1) * limit
    end = start + limit
    monitors = monitors[start:end]
    # Get action type for each monitor
    monitor_ids = [m.id for m in monitors]
    action_map = {}
    if monitor_ids:
        # First, get the largest interim for each monitor
        subq = (
            select(
                WeeklyQualityCheck.monitor_id,
                func.max(func.lower(WeeklyQualityCheck.interim)).label('max_interim')
            )
            .where(WeeklyQualityCheck.monitor_id.in_(monitor_ids))
            .group_by(WeeklyQualityCheck.monitor_id)
            .subquery()
        )
        
        # Get actions for the largest interim
        action_rows = await db.execute(
            select(WeeklyQualityCheck.monitor_id, WeeklyQualityCheck.interim, WeeklyQualityCheck.actions)
            .join(subq, (WeeklyQualityCheck.monitor_id == subq.c.monitor_id) & (func.lower(WeeklyQualityCheck.interim) == func.lower(subq.c.max_interim)))
        )
        
        # Create a map of monitor_id to (interim, actions) for largest interim
        largest_interim_map = {}
        for row in action_rows:
            largest_interim_map[row.monitor_id] = (row.interim, row.actions)
        
        # For monitors where largest interim has no action, get the second largest interim
        monitors_without_action = []
        for mid in monitor_ids:
            if mid in largest_interim_map:
                action_value = largest_interim_map[mid][1]
                extracted_action = extract_action(action_value)
                if not extracted_action:
                    monitors_without_action.append(mid)
        
        if monitors_without_action:
            # Get the second largest interim for these monitors
            subq2 = (
                select(
                    WeeklyQualityCheck.monitor_id,
                    func.max(func.lower(WeeklyQualityCheck.interim)).label('second_max_interim')
                )
                .where(
                    WeeklyQualityCheck.monitor_id.in_(monitors_without_action),
                    func.lower(WeeklyQualityCheck.interim) < func.lower(subq.c.max_interim)
                )
                .group_by(WeeklyQualityCheck.monitor_id)
                .subquery()
            )
            
            second_action_rows = await db.execute(
                select(WeeklyQualityCheck.monitor_id, WeeklyQualityCheck.interim, WeeklyQualityCheck.actions)
                .join(subq2, (WeeklyQualityCheck.monitor_id == subq2.c.monitor_id) & (func.lower(WeeklyQualityCheck.interim) == func.lower(subq2.c.second_max_interim)))
            )
            
            # Update action_map with fallback actions
            for row in second_action_rows:
                if row.monitor_id in monitors_without_action:
                    action_map[row.monitor_id] = row.actions
        
        # Add actions from largest interim (if they exist)
        for monitor_id, (interim, actions) in largest_interim_map.items():
            if extract_action(actions):  # Only use if action exists
                action_map[monitor_id] = actions
        
        # Ensure all actions in action_map are extracted strings
        final_action_map = {}
        for monitor_id in monitor_ids:
            action_data = action_map.get(monitor_id)
            extracted = extract_action(action_data) if action_data else ''
            final_action_map[monitor_id] = extracted
        
        action_map = final_action_map
    # Concatenate return
    data = []
    for m in monitors:
        action_value = action_map.get(m.id, '')
        m.action = action_value if action_value else ''
        data.append(monitorSchemas.MonitorSchema.from_orm(m).dict())
    return {"data": data, "total": total, "page": page, "limit": limit}

@router.post("/monitors/by_ids", response_model=List[monitorSchemas.MonitorSchema])
async def get_monitors_by_ids(ids: List[int], db: AsyncSession = Depends(get_session)):
    """
    Fetch multiple monitors by their primary key IDs, sorted by natural order of monitor_name.
    """
    if not ids:
        return []
    query = select(Monitor).where(Monitor.id.in_(ids))
    result = await db.execute(query)
    monitors = result.scalars().all()
    # Natural sort: FM1, FM2, ..., FM10, FM11, FM100
    import re
    def natural_key(s):
        m = re.match(r'([A-Za-z]+)(\d+)', s)
        if m:
            return (m.group(1), int(m.group(2)))
        return (s, 0)
    monitors.sort(key=lambda m: natural_key(m.monitor_name))
    monitor_ids = [m.id for m in monitors]
    action_map = {}
    if monitor_ids:
        # First, get the largest interim for each monitor
        subq = (
            select(
                WeeklyQualityCheck.monitor_id,
                func.max(func.lower(WeeklyQualityCheck.interim)).label('max_interim')
            )
            .where(WeeklyQualityCheck.monitor_id.in_(monitor_ids))
            .group_by(WeeklyQualityCheck.monitor_id)
            .subquery()
        )
        
        # Get actions for the largest interim
        action_rows = await db.execute(
            select(WeeklyQualityCheck.monitor_id, WeeklyQualityCheck.interim, WeeklyQualityCheck.actions)
            .join(subq, (WeeklyQualityCheck.monitor_id == subq.c.monitor_id) & (func.lower(WeeklyQualityCheck.interim) == func.lower(subq.c.max_interim)))
        )
        
        # Create a map of monitor_id to (interim, actions) for largest interim
        largest_interim_map = {}
        for row in action_rows:
            largest_interim_map[row.monitor_id] = (row.interim, row.actions)
        
        # For monitors where largest interim has no action, get the second largest interim
        monitors_without_action = [mid for mid in monitor_ids if mid in largest_interim_map and not extract_action(largest_interim_map[mid][1])]
        
        if monitors_without_action:
            # Get the second largest interim for these monitors
            subq2 = (
                select(
                    WeeklyQualityCheck.monitor_id,
                    func.max(func.lower(WeeklyQualityCheck.interim)).label('second_max_interim')
                )
                .where(
                    WeeklyQualityCheck.monitor_id.in_(monitors_without_action),
                    func.lower(WeeklyQualityCheck.interim) < func.lower(subq.c.max_interim)
                )
                .group_by(WeeklyQualityCheck.monitor_id)
                .subquery()
            )
            
            second_action_rows = await db.execute(
                select(WeeklyQualityCheck.monitor_id, WeeklyQualityCheck.interim, WeeklyQualityCheck.actions)
                .join(subq2, (WeeklyQualityCheck.monitor_id == subq2.c.monitor_id) & (func.lower(WeeklyQualityCheck.interim) == func.lower(subq2.c.second_max_interim)))
            )
            
            # Update action_map with fallback actions
            for row in second_action_rows:
                if row.monitor_id in monitors_without_action:
                    action_map[row.monitor_id] = row.actions
        
        # Add actions from largest interim (if they exist)
        for monitor_id, (interim, actions) in largest_interim_map.items():
            if extract_action(actions):  # Only use if action exists
                action_map[monitor_id] = actions
    data = []
    for m in monitors:
        m.action = extract_action(action_map.get(m.id, ''))
        data.append(monitorSchemas.MonitorSchema.from_orm(m).dict())
    return data

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

def extract_action(actions):
    if not actions:
        return ''
    if isinstance(actions, dict):
        # compatible with JSON object in database
        return actions.get('actions', '')
    return str(actions)