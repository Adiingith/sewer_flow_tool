from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import aliased
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case, String, or_, cast
from backend.core.db_connect import get_session
from backend.models.monitor import Monitor
from backend.models.ActionResponsibility import ActionResponsibility
from backend.models.WeeklyQualityCheck import WeeklyQualityCheck
from backend.schemas.monitorSchemas import MonitorSchema, DashboardSummary
from typing import List
from datetime import datetime, timedelta, timezone
from collections import Counter
from backend.models.measurement import Measurement
from backend.models.rain_gauge import RainGauge
from sqlalchemy import func, cast, Integer
from backend.models.weatherEvent import WeatherEvent
from backend.schemas.monitorSchemas import WeatherEventSchema
from backend.services.data_processor import get_max_interim

router = APIRouter()

# fixed path interface first
@router.get("/visualization/dashboard_summary", response_model=DashboardSummary, tags=["Monitors"])
async def get_dashboard_summary(db: AsyncSession = Depends(get_session)):
    # find all monitor_id with actions as removal (unique, distinct, case-insensitive, support string or JSON)
    removal_id_query = (
        select(WeeklyQualityCheck.monitor_id)
        .where(
            or_(
                func.lower(func.trim(cast(WeeklyQualityCheck.actions, String), '"')) == 'removal',
                func.lower(cast(WeeklyQualityCheck.actions['actions'], String)) == 'removal'
            )
        )
        .distinct()
    )
    removal_ids_result = await db.execute(removal_id_query)
    removal_ids = set(row[0] for row in removal_ids_result.fetchall())

    # find all non-scrapped monitors
    monitors_query = (
        select(Monitor.id, Monitor.monitor_name, Monitor.status)
        .where(Monitor.status != 'scrapped')
    )
    monitors_result = await db.execute(monitors_query)
    monitors = monitors_result.fetchall()

    # count removal_count for each category
    category_counts = {"FM": 0, "DM": 0, "PL": 0, "RG": 0}
    category_removal_counts = {"FM": 0, "DM": 0, "PL": 0, "RG": 0}
    for m_id, m_name, _ in monitors:
        for prefix in category_counts.keys():
            if m_name.startswith(prefix):
                category_counts[prefix] += 1
                if m_id in removal_ids:
                    category_removal_counts[prefix] += 1
                break

    # count total removal_count and retain_count
    total_removal_count = sum(category_removal_counts.values())
    total_monitor_count = sum(category_counts.values())
    total_retain_count = total_monitor_count - total_removal_count

    response_data = {
        "removal_count": total_removal_count,
        "retain_count": total_retain_count,
        "category_counts": category_counts,
        "category_removal_counts": category_removal_counts
    }
    return response_data

@router.get("/visualization/daily_removals", response_model=List[dict], tags=["Monitors"])
async def get_daily_removals(db: AsyncSession = Depends(get_session)):
    #  Get all monitor_ids with actions as removal
    removal_monitor_ids_query = select(WeeklyQualityCheck.monitor_id).where(
        or_(
            func.lower(func.replace(func.replace(WeeklyQualityCheck.actions.cast(String), '"', ''), "'", '')) == 'removal',
            func.lower(WeeklyQualityCheck.actions['actions'].as_string()) == 'removal'
        )
    ).distinct()
    result = await db.execute(removal_monitor_ids_query)
    removal_monitor_ids = [row[0] for row in result.fetchall()]
    if not removal_monitor_ids:
        return []

    # 2. Get the latest removal_date for each monitor_id
    latest_ar_subq = (
        select(
            ActionResponsibility.monitor_id,
            func.max(ActionResponsibility.id).label('max_id')
        )
        .where(
            ActionResponsibility.monitor_id.in_(removal_monitor_ids),
            ActionResponsibility.removal_date != None
        )
        .group_by(ActionResponsibility.monitor_id)
        .subquery()
    )
    # Get the removal_date for these monitor_ids, and then count by date
    latest_removal_query = (
        select(
            func.count().label('count'),
            func.date(ActionResponsibility.removal_date).label('date')
        )
        .join(latest_ar_subq, ActionResponsibility.id == latest_ar_subq.c.max_id)
        .group_by(func.date(ActionResponsibility.removal_date))
        .order_by(func.date(ActionResponsibility.removal_date))
    )
    result = await db.execute(latest_removal_query)
    raw_results = result.fetchall()
    response_data = [
        {
            "date": row.date.strftime('%Y-%m-%d'),
            "count": row.count
        }
        for row in raw_results
    ]
    return response_data

@router.get("/monitors/ids", response_model=List[int])
async def get_all_monitor_ids(db: AsyncSession = Depends(get_session)):
    query = select(Monitor.id).order_by(Monitor.id)
    result = await db.execute(query)
    ids = result.scalars().all()
    return ids

# parameter path interface last, and parameter name is id
@router.get("/monitors/{id}", response_model=MonitorSchema)
async def get_monitor_details(id: int, db: AsyncSession = Depends(get_session)):
    query = select(Monitor).where(Monitor.id == id)
    result = await db.execute(query)
    monitor = result.scalars().first()
    if not monitor:
        raise HTTPException(status_code=404, detail="Monitor not found")
    return monitor

@router.get("/monitors/{id}/available-interims", tags=["Monitors"])
async def get_available_interims(id: int, db: AsyncSession = Depends(get_session)):
    """Get all available interim values for a monitor"""
    # Get all interim values for measurements
    measurement_query = (
        select(Measurement.interim)
        .where(Measurement.monitor_id == id)
        .distinct()
        .order_by(Measurement.interim)
    )
    result = await db.execute(measurement_query)
    measurement_interims = [row[0] for row in result.fetchall()]
    
    # Get all interim values for rain gauge (if monitor has assigned rain gauge)
    monitor_result = await db.execute(select(Monitor).where(Monitor.id == id))
    monitor = monitor_result.scalars().first()
    rain_gauge_interims = []
    
    if monitor and monitor.assigned_rain_gauge_id:
        rain_gauge_query = (
            select(RainGauge.interim)
            .where(RainGauge.monitor_id == monitor.assigned_rain_gauge_id)
            .distinct()
            .order_by(RainGauge.interim)
        )
        result = await db.execute(rain_gauge_query)
        rain_gauge_interims = [row[0] for row in result.fetchall()]
    
    # Get max interim for both
    max_measurement_interim = await get_max_interim(db, Measurement, 'monitor_id', id)
    max_rain_gauge_interim = None
    if monitor and monitor.assigned_rain_gauge_id:
        max_rain_gauge_interim = await get_max_interim(db, RainGauge, 'monitor_id', monitor.assigned_rain_gauge_id)
    
    return {
        "measurement_interims": measurement_interims,
        "rain_gauge_interims": rain_gauge_interims,
        "max_measurement_interim": max_measurement_interim,
        "max_rain_gauge_interim": max_rain_gauge_interim
    }

# measurement and rain_gauge data interface
@router.get("/monitors/{id}/measurements", tags=["Monitors"])
async def get_measurements(
    id: int,
    start: datetime = Query(None, description="start time"),
    end: datetime = Query(None, description="end time"),
    interim: str = Query(None, description="interim value (e.g., 'interim1', 'interim2')"),
    db: AsyncSession = Depends(get_session)
):
    # Get max interim if not specified
    if interim is None:
        max_interim = await get_max_interim(db, Measurement, 'monitor_id', id)
        if max_interim is None:
            return {"data": [], "min_time": None, "max_time": None, "interim": None}
        interim = max_interim
    
    # If both start and end are provided, return data for the time range
    if start is not None and end is not None:
        query = select(Measurement).where(
            Measurement.monitor_id == id,
            Measurement.interim == interim,
            Measurement.time >= start,
            Measurement.time <= end
        ).order_by(Measurement.time.asc())
        result = await db.execute(query)
        data = result.scalars().all()
        if data:
            min_time = data[0].time
            max_time = data[-1].time
        else:
            min_time = max_time = None
        return {
            "data": [m.to_dict() for m in data],
            "min_time": min_time.isoformat() if min_time else None,
            "max_time": max_time.isoformat() if max_time else None,
            "interim": interim
        }
    else:
        # Default: return first 3 days of data for the specified interim
        query = select(Measurement).where(
            Measurement.monitor_id == id,
            Measurement.interim == interim
        ).order_by(Measurement.time.asc())
        result = await db.execute(query)
        all_data = result.scalars().all()
        
        if not all_data:
            return {
                "data": [],
                "min_time": None,
                "max_time": None,
                "interim": interim
            }
        
        # Get first 3 days of data
        first_time = all_data[0].time
        three_days_later = first_time + timedelta(days=3)
        
        # Filter data for first 3 days
        three_days_data = [m for m in all_data if m.time <= three_days_later]
        
        min_time = all_data[0].time
        max_time = all_data[-1].time
        
        return {
            "data": [m.to_dict() for m in three_days_data],
            "min_time": min_time.isoformat() if min_time else None,
            "max_time": max_time.isoformat() if max_time else None,
            "interim": interim
        }

@router.get("/monitors/{id}/rain-gauge", tags=["Monitors"])
async def get_rain_gauge(
    id: int,
    start: datetime = Query(None, description="start time"),
    end: datetime = Query(None, description="end time"),
    interim: str = Query(None, description="interim value (e.g., 'interim1', 'interim2')"),
    db: AsyncSession = Depends(get_session)
):
    # First get monitor, then get assigned_rain_gauge_id
    monitor_result = await db.execute(select(Monitor).where(Monitor.id == id))
    monitor = monitor_result.scalars().first()
    if not monitor or not monitor.assigned_rain_gauge_id:
        return {"data": [], "min_time": None, "max_time": None, "interim": None}
    rain_gauge_id = monitor.assigned_rain_gauge_id
    
    # Get max interim if not specified
    if interim is None:
        max_interim = await get_max_interim(db, RainGauge, 'monitor_id', rain_gauge_id)
        if max_interim is None:
            return {"data": [], "min_time": None, "max_time": None, "interim": None}
        interim = max_interim

    # If both start and end are provided, return data for the time range
    if start is not None and end is not None:
        query = select(RainGauge).where(
            RainGauge.monitor_id == rain_gauge_id,
            RainGauge.interim == interim,
            RainGauge.timestamp >= start,
            RainGauge.timestamp <= end
        ).order_by(RainGauge.timestamp.asc())
        result = await db.execute(query)
        data = result.scalars().all()
        if data:
            min_time = data[0].timestamp
            max_time = data[-1].timestamp
        else:
            min_time = max_time = None
        return {
            "data": [r.to_dict() for r in data],
            "min_time": min_time.isoformat() if min_time else None,
            "max_time": max_time.isoformat() if max_time else None,
            "interim": interim
        }
    else:
        # Default: return first 3 days of data for the specified interim
        query = select(RainGauge).where(
            RainGauge.monitor_id == rain_gauge_id,
            RainGauge.interim == interim
        ).order_by(RainGauge.timestamp.asc())
        result = await db.execute(query)
        all_data = result.scalars().all()
        
        if not all_data:
            return {
                "data": [],
                "min_time": None,
                "max_time": None,
                "interim": interim
            }
        
        # Get first 3 days of data
        first_time = all_data[0].timestamp
        three_days_later = first_time + timedelta(days=3)
        
        # Filter data for first 3 days
        three_days_data = [r for r in all_data if r.timestamp <= three_days_later]
        
        min_time = all_data[0].timestamp
        max_time = all_data[-1].timestamp
        
        return {
            "data": [r.to_dict() for r in three_days_data],
            "min_time": min_time.isoformat() if min_time else None,
            "max_time": max_time.isoformat() if max_time else None,
            "interim": interim
        }

@router.get("/monitors/{id}/rain-gauge-by-time", tags=["Monitors"])
async def get_rain_gauge_by_time(
    id: int,
    start: datetime = Query(..., description="start time"),
    end: datetime = Query(..., description="end time"),
    db: AsyncSession = Depends(get_session)
):
    """
    Get rain gauge data by time range only, without interim dependency.
    Missing data points will be filled with 0 values.
    """
    # First get monitor, then get assigned_rain_gauge_id
    monitor_result = await db.execute(select(Monitor).where(Monitor.id == id))
    monitor = monitor_result.scalars().first()
    if not monitor:
        return {"data": [], "min_time": None, "max_time": None, "error": "Monitor not found"}
    
    if not monitor.assigned_rain_gauge_id:
        return {"data": [], "min_time": None, "max_time": None, "error": "No rain gauge assigned"}
    
    rain_gauge_id = monitor.assigned_rain_gauge_id
    
    # If assigned_rain_gauge_id equals the current monitor id, it's probably wrong
    if rain_gauge_id == id:
        # Find all rain gauge data and get the most common monitor_id
        rain_gauge_query = select(RainGauge.monitor_id, func.count(RainGauge.monitor_id).label('count')).group_by(RainGauge.monitor_id).order_by(func.count(RainGauge.monitor_id).desc())
        rain_gauge_result = await db.execute(rain_gauge_query)
        rain_gauge_counts = rain_gauge_result.fetchall()
        
        if rain_gauge_counts and rain_gauge_counts[0].monitor_id != id:
            # Use the rain gauge with the most data
            rain_gauge_id = rain_gauge_counts[0].monitor_id
        else:
            # If no suitable rain gauge found, return empty data
            return {
                "data": [],
                "min_time": start.isoformat(),
                "max_time": end.isoformat()
            }
    
    # Get all rain gauge data for the time range, regardless of interim
    # rain_gauge_id is the assigned rain gauge's monitor ID
    query = select(RainGauge).where(
        RainGauge.monitor_id == rain_gauge_id,
        RainGauge.timestamp >= start,
        RainGauge.timestamp <= end
    ).order_by(RainGauge.timestamp.asc())
    result = await db.execute(query)
    raw_data = result.scalars().all()
    
    if not raw_data:
        return {
            "data": [],
            "min_time": start.isoformat(),
            "max_time": end.isoformat()
        }
    
    # Convert to dictionary and create a map for quick lookup
    # Use timestamp as string key to avoid datetime precision issues
    data_dict = {r.timestamp.isoformat(): r.to_dict() for r in raw_data}
    
    # Get the actual timestamps from the data to determine the interval
    timestamps = sorted([r.timestamp for r in raw_data])
    
    # Determine the most common interval between data points
    intervals = []
    for i in range(1, len(timestamps)):
        interval = (timestamps[i] - timestamps[i-1]).total_seconds()
        intervals.append(interval)
    
    # Use the most common interval, or default to 2 minutes if no pattern found
    if intervals:
        most_common_interval = Counter(intervals).most_common(1)[0][0]
        interval_seconds = most_common_interval
    else:
        interval_seconds = 120  # Default to 2 minutes
    
    # Generate time series with the determined interval
    current_time = start
    
    # Make sure both start and end times have timezone info to match database timestamps
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    
    filled_data = []
    
    while current_time <= end:
        # Generate time string with timezone to match database format
        current_time_str = current_time.isoformat()
        
        if current_time_str in data_dict:
            # Use actual data
            filled_data.append(data_dict[current_time_str])
        else:
            # Fill with 0 values
            filled_data.append({
                "id": None,
                "monitor_id": rain_gauge_id,
                "timestamp": current_time_str,
                "intensity_mm_per_hr": 0.0,
                "depth_mm": 0.0,
                "interim": None
            })
        current_time += timedelta(seconds=interval_seconds)
    
    return {
        "data": filled_data,
        "min_time": start.isoformat(),
        "max_time": end.isoformat()
    }

@router.get("/monitors/{id}/storms", response_model=List[WeatherEventSchema], tags=["Monitors"])
async def get_monitor_storm_events(id: int, db: AsyncSession = Depends(get_session)):
    monitor_result = await db.execute(select(Monitor).where(Monitor.id == id))
    monitor = monitor_result.scalars().first()
    if not monitor or not monitor.assigned_rain_gauge_id:
        return []
    
    rain_gauge_id = monitor.assigned_rain_gauge_id
    
    # If assigned_rain_gauge_id equals the current monitor id, it's probably wrong
    if rain_gauge_id == id:
        # Find all rain gauge data and get the most common monitor_id
        rain_gauge_query = select(RainGauge.monitor_id, func.count(RainGauge.monitor_id).label('count')).group_by(RainGauge.monitor_id).order_by(func.count(RainGauge.monitor_id).desc())
        rain_gauge_result = await db.execute(rain_gauge_query)
        rain_gauge_counts = rain_gauge_result.fetchall()
        
        if rain_gauge_counts and rain_gauge_counts[0].monitor_id != id:
            rain_gauge_id = rain_gauge_counts[0].monitor_id
        else:
            return []
    
    # Get max interim for the rain gauge
    max_interim = await get_max_interim(db, WeatherEvent, 'rain_gauge_monitor_id', rain_gauge_id)
    if not max_interim:
        return []
    result = await db.execute(
        select(WeatherEvent)
        .where(
            WeatherEvent.rain_gauge_monitor_id == rain_gauge_id,
            WeatherEvent.event_type == 'weekly_storm',
            WeatherEvent.interim == max_interim
        )
        .order_by(WeatherEvent.start_time.asc())
    )
    events = result.scalars().all()
    return events

@router.get("/monitors/{id}/stats", tags=["Monitors"])
async def get_monitor_stats(
    id: int,
    start: datetime = Query(None, description="start time"),
    end: datetime = Query(None, description="end time"),
    interim: str = Query(None, description="interim value"),
    db: AsyncSession = Depends(get_session)
):
    # Get interim for measurement (use provided interim or max interim)
    if interim is None:
        max_interim = await get_max_interim(db, Measurement, 'monitor_id', id)
        if max_interim is None:
            return {}
        measurement_interim = max_interim
    else:
        measurement_interim = interim

    # Get assigned rain gauge id
    monitor_result = await db.execute(select(Monitor).where(Monitor.id == id))
    monitor = monitor_result.scalars().first()
    rain_gauge_id = monitor.assigned_rain_gauge_id if monitor else None
    
    # If assigned_rain_gauge_id equals the current monitor id, it's probably wrong
    if rain_gauge_id == id:
        # Find all rain gauge data and get the most common monitor_id
        rain_gauge_query = select(RainGauge.monitor_id, func.count(RainGauge.monitor_id).label('count')).group_by(RainGauge.monitor_id).order_by(func.count(RainGauge.monitor_id).desc())
        rain_gauge_result = await db.execute(rain_gauge_query)
        rain_gauge_counts = rain_gauge_result.fetchall()
        
        if rain_gauge_counts and rain_gauge_counts[0].monitor_id != id:
            rain_gauge_id = rain_gauge_counts[0].monitor_id
        else:
            rain_gauge_id = None

    # Query measurement data with time range filter if provided
    measurement_query = select(Measurement).where(
        Measurement.monitor_id == id,
        Measurement.interim == measurement_interim
    )
    if start:
        measurement_query = measurement_query.where(Measurement.time >= start)
    if end:
        measurement_query = measurement_query.where(Measurement.time <= end)
    measurement_query = measurement_query.order_by(Measurement.time.asc())
    result = await db.execute(measurement_query)
    measurements = result.scalars().all()

    # Query rain gauge data with time range filter if provided
    rain_gauge_data = []
    if rain_gauge_id:
        # Get interim for rain_gauge (use provided interim or max interim)
        if interim is None:
            max_interim_rg = await get_max_interim(db, RainGauge, 'monitor_id', rain_gauge_id)
            if max_interim_rg is None:
                rain_gauge_data = []
            else:
                rain_gauge_interim = max_interim_rg
        else:
            rain_gauge_interim = interim
            
        if rain_gauge_interim:
            rain_gauge_query = select(RainGauge).where(
                RainGauge.monitor_id == rain_gauge_id,
                RainGauge.interim == rain_gauge_interim
            )
            if start:
                rain_gauge_query = rain_gauge_query.where(RainGauge.timestamp >= start)
            if end:
                rain_gauge_query = rain_gauge_query.where(RainGauge.timestamp <= end)
            rain_gauge_query = rain_gauge_query.order_by(RainGauge.timestamp.asc())
            result = await db.execute(rain_gauge_query)
            rain_gauge_data = result.scalars().all()

    # Compute statistics
    # Rainfall
    rainfall_peak = max([float(r.intensity_mm_per_hr) for r in rain_gauge_data if r.intensity_mm_per_hr is not None], default=None)
    rainfall_avg = (sum([float(r.intensity_mm_per_hr) for r in rain_gauge_data if r.intensity_mm_per_hr is not None]) / len([r for r in rain_gauge_data if r.intensity_mm_per_hr is not None])) if rain_gauge_data else None
    # Depth (convert mm to m)
    depth_vals = [float(m.depth) / 1000 for m in measurements if m.depth is not None]
    depth_min = min(depth_vals) if depth_vals else None
    depth_max = max(depth_vals) if depth_vals else None
    # Flow
    flow_vals = [float(m.flow) for m in measurements if m.flow is not None]
    flow_min = min(flow_vals) if flow_vals else None
    flow_max = max(flow_vals) if flow_vals else None
    # Volume (m3): sum(flow * dt)
    volume = None
    if len(measurements) > 1:
        volume = 0.0
        for i in range(1, len(measurements)):
            prev = measurements[i-1]
            curr = measurements[i]
            if prev.flow is not None and curr.flow is not None:
                dt = (curr.time - prev.time).total_seconds()
                avg_flow = (float(prev.flow) + float(curr.flow)) / 2
                volume += avg_flow * dt
    # Velocity
    velocity_vals = [float(m.velocity) for m in measurements if m.velocity is not None]
    velocity_min = min(velocity_vals) if velocity_vals else None
    velocity_max = max(velocity_vals) if velocity_vals else None

    return {
        "rainfall": {
            "peak": rainfall_peak,
            "average": rainfall_avg
        },
        "depth": {
            "min": depth_min,
            "max": depth_max
        },
        "flow": {
            "min": flow_min,
            "max": flow_max,
            "volume": volume
        },
        "velocity": {
            "min": velocity_min,
            "max": velocity_max
        }
    } 


@router.get("/monitors/{id}/rg-available-interims", tags=["RG Monitors"])
async def get_rg_available_interims(id: int, db: AsyncSession = Depends(get_session)):
    """Get all available interim values for a RG monitor (rain gauge data only)"""
    # Get all interim values for rain gauge data
    rain_gauge_query = (
        select(RainGauge.interim)
        .where(RainGauge.monitor_id == id)
        .distinct()
        .order_by(RainGauge.interim)
    )
    result = await db.execute(rain_gauge_query)
    rain_gauge_interims = [row[0] for row in result.fetchall()]
    
    # Get max interim for rain gauge
    max_rain_gauge_interim = await get_max_interim(db, RainGauge, 'monitor_id', id)
    
    return {
        "rain_gauge_interims": rain_gauge_interims,
        "max_rain_gauge_interim": max_rain_gauge_interim
    }

@router.get("/monitors/{id}/rg-rain-gauge", tags=["RG Monitors"])
async def get_rg_rain_gauge(
    id: int,
    start: datetime = Query(None, description="start time"),
    end: datetime = Query(None, description="end time"),
    interim: str = Query(None, description="interim value (e.g., 'interim1', 'interim2')"),
    db: AsyncSession = Depends(get_session)
):
    """Get rain gauge data for RG monitor with interim and time range support"""
    # Get max interim if not specified
    if interim is None:
        max_interim = await get_max_interim(db, RainGauge, 'monitor_id', id)
        if max_interim is None:
            return {"data": [], "min_time": None, "max_time": None, "interim": None}
        interim = max_interim

    # If both start and end are provided, return data for the time range
    if start is not None and end is not None:
        query = select(RainGauge).where(
            RainGauge.monitor_id == id,
            RainGauge.interim == interim,
            RainGauge.timestamp >= start,
            RainGauge.timestamp <= end
        ).order_by(RainGauge.timestamp.asc())
        result = await db.execute(query)
        data = result.scalars().all()
        if data:
            min_time = data[0].timestamp
            max_time = data[-1].timestamp
        else:
            min_time = max_time = None
        return {
            "data": [r.to_dict() for r in data],
            "min_time": min_time.isoformat() if min_time else None,
            "max_time": max_time.isoformat() if max_time else None,
            "interim": interim
        }
    else:
        # Default: return first 3 days of data for the specified interim
        query = select(RainGauge).where(
            RainGauge.monitor_id == id,
            RainGauge.interim == interim
        ).order_by(RainGauge.timestamp.asc())
        result = await db.execute(query)
        all_data = result.scalars().all()
        
        if not all_data:
            return {
                "data": [],
                "min_time": None,
                "max_time": None,
                "interim": interim
            }
        
        # Get first 3 days of data
        first_time = all_data[0].timestamp
        three_days_later = first_time + timedelta(days=3)
        
        # Filter data for first 3 days
        three_days_data = [r for r in all_data if r.timestamp <= three_days_later]
        
        min_time = all_data[0].timestamp
        max_time = all_data[-1].timestamp
        
        return {
            "data": [r.to_dict() for r in three_days_data],
            "min_time": min_time.isoformat() if min_time else None,
            "max_time": max_time.isoformat() if max_time else None,
            "interim": interim
        }

@router.get("/monitors/{id}/rg-stats", tags=["RG Monitors"])
async def get_rg_stats(
    id: int,
    start: datetime = Query(None, description="start time"),
    end: datetime = Query(None, description="end time"),
    interim: str = Query(None, description="interim value"),
    db: AsyncSession = Depends(get_session)
):
    """Get statistics for RG monitor (rain gauge data only)"""
    # Get interim for rain_gauge (use provided interim or max interim)
    if interim is None:
        max_interim = await get_max_interim(db, RainGauge, 'monitor_id', id)
        if max_interim is None:
            return {}
        rain_gauge_interim = max_interim
    else:
        rain_gauge_interim = interim

    # Query rain gauge data with time range filter if provided
    rain_gauge_query = select(RainGauge).where(
        RainGauge.monitor_id == id,
        RainGauge.interim == rain_gauge_interim
    )
    if start:
        rain_gauge_query = rain_gauge_query.where(RainGauge.timestamp >= start)
    if end:
        rain_gauge_query = rain_gauge_query.where(RainGauge.timestamp <= end)
    rain_gauge_query = rain_gauge_query.order_by(RainGauge.timestamp.asc())
    result = await db.execute(rain_gauge_query)
    rain_gauge_data = result.scalars().all()

    # Calculate statistics for rain gauge data
    rainfall_stats = {}
    if rain_gauge_data:
        intensities = [float(r.intensity_mm_per_hr) for r in rain_gauge_data if r.intensity_mm_per_hr is not None]
        if intensities:
            rainfall_stats = {
                "peak": max(intensities),
                "average": sum(intensities) / len(intensities)
            }

    return {
        "rainfall": rainfall_stats
    } 