from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, Integer
from backend.core.db_connect import get_session
from backend.models.monitor import Monitor
from typing import List, Dict, Any

router = APIRouter(prefix="/monitor-analysis")

@router.get("/areas", tags=["Monitor Analysis"])
async def get_areas(db: AsyncSession = Depends(get_session)):
    """Get all available areas from monitors"""
    # Get distinct areas from monitor table (excluding scrapped devices)
    query = select(Monitor.area).distinct().where(
        Monitor.area.isnot(None),
        Monitor.status != 'scrapped'
    )
    result = await db.execute(query)
    areas = [row[0] for row in result.fetchall() if row[0]]
    
    return {"areas": areas}

@router.get("/monitors/with-rain-gauge", tags=["Monitor Analysis"])
async def get_monitors_with_rain_gauge(
    area: str = Query(..., description="Area name"),
    db: AsyncSession = Depends(get_session)
):
    """Get all monitors in an area with their assigned rain gauge information"""
    query = select(Monitor).where(
        Monitor.area == area,
        Monitor.monitor_name.notlike('RG%'),  # Exclude rain gauges (RG devices)
        Monitor.status != 'scrapped'  # Exclude scrapped devices
    ).order_by(
        func.substring(Monitor.monitor_name, 1, 2),  # Sort by prefix (FM, DM, etc.)
        func.cast(func.substring(Monitor.monitor_name, 3), Integer)  # Sort by number part
    )
    
    result = await db.execute(query)
    monitors = result.scalars().all()
    
    return {
        "monitors": [
            {
                "monitor_id": monitor.id,
                "monitor_name": monitor.monitor_name,
                "assigned_rain_gauge_id": monitor.assigned_rain_gauge_id
            }
            for monitor in monitors
        ]
    }

@router.get("/monitors/rain-gauges", tags=["Monitor Analysis"])
async def get_rain_gauges(
    area: str = Query(..., description="Area name"),
    db: AsyncSession = Depends(get_session)
):
    """Get all rain gauges in an area"""
    query = select(Monitor).where(
        Monitor.area == area,
        Monitor.monitor_name.like('RG%'),  # Only rain gauges (RG devices)
        Monitor.status != 'scrapped'  # Exclude scrapped devices
    ).order_by(
        func.substring(Monitor.monitor_name, 1, 2),  # Sort by prefix (RG)
        func.cast(func.substring(Monitor.monitor_name, 3), Integer)  # Sort by number part
    )
    
    result = await db.execute(query)
    rain_gauges = result.scalars().all()
    
    return {
        "rain_gauges": [
            {
                "id": rg.id,
                "monitor_id": rg.id,
                "monitor_name": rg.monitor_name
            }
            for rg in rain_gauges
        ]
    }

@router.post("/monitor/{monitor_id}/assign-rain-gauge", tags=["Monitor Analysis"])
async def assign_rain_gauge(
    monitor_id: int,
    interim: str = Query(..., description="Rain gauge ID and interim, format: rg_id:interim"),
    db: AsyncSession = Depends(get_session)
):
    """Assign a rain gauge to a monitor"""
    try:
        # Parse interim parameter (format: "rg_id:interim")
        parts = interim.split(':')
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid interim format. Expected 'rg_id:interim'")
        
        rg_id = int(parts[0])
        interim_value = parts[1]
        
        # Verify monitor exists and is not scrapped
        monitor_query = select(Monitor).where(
            Monitor.id == monitor_id,
            Monitor.status != 'scrapped'
        )
        monitor_result = await db.execute(monitor_query)
        monitor = monitor_result.scalars().first()
        
        if not monitor:
            raise HTTPException(status_code=404, detail="Monitor not found or is scrapped")
        
        # Verify rain gauge exists and is not scrapped
        rg_query = select(Monitor).where(
            Monitor.id == rg_id,
            Monitor.status != 'scrapped'
        )
        rg_result = await db.execute(rg_query)
        rain_gauge = rg_result.scalars().first()
        
        if not rain_gauge:
            raise HTTPException(status_code=404, detail="Rain gauge not found or is scrapped")
        
        # Update monitor with assigned rain gauge
        monitor.assigned_rain_gauge_id = rg_id
        await db.commit()
        
        return {"message": "Rain gauge assigned successfully"}
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid rain gauge ID")
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Assignment failed: {str(e)}")

@router.delete("/monitor/{monitor_id}/unassign-rain-gauge", tags=["Monitor Analysis"])
async def unassign_rain_gauge(
    monitor_id: int,
    db: AsyncSession = Depends(get_session)
):
    """Unassign rain gauge from a monitor"""
    try:
        # Get monitor and verify it's not scrapped
        monitor_query = select(Monitor).where(
            Monitor.id == monitor_id,
            Monitor.status != 'scrapped'
        )
        monitor_result = await db.execute(monitor_query)
        monitor = monitor_result.scalars().first()
        
        if not monitor:
            raise HTTPException(status_code=404, detail="Monitor not found or is scrapped")
        
        # Remove rain gauge assignment
        monitor.assigned_rain_gauge_id = None
        await db.commit()
        
        return {"message": "Rain gauge unassigned successfully"}
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Unassignment failed: {str(e)}") 