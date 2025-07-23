from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from backend.core.db_connect import get_session
from backend.models.monitor import Monitor
from backend.models.measurement import Measurement
from backend.models.rain_gauge import RainGauge
from backend.models.WeeklyQualityCheck import WeeklyQualityCheck
from backend.models.weatherEvent import WeatherEvent
from backend.services.signal_rule_engine import process_all_rules
from backend.services.storm_event_selector import StormEventSelector
from backend.machine_learning.inference.predictor import predict_storm_response
import re
import logging
from sqlalchemy import func, delete

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/ai/predict-action")
async def ai_predict_action(
    monitor_ids: List[str] = Body(...),
    db: AsyncSession = Depends(get_session)
):
    """
    Predict action for selected monitors using rule engine and event extraction.
    - monitor_ids: list of monitor.monitor_id (string, not PK)
    - start_time, end_time: optional time range (default: last 7 days)
    """
    if not monitor_ids:
        raise HTTPException(status_code=400, detail="No monitor_ids provided.")

    # 1. Get all distinct interim values from WeeklyQualityCheck table (consistent with other APIs)
    interim_rows = (await db.execute(
        select(WeeklyQualityCheck.interim)
        .where(WeeklyQualityCheck.interim.isnot(None))
        .distinct()
    )).all()
    interim_list = [row[0] for row in interim_rows if row[0]]
    
    if not interim_list:
        return {"results": [
            {"monitor_id": m, "error": "No interim found in the system. Please upload or check weekly quality data (no interim present)."}
            for m in monitor_ids
        ]}
    
    # Extract numeric part and sort (format: Interim1, Interim2, etc.)
    def extract_interim_number(interim):
        match = re.search(r'Interim(\d+)', interim, re.IGNORECASE)
        return int(match.group(1)) if match else -1
    interim_list_sorted = sorted(interim_list, key=extract_interim_number, reverse=True)
    latest_interim = interim_list_sorted[0]

    results = []
    try:
        for monitor_business_id in monitor_ids:
            # 2. Get Monitor by monitor_id (business ID)
            monitor_obj = (await db.execute(select(Monitor).where(Monitor.monitor_id == monitor_business_id))).scalar_one_or_none()
            if not monitor_obj:
                results.append({"monitor_id": monitor_business_id, "error": "Monitor not found."})
                continue
            monitor_pk = monitor_obj.id

            # 3. Fetch data for this monitor and interim (case-insensitive)
            # Check if this is an RG device
            is_rg_device = monitor_obj.monitor_name and monitor_obj.monitor_name.startswith('RG')
            
            if is_rg_device:
                # For RG devices, check rain_gauge table
                rain_gauges = (await db.execute(
                    select(RainGauge).where(
                        RainGauge.monitor_id == monitor_pk,
                        func.lower(RainGauge.interim) == latest_interim.lower()
                    ).order_by(RainGauge.timestamp)
                )).scalars().all()
                
                # Validation: if no rain gauge data, prompt user to upload
                if not rain_gauges:
                    # Upsert WeeklyQualityCheck: update if exists, else insert
                    existing = await db.execute(
                        select(WeeklyQualityCheck).where(
                            WeeklyQualityCheck.monitor_id == monitor_pk,
                            WeeklyQualityCheck.interim == latest_interim
                        )
                    )
                    wqc = existing.scalars().first()
                    
                    # If not found, try case-insensitive match
                    if not wqc:
                        existing = await db.execute(
                            select(WeeklyQualityCheck).where(
                                WeeklyQualityCheck.monitor_id == monitor_pk,
                                func.lower(WeeklyQualityCheck.interim) == latest_interim.lower()
                            )
                        )
                        wqc = existing.scalars().first()
                    
                    if wqc:
                        wqc.actions = {
                            "actions": "inspect_and_reboot:No Data",
                            "rule_engine_action": None,
                            "ai_predicted_action": None,
                            "ai_suggestion": None
                        }
                        wqc.device_status = None
                        wqc.device_status_reason = None
                        wqc.check_date = datetime.utcnow().date()
                        db.add(wqc)
                    else:
                        weekly_check = WeeklyQualityCheck(
                            monitor_id=monitor_pk,
                            check_date=datetime.utcnow().date(),
                            actions={
                                "actions": "inspect_and_reboot:No Data",
                                "rule_engine_action": None,
                                "ai_predicted_action": None,
                                "ai_suggestion": None
                            },
                            device_status=None,
                            device_status_reason=None,
                            interim=latest_interim
                        )
                        db.add(weekly_check)
                    results.append({
                        "monitor_id": monitor_business_id,
                        "device_status": None,
                        "device_status_reason": None,
                        "final_action": "inspect_and_reboot:No Data",
                        "source": "system",
                        "storm_events": [],
                        "dry_day_events": []
                    })
                    continue
                
                # Convert rain gauge data to DataFrame for rule engine
                df = pd.DataFrame([r.to_dict() for r in rain_gauges])
                # For RG devices, we need to adapt the data structure for rule engine
                # Rule engine expects 'rainfall' and 'rg' columns, but rain_gauge has 'intensity_mm_per_hr'
                if not df.empty:
                    # Rename columns to match rule engine expectations
                    df = df.rename(columns={
                        'intensity_mm_per_hr': 'rainfall'
                    })
                    # Add 'rg' column (rain gauge data) - for RG devices, this is the same as rainfall
                    df['rg'] = df['rainfall']
                    # Add dummy columns for other checks (these won't be used for RG devices)
                    df['velocity'] = None
                    df['depth'] = None
                    df['flow'] = None
                
            else:
                # For regular devices, check measurement table
                measurements = (await db.execute(
                    select(Measurement).where(
                        Measurement.monitor_id == monitor_pk,
                        func.lower(Measurement.interim) == latest_interim.lower()
                    ).order_by(Measurement.time)
                )).scalars().all()

                # Validation: if no measurement data, prompt user to upload
                if not measurements:
                    # Upsert WeeklyQualityCheck: update if exists, else insert
                    existing = await db.execute(
                        select(WeeklyQualityCheck).where(
                            WeeklyQualityCheck.monitor_id == monitor_pk,
                            WeeklyQualityCheck.interim == latest_interim
                        )
                    )
                    wqc = existing.scalars().first()
                    
                    # If not found, try case-insensitive match
                    if not wqc:
                        existing = await db.execute(
                            select(WeeklyQualityCheck).where(
                                WeeklyQualityCheck.monitor_id == monitor_pk,
                                func.lower(WeeklyQualityCheck.interim) == latest_interim.lower()
                            )
                        )
                        wqc = existing.scalars().first()
                    
                    if wqc:
                        wqc.actions = {
                            "actions": "inspect_and_reboot:No Data",
                            "rule_engine_action": None,
                            "ai_predicted_action": None,
                            "ai_suggestion": None
                        }
                        wqc.device_status = None
                        wqc.device_status_reason = None
                        wqc.check_date = datetime.utcnow().date()
                        db.add(wqc)
                    else:
                        weekly_check = WeeklyQualityCheck(
                            monitor_id=monitor_pk,
                            check_date=datetime.utcnow().date(),
                            actions={
                                "actions": "inspect_and_reboot:No Data",
                                "rule_engine_action": None,
                                "ai_predicted_action": None,
                                "ai_suggestion": None
                            },
                            device_status=None,
                            device_status_reason=None,
                            interim=latest_interim
                        )
                        db.add(weekly_check)
                    results.append({
                        "monitor_id": monitor_business_id,
                        "device_status": None,
                        "device_status_reason": None,
                        "final_action": "inspect_and_reboot:No Data",
                        "source": "system",
                        "storm_events": [],
                        "dry_day_events": []
                    })
                    continue
                
                df = pd.DataFrame([m.to_dict() for m in measurements])

            # 4. Run rule engine
            rule_result_df = process_all_rules(df, monitor_name=monitor_obj.monitor_name) if not df.empty else df
            if not rule_result_df.empty:
                last_row = rule_result_df.iloc[-1]
                device_status = last_row.get('device_status')
                device_status_reason = last_row.get('device_status_reason')
                device_action = last_row.get('device_action')
            else:
                device_status = None
                device_status_reason = None
                device_action = None

                        # 5. Storm event extraction for regular devices with assigned rain gauge
            storm_events = []
            
            if monitor_obj.assigned_rain_gauge_id:
                # For regular devices, use assigned rain gauge data
                rain_gauges = (await db.execute(
                    select(RainGauge).where(
                        RainGauge.monitor_id == monitor_obj.assigned_rain_gauge_id,
                        func.lower(RainGauge.interim) == latest_interim.lower()
                    ).order_by(RainGauge.timestamp)
                )).scalars().all()
                
                # Note: No rain gauge data doesn't prevent device status processing
                if rain_gauges:
                    rain_df = pd.DataFrame([r.to_dict() for r in rain_gauges]) if rain_gauges else pd.DataFrame()
                    if not rain_df.empty:
                        # Always delete existing storm events for this interim to ensure only top 3 largest events
                        await db.execute(
                            delete(WeatherEvent).where(
                                WeatherEvent.rain_gauge_monitor_id == monitor_obj.assigned_rain_gauge_id,
                                WeatherEvent.interim == latest_interim,
                                WeatherEvent.event_type == 'weekly_storm'
                            )
                        )
                        
                        selector = StormEventSelector()
                        rain_series = rain_df['intensity_mm_per_hr']
                        timestamps = pd.to_datetime(rain_df['timestamp'])
                        storm_events = selector.extract_storm_events(rain_series, timestamps, area=monitor_obj.area or '', interim=latest_interim)
                        
                        for event in storm_events:
                            event_type = event.get('event_type', 'weekly_storm')
                            start_time = event.get('start_time')
                            end_time = event.get('end_time')
                            area = event.get('area', '')
                            interim = event.get('interim', '')
                            rain_gauge_monitor_id = monitor_obj.assigned_rain_gauge_id

                            weather_event = WeatherEvent(
                                event_type=event_type,
                                start_time=start_time,
                                end_time=end_time,
                                area=area,  # use value from event
                                interim=interim,  # use value from event
                                storm_type=event.get('storm_type'),
                                dry_day_number=None,
                                coverage=event.get('coverage'),
                                event_comment=event.get('event_comment'),
                                rain_gauge_monitor_id=rain_gauge_monitor_id
                            )
                            db.add(weather_event)

            # Handle all statuses - update monitor table for all cases
            if device_status in ('unusable', 'usable_with_warning', 'system_error'):
                # Use device_action if available, otherwise use a default action based on status
                action_to_save = device_action if device_action else f"Action required for {device_status} status"
            else:
                # For normal status (usable or None), clear any previous error states
                action_to_save = None
            
            # Always update monitor status regardless of device_status value
            monitor_obj.status = device_status
            monitor_obj.status_reason = device_status_reason
            db.add(monitor_obj)

            # 7. TODO: Dry day event extraction (leave blank for now)
            dry_day_events = []  # TODO: implement dry day event extraction

            # 8. AI model prediction for storm response classification
            ai_predicted_action = None
            ai_suggestion = None
            
            try:
                # Call T-GNN model for storm response prediction
                prediction_result = await predict_storm_response(db, monitor_business_id, latest_interim)
                
                if 'error' not in prediction_result:
                    predicted_label = prediction_result.get('predicted_label', 'unknown')
                    confidence = prediction_result.get('confidence', 0.0)
                    confidence_level = prediction_result.get('confidence_level', 'low')
                    suggestion = prediction_result.get('suggestion', '')
                    
                    # Use the predicted action directly (no mapping needed)
                    ai_predicted_action = predicted_label
                    ai_suggestion = f"{suggestion} (Confidence: {confidence_level}, {confidence:.2f})"
                    
                    logger.info(f"AI prediction for {monitor_business_id}: {predicted_label} ({confidence:.2f})")
                else:
                    logger.warning(f"AI prediction failed for {monitor_business_id}: {prediction_result['error']}")
                    ai_suggestion = f"AI prediction unavailable: {prediction_result['error']}"
                    
            except Exception as ai_error:
                logger.error(f"AI prediction error for {monitor_business_id}: {ai_error}")
                ai_suggestion = f"AI prediction error: {str(ai_error)}"

            # 9. Save result to WeeklyQualityCheck (if no rule engine action, use AI/model action or leave blank)
            final_action = device_action or ai_predicted_action or None  # Prioritize rule engine action
            
            # Upsert WeeklyQualityCheck: update if exists, else insert
            # First, try exact match
            existing = await db.execute(
                select(WeeklyQualityCheck).where(
                    WeeklyQualityCheck.monitor_id == monitor_pk,
                    WeeklyQualityCheck.interim == latest_interim
                )
            )
            wqc = existing.scalars().first()
            
            # If not found, try case-insensitive match
            if not wqc:
                existing = await db.execute(
                    select(WeeklyQualityCheck).where(
                        WeeklyQualityCheck.monitor_id == monitor_pk,
                        func.lower(WeeklyQualityCheck.interim) == latest_interim.lower()
                    )
                )
                wqc = existing.scalars().first()
            
            if wqc:
                # Update existing record
                wqc.actions = {
                    "actions": final_action or "No action required",
                    "rule_engine_action": device_action,
                    "ai_predicted_action": ai_predicted_action,
                    "ai_suggestion": ai_suggestion
                }
                wqc.device_status = device_status
                wqc.device_status_reason = device_status_reason
                wqc.check_date = datetime.utcnow().date()
                db.add(wqc)
            else:
                # Insert new record
                weekly_check = WeeklyQualityCheck(
                    monitor_id=monitor_pk,
                    check_date=datetime.utcnow().date(),
                    actions={
                        "actions": final_action or "No action required",
                        "rule_engine_action": device_action,
                        "ai_predicted_action": ai_predicted_action,
                        "ai_suggestion": ai_suggestion
                    },
                    device_status=device_status,
                    device_status_reason=device_status_reason,
                    interim=latest_interim
                )
                db.add(weekly_check)

            results.append({
                "monitor_id": monitor_business_id,
                "device_status": device_status,
                "device_status_reason": device_status_reason,
                "final_action": final_action,
                "source": "ai_model_or_none",
                "storm_events": storm_events,
                "dry_day_events": dry_day_events
            })

        await db.commit()
    except Exception as e:
        import traceback
        traceback.print_exc()
        await db.rollback()
        raise
    return {"results": results} 