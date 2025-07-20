import os
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.measurement import Measurement
from backend.models.rain_gauge import RainGauge
from datetime import datetime, timedelta, timezone
from typing import List, Dict
import boto3
from botocore.exceptions import ClientError
from backend.core.config import get_settings
import pandas as pd
from backend.services.signal_rule_engine import process_all_rules
from backend.models.WeeklyQualityCheck import WeeklyQualityCheck
from backend.models.monitor import Monitor
from backend.models.ActionResponsibility import ActionResponsibility
import re
from collections import Counter
from sqlalchemy import select, func, delete
from backend.models.WeeklyQualityCheck import WeeklyQualityCheck
class TimeSeriesProcessor:
    def __init__(self, bucket_name='sewer-timeseries'):
        settings = get_settings()
        self.s3_client = boto3.client(
            's3',
            endpoint_url=f'http://{settings.MINIO_ENDPOINT}',
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
            region_name='us-east-1',
        )
        self.bucket_name = bucket_name
        # Automatically create bucket if it does not exist
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            self.s3_client.create_bucket(Bucket=self.bucket_name)

    async def save_file(self, file: UploadFile, area: str, type_: str, interim: str = None) -> str:
        """
        Upload fdv file to S3-compatible object storage.
        Return file URL and object_name.
        """
        filename = file.filename.lower()
        if filename.endswith(('.csv', '.xlsx', '.xls', '.xlsm')):
            object_name = f"{area}/{file.filename}"
        elif interim:
            object_name = f"{area}/{type_}/{interim}/{file.filename}"
        else:
            object_name = f"{area}/{type_}/{file.filename}"
        content = await file.read()
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=object_name,
            Body=content,
            ContentType=file.content_type or 'application/octet-stream'
        )
        url = f"http://{get_settings().MINIO_ENDPOINT}/{self.bucket_name}/{object_name}"
        return url, object_name

    async def parse_fdv(self, db: AsyncSession, object_key: str, monitor_id: int, interim: str, start_time_str: str, monitor_id_str: str, is_rain_gauge: bool = False) -> Dict[str, List[Dict]]:
        """
        Parse FDV file and extract time series data.
        For RG-prefixed monitor_id, parse as rain_gauge format, otherwise as measurement format.
        Automatically find the maximum time of the last interim, correct start_time.
        """
        data = []
        # Get object content from minio
        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=object_key)
        content = obj['Body'].read().decode('utf-8', errors='ignore')
        lines = content.splitlines()
        # Find data between *CEND and *END
        in_data = False
        data_lines = []
        for line in lines:
            if line.strip().startswith('*CEND'):
                in_data = True
                continue
            if line.strip().startswith('*END'):
                break
            if in_data:
                if line.strip():
                    data_lines.append(line.strip())
        # find the maximum time of the last interim
        last_end_time = None
        if is_rain_gauge:
            from backend.models.rain_gauge import RainGauge
            subq = select(func.max(RainGauge.interim)).where(
                RainGauge.monitor_id == monitor_id,
                RainGauge.interim < interim
            )
            result = await db.execute(subq)
            last_interim = result.scalar()
            if last_interim:
                result2 = await db.execute(
                    select(func.max(RainGauge.timestamp)).where(
                        RainGauge.monitor_id == monitor_id,
                        RainGauge.interim == last_interim
                    )
                )
                last_end_time = result2.scalar()
        else:
            from backend.models.measurement import Measurement
            subq = select(func.max(Measurement.interim)).where(
                Measurement.monitor_id == monitor_id,
                Measurement.interim < interim
            )
            result = await db.execute(subq)
            last_interim = result.scalar()
            if last_interim:
                result2 = await db.execute(
                    select(func.max(Measurement.time)).where(
                        Measurement.monitor_id == monitor_id,
                        Measurement.interim == last_interim
                    )
                )
                last_end_time = result2.scalar()
        
        # Parse start_time from file
        start_time = datetime.strptime(start_time_str, '%y%m%d%H%M')
        
        current_time = start_time
        number_pattern = re.compile(r'-?\d+(?:\.\d+)?')
        if monitor_id_str.startswith('RG'):
            # Rain gauge data: one number per line, representing intensity
            for line in data_lines:
                vals = []
                for x in number_pattern.findall(line):
                    try:
                        if x.strip():  # ensure not empty string
                            vals.append(float(x))
                    except (ValueError, TypeError):
                        continue  # skip invalid values
                for val in vals:
                    row = {
                        'timestamp': current_time,
                        'monitor_id': monitor_id,
                        'intensity_mm_per_hr': val,
                        'interim': interim
                    }
                    data.append(row)
                    current_time += timedelta(minutes=2)
        else:
            # Original logic
            for line in data_lines:
                nums = []
                for x in number_pattern.findall(line):
                    try:
                        if x.strip():  # ensure not empty string
                            nums.append(float(x))
                    except (ValueError, TypeError):
                        continue  # skip invalid values
                
                if len(nums) != 15:
                    continue
                for i in range(5):
                    flow = nums[i*3]
                    depth = nums[i*3+1]
                    velocity = nums[i*3+2]
                    row = {
                        'time': current_time,
                        'monitor_id': monitor_id,
                        'flow': flow,
                        'depth': depth,
                        'velocity': velocity,
                        'interim': interim
                    }
                    data.append(row)
                    current_time += timedelta(minutes=2)
        
        # Full replacement mode: assign data to appropriate interim based on existing boundaries
        if data:
            # Get existing interim boundaries for this monitor
            interim_boundaries = await self.get_interim_boundaries(db, monitor_id, is_rain_gauge)
            
            # Categorize data by interim based on time boundaries
            categorized_data = await self.categorize_data_by_interim(
                data, interim, interim_boundaries, is_rain_gauge
            )
            
            return categorized_data
        
        return {interim: data}  # Return data grouped by interim

    async def get_interim_boundaries(self, db: AsyncSession, monitor_id: int, is_rain_gauge: bool) -> List[Dict]:
        """
        Get existing interim boundaries for a monitor.
        
        Args:
            db: Database session
            monitor_id: Monitor ID (primary key)
            is_rain_gauge: Whether this is a rain gauge device
            
        Returns:
            List[Dict]: List of interim boundaries with format:
                [{'interim': 'interim1', 'start_time': datetime, 'end_time': datetime}, ...]
        """
        if is_rain_gauge:
            # Query rain gauge data
            result = await db.execute(
                select(
                    RainGauge.interim,
                    func.min(RainGauge.timestamp).label('start_time'),
                    func.max(RainGauge.timestamp).label('end_time')
                )
                .where(RainGauge.monitor_id == monitor_id)
                .group_by(RainGauge.interim)
                .order_by('start_time')
            )
        else:
            # Query measurement data
            result = await db.execute(
                select(
                    Measurement.interim,
                    func.min(Measurement.time).label('start_time'),
                    func.max(Measurement.time).label('end_time')
                )
                .where(Measurement.monitor_id == monitor_id)
                .group_by(Measurement.interim)
                .order_by('start_time')
            )
        
        boundaries = []
        for row in result:
            boundaries.append({
                'interim': row.interim,
                'start_time': row.start_time,
                'end_time': row.end_time
            })
        
        return boundaries

    async def categorize_data_by_interim(self, data: List[Dict], current_interim: str, 
                                       boundaries: List[Dict], is_rain_gauge: bool) -> Dict[str, List[Dict]]:
        """
        Categorize data points into appropriate interims based on existing boundaries.
        
        Args:
            data: List of data points
            current_interim: The interim from file upload
            boundaries: Existing interim boundaries
            is_rain_gauge: Whether this is rain gauge data
            
        Returns:
            Dict[str, List[Dict]]: Data categorized by interim
        """
        timestamp_field = 'timestamp' if is_rain_gauge else 'time'
        categorized = {}
        
        for row in data:
            row_time = row[timestamp_field]
            assigned_interim = self.determine_interim_by_time(row_time, current_interim, boundaries)
            
            # Assign the interim to this data point
            row['interim'] = assigned_interim
            
            if assigned_interim not in categorized:
                categorized[assigned_interim] = []
            categorized[assigned_interim].append(row)
        
        return categorized

    def determine_interim_by_time(self, row_time: datetime, current_interim: str, boundaries: List[Dict]) -> str:
        """
        Determine which interim a data point should belong to based on its timestamp.
        
        Args:
            row_time: Timestamp of the data point
            current_interim: The interim from current upload
            boundaries: List of existing interim boundaries
            
        Returns:
            str: The interim this data point should belong to
        """
        # Check if this time falls within any existing interim boundary
        for boundary in boundaries:
            start_time = boundary['start_time']
            end_time = boundary['end_time']
            
            # Ensure timezone consistency for comparison
            if start_time.tzinfo is None:
                row_time_compare = row_time.replace(tzinfo=None) if row_time.tzinfo else row_time
            else:
                if row_time.tzinfo is None:
                    row_time_compare = row_time.replace(tzinfo=timezone.utc)
                else:
                    row_time_compare = row_time
            
            if start_time <= row_time_compare <= end_time:
                return boundary['interim']
        
        # If not in any existing interim, assign to current interim
        return current_interim

    async def insert_time_series_data(self, db: AsyncSession, categorized_data: Dict[str, List[Dict]], is_rain_gauge: bool = False):
        """
        Batch insert into measurement or rain_gauge table with full replacement per interim.
        
        Args:
            db: Database session
            categorized_data: Data grouped by interim
            is_rain_gauge: Whether this is rain gauge data
        """
        if not categorized_data:
            return
        
        for interim, data in categorized_data.items():
            if not data:
                continue
                
            monitor_id = data[0]['monitor_id']
            
            # Delete existing data for this interim (full replacement)
            if is_rain_gauge:
                await db.execute(
                    delete(RainGauge).where(
                        RainGauge.monitor_id == monitor_id,
                        RainGauge.interim == interim
                    )
                )
                
                # Insert new data
                objs = [RainGauge(**d) for d in data]
            else:
                await db.execute(
                    delete(Measurement).where(
                        Measurement.monitor_id == monitor_id,
                        Measurement.interim == interim
                    )
                )
                
                # Insert new data
                objs = [Measurement(**d) for d in data]
            
            db.add_all(objs)

        await db.commit()

        # Insert/update WeeklyQualityCheck records for each interim
        all_unique_pairs = set()
        for interim, data in categorized_data.items():
            if data:
                monitor_id = data[0]['monitor_id']
                all_unique_pairs.add((monitor_id, interim))
        
        for monitor_id, interim in all_unique_pairs:
            # Check if already exists to avoid duplicates
            exists = await db.execute(
                select(WeeklyQualityCheck).where(
                    WeeklyQualityCheck.monitor_id == monitor_id,
                    WeeklyQualityCheck.interim == interim
                )
            )
            if not exists.scalars().first():
                wqc = WeeklyQualityCheck(monitor_id=monitor_id, interim=interim)
                db.add(wqc)
        
        await db.commit() 