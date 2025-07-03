import os
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.measurement import Measurement
from datetime import datetime, timedelta
from typing import List, Dict
import boto3
from botocore.exceptions import ClientError
from backend.core.config import get_settings

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
        if interim:
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

    def parse_fdv(self, object_key: str, monitor_id: int, interim: str, start_time_str: str) -> List[Dict]:
        """
        Parse FDV file and extract time series data.
        Only process data between *CEND and *END, each line has 15 numbers, divided into 5 groups, each group is FLOW, DEPTH, VELOCITY.
        The start time is specified by start_time_str, monitor_id and interim are passed as parameters.
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
        # Parse start time
        # start_time_str format like 2408050000, convert to datetime
        start_time = datetime.strptime(start_time_str, '%y%m%d%H%M')
        current_time = start_time
        for line in data_lines:
            # Each line has 15 numbers, divided into 5 groups
            nums = [float(x) for x in line.split() if x.replace('.', '', 1).replace('-', '', 1).isdigit()]
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
        return data

    async def insert_measurements(self, db: AsyncSession, measurements: List[Dict]):
        """
        Batch insert into measurement table.
        """
        objs = [Measurement(**m) for m in measurements]
        db.add_all(objs)
        await db.commit() 