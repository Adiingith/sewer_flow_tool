"""
Data loading and preprocessing module for T-GNN storm response classification
Handles monitor relationships, null values, and data formatting
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from datetime import datetime, timedelta
import asyncio
import logging
import json
from decimal import Decimal

from backend.models.monitor import Monitor
from backend.models.measurement import Measurement
from backend.models.rain_gauge import RainGauge
from backend.models.WeeklyQualityCheck import WeeklyQualityCheck
from backend.machine_learning.data.minio_manager import get_minio_cache
from backend.machine_learning.utils.logger_config import get_data_logger, log_function_call

logger = get_data_logger()

class StormResponseDataLoader:
    """
    Data loader for storm response classification model
    Handles the complex relationships between monitors, measurements, rain gauges
    """
    
    def __init__(self, db_session: AsyncSession, use_cache: bool = True):
        self.db = db_session
        self.use_cache = use_cache
        self.minio_cache = get_minio_cache() if use_cache else None
    
    @log_function_call(logger)
    async def get_available_interims(self) -> List[str]:
        """Get all available interim periods from the system"""
        interim_rows = (await self.db.execute(
            select(WeeklyQualityCheck.interim)
            .where(WeeklyQualityCheck.interim.isnot(None))
            .distinct()
        )).all()
        
        interim_list = [row[0] for row in interim_rows if row[0]]
        
        # Sort by numeric part (format: Interim1, Interim2, etc.)
        def extract_interim_number(interim):
            import re
            match = re.search(r'Interim(\d+)', interim, re.IGNORECASE)
            return int(match.group(1)) if match else -1
        
        return sorted(interim_list, key=extract_interim_number)
    
    async def get_monitor_metadata(self, monitor_ids: List[str] = None) -> pd.DataFrame:
        """
        Get monitor metadata including spatial features and relationships
        
        Args:
            monitor_ids: List of monitor business IDs. If None, get all monitors
            
        Returns:
            DataFrame with monitor metadata
        """
        query = select(Monitor)
        if monitor_ids:
            query = query.where(Monitor.monitor_id.in_(monitor_ids))
        
        monitors = (await self.db.execute(query)).scalars().all()
        
        monitor_data = []
        for monitor in monitors:
            # Handle spatial features
            spatial_features = {
                'monitor_id': monitor.monitor_id,
                'monitor_pk': monitor.id,  # Primary key for relationships
                'monitor_name': monitor.monitor_name,
                'type': monitor.type,
                'w3w': monitor.w3w,
                'area': monitor.area,
                'location': monitor.location,
                'mh_reference': monitor.mh_reference,
                'pipe': monitor.pipe,
                'height_mm': monitor.height_mm,
                'width_mm': monitor.width_mm,
                'shape': monitor.shape,
                'depth_mm': monitor.depth_mm,
                'assigned_rain_gauge_id': monitor.assigned_rain_gauge_id,
                'is_rain_gauge': monitor.monitor_name.startswith('RG') if monitor.monitor_name else False
            }
            monitor_data.append(spatial_features)
        
        return pd.DataFrame(monitor_data)
    
    async def get_time_series_data(self, monitor_pk: int, interim: str) -> pd.DataFrame:
        """
        Get time series data for a specific monitor and interim
        
        Args:
            monitor_pk: Monitor primary key (id field)
            interim: Interim period identifier
            
        Returns:
            DataFrame with time series data
        """
        # Check if this is a rain gauge device
        monitor = (await self.db.execute(
            select(Monitor).where(Monitor.id == monitor_pk)
        )).scalar_one_or_none()
        
        if not monitor:
            return pd.DataFrame()
        
        is_rg_device = monitor.monitor_name and monitor.monitor_name.startswith('RG')
        
        if is_rg_device:
            # Get rain gauge data
            rain_data = (await self.db.execute(
                select(RainGauge).where(
                    RainGauge.monitor_id == monitor_pk,
                    func.lower(RainGauge.interim) == interim.lower()
                ).order_by(RainGauge.timestamp)
            )).scalars().all()
            
            if not rain_data:
                return pd.DataFrame()
            
            # Convert to DataFrame and format for T-GNN
            df = pd.DataFrame([r.to_dict() for r in rain_data])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'intensity_mm_per_hr': 'rainfall'
            })
            
            # Add missing columns for consistency
            df['depth'] = None
            df['flow'] = None
            df['velocity'] = None
            
        else:
            # Get measurement data
            measurements = (await self.db.execute(
                select(Measurement).where(
                    Measurement.monitor_id == monitor_pk,
                    func.lower(Measurement.interim) == interim.lower()
                ).order_by(Measurement.time)
            )).scalars().all()
            
            if not measurements:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame([m.to_dict() for m in measurements])
            df['timestamp'] = pd.to_datetime(df['time'])
            
            # Get associated rain gauge data if available
            if monitor.assigned_rain_gauge_id:
                rain_data = (await self.db.execute(
                    select(RainGauge).where(
                        RainGauge.monitor_id == monitor.assigned_rain_gauge_id,
                        func.lower(RainGauge.interim) == interim.lower()
                    ).order_by(RainGauge.timestamp)
                )).scalars().all()
                
                if rain_data:
                    rain_df = pd.DataFrame([r.to_dict() for r in rain_data])
                    rain_df['timestamp'] = pd.to_datetime(rain_df['timestamp'])
                    rain_df = rain_df.rename(columns={'intensity_mm_per_hr': 'rainfall'})
                    
                    # Merge rainfall data with measurements (using nearest timestamp)
                    df = pd.merge_asof(
                        df.sort_values('timestamp'),
                        rain_df[['timestamp', 'rainfall']].sort_values('timestamp'),
                        on='timestamp',
                        direction='nearest'
                    )
                else:
                    df['rainfall'] = None
            else:
                df['rainfall'] = None
        
        return df
    
    def clean_and_preprocess_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess time series data
        
        Args:
            df: Raw time series DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        # Handle string values with spaces
        for col in ['depth', 'flow', 'velocity', 'rainfall']:
            if col in df.columns:
                # Strip whitespace and convert to numeric
                df[col] = df[col].astype(str).str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing values
        numeric_cols = ['depth', 'flow', 'velocity', 'rainfall']
        for col in numeric_cols:
            if col in df.columns:
                # Linear interpolation for missing values
                df[col] = df[col].interpolate(method='linear')
                # Fill remaining NaNs with forward fill, then backward fill
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Ensure we have the expected 4 channels
        expected_channels = ['depth', 'flow', 'velocity', 'rainfall']
        for channel in expected_channels:
            if channel not in df.columns:
                df[channel] = 0.0
        
        return df
    
    async def get_labels_from_quality_checks(self, monitor_pk: int, interim: str) -> Optional[str]:
        """
        Extract action labels from WeeklyQualityCheck actions
        
        Args:
            monitor_pk: Monitor primary key
            interim: Interim period
            
        Returns:
            Label string (one of 7 action categories) or None
        """
        wqc = (await self.db.execute(
            select(WeeklyQualityCheck).where(
                WeeklyQualityCheck.monitor_id == monitor_pk,
                func.lower(WeeklyQualityCheck.interim) == interim.lower()
            )
        )).scalar_one_or_none()
        
        if not wqc or not wqc.actions:
            return None
        
        # Extract label from actions.actions field
        try:
            if isinstance(wqc.actions, dict) and 'actions' in wqc.actions:
                action_text = wqc.actions['actions']
                if action_text:
                    action_text = str(action_text).strip()
                    
                    # Direct mapping if action matches our categories
                    action_categories = [
                        'no_action_continue_monitoring',
                        'investigate_storm_failure',
                        'investigate_dryday_failure',
                        'sensor_fault_or_ragging',
                        'partial_data_needs_review',
                        'partial_data_no_action',
                        'recommend_remove_or_relocate'
                    ]
                    
                    # Check for exact match first
                    if action_text in action_categories:
                        return action_text
                    
                    # Fallback mapping based on keywords
                    action_lower = action_text.lower()
                    if 'no action' in action_lower or 'continue monitoring' in action_lower:
                        return 'no_action_continue_monitoring'
                    elif 'storm' in action_lower and ('failure' in action_lower or 'investigate' in action_lower):
                        return 'investigate_storm_failure'
                    elif 'dry' in action_lower and ('failure' in action_lower or 'investigate' in action_lower):
                        return 'investigate_dryday_failure'
                    elif 'sensor' in action_lower or 'fault' in action_lower or 'ragging' in action_lower:
                        return 'sensor_fault_or_ragging'
                    elif 'partial' in action_lower and 'review' in action_lower:
                        return 'partial_data_needs_review'
                    elif 'partial' in action_lower and 'no action' in action_lower:
                        return 'partial_data_no_action'
                    elif 'remove' in action_lower or 'relocate' in action_lower:
                        return 'recommend_remove_or_relocate'
                    else:
                        # Default fallback based on action type
                        if 'inspect' in action_lower or 'investigate' in action_lower:
                            return 'investigate_storm_failure'
                        elif 'partial' in action_lower:
                            return 'partial_data_needs_review'
                        else:
                            return 'no_action_continue_monitoring'
                            
        except Exception as e:
            logger.warning(f"Error parsing actions for monitor {monitor_pk}, interim {interim}: {e}")
        
        return None
    
    async def build_graph_structure(self, monitor_metadata: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build graph structure based on monitor relationships
        
        Args:
            monitor_metadata: DataFrame with monitor metadata
            
        Returns:
            Tuple of (adjacency_matrix, edge_list)
        """
        n_monitors = len(monitor_metadata)
        adjacency_matrix = np.zeros((n_monitors, n_monitors))
        edge_list = []
        
        # Create mapping from monitor_id to index
        id_to_idx = {row['monitor_id']: idx for idx, row in monitor_metadata.iterrows()}
        
        # Build edges based on area relationships and rain gauge assignments
        for idx, row in monitor_metadata.iterrows():
            current_id = row['monitor_id']
            
            # Connect monitors in the same area
            same_area_monitors = monitor_metadata[
                (monitor_metadata['area'] == row['area']) & 
                (monitor_metadata['monitor_id'] != current_id) &
                (monitor_metadata['area'].notna())
            ]
            
            for _, neighbor_row in same_area_monitors.iterrows():
                neighbor_idx = id_to_idx[neighbor_row['monitor_id']]
                adjacency_matrix[idx, neighbor_idx] = 1
                adjacency_matrix[neighbor_idx, idx] = 1  # Undirected graph
                edge_list.append((idx, neighbor_idx))
            
            # Connect monitors to their assigned rain gauges
            if row['assigned_rain_gauge_id']:
                # Find the rain gauge monitor
                rain_gauge_monitors = monitor_metadata[
                    monitor_metadata['monitor_pk'] == row['assigned_rain_gauge_id']
                ]
                for _, rg_row in rain_gauge_monitors.iterrows():
                    rg_idx = id_to_idx[rg_row['monitor_id']]
                    adjacency_matrix[idx, rg_idx] = 1
                    adjacency_matrix[rg_idx, idx] = 1
                    edge_list.append((idx, rg_idx))
        
        return adjacency_matrix, np.array(edge_list)
    
    @log_function_call(logger)
    async def load_dataset_for_training(self, interims: List[str] = None, 
                                       monitor_ids: List[str] = None) -> Dict[str, Any]:
        """
        Load complete dataset for model training with MinIO caching support
        
        Args:
            interims: List of interim periods to include. If None, use all available
            monitor_ids: List of monitor IDs to include. If None, use all monitors
            
        Returns:
            Dictionary containing all data needed for T-GNN training
        """
        if interims is None:
            interims = await self.get_available_interims()
        
        # Try to get cached data first
        if self.use_cache and self.minio_cache:
            logger.info("Checking MinIO cache for preprocessed data...")
            cached_data = await self.minio_cache.get_cached_data(interims, monitor_ids)
            
            if cached_data and 'data' in cached_data:
                logger.info("Using cached preprocessed data")
                return cached_data['data']
        
        logger.info("Loading fresh data from database...")
        
        # Get monitor metadata
        monitor_metadata = await self.get_monitor_metadata(monitor_ids)
        
        if monitor_metadata.empty:
            logger.warning("No monitors found in the system")
            return {'error': 'No monitors found'}
        
        # Build graph structure
        adjacency_matrix, edge_list = await self.build_graph_structure(monitor_metadata)
        
        # Load time series data and labels for each monitor-interim pair
        dataset = {
            'monitor_metadata': monitor_metadata,
            'graph_structure': {
                'adjacency_matrix': adjacency_matrix,
                'edge_list': edge_list
            },
            'time_series_data': {},
            'labels': {},
            'interims': interims
        }
        
        total_samples = 0
        for interim in interims:
            interim_data = {}
            interim_labels = {}
            
            for _, monitor_row in monitor_metadata.iterrows():
                monitor_pk = monitor_row['monitor_pk']
                monitor_id = monitor_row['monitor_id']
                
                # Load time series data
                ts_data = await self.get_time_series_data(monitor_pk, interim)
                if not ts_data.empty:
                    cleaned_data = self.clean_and_preprocess_timeseries(ts_data)
                    interim_data[monitor_id] = cleaned_data
                
                # Load labels
                label = await self.get_labels_from_quality_checks(monitor_pk, interim)
                if label:
                    interim_labels[monitor_id] = label
                    total_samples += 1
            
            dataset['time_series_data'][interim] = interim_data
            dataset['labels'][interim] = interim_labels
        
        logger.info(f"Loaded {total_samples} samples from database")
        
        # Cache the loaded data
        if self.use_cache and self.minio_cache and total_samples > 0:
            logger.info("Caching preprocessed data to MinIO...")
            cache_success = await self.minio_cache.cache_data(dataset, interims, monitor_ids)
            if cache_success:
                logger.info("Data successfully cached")
            else:
                logger.warning("Failed to cache data")
        
        return dataset
    
    async def load_data_for_prediction(self, monitor_id: str, interim: str) -> Dict[str, Any]:
        """
        Load data for real-time prediction
        
        Args:
            monitor_id: Monitor business ID
            interim: Interim period
            
        Returns:
            Dictionary containing prediction data
        """
        # Get monitor metadata
        monitor_metadata = await self.get_monitor_metadata([monitor_id])
        
        if monitor_metadata.empty:
            return {'error': f'Monitor {monitor_id} not found'}
        
        monitor_row = monitor_metadata.iloc[0]
        monitor_pk = monitor_row['monitor_pk']
        
        # Load time series data
        ts_data = await self.get_time_series_data(monitor_pk, interim)
        
        if ts_data.empty:
            return {'error': f'No data found for monitor {monitor_id}, interim {interim}'}
        
        # Clean and preprocess
        cleaned_data = self.clean_and_preprocess_timeseries(ts_data)
        
        # Get all monitors for graph structure
        all_monitor_metadata = await self.get_monitor_metadata()
        adjacency_matrix, edge_list = await self.build_graph_structure(all_monitor_metadata)
        
        return {
            'monitor_metadata': monitor_row.to_dict(),
            'time_series_data': cleaned_data,
            'graph_structure': {
                'adjacency_matrix': adjacency_matrix,
                'edge_list': edge_list,
                'all_monitors': all_monitor_metadata
            },
            'monitor_index': all_monitor_metadata[
                all_monitor_metadata['monitor_id'] == monitor_id
            ].index[0] if not all_monitor_metadata.empty else 0
        }