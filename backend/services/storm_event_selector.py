import pandas as pd
import numpy as np
from scipy import signal
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass


# Remove logging import and logger definition

@dataclass
class StormEventConfig:
    """storm event extraction configuration"""
    min_peak_threshold: float = 3.0  # mm/h
    min_duration_minutes: int = 30
    rolling_window: int = 3
    min_peak_distance_minutes: int = 60
    expansion_before_minutes: int = 30
    expansion_after_minutes: int = 60
    merge_interval_minutes: int = 30
    min_rain_threshold: float = 0.2  # mm/h, used to expand event boundaries

@dataclass
class StormEvent:
    """storm event data structure"""
    start_time: datetime
    end_time: datetime
    total_rain: float
    peak_intensity: float
    duration_minutes: int
    storm_type: str  # A/B/C based on intensity

class StormEventSelector:
    """storm event selector"""
    
    def __init__(self, config: Optional[StormEventConfig] = None):
        self.config = config or StormEventConfig()
    
    def validate_rain_data(self, rain_series: pd.Series, timestamps: pd.Series) -> bool:
        """
        validate rain data quality
        
        Args:
            rain_series: rain data series
            timestamps: timestamp series
            
        Returns:
            bool: whether the data is valid
        """
        if len(rain_series) != len(timestamps):
            return False
        
        if rain_series.isnull().all():
            return False
        
        if len(rain_series) < self.config.min_duration_minutes:
            return False
        
        # check time continuity - allow different but consistent intervals
        time_diff = timestamps.diff().dropna()
        if len(time_diff) == 0:
            return False
        
        # Get the most common time interval
        most_common_interval = time_diff.mode()
        if len(most_common_interval) == 0:
            return False
        
        expected_interval = most_common_interval[0]
        
        # Allow up to 10% of data points to have different intervals (for minor inconsistencies)
        consistent_intervals = (time_diff == expected_interval).sum()
        total_intervals = len(time_diff)
        consistency_ratio = consistent_intervals / total_intervals
        
        if consistency_ratio < 0.9:  # Require at least 90% consistency
            return False
        
        return True
    
    def smooth_rain_data(self, rain_series: pd.Series) -> pd.Series:
        """
        smooth rain data
        
        Args:
            rain_series: original rain data
            
        Returns:
            pd.Series: smoothed data
        """
        return rain_series.rolling(window=self.config.rolling_window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    
    def find_rain_peaks(self, rain_series: pd.Series, time_interval_minutes: int = 1) -> np.ndarray:
        """
        find rain peaks
        
        Args:
            rain_series: rain data series
            time_interval_minutes: time interval between data points in minutes
            
        Returns:
            np.ndarray: peak index array
        """
        # Calculate distance in data points based on actual time interval
        distance_in_points = self.config.min_peak_distance_minutes // time_interval_minutes
        
        # use scipy.signal.find_peaks to find peaks
        peaks, _ = signal.find_peaks(
            rain_series.values,
            height=self.config.min_peak_threshold,
            distance=distance_in_points
        )
        return peaks
    
    def expand_event_boundaries(self, peak_idx: int, rain_series: pd.Series, time_interval_minutes: int = 1) -> Tuple[int, int]:
        """
        expand event boundaries
        
        Args:
            peak_idx: peak index
            rain_series: rain data series
            time_interval_minutes: time interval between data points in minutes
            
        Returns:
            Tuple[int, int]: (start index, end index)
        """
        start_idx = peak_idx
        end_idx = peak_idx
        
        # Calculate expansion range in data points
        expansion_before_points = self.config.expansion_before_minutes // time_interval_minutes
        expansion_after_points = self.config.expansion_after_minutes // time_interval_minutes
        
        # expand forward (to earlier times)
        for i in range(peak_idx - 1, max(0, peak_idx - expansion_before_points), -1):
            if rain_series.iloc[i] > self.config.min_rain_threshold:
                start_idx = i
            else:
                break
        
        # expand backward (to later times)
        for i in range(peak_idx + 1, min(len(rain_series), peak_idx + expansion_after_points)):
            if rain_series.iloc[i] > self.config.min_rain_threshold:
                end_idx = i
            else:
                break
        
        # Ensure start_idx <= end_idx
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        
        return start_idx, end_idx
    
    def merge_close_events(self, events: List[StormEvent]) -> List[StormEvent]:
        """
        merge close events
        
        Args:
            events: event list
            
        Returns:
            List[StormEvent]: merged event list
        """
        if not events:
            return events
        
        merged_events = []
        current_event = events[0]
        
        for next_event in events[1:]:
            time_gap = (next_event.start_time - current_event.end_time).total_seconds() / 60
            
            if time_gap <= self.config.merge_interval_minutes:
                # merge events
                current_event.end_time = next_event.end_time
                current_event.total_rain += next_event.total_rain
                current_event.peak_intensity = max(current_event.peak_intensity, next_event.peak_intensity)
                current_event.duration_minutes = int((current_event.end_time - current_event.start_time).total_seconds() / 60)
                current_event.storm_type = self.determine_storm_type(max(current_event.peak_intensity, next_event.peak_intensity))
            else:
                merged_events.append(current_event)
                current_event = next_event
        
        merged_events.append(current_event)
        return merged_events
    
    def determine_storm_type(self, peak_intensity: float) -> str:
        """
        determine storm type based on peak intensity
        
        Args:
            peak_intensity: peak intensity (mm/h)
            
        Returns:
            str: storm type A/B/C
        """
        if peak_intensity >= 50:
            return "A"
        elif peak_intensity >= 25:
            return "B"
        else:
            return "C"
    
    def calculate_event_metrics(self, start_idx: int, end_idx: int, 
                               rain_series: pd.Series, timestamps: pd.Series) -> StormEvent:
        """
        calculate event metrics
        
        Args:
            start_idx: start index
            end_idx: end index
            rain_series: rain data series
            timestamps: timestamp series
            
        Returns:
            StormEvent: event object
        """
        # Ensure start_idx <= end_idx
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        
        event_rain = rain_series.iloc[start_idx:end_idx + 1]
        event_times = timestamps.iloc[start_idx:end_idx + 1]
        
        # Ensure times are in correct order
        start_time = event_times.iloc[0]
        end_time = event_times.iloc[-1]
        
        # Validate that start_time <= end_time
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        
        interval_minutes = 2  # data interval is 2 minutes
        total_rain = event_rain.sum() * interval_minutes / 60  # mm
        peak_intensity = event_rain.max()
        duration_minutes = int((end_time - start_time).total_seconds() / 60) + interval_minutes
        storm_type = self.determine_storm_type(peak_intensity)
        return StormEvent(
            start_time=start_time,
            end_time=end_time,
            total_rain=total_rain,
            peak_intensity=peak_intensity,
            duration_minutes=duration_minutes,
            storm_type=storm_type
        )
    
    def extract_storm_events(self, rain_series: pd.Series, timestamps: pd.Series, 
                           max_events: int = 3, area: str = "default", 
                           interim: str = "default") -> List[Dict]:
        """
        extract storm events
        
        Args:
            rain_series: minute-level rain data (mm/h)
            timestamps: timestamp series
            max_events: maximum number of events
            area: area identifier
            interim: interim identifier
            
        Returns:
            List[Dict]: storm event list, each event contains fields compatible with WeatherEvent model
        """
        # data validation
        if not self.validate_rain_data(rain_series, timestamps):
            return []
        
        # data smoothing
        smoothed_rain = self.smooth_rain_data(rain_series)
        
        # find peaks
        # Calculate time interval from timestamps
        time_diff = timestamps.diff().dropna()
        time_interval_minutes = int(time_diff.mode()[0].total_seconds() / 60)
        
        peak_indices = self.find_rain_peaks(smoothed_rain, time_interval_minutes)
        
        if len(peak_indices) == 0:
            return []
        
        # expand event boundaries
        events = []
        for peak_idx in peak_indices:
            start_idx, end_idx = self.expand_event_boundaries(peak_idx, rain_series, time_interval_minutes)
            
            # check event duration (convert data points to actual minutes)
            duration_points = end_idx - start_idx + 1
            duration_minutes = duration_points * time_interval_minutes
            
            if duration_minutes >= self.config.min_duration_minutes:
                event = self.calculate_event_metrics(start_idx, end_idx, rain_series, timestamps)
                events.append(event)
        
        if not events:
            return []
        
        # merge close events
        events = self.merge_close_events(events)
        
        # sort by total rain, select top-K
        events.sort(key=lambda x: x.total_rain, reverse=True)
        selected_events = events[:max_events]
        
        # format output
        formatted_events = []
        for event in selected_events:
            formatted_event = self.format_storm_event(event, area, interim)
            formatted_events.append(formatted_event)
        
        return formatted_events
    
    def format_storm_event(self, event: StormEvent, area: str, interim: str) -> Dict:
        """
        format storm event data to match WeatherEvent model
        
        Args:
            event: storm event object
            area: area identifier
            interim: interim identifier
            
        Returns:
            Dict: formatted event data
        """
        return {
            "event_type": "weekly_storm",
            "start_time": event.start_time,
            "end_time": event.end_time,
            "area": area,
            "interim": interim,
            "storm_type": event.storm_type,
            "dry_day_number": None,
            "coverage": None,
            "event_comment": {
                "total_rain": float(event.total_rain),
                "peak_intensity": float(event.peak_intensity),
                "duration_minutes": event.duration_minutes,
                "extraction_method": "storm_event_selector",
                "config_used": {
                    "min_peak_threshold": self.config.min_peak_threshold,
                    "min_duration_minutes": self.config.min_duration_minutes,
                    "rolling_window": self.config.rolling_window
                }
            },
            "rain_gauge_monitor_id": None,
            "rain_gauge_interim": None
        }
    
    def process_rain_gauge_data(self, rain_gauge_data: pd.DataFrame, 
                               monitor_id: int, interim: str,
                               max_events: int = 3) -> List[Dict]:
        """
            process rain gauge data and extract storm events
        
        Args:
            rain_gauge_data: rain gauge data DataFrame, contains timestamp and intensity_mm_per_hr columns
            monitor_id: rain gauge monitor ID
            interim: interim identifier
            max_events: maximum number of events
            
        Returns:
            List[Dict]: storm event list
        """
        if rain_gauge_data.empty:
            return []
        
        # ensure data is sorted by time
        rain_gauge_data = rain_gauge_data.sort_values('timestamp')
        
        # Validate that timestamps are in ascending order
        timestamps = rain_gauge_data['timestamp']
        if not timestamps.is_monotonic_increasing:
            # Re-sort to ensure monotonic order
            rain_gauge_data = rain_gauge_data.sort_values('timestamp').reset_index(drop=True)
            timestamps = rain_gauge_data['timestamp']
        
        # extract rain data and timestamps
        rain_series = rain_gauge_data['intensity_mm_per_hr'].fillna(0)
        
        # extract storm events
        events = self.extract_storm_events(rain_series, timestamps, max_events, "default", interim)
        
        # set rain_gauge_id - need to find the corresponding rain_gauge.id based on monitor_id and interim
        # here set to None, will be processed in calling function
        for event in events:
            event['rain_gauge_monitor_id'] = monitor_id
            event['rain_gauge_interim'] = interim
        
        return events 