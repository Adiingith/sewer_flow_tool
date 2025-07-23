"""
Feature engineering module for T-GNN storm response classification
Handles time series feature extraction and spatial feature combination
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import logging
from scipy import signal
from scipy.stats import skew, kurtosis
from backend.machine_learning.utils.logger_config import get_ml_logger, log_function_call

logger = get_ml_logger("feature_engineering")

class TimeSeriesFeatureExtractor:
    """
    Extract features from time series data for T-GNN input
    """
    
    def __init__(self, sequence_length: int = 2016):  # One week at 5-min intervals
        self.sequence_length = sequence_length
        self.scalers = {}
    
    def extract_statistical_features(self, ts_data: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from time series
        
        Args:
            ts_data: Time series data (n_samples, n_channels)
            
        Returns:
            Statistical features array
        """
        features = []
        
        for channel in range(ts_data.shape[1]):
            channel_data = ts_data[:, channel]
            channel_data = channel_data[~np.isnan(channel_data)]  # Remove NaN values
            
            if len(channel_data) == 0:
                # Handle empty channel
                channel_features = [0] * 12
            else:
                channel_features = [
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.median(channel_data),
                    np.min(channel_data),
                    np.max(channel_data),
                    np.percentile(channel_data, 25),
                    np.percentile(channel_data, 75),
                    skew(channel_data),
                    kurtosis(channel_data),
                    np.sum(channel_data),
                    len(channel_data[channel_data > 0]) / len(channel_data),  # Non-zero ratio
                    np.var(channel_data)
                ]
            
            features.extend(channel_features)
        
        return np.array(features)
    
    def extract_temporal_patterns(self, ts_data: np.ndarray, timestamps: pd.DatetimeIndex = None) -> np.ndarray:
        """
        Extract temporal patterns and trends
        
        Args:
            ts_data: Time series data
            timestamps: Corresponding timestamps
            
        Returns:
            Temporal features array
        """
        features = []
        
        for channel in range(ts_data.shape[1]):
            channel_data = ts_data[:, channel]
            
            # Trend features
            if len(channel_data) > 1:
                # Linear trend slope
                x = np.arange(len(channel_data))
                slope = np.polyfit(x, channel_data, 1)[0]
                features.append(slope)
                
                # First difference statistics
                diff = np.diff(channel_data)
                features.extend([
                    np.mean(diff),
                    np.std(diff),
                    np.sum(diff > 0) / len(diff) if len(diff) > 0 else 0  # Increasing ratio
                ])
            else:
                features.extend([0, 0, 0, 0])
        
        # Add temporal features if timestamps available
        if timestamps is not None:
            # Hour of day patterns
            hours = timestamps.hour
            features.extend([
                np.mean(hours),
                np.std(hours)
            ])
            
            # Day of week patterns
            dow = timestamps.dayofweek
            features.extend([
                np.mean(dow),
                np.std(dow)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features)
    
    def extract_frequency_features(self, ts_data: np.ndarray, sampling_rate: float = 1/300) -> np.ndarray:
        """
        Extract frequency domain features
        
        Args:
            ts_data: Time series data
            sampling_rate: Sampling rate in Hz (default: 1/300 for 5-min intervals)
            
        Returns:
            Frequency features array
        """
        features = []
        
        for channel in range(ts_data.shape[1]):
            channel_data = ts_data[:, channel]
            
            if len(channel_data) > 8:  # Need sufficient data for FFT
                try:
                    # FFT
                    fft = np.fft.fft(channel_data)
                    freqs = np.fft.fftfreq(len(channel_data), 1/sampling_rate)
                    power_spectrum = np.abs(fft) ** 2
                    
                    # Spectral features
                    features.extend([
                        np.sum(power_spectrum),  # Total power
                        freqs[np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1],  # Dominant frequency
                        np.sum(power_spectrum[:len(power_spectrum)//4]) / np.sum(power_spectrum),  # Low frequency ratio
                        np.std(power_spectrum)  # Spectral variability
                    ])
                except:
                    features.extend([0, 0, 0, 0])
            else:
                features.extend([0, 0, 0, 0])
        
        return np.array(features)
    
    def pad_or_truncate_sequence(self, ts_data: np.ndarray) -> np.ndarray:
        """
        Pad or truncate time series to fixed length
        
        Args:
            ts_data: Time series data
            
        Returns:
            Fixed-length time series
        """
        if len(ts_data) >= self.sequence_length:
            # Truncate to last sequence_length samples
            return ts_data[-self.sequence_length:]
        else:
            # Pad with zeros at the beginning
            padding = np.zeros((self.sequence_length - len(ts_data), ts_data.shape[1]))
            return np.vstack([padding, ts_data])
    
    def normalize_time_series(self, ts_data: np.ndarray, monitor_id: str, fit: bool = True) -> np.ndarray:
        """
        Normalize time series data per monitor
        
        Args:
            ts_data: Time series data
            monitor_id: Monitor identifier for scaler storage
            fit: Whether to fit the scaler
            
        Returns:
            Normalized time series
        """
        if monitor_id not in self.scalers:
            self.scalers[monitor_id] = StandardScaler()
        
        scaler = self.scalers[monitor_id]
        
        if fit:
            normalized = scaler.fit_transform(ts_data)
        else:
            normalized = scaler.transform(ts_data)
        
        return normalized
    
    def extract_all_features(self, ts_data: pd.DataFrame, monitor_id: str, 
                           fit_scaler: bool = True) -> Dict[str, np.ndarray]:
        """
        Extract all types of features from time series data
        
        Args:
            ts_data: Time series DataFrame with columns [depth, flow, velocity, rainfall]
            monitor_id: Monitor identifier
            fit_scaler: Whether to fit the normalizer
            
        Returns:
            Dictionary containing different feature types
        """
        # Extract the 4 channels
        channels = ['depth', 'flow', 'velocity', 'rainfall']
        ts_array = ts_data[channels].values
        
        # Handle NaN values
        ts_array = np.nan_to_num(ts_array, nan=0.0)
        
        # Pad or truncate to fixed length
        ts_sequence = self.pad_or_truncate_sequence(ts_array)
        
        # Normalize the sequence
        ts_normalized = self.normalize_time_series(ts_sequence, monitor_id, fit=fit_scaler)
        
        # Extract different types of features
        statistical_features = self.extract_statistical_features(ts_sequence)
        temporal_features = self.extract_temporal_patterns(
            ts_sequence, 
            ts_data['timestamp'] if 'timestamp' in ts_data.columns else None
        )
        frequency_features = self.extract_frequency_features(ts_sequence)
        
        return {
            'raw_sequence': ts_sequence,
            'normalized_sequence': ts_normalized,
            'statistical_features': statistical_features,
            'temporal_features': temporal_features,
            'frequency_features': frequency_features
        }

class SpatialFeatureProcessor:
    """
    Process spatial features for monitors
    """
    
    def __init__(self):
        self.spatial_scaler = StandardScaler()
        self.categorical_encodings = {}
    
    def encode_categorical_features(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical spatial features
        
        Args:
            metadata_df: Monitor metadata DataFrame
            
        Returns:
            DataFrame with encoded features
        """
        encoded_df = metadata_df.copy()
        
        # One-hot encode categorical features
        categorical_cols = ['type', 'shape', 'area']
        
        for col in categorical_cols:
            if col in encoded_df.columns:
                # Get unique values
                unique_vals = encoded_df[col].dropna().unique()
                
                # Create one-hot encoding
                for val in unique_vals:
                    encoded_df[f'{col}_{val}'] = (encoded_df[col] == val).astype(int)
                
                # Store encoding for later use
                self.categorical_encodings[col] = unique_vals
        
        return encoded_df
    
    def extract_spatial_features(self, metadata_df: pd.DataFrame) -> np.ndarray:
        """
        Extract and normalize spatial features
        
        Args:
            metadata_df: Monitor metadata DataFrame
            
        Returns:
            Spatial features array (n_monitors, n_spatial_features)
        """
        # Encode categorical features
        encoded_df = self.encode_categorical_features(metadata_df)
        
        # Select numeric spatial features
        numeric_features = [
            'height_mm', 'width_mm', 'depth_mm'
        ]
        
        # Add one-hot encoded features
        categorical_feature_cols = [col for col in encoded_df.columns 
                                  if any(cat in col for cat in ['type_', 'shape_', 'area_'])]
        
        feature_cols = numeric_features + categorical_feature_cols
        
        # Extract features, fill NaN with 0
        spatial_array = encoded_df[feature_cols].fillna(0).values
        
        # Normalize numeric features only
        if hasattr(self.spatial_scaler, 'n_features_in_'):
            spatial_normalized = self.spatial_scaler.transform(spatial_array)
        else:
            spatial_normalized = self.spatial_scaler.fit_transform(spatial_array)
        
        return spatial_normalized
    
    def get_node_features(self, time_features: Dict[str, np.ndarray], 
                         spatial_features: np.ndarray, monitor_idx: int) -> np.ndarray:
        """
        Combine time series and spatial features for a node
        
        Args:
            time_features: Dictionary of time series features
            spatial_features: Spatial features array
            monitor_idx: Index of the monitor
            
        Returns:
            Combined node features
        """
        # Combine different feature types
        combined_features = np.concatenate([
            time_features['statistical_features'],
            time_features['temporal_features'],
            time_features['frequency_features'],
            spatial_features[monitor_idx]
        ])
        
        return combined_features

class TGNNDataProcessor:
    """
    Main processor that combines all feature engineering for T-GNN
    """
    
    def __init__(self, sequence_length: int = 2016):
        self.ts_extractor = TimeSeriesFeatureExtractor(sequence_length)
        self.spatial_processor = SpatialFeatureProcessor()
        # 7 action categories based on Action和Data_quality映射关系.md
        self.label_encoder = {
            'no_action_continue_monitoring': 0,
            'investigate_storm_failure': 1,
            'investigate_dryday_failure': 2,
            'sensor_fault_or_ragging': 3,
            'partial_data_needs_review': 4,
            'partial_data_no_action': 5,
            'recommend_remove_or_relocate': 6
        }
        self.label_decoder = {v: k for k, v in self.label_encoder.items()}
    
    @log_function_call(logger)
    def process_dataset_for_training(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process complete dataset for T-GNN training
        
        Args:
            dataset: Raw dataset from data loader
            
        Returns:
            Processed dataset ready for T-GNN
        """
        if 'error' in dataset:
            return dataset
        
        monitor_metadata = dataset['monitor_metadata']
        time_series_data = dataset['time_series_data']
        labels = dataset['labels']
        interims = dataset['interims']
        
        # Process spatial features
        spatial_features = self.spatial_processor.extract_spatial_features(monitor_metadata)
        
        processed_data = {
            'graph_structure': dataset['graph_structure'],
            'spatial_features': spatial_features,
            'monitor_metadata': monitor_metadata,
            'samples': []
        }
        
        # Process each interim
        for interim in interims:
            if interim not in time_series_data or interim not in labels:
                continue
            
            interim_ts = time_series_data[interim]
            interim_labels = labels[interim]
            
            # Process each monitor in this interim
            for monitor_id, ts_df in interim_ts.items():
                if monitor_id not in interim_labels:
                    continue  # Skip monitors without labels
                
                # Find monitor index
                monitor_idx = monitor_metadata[
                    monitor_metadata['monitor_id'] == monitor_id
                ].index[0]
                
                # Extract time series features
                ts_features = self.ts_extractor.extract_all_features(
                    ts_df, monitor_id, fit_scaler=True
                )
                
                # Combine with spatial features
                node_features = self.spatial_processor.get_node_features(
                    ts_features, spatial_features, monitor_idx
                )
                
                # Encode label
                label_str = interim_labels[monitor_id]
                label_encoded = self.label_encoder.get(label_str, 1)  # Default to 'partial'
                
                # Create sample
                sample = {
                    'monitor_id': monitor_id,
                    'monitor_idx': monitor_idx,
                    'interim': interim,
                    'time_series_sequence': ts_features['normalized_sequence'],
                    'node_features': node_features,
                    'label': label_encoded,
                    'label_str': label_str
                }
                
                processed_data['samples'].append(sample)
        
        logger.info(f"Processed {len(processed_data['samples'])} samples for training")
        return processed_data
    
    def process_data_for_prediction(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data for real-time prediction
        
        Args:
            prediction_data: Raw prediction data from data loader
            
        Returns:
            Processed data ready for T-GNN prediction
        """
        if 'error' in prediction_data:
            return prediction_data
        
        monitor_metadata = prediction_data['monitor_metadata']
        ts_data = prediction_data['time_series_data']
        graph_structure = prediction_data['graph_structure']
        monitor_index = prediction_data['monitor_index']
        
        # Process spatial features for all monitors
        all_monitors_df = graph_structure['all_monitors']
        spatial_features = self.spatial_processor.extract_spatial_features(all_monitors_df)
        
        # Extract time series features for the target monitor
        monitor_id = monitor_metadata['monitor_id']
        ts_features = self.ts_extractor.extract_all_features(
            ts_data, monitor_id, fit_scaler=False  # Don't refit during prediction
        )
        
        # Combine features
        node_features = self.spatial_processor.get_node_features(
            ts_features, spatial_features, monitor_index
        )
        
        return {
            'monitor_id': monitor_id,
            'monitor_idx': monitor_index,
            'time_series_sequence': ts_features['normalized_sequence'],
            'node_features': node_features,
            'spatial_features': spatial_features,
            'graph_structure': graph_structure,
            'all_monitors': all_monitors_df
        }
    
    def create_batches(self, samples: List[Dict], batch_size: int = 32) -> List[Dict]:
        """
        Create batches for training
        
        Args:
            samples: List of processed samples
            batch_size: Batch size
            
        Returns:
            List of batches
        """
        batches = []
        
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]
            
            # Stack sequences and features
            sequences = np.stack([s['time_series_sequence'] for s in batch_samples])
            node_features = np.stack([s['node_features'] for s in batch_samples])
            labels = np.array([s['label'] for s in batch_samples])
            monitor_indices = np.array([s['monitor_idx'] for s in batch_samples])
            
            batch = {
                'sequences': sequences,
                'node_features': node_features,
                'labels': labels,
                'monitor_indices': monitor_indices,
                'monitor_ids': [s['monitor_id'] for s in batch_samples],
                'interims': [s['interim'] for s in batch_samples]
            }
            
            batches.append(batch)
        
        return batches