"""
MinIO manager for caching preprocessed machine learning data
Avoids reprocessing data multiple times during model training
"""
import pickle
import gzip
import json
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import os
from pathlib import Path

try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    logging.warning("MinIO client not available. Install with: pip install minio")

from backend.machine_learning.utils.logger_config import get_cache_logger, log_function_call

logger = get_cache_logger()

class MinIODataCache:
    """
    MinIO-based cache for preprocessed machine learning data
    """
    
    def __init__(self, 
                 endpoint: str = "localhost:9000",
                 access_key: str = None,
                 secret_key: str = None,
                 bucket_name: str = "ml-data-cache",
                 secure: bool = False):
        """
        Initialize MinIO cache manager
        
        Args:
            endpoint: MinIO server endpoint
            access_key: MinIO access key
            secret_key: MinIO secret key
            bucket_name: Bucket name for caching
            secure: Use HTTPS
        """
        if not MINIO_AVAILABLE:
            raise ImportError("MinIO client not available. Install with: pip install minio")
        
        # Get credentials from environment if not provided
        self.access_key = access_key or os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
        self.secret_key = secret_key or os.getenv('MINIO_SECRET_KEY', 'minioadmin')
        self.endpoint = endpoint or os.getenv('MINIO_ENDPOINT', 'localhost:9000')
        self.bucket_name = bucket_name
        
        # Initialize MinIO client
        try:
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=secure
            )
            
            # Create bucket if it doesn't exist
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created MinIO bucket: {self.bucket_name}")
            
            self.is_available = True
            logger.info(f"MinIO cache initialized: {self.endpoint}/{self.bucket_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {e}")
            self.is_available = False
            self.client = None
    
    def _generate_cache_key(self, data_params: Dict[str, Any]) -> str:
        """
        Generate cache key based on data parameters
        
        Args:
            data_params: Parameters that define the data
            
        Returns:
            Cache key string
        """
        # Create a hash from the parameters
        param_str = json.dumps(data_params, sort_keys=True)
        cache_hash = hashlib.md5(param_str.encode()).hexdigest()
        
        # Include timestamp for cache versioning
        timestamp = datetime.now().strftime("%Y%m%d")
        
        return f"preprocessed_data_{timestamp}_{cache_hash}.pkl.gz"
    
    def _get_data_signature(self, interims: List[str], 
                           monitor_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create data signature for cache key generation
        
        Args:
            interims: List of interim periods
            monitor_ids: List of monitor IDs (optional)
            
        Returns:
            Data signature dictionary
        """
        signature = {
            'interims': sorted(interims) if interims else [],
            'monitor_ids': sorted(monitor_ids) if monitor_ids else 'all',
            'data_version': '1.0',  # Increment when data structure changes
            'feature_version': '1.0'  # Increment when feature extraction changes
        }
        return signature
    
    @log_function_call(logger)
    async def get_cached_data(self, interims: List[str], 
                            monitor_ids: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached preprocessed data
        
        Args:
            interims: List of interim periods
            monitor_ids: List of monitor IDs (optional)
            
        Returns:
            Cached data or None if not found
        """
        if not self.is_available:
            return None
        
        try:
            # Generate cache key
            data_signature = self._get_data_signature(interims, monitor_ids)
            cache_key = self._generate_cache_key(data_signature)
            
            # Check if object exists
            try:
                self.client.stat_object(self.bucket_name, cache_key)
            except S3Error as e:
                if e.code == 'NoSuchKey':
                    logger.info(f"Cache miss: {cache_key}")
                    return None
                raise
            
            # Download and decompress data
            response = self.client.get_object(self.bucket_name, cache_key)
            compressed_data = response.read()
            response.close()
            
            # Decompress and deserialize
            decompressed_data = gzip.decompress(compressed_data)
            cached_data = pickle.loads(decompressed_data)
            
            logger.info(f"Cache hit: {cache_key} ({len(compressed_data)} bytes)")
            return cached_data
            
        except Exception as e:
            logger.error(f"Error retrieving cached data: {e}")
            return None
    
    @log_function_call(logger)
    async def cache_data(self, data: Dict[str, Any], interims: List[str],
                        monitor_ids: Optional[List[str]] = None) -> bool:
        """
        Cache preprocessed data
        
        Args:
            data: Preprocessed data to cache
            interims: List of interim periods
            monitor_ids: List of monitor IDs (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available:
            return False
        
        try:
            # Generate cache key
            data_signature = self._get_data_signature(interims, monitor_ids)
            cache_key = self._generate_cache_key(data_signature)
            
            # Add metadata to data
            cached_data = {
                'data': data,
                'metadata': {
                    'cached_at': datetime.utcnow().isoformat(),
                    'data_signature': data_signature,
                    'cache_version': '1.0'
                }
            }
            
            # Serialize and compress
            serialized_data = pickle.dumps(cached_data)
            compressed_data = gzip.compress(serialized_data)
            
            # Upload to MinIO
            from io import BytesIO
            data_stream = BytesIO(compressed_data)
            
            self.client.put_object(
                self.bucket_name,
                cache_key,
                data_stream,
                length=len(compressed_data),
                content_type='application/octet-stream'
            )
            
            logger.info(f"Data cached: {cache_key} ({len(compressed_data)} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Error caching data: {e}")
            return False
    
    def clear_cache(self, pattern: str = None) -> int:
        """
        Clear cached data
        
        Args:
            pattern: Pattern to match object names (optional)
            
        Returns:
            Number of objects deleted
        """
        if not self.is_available:
            return 0
        
        try:
            objects = self.client.list_objects(self.bucket_name, recursive=True)
            deleted_count = 0
            
            for obj in objects:
                if pattern is None or pattern in obj.object_name:
                    self.client.remove_object(self.bucket_name, obj.object_name)
                    deleted_count += 1
                    logger.info(f"Deleted cache object: {obj.object_name}")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.is_available:
            return {'error': 'MinIO not available'}
        
        try:
            objects = list(self.client.list_objects(self.bucket_name, recursive=True))
            
            total_size = 0
            oldest_date = None
            newest_date = None
            
            for obj in objects:
                total_size += obj.size
                
                if oldest_date is None or obj.last_modified < oldest_date:
                    oldest_date = obj.last_modified
                
                if newest_date is None or obj.last_modified > newest_date:
                    newest_date = obj.last_modified
            
            return {
                'total_objects': len(objects),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'oldest_cache': oldest_date.isoformat() if oldest_date else None,
                'newest_cache': newest_date.isoformat() if newest_date else None,
                'bucket_name': self.bucket_name,
                'endpoint': self.endpoint
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}
    
    def health_check(self) -> bool:
        """
        Check if MinIO connection is healthy
        
        Returns:
            True if healthy, False otherwise
        """
        if not self.is_available:
            return False
        
        try:
            # Try to list objects (minimal operation)
            list(self.client.list_objects(self.bucket_name, recursive=False))
            return True
        except Exception as e:
            logger.error(f"MinIO health check failed: {e}")
            return False

# Global cache instance
_cache_instance = None

def get_minio_cache(
    endpoint: str = None,
    access_key: str = None,
    secret_key: str = None,
    bucket_name: str = "ml-data-cache"
) -> Optional[MinIODataCache]:
    """
    Get global MinIO cache instance
    
    Args:
        endpoint: MinIO endpoint
        access_key: Access key
        secret_key: Secret key
        bucket_name: Bucket name
        
    Returns:
        MinIODataCache instance or None if MinIO not available
    """
    global _cache_instance
    
    if _cache_instance is None and MINIO_AVAILABLE:
        try:
            _cache_instance = MinIODataCache(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                bucket_name=bucket_name
            )
        except Exception as e:
            logger.warning(f"Could not initialize MinIO cache: {e}")
            _cache_instance = None
    
    return _cache_instance