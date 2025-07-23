"""
MinIO cache management script for machine learning data
Use this to manage cached preprocessed data
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.machine_learning.data.minio_manager import get_minio_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def show_cache_stats():
    """Show cache statistics"""
    cache = get_minio_cache()
    
    if cache is None:
        print("MinIO cache not available")
        return
    
    if not cache.health_check():
        print("MinIO cache health check failed")
        return
    
    stats = cache.get_cache_stats()
    
    if 'error' in stats:
        print(f"Error getting cache stats: {stats['error']}")
        return
    
    print("\nMinIO Cache Statistics:")
    print("=" * 50)
    print(f"Endpoint: {stats['endpoint']}")
    print(f"Bucket: {stats['bucket_name']}")
    print(f"Total Objects: {stats['total_objects']}")
    print(f"Total Size: {stats['total_size_mb']:.2f} MB ({stats['total_size_bytes']} bytes)")
    
    if stats['oldest_cache']:
        print(f"Oldest Cache: {stats['oldest_cache']}")
    if stats['newest_cache']:
        print(f"Newest Cache: {stats['newest_cache']}")

def clear_cache(pattern=None):
    """Clear cache with optional pattern"""
    cache = get_minio_cache()
    
    if cache is None:
        print("MinIO cache not available")
        return
    
    if not cache.health_check():
        print("MinIO cache health check failed")
        return
    
    print(f"Clearing cache{' with pattern: ' + pattern if pattern else ''}...")
    deleted_count = cache.clear_cache(pattern)
    print(f"Deleted {deleted_count} cache objects")

def test_cache():
    """Test cache functionality"""
    cache = get_minio_cache()
    
    if cache is None:
        print("MinIO cache not available")
        return
    
    print("Testing MinIO cache functionality...")
    
    # Health check
    if cache.health_check():
        print("✓ Health check passed")
    else:
        print("✗ Health check failed")
        return
    
    # Test data caching
    test_data = {
        'test': True,
        'data': [1, 2, 3, 4, 5],
        'metadata': {'created_at': '2024-01-01'}
    }
    
    test_interims = ['Interim1', 'Interim2']
    test_monitors = ['TEST001', 'TEST002']
    
    print("Testing data caching...")
    
    # Cache test data
    cache_result = asyncio.run(
        cache.cache_data(test_data, test_interims, test_monitors)
    )
    
    if cache_result:
        print("✓ Data caching successful")
    else:
        print("✗ Data caching failed")
        return
    
    # Retrieve test data
    retrieved_data = asyncio.run(
        cache.get_cached_data(test_interims, test_monitors)
    )
    
    if retrieved_data and 'data' in retrieved_data:
        if retrieved_data['data'] == test_data:
            print("✓ Data retrieval successful")
        else:
            print("✗ Retrieved data doesn't match original")
    else:
        print("✗ Data retrieval failed")
    
    # Clean up test data
    cache.clear_cache('preprocessed_data_')
    print("✓ Test data cleaned up")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Manage MinIO cache for ML data')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show cache statistics')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear cache')
    clear_parser.add_argument('--pattern', type=str, help='Pattern to match for deletion')
    clear_parser.add_argument('--confirm', action='store_true', help='Confirm deletion')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test cache functionality')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check cache health')
    
    args = parser.parse_args()
    
    if args.command == 'stats':
        show_cache_stats()
    
    elif args.command == 'clear':
        if not args.confirm:
            print("This will delete cached data. Use --confirm to proceed.")
            return
        clear_cache(args.pattern)
    
    elif args.command == 'test':
        test_cache()
    
    elif args.command == 'health':
        cache = get_minio_cache()
        if cache is None:
            print("MinIO cache not available")
        elif cache.health_check():
            print("MinIO cache is healthy")
        else:
            print("MinIO cache health check failed")
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python manage_cache.py stats                    # Show cache statistics")
        print("  python manage_cache.py health                   # Check cache health")
        print("  python manage_cache.py test                     # Test cache functionality")
        print("  python manage_cache.py clear --confirm          # Clear all cache")
        print("  python manage_cache.py clear --pattern data --confirm  # Clear specific pattern")

if __name__ == "__main__":
    main()