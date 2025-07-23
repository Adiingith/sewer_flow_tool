"""
Test script for T-GNN storm response prediction
Use this script to test predictions on specific monitors
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.DB_connect import get_session
from backend.machine_learning.inference.predictor import get_predictor, predict_storm_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_single_prediction(monitor_id: str, interim: str):
    """
    Test prediction for a single monitor
    
    Args:
        monitor_id: Monitor business ID
        interim: Interim period
    """
    async with get_session() as db_session:
        logger.info(f"Testing prediction for monitor {monitor_id}, interim {interim}")
        
        result = await predict_storm_response(db_session, monitor_id, interim)
        
        print(f"\nPrediction Result for {monitor_id} ({interim}):")
        print("=" * 60)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Predicted Label: {result['predicted_label']}")
            print(f"Confidence: {result['confidence']:.3f} ({result['confidence_level']})")
            print(f"Probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob:.3f}")
            print(f"Suggestion: {result['suggestion']}")
            print(f"Model: {result['model_info']['model_name']} "
                  f"({result['model_info']['model_stage']})")

async def test_batch_prediction(monitor_ids: list, interim: str):
    """
    Test batch prediction for multiple monitors
    
    Args:
        monitor_ids: List of monitor business IDs
        interim: Interim period
    """
    async with get_session() as db_session:
        predictor = await get_predictor()
        
        logger.info(f"Testing batch prediction for {len(monitor_ids)} monitors, interim {interim}")
        
        results = await predictor.predict_batch(db_session, monitor_ids, interim)
        
        print(f"\nBatch Prediction Results ({interim}):")
        print("=" * 80)
        
        for result in results:
            monitor_id = result.get('monitor_id', 'Unknown')
            
            if 'error' in result:
                print(f"{monitor_id}: ERROR - {result['error']}")
            else:
                predicted = result['predicted_label']
                confidence = result['confidence']
                conf_level = result['confidence_level']
                print(f"{monitor_id}: {predicted} (conf: {confidence:.3f}, {conf_level})")

async def test_model_info():
    """
    Test model information retrieval
    """
    predictor = await get_predictor()
    model_info = predictor.get_model_info()
    
    print("\nModel Information:")
    print("=" * 50)
    
    if 'error' in model_info:
        print(f"Error: {model_info['error']}")
    else:
        print(f"Model Name: {model_info['model_name']}")
        print(f"Model Version: {model_info['model_version']}")
        print(f"Model Stage: {model_info['model_stage']}")
        print(f"Device: {model_info['device']}")
        print(f"Total Parameters: {model_info['parameters']:,}")
        print(f"Trainable Parameters: {model_info['trainable_parameters']:,}")
        
        if 'config' in model_info and model_info['config']:
            print(f"\nModel Configuration:")
            config = model_info['config']
            for key, value in config.items():
                print(f"  {key}: {value}")

async def get_available_data():
    """
    Get information about available data in the system
    """
    from backend.machine_learning.data.data_loader import StormResponseDataLoader
    
    async with get_session() as db_session:
        data_loader = StormResponseDataLoader(db_session)
        
        # Get available interims
        interims = await data_loader.get_available_interims()
        print(f"\nAvailable Interims: {len(interims)}")
        for interim in interims[-10:]:  # Show last 10
            print(f"  {interim}")
        
        # Get monitor metadata
        monitors = await data_loader.get_monitor_metadata()
        print(f"\nTotal Monitors: {len(monitors)}")
        print(f"Monitor Types:")
        type_counts = monitors['type'].value_counts()
        for monitor_type, count in type_counts.items():
            print(f"  {monitor_type}: {count}")
        
        # Show some example monitor IDs
        print(f"\nExample Monitor IDs:")
        for monitor_id in monitors['monitor_id'].head(10):
            print(f"  {monitor_id}")

async def main():
    """
    Main test function
    """
    parser = argparse.ArgumentParser(description='Test T-GNN storm response prediction')
    
    parser.add_argument('--monitor-id', type=str, help='Single monitor ID to test')
    parser.add_argument('--monitor-ids', nargs='+', help='Multiple monitor IDs for batch test')
    parser.add_argument('--interim', type=str, help='Interim period (e.g., Interim1)')
    parser.add_argument('--model-info', action='store_true', help='Show model information')
    parser.add_argument('--data-info', action='store_true', help='Show available data information')
    parser.add_argument('--model-name', type=str, default='TGNNStormClassifier', help='Model name')
    parser.add_argument('--model-stage', type=str, default='Production', help='Model stage')
    
    args = parser.parse_args()
    
    try:
        if args.data_info:
            await get_available_data()
            return
        
        if args.model_info:
            await test_model_info()
            return
        
        if not args.interim:
            logger.error("Please provide --interim parameter")
            return
        
        if args.monitor_id:
            # Test single prediction
            await test_single_prediction(args.monitor_id, args.interim)
        
        elif args.monitor_ids:
            # Test batch prediction
            await test_batch_prediction(args.monitor_ids, args.interim)
        
        else:
            logger.error("Please provide either --monitor-id or --monitor-ids")
            print("\nUsage examples:")
            print("  # Show available data")
            print("  python test_prediction.py --data-info")
            print("")
            print("  # Show model information")
            print("  python test_prediction.py --model-info")
            print("")
            print("  # Test single monitor")
            print("  python test_prediction.py --monitor-id FM001 --interim Interim1")
            print("")
            print("  # Test multiple monitors")
            print("  python test_prediction.py --monitor-ids FM001 FM002 FM003 --interim Interim1")
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())