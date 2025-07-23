"""
Training script for T-GNN storm response classification model
Run this script to train the model with data from the database
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.DB_connect import get_session
from backend.machine_learning.data.data_loader import StormResponseDataLoader
from backend.machine_learning.features.feature_engineering import TGNNDataProcessor
from backend.machine_learning.training.trainer import TGNNTrainer
from backend.machine_learning.models.tgnn_model import DEFAULT_TGNN_CONFIG
from backend.machine_learning.utils.logger_config import get_training_logger, get_logger_manager

# Logger will be initialized with run name later
logger = None

async def load_and_process_data(interims=None, monitor_ids=None, use_cache=True):
    """
    Load and process data for training
    
    Args:
        interims: List of interim periods to include
        monitor_ids: List of monitor IDs to include
        use_cache: Whether to use MinIO caching
        
    Returns:
        Processed dataset
    """
    async with get_session() as db_session:
        # Load data
        data_loader = StormResponseDataLoader(db_session, use_cache=use_cache)
        dataset = await data_loader.load_dataset_for_training(interims, monitor_ids)
        
        if 'error' in dataset:
            logger.error(f"Failed to load dataset: {dataset['error']}")
            return None
        
        # Count samples
        total_samples = sum(
            len(labels) for labels in dataset['labels'].values()
        )
        logger.info(f"Loaded dataset with {total_samples} labeled samples")
        
        # Process features
        processor = TGNNDataProcessor()
        processed_data = processor.process_dataset_for_training(dataset)
        
        logger.info(f"Processed {len(processed_data['samples'])} samples for training")
        return processed_data

def create_training_config(args) -> Dict[str, Any]:
    """
    Create training configuration from arguments
    
    Args:
        args: Command line arguments
        
    Returns:
        Training configuration dictionary
    """
    config = DEFAULT_TGNN_CONFIG.copy()
    
    # Update with command line arguments
    if args.hidden_dim:
        config['temporal_hidden_dim'] = args.hidden_dim
        config['spatial_hidden_dim'] = args.hidden_dim
    
    if args.temporal_encoder:
        config['temporal_encoder_type'] = args.temporal_encoder
    
    if args.spatial_encoder:
        config['spatial_encoder_type'] = args.spatial_encoder
    
    if args.epochs:
        config['epochs'] = args.epochs
    
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    
    if args.dropout:
        config['dropout'] = args.dropout
    
    # Add training-specific parameters
    config.update({
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'weight_decay': args.weight_decay,
        'patience': args.patience
    })
    
    return config

async def main():
    """
    Main training function
    """
    parser = argparse.ArgumentParser(description='Train T-GNN storm response classification model')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--temporal-encoder', choices=['gru', 'tcn', 'cnn'], default='gru', help='Temporal encoder type')
    parser.add_argument('--spatial-encoder', choices=['gcn', 'gat', 'sage'], default='gcn', help='Spatial encoder type')
    parser.add_argument('--temporal-layers', type=int, default=2, help='Number of temporal layers')
    parser.add_argument('--spatial-layers', type=int, default=2, help='Number of spatial layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', choices=['adamw', 'adam', 'sgd'], default='adamw', help='Optimizer')
    parser.add_argument('--scheduler', choices=['cosine', 'step', 'plateau'], default='cosine', help='Learning rate scheduler')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    # Data parameters
    parser.add_argument('--interims', nargs='+', help='Specific interim periods to use (e.g., Interim1 Interim2)')
    parser.add_argument('--monitor-ids', nargs='+', help='Specific monitor IDs to use')
    parser.add_argument('--no-cache', action='store_true', help='Disable MinIO caching')
    parser.add_argument('--cross-validate', action='store_true', help='Perform cross-validation')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds')
    
    # MLflow parameters
    parser.add_argument('--run-name', type=str, help='MLflow run name')
    parser.add_argument('--experiment-name', type=str, default='storm_response_tgnn', help='MLflow experiment name')
    
    # Device
    parser.add_argument('--device', type=str, help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize logger with run name
    global logger
    run_name = args.run_name or f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = get_training_logger(run_name)
    
    logger.info("Starting T-GNN training script")
    logger.info(f"Run name: {run_name}")
    
    try:
        # Load and process data
        logger.info("Loading and processing data...")
        use_cache = not args.no_cache
        processed_data = await load_and_process_data(
            args.interims, 
            args.monitor_ids, 
            use_cache
        )
        
        if processed_data is None:
            logger.error("Failed to load data")
            return
        
        if len(processed_data['samples']) == 0:
            logger.error("No samples found in dataset")
            return
        
        # Create training configuration
        config = create_training_config(args)
        logger.info(f"Training configuration: {json.dumps(config, indent=2)}")
        
        # Initialize trainer
        trainer = TGNNTrainer(config, device=args.device, run_name=run_name)
        
        if args.cross_validate:
            # Perform cross-validation
            logger.info(f"Starting {args.cv_folds}-fold cross-validation...")
            cv_results = trainer.cross_validate(processed_data, n_splits=args.cv_folds)
            
            # Print CV results
            print("\nCross-Validation Results:")
            print("=" * 50)
            for metric, values in cv_results.items():
                mean_val = sum(values) / len(values)
                std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
                print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        else:
            # Train single model
            logger.info("Starting model training...")
            model, history = await trainer.train(processed_data, args.run_name)
            
            # Print final results
            print("\nTraining completed!")
            print("=" * 50)
            print(f"Final validation F1 score: {history['val_f1_macro'][-1]:.4f}")
            print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
            
            logger.info("Model training completed successfully")
    
    except Exception as e:
        get_logger_manager().log_exception(logger, e, "Training script execution")
        raise

if __name__ == "__main__":
    asyncio.run(main())