"""
Example usage of the unified logging system
This file demonstrates how to use the ML module's logging features
"""
import asyncio
import time
from backend.machine_learning.utils.logger_config import (
    get_training_logger, 
    get_prediction_logger, 
    get_data_logger,
    get_cache_logger,
    get_ml_logger,
    log_function_call,
    get_logger_manager
)

# Example 1: Basic logger usage
def basic_logging_example():
    """Demonstrate basic logging functionality"""
    logger = get_ml_logger("example")
    
    logger.info("Starting basic logging example")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Log with structured data
    metrics = {"accuracy": 0.95, "loss": 0.05, "epoch": 10}
    get_logger_manager().log_performance_metrics(
        logger, metrics, "Training epoch completion"
    )

# Example 2: Specialized loggers
def specialized_loggers_example():
    """Demonstrate specialized loggers for different operations"""
    
    # Training logger
    training_logger = get_training_logger("example_run")
    training_logger.info("Model training started")
    training_logger.info("Epoch 1/10 completed")
    
    # Prediction logger
    prediction_logger = get_prediction_logger()
    prediction_logger.info("Processing prediction request for monitor FM001")
    
    # Data processing logger
    data_logger = get_data_logger()
    data_logger.debug("Loading data from database")
    data_logger.info("Processed 1000 samples")
    
    # Cache logger
    cache_logger = get_cache_logger()
    cache_logger.info("Checking MinIO cache for key: data_20240101")

# Example 3: Function call logging decorator
@log_function_call(get_ml_logger("decorated_functions"))
def example_function(param1: str, param2: int = 10):
    """Example function with logging decorator"""
    time.sleep(0.1)  # Simulate some work
    result = f"Processed {param1} with value {param2}"
    return result

@log_function_call(get_data_logger())
async def async_example_function(data_size: int):
    """Example async function with logging"""
    await asyncio.sleep(0.05)  # Simulate async work
    return f"Processed {data_size} records"

# Example 4: Exception logging
def exception_logging_example():
    """Demonstrate exception logging"""
    logger = get_ml_logger("exception_example")
    
    try:
        # Simulate an error
        result = 10 / 0
    except Exception as e:
        get_logger_manager().log_exception(
            logger, e, "Division by zero in calculation"
        )

# Example 5: Model information logging
def model_info_logging_example():
    """Demonstrate model information logging"""
    logger = get_training_logger("model_info_example")
    
    model_info = {
        "model_type": "T-GNN",
        "parameters": 1_250_000,
        "hidden_dim": 64,
        "num_layers": 3,
        "device": "cuda",
        "created_at": "2024-01-01T10:00:00"
    }
    
    get_logger_manager().log_model_info(logger, model_info)

# Example 6: Performance metrics logging
def performance_logging_example():
    """Demonstrate performance metrics logging"""
    logger = get_training_logger("performance_example")
    
    # Training metrics
    training_metrics = {
        "epoch": 5,
        "train_loss": 0.234,
        "train_accuracy": 0.856,
        "learning_rate": 0.001,
        "batch_size": 32,
        "gpu_memory_mb": 2048
    }
    
    get_logger_manager().log_performance_metrics(
        logger, training_metrics, "Training epoch 5"
    )
    
    # Validation metrics
    validation_metrics = {
        "val_loss": 0.287,
        "val_accuracy": 0.834,
        "val_f1_macro": 0.798,
        "inference_time_ms": 45.6
    }
    
    get_logger_manager().log_performance_metrics(
        logger, validation_metrics, "Validation epoch 5"
    )

async def main():
    """Run all logging examples"""
    print("Running logging examples...")
    print("Check the logs/ directory for output files")
    
    # Run examples
    basic_logging_example()
    specialized_loggers_example()
    
    # Function decorator examples
    result1 = example_function("test_data", 42)
    result2 = await async_example_function(1000)
    
    exception_logging_example()
    model_info_logging_example()
    performance_logging_example()
    
    print("Examples completed. Check logs for output.")

if __name__ == "__main__":
    asyncio.run(main())