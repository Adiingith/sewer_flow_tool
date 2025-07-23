"""
Unified logging configuration for machine learning module
Provides centralized logging with file output and proper formatting
"""
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import traceback
import json

# Get the machine learning module root directory
ML_ROOT = Path(__file__).parent.parent
LOGS_DIR = ML_ROOT / "logs"

# Ensure logs directory exists
LOGS_DIR.mkdir(exist_ok=True)

class MLLoggerFormatter(logging.Formatter):
    """
    Custom formatter for ML module logs with enhanced formatting
    """
    
    def __init__(self):
        # Color codes for console output
        self.colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
            'RESET': '\033[0m'      # Reset
        }
        
        # Base format
        self.base_format = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        super().__init__(self.base_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record):
        # Add module and function info for better traceability
        if hasattr(record, 'funcName'):
            record.location = f"{record.filename}:{record.funcName}:{record.lineno}"
        else:
            record.location = f"{record.filename}:{record.lineno}"
        
        # Format the base message
        formatted = super().format(record)
        
        # Add exception info if present
        if record.exc_info:
            formatted += f"\nException: {self.formatException(record.exc_info)}"
        
        return formatted
    
    def format_console(self, record):
        """Format with colors for console output"""
        level_color = self.colors.get(record.levelname, '')
        reset_color = self.colors['RESET']
        
        # Create colored format
        colored_format = f'{level_color}%(asctime)s | %(name)s | %(levelname)s{reset_color} | %(message)s'
        formatter = logging.Formatter(colored_format, datefmt='%Y-%m-%d %H:%M:%S')
        
        return formatter.format(record)

class MLLogger:
    """
    Unified logger for machine learning module
    """
    
    def __init__(self):
        self.loggers: Dict[str, logging.Logger] = {}
        self._setup_default_config()
    
    def _setup_default_config(self):
        """Setup default logging configuration"""
        # Set root logging level
        logging.getLogger().setLevel(logging.DEBUG)
        
        # Disable matplotlib debug logs
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        
        # Configure MLflow logging
        logging.getLogger('mlflow').setLevel(logging.INFO)
    
    def get_logger(self, 
                   name: str, 
                   log_level: str = "INFO",
                   log_to_file: bool = True,
                   log_to_console: bool = True,
                   file_prefix: Optional[str] = None) -> logging.Logger:
        """
        Get or create a logger with unified configuration
        
        Args:
            name: Logger name (usually module name)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
            file_prefix: Custom prefix for log file
            
        Returns:
            Configured logger instance
        """
        logger_key = f"{name}_{log_level}_{log_to_file}_{log_to_console}"
        
        if logger_key in self.loggers:
            return self.loggers[logger_key]
        
        # Create new logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        logger.propagate = False
        
        formatter = MLLoggerFormatter()
        
        # File handler
        if log_to_file:
            file_handler = self._create_file_handler(name, file_prefix)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Console handler
        if log_to_console:
            console_handler = self._create_console_handler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Store logger
        self.loggers[logger_key] = logger
        
        return logger
    
    def _create_file_handler(self, name: str, file_prefix: Optional[str] = None) -> logging.Handler:
        """Create rotating file handler"""
        # Generate filename
        if file_prefix:
            filename = f"{file_prefix}_{name}_{datetime.now().strftime('%Y%m%d')}.log"
        else:
            filename = f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        filepath = LOGS_DIR / filename
        
        # Create rotating file handler (10MB max, keep 5 backups)
        handler = logging.handlers.RotatingFileHandler(
            filepath,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        return handler
    
    def _create_console_handler(self) -> logging.Handler:
        """Create console handler with colors"""
        handler = logging.StreamHandler(sys.stdout)
        
        # Use colored formatter for console
        class ColoredFormatter(MLLoggerFormatter):
            def format(self, record):
                return self.format_console(record)
        
        handler.setFormatter(ColoredFormatter())
        return handler
    
    def create_training_logger(self, run_name: str = None) -> logging.Logger:
        """Create specialized logger for training runs"""
        run_name = run_name or f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self.get_logger(
            name="ml_training",
            log_level="INFO",
            file_prefix=run_name
        )
    
    def create_prediction_logger(self) -> logging.Logger:
        """Create specialized logger for predictions"""
        return self.get_logger(
            name="ml_prediction",
            log_level="INFO",
            file_prefix="prediction"
        )
    
    def create_data_logger(self) -> logging.Logger:
        """Create specialized logger for data processing"""
        return self.get_logger(
            name="ml_data",
            log_level="DEBUG",
            file_prefix="data_processing"
        )
    
    def create_cache_logger(self) -> logging.Logger:
        """Create specialized logger for cache operations"""
        return self.get_logger(
            name="ml_cache",
            log_level="INFO",
            file_prefix="cache"
        )
    
    def log_exception(self, logger: logging.Logger, 
                     exception: Exception, 
                     context: str = ""):
        """
        Log exception with full traceback and context
        
        Args:
            logger: Logger instance to use
            exception: Exception to log
            context: Additional context information
        """
        exc_info = {
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc()
        }
        
        logger.error(f"Exception occurred: {context}")
        logger.error(f"Exception type: {exc_info['exception_type']}")
        logger.error(f"Exception message: {exc_info['exception_message']}")
        logger.error(f"Full traceback:\n{exc_info['traceback']}")
    
    def log_performance_metrics(self, logger: logging.Logger,
                               metrics: Dict[str, Any],
                               context: str = ""):
        """
        Log performance metrics in structured format
        
        Args:
            logger: Logger instance
            metrics: Dictionary of metrics
            context: Context description
        """
        metrics_log = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'metrics': metrics
        }
        
        logger.info(f"Performance metrics - {context}: {json.dumps(metrics, indent=2)}")
    
    def log_model_info(self, logger: logging.Logger,
                      model_info: Dict[str, Any]):
        """
        Log model information in structured format
        
        Args:
            logger: Logger instance
            model_info: Model information dictionary
        """
        logger.info("=" * 60)
        logger.info("MODEL INFORMATION")
        logger.info("=" * 60)
        
        for key, value in model_info.items():
            logger.info(f"{key}: {value}")
        
        logger.info("=" * 60)
    
    def setup_mlflow_logging(self, experiment_name: str = None):
        """
        Setup MLflow logging integration
        
        Args:
            experiment_name: MLflow experiment name
        """
        mlflow_logger = self.get_logger(
            name="mlflow_integration",
            log_level="INFO",
            file_prefix="mlflow"
        )
        
        if experiment_name:
            mlflow_logger.info(f"MLflow experiment: {experiment_name}")
        
        return mlflow_logger

# Global logger manager instance
_logger_manager = None

def get_logger_manager() -> MLLogger:
    """Get global logger manager instance"""
    global _logger_manager
    if _logger_manager is None:
        _logger_manager = MLLogger()
    return _logger_manager

def get_ml_logger(name: str, **kwargs) -> logging.Logger:
    """
    Convenience function to get ML logger
    
    Args:
        name: Logger name
        **kwargs: Additional arguments for logger configuration
        
    Returns:
        Configured logger
    """
    manager = get_logger_manager()
    return manager.get_logger(name, **kwargs)

def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls with arguments and execution time
    
    Args:
        logger: Logger instance to use
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            func_name = func.__name__
            
            # Log function start
            logger.debug(f"Starting {func_name}")
            if args or kwargs:
                logger.debug(f"Arguments: args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.debug(f"Completed {func_name} in {execution_time:.2f}s")
                return result
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"Failed {func_name} after {execution_time:.2f}s")
                get_logger_manager().log_exception(logger, e, f"Function: {func_name}")
                raise
        
        return wrapper
    return decorator

# Pre-configured loggers for common use cases
def get_training_logger(run_name: str = None) -> logging.Logger:
    """Get logger for training operations"""
    return get_logger_manager().create_training_logger(run_name)

def get_prediction_logger() -> logging.Logger:
    """Get logger for prediction operations"""
    return get_logger_manager().create_prediction_logger()

def get_data_logger() -> logging.Logger:
    """Get logger for data processing operations"""
    return get_logger_manager().create_data_logger()

def get_cache_logger() -> logging.Logger:
    """Get logger for cache operations"""
    return get_logger_manager().create_cache_logger()