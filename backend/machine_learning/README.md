# T-GNN Action Classification Module

This module implements a Temporal Graph Neural Network (T-GNN) for action prediction in sewer monitoring systems. The model predicts the appropriate action for each monitor device based on time series data and device relationships, classifying into 7 action categories based on data quality analysis.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Loader   │───▶│ Feature Engineer │───▶│   T-GNN Model   │
│                 │    │                  │    │                 │
│ - Monitor data  │    │ - Time series    │    │ - Temporal GRU  │
│ - Measurements  │    │ - Spatial info   │    │ - Graph Conv    │
│ - Rain gauge    │    │ - Graph struct   │    │ - Classification│
│ - Quality labels│    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Database      │    │     MLflow       │    │   Prediction    │
│   Integration   │    │   Tracking       │    │   API           │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Key Components

### 1. Data Pipeline (`data/`)
- **`data_loader.py`**: Loads time series data from monitors, rain gauges, and quality checks
- Handles monitor relationships and spatial features
- Manages data preprocessing and null value handling

### 2. Feature Engineering (`features/`)
- **`feature_engineering.py`**: Extracts time series and spatial features
- Combines statistical, temporal, and frequency domain features
- Manages graph structure construction based on monitor relationships

### 3. Model Architecture (`models/`)
- **`tgnn_model.py`**: T-GNN implementation with PyTorch and PyTorch Geometric
- Temporal encoder options: GRU, TCN, 1D CNN
- Spatial encoder options: GCN, GAT, GraphSAGE
- Configurable fusion strategies and classification head

### 4. Training Pipeline (`training/`)
- **`trainer.py`**: Complete training workflow with early stopping
- **`mlflow_manager.py`**: MLflow integration for experiment tracking
- Cross-validation support and model evaluation

### 5. Inference Engine (`inference/`)
- **`predictor.py`**: Real-time prediction interface
- Model loading from MLflow registry
- Confidence estimation and suggestion generation

## Usage

### 1. Training a Model

```bash
cd backend/machine_learning

# Basic training (with MinIO caching)
python train_model.py --epochs 100 --batch-size 32

# Training with specific interim periods
python train_model.py --interims Interim1 Interim2 Interim3 --epochs 100

# Training without cache (force reload from database)
python train_model.py --no-cache --epochs 100

# Advanced configuration with cross-validation
python train_model.py \
    --hidden-dim 128 \
    --temporal-encoder gru \
    --spatial-encoder gcn \
    --epochs 200 \
    --learning-rate 0.001 \
    --cross-validate \
    --run-name "tgnn_experiment_1"
```

### 2. Testing Predictions

```bash
# Check available data
python test_prediction.py --data-info

# Test single monitor
python test_prediction.py --monitor-id FM001 --interim Interim1

# Test multiple monitors
python test_prediction.py --monitor-ids FM001 FM002 FM003 --interim Interim1

# Check model information
python test_prediction.py --model-info
```

### 3. Integration with API

The model is automatically integrated with the existing `aiPredictApi.py`. When you call the `/ai/predict-action` endpoint, it will:

1. Run the rule engine (existing functionality)
2. Call the T-GNN model for storm response prediction
3. Combine results and save to `WeeklyQualityCheck` table

## Configuration

## Action Categories

The model predicts 7 action categories based on the Action和Data_quality映射关系.md:

1. **`no_action_continue_monitoring`** - Data normal, continue monitoring
2. **`investigate_storm_failure`** - Storm response failure, dispatch field team
3. **`investigate_dryday_failure`** - Dry day flow anomaly, check for sedimentation
4. **`sensor_fault_or_ragging`** - Device fault or debris, needs cleaning/reinstallation
5. **`partial_data_needs_review`** - Partial data quality, requires manual assessment
6. **`partial_data_no_action`** - Partial data acceptable, continue with limitations
7. **`recommend_remove_or_relocate`** - Long-term ineffective, consider removal/relocation

### Model Configuration (`DEFAULT_TGNN_CONFIG`)

```python
{
    'sequence_length': 2016,        # One week at 5-min intervals
    'input_channels': 4,            # depth, flow, velocity, rainfall
    'node_feature_dim': 100,        # Combined features per node
    'temporal_hidden_dim': 64,      # Temporal encoder dimension
    'spatial_hidden_dim': 64,       # Spatial encoder dimension
    'num_classes': 7,               # 7 action categories
    'temporal_encoder_type': 'gru', # gru, tcn, cnn
    'spatial_encoder_type': 'gcn',  # gcn, gat, sage
    'temporal_layers': 2,
    'spatial_layers': 2,
    'dropout': 0.2,
    'fusion_type': 'concat'         # concat, add, attention
}
```

### Training Configuration

```python
{
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'optimizer': 'adamw',           # adamw, adam, sgd
    'scheduler': 'cosine',          # cosine, step, plateau
    'weight_decay': 0.01,
    'patience': 20                  # Early stopping patience
}
```

## MLflow Integration

The module uses MLflow for experiment tracking and model management:

- **Experiments**: All runs are tracked under `storm_response_tgnn` experiment
- **Metrics**: Training/validation loss, accuracy, F1-scores
- **Artifacts**: Model checkpoints, configuration files, plots
- **Model Registry**: Production models are registered for deployment

Access MLflow UI at: `http://localhost:5002`

## Unified Logging System

The module includes a comprehensive logging system that outputs structured logs to the `logs/` directory:

### Log Files Structure

```
logs/
├── ml_training_YYYYMMDD.log      # Training operations
├── ml_prediction_YYYYMMDD.log    # Prediction operations  
├── ml_data_YYYYMMDD.log          # Data processing operations
├── ml_cache_YYYYMMDD.log         # MinIO cache operations
└── ml_errors_YYYYMMDD.log        # All error messages
```

### Log Management Commands

```bash
# List all log files with statistics
python manage_logs.py list

# View last 50 lines of training logs
python manage_logs.py tail training

# Search for specific patterns
python manage_logs.py search "ERROR" --context 3

# Analyze error patterns from last 7 days
python manage_logs.py analyze --days 7

# Compress old log files (older than 7 days)
python manage_logs.py compress --days 7

# Clean old log files (older than 30 days)
python manage_logs.py clean --days 30 --confirm

# Export recent logs to single file
python manage_logs.py export ml_logs_export.txt --days 7
```

### Logging Features

- **Structured Format**: Consistent timestamp, logger name, level, and message format
- **Automatic Rotation**: Files rotate when they reach 10MB, keeping 5 backups
- **Color Console Output**: Different colors for different log levels in console
- **Exception Tracking**: Full stacktraces with context information
- **Performance Metrics**: Structured logging of training metrics and timings
- **Function Call Tracing**: Decorator-based logging of function calls and execution times

## MinIO Data Caching

The module includes MinIO integration for caching preprocessed data, significantly speeding up training when using the same datasets:

### Cache Management

```bash
# Check cache statistics
python manage_cache.py stats

# Test cache functionality
python manage_cache.py test

# Clear all cache (with confirmation)
python manage_cache.py clear --confirm

# Clear specific pattern
python manage_cache.py clear --pattern "preprocessed_data_20240101" --confirm

# Check cache health
python manage_cache.py health
```

### Cache Configuration

Set environment variables for MinIO connection:
```bash
export MINIO_ENDPOINT="localhost:9000"
export MINIO_ACCESS_KEY="minioadmin"
export MINIO_SECRET_KEY="minioadmin"
```

## Data Requirements

### Input Data
- **Time Series**: depth, flow, velocity, rainfall (2016 timesteps per week)
- **Spatial Features**: monitor metadata (location, pipe dimensions, etc.)
- **Graph Structure**: monitor relationships (area, rain gauge assignments)
- **Labels**: Action categories from `WeeklyQualityCheck.actions.actions` (7 categories)

### Data Processing
- Handles missing values with interpolation
- Normalizes time series per monitor
- Constructs graph edges based on spatial relationships
- Processes categorical spatial features with one-hot encoding

## Model Performance

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Macro and weighted F1-scores
- **Per-Class Metrics**: Precision, recall, F1 for each class
- **Confusion Matrix**: Detailed classification breakdown

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Database Connection**: Check database configuration in `.env`
3. **MLflow Connection**: Verify MLflow server is running on port 5002
4. **Memory Issues**: Reduce batch size or sequence length for large datasets
5. **CUDA Issues**: Specify `--device cpu` if GPU is not available
6. **Log Permission Issues**: Ensure the `logs/` directory is writable
7. **Log File Too Large**: Use `manage_logs.py compress` to compress old logs

### Dependencies

The module requires additional Python packages:
```
torch>=1.12.0
torch-geometric>=2.0.0
mlflow>=2.0.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0
minio>=7.0.0       # For data caching
```

Install with:
```bash
cd backend/machine_learning
pip install -r requirements.txt
```

## Future Enhancements

1. **Multi-Task Learning**: Extend to predict multiple aspects (dry day response, device health)
2. **Attention Visualization**: Add attention heatmaps for interpretability
3. **Online Learning**: Implement incremental learning for new data
4. **Edge Computing**: Deploy lightweight models on edge devices
5. **Real-Time Streaming**: Add support for real-time data streams

## API Integration

The prediction results are integrated into the existing system:

```python
# In aiPredictApi.py, the model provides:
{
    'ai_predicted_action': 'no_action_continue_monitoring',  # Direct action prediction
    'ai_suggestion': 'Data appears normal. Continue regular monitoring...'  # Detailed suggestion
}
```

Action categories map directly to the 7 categories defined in Action和Data_quality映射关系.md, providing clear guidance for field operations and maintenance scheduling.