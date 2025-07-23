"""
MLflow integration for T-GNN model training and management
Provides experiment tracking, model versioning, and deployment management
"""
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import os
from datetime import datetime
import uuid
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

class MLFlowManager:
    """
    Manages MLflow experiments and model tracking for T-GNN
    """
    
    def __init__(self, tracking_uri: str = "http://localhost:5002", 
                 experiment_name: str = "storm_response_tgnn"):
        """
        Initialize MLflow manager
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the MLflow experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.client = MlflowClient(tracking_uri)
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
        
        logger.info(f"Using MLflow experiment: {experiment_name} (ID: {self.experiment_id})")
    
    def start_run(self, run_name: Optional[str] = None, 
                  tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
            
        Returns:
            Run ID
        """
        if run_name is None:
            run_name = f"tgnn_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Default tags
        default_tags = {
            "model_type": "T-GNN",
            "task": "storm_response_classification",
            "created_by": "sewer_flow_tool"
        }
        
        if tags:
            default_tags.update(tags)
        
        run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=default_tags
        )
        
        logger.info(f"Started MLflow run: {run_name} (ID: {run.info.run_id})")
        return run.info.run_id
    
    def log_model_config(self, config: Dict[str, Any]):
        """
        Log model configuration parameters
        
        Args:
            config: Model configuration dictionary
        """
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(key, value)
            else:
                mlflow.log_param(key, str(value))
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """
        Log dataset information
        
        Args:
            dataset_info: Dataset information dictionary
        """
        # Log dataset statistics
        mlflow.log_param("num_monitors", dataset_info.get('num_monitors', 0))
        mlflow.log_param("num_interims", dataset_info.get('num_interims', 0))
        mlflow.log_param("total_samples", dataset_info.get('total_samples', 0))
        mlflow.log_param("sequence_length", dataset_info.get('sequence_length', 0))
        
        # Log class distribution
        class_dist = dataset_info.get('class_distribution', {})
        for class_name, count in class_dist.items():
            mlflow.log_param(f"class_{class_name}_count", count)
        
        # Log time range
        if 'time_range' in dataset_info:
            mlflow.log_param("data_start_date", dataset_info['time_range'].get('start'))
            mlflow.log_param("data_end_date", dataset_info['time_range'].get('end'))
    
    def log_training_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Log training metrics for an epoch
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics (loss, accuracy, etc.)
        """
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value, step=epoch)
    
    def log_validation_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Log validation metrics for an epoch
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"val_{metric_name}", value, step=epoch)
    
    def log_test_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_prob: Optional[np.ndarray] = None,
                        class_names: List[str] = ['good', 'partial', 'bad']):
        """
        Log comprehensive test results
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            class_names: Class names for reporting
        """
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Log main metrics
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_macro", f1_macro)
        mlflow.log_metric("test_f1_weighted", f1_weighted)
        
        # Log per-class F1 scores
        f1_per_class = f1_score(y_true, y_pred, average=None)
        for i, class_name in enumerate(class_names):
            if i < len(f1_per_class):
                mlflow.log_metric(f"test_f1_{class_name}", f1_per_class[i])
        
        # Generate and log classification report
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Save classification report as artifact
        report_path = "test_classification_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)
        os.remove(report_path)
        
        # Generate and log confusion matrix
        self._log_confusion_matrix(y_true, y_pred, class_names)
        
        # Log probability distribution if available
        if y_prob is not None:
            self._log_probability_analysis(y_true, y_pred, y_prob, class_names)
    
    def _log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             class_names: List[str]):
        """
        Generate and log confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save and log plot
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)
        plt.close()
        
        # Log confusion matrix values
        for i, true_class in enumerate(class_names):
            for j, pred_class in enumerate(class_names):
                mlflow.log_metric(f"cm_{true_class}_to_{pred_class}", cm[i, j])
    
    def _log_probability_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_prob: np.ndarray, class_names: List[str]):
        """
        Log probability analysis
        """
        # Calculate confidence statistics
        max_probs = np.max(y_prob, axis=1)
        correct_predictions = (y_true == y_pred)
        
        mlflow.log_metric("avg_confidence_correct", np.mean(max_probs[correct_predictions]))
        mlflow.log_metric("avg_confidence_incorrect", np.mean(max_probs[~correct_predictions]))
        mlflow.log_metric("avg_confidence_overall", np.mean(max_probs))
        
        # Plot probability distributions
        plt.figure(figsize=(12, 4))
        
        for i, class_name in enumerate(class_names):
            plt.subplot(1, 3, i+1)
            class_probs = y_prob[:, i]
            plt.hist(class_probs, bins=20, alpha=0.7, label=f'{class_name} prob')
            plt.title(f'Probability Distribution - {class_name}')
            plt.xlabel('Probability')
            plt.ylabel('Count')
        
        plt.tight_layout()
        prob_dist_path = "probability_distributions.png"
        plt.savefig(prob_dist_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(prob_dist_path)
        os.remove(prob_dist_path)
        plt.close()
    
    def log_model(self, model: torch.nn.Module, model_config: Dict[str, Any],
                  model_name: str = "tgnn_storm_classifier",
                  signature: Optional[mlflow.models.ModelSignature] = None):
        """
        Log the trained model
        
        Args:
            model: Trained PyTorch model
            model_config: Model configuration
            model_name: Name for the model
            signature: MLflow model signature (optional)
        """
        # Save model configuration
        config_path = "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        mlflow.log_artifact(config_path)
        os.remove(config_path)
        
        # Log the PyTorch model
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=model_name,
            signature=signature,
            extra_files=[config_path] if os.path.exists(config_path) else None
        )
        
        logger.info(f"Model logged as: {model_name}")
    
    def register_model(self, run_id: str, model_name: str = "TGNNStormClassifier",
                      model_version_description: Optional[str] = None) -> str:
        """
        Register model in MLflow Model Registry
        
        Args:
            run_id: MLflow run ID
            model_name: Name for the registered model
            model_version_description: Description for this version
            
        Returns:
            Model URI
        """
        model_uri = f"runs:/{run_id}/tgnn_storm_classifier"
        
        try:
            # Register the model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags={"framework": "pytorch", "task": "classification"}
            )
            
            # Add description if provided
            if model_version_description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=model_version_description
                )
            
            logger.info(f"Model registered: {model_name} v{model_version.version}")
            return model_uri
            
        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Failed to register model: {e}")
            return model_uri
    
    def load_model(self, model_name: str, version: Optional[str] = None,
                   stage: Optional[str] = None) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Load a registered model
        
        Args:
            model_name: Name of the registered model
            version: Specific version to load (optional)
            stage: Stage to load from (e.g., 'Production', 'Staging')
            
        Returns:
            Tuple of (model, config)
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
        elif stage:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            model_uri = f"models:/{model_name}/latest"
        
        # Load the model
        model = mlflow.pytorch.load_model(model_uri)
        
        # Try to load configuration
        try:
            # Get the run info to download artifacts
            model_version = self.client.get_latest_versions(model_name, stages=[stage] if stage else None)[0]
            run_id = model_version.run_id
            
            # Download model config
            config_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, 
                artifact_path="model_config.json"
            )
            
            with open(config_path, 'r') as f:
                config = json.load(f)
                
        except Exception as e:
            logger.warning(f"Could not load model config: {e}")
            config = {}
        
        return model, config
    
    def get_best_model(self, metric_name: str = "val_f1_macro", 
                      ascending: bool = False) -> Optional[str]:
        """
        Get the best model based on a specific metric
        
        Args:
            metric_name: Metric to optimize for
            ascending: Whether to sort in ascending order
            
        Returns:
            Run ID of the best model
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string="",
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
        )
        
        if runs:
            best_run = runs[0]
            logger.info(f"Best model run: {best_run.info.run_id} "
                       f"({metric_name}: {best_run.data.metrics.get(metric_name, 'N/A')})")
            return best_run.info.run_id
        
        return None
    
    def compare_models(self, run_ids: List[str], 
                      metrics: List[str] = ['val_accuracy', 'val_f1_macro']) -> pd.DataFrame:
        """
        Compare multiple model runs
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            
            row_data = {
                'run_id': run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'status': run.info.status
            }
            
            # Add metrics
            for metric in metrics:
                row_data[metric] = run.data.metrics.get(metric, None)
            
            # Add key parameters
            for param_name in ['temporal_encoder_type', 'spatial_encoder_type', 
                              'temporal_hidden_dim', 'spatial_hidden_dim']:
                row_data[param_name] = run.data.params.get(param_name, None)
            
            comparison_data.append(row_data)
        
        return pd.DataFrame(comparison_data)
    
    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()
        logger.info("MLflow run ended")
    
    def cleanup_failed_runs(self):
        """Clean up failed or incomplete runs"""
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string="attributes.status = 'FAILED'"
        )
        
        for run in runs:
            self.client.delete_run(run.info.run_id)
            logger.info(f"Deleted failed run: {run.info.run_id}")

# Singleton instance for global use
_mlflow_manager = None

def get_mlflow_manager(tracking_uri: str = "http://localhost:5002", 
                      experiment_name: str = "storm_response_tgnn") -> MLFlowManager:
    """
    Get the global MLflow manager instance
    
    Args:
        tracking_uri: MLflow tracking server URI
        experiment_name: Experiment name
        
    Returns:
        MLFlowManager instance
    """
    global _mlflow_manager
    
    if _mlflow_manager is None:
        _mlflow_manager = MLFlowManager(tracking_uri, experiment_name)
    
    return _mlflow_manager