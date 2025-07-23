"""
Training module for T-GNN storm response classification model
Handles model training, validation, and cross-validation
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
import copy
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from backend.machine_learning.models.tgnn_model import TGNNClassifier, TGNNLoss, DEFAULT_TGNN_CONFIG
from backend.machine_learning.training.mlflow_manager import get_mlflow_manager
from backend.machine_learning.utils.logger_config import get_training_logger, log_function_call, get_logger_manager

logger = get_training_logger()

class StormResponseDataset(Dataset):
    """
    PyTorch Dataset for storm response classification
    """
    
    def __init__(self, samples: List[Dict[str, Any]], graph_structure: Dict[str, Any]):
        self.samples = samples
        self.adjacency_matrix = torch.FloatTensor(graph_structure['adjacency_matrix'])
        self.edge_list = torch.LongTensor(graph_structure['edge_list']).t().contiguous()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        return {
            'sequence': torch.FloatTensor(sample['time_series_sequence']),
            'node_features': torch.FloatTensor(sample['node_features']),
            'label': torch.LongTensor([sample['label']]),
            'monitor_idx': sample['monitor_idx'],
            'monitor_id': sample['monitor_id'],
            'interim': sample['interim']
        }

def collate_fn(batch):
    """
    Custom collate function for batching graph data
    """
    sequences = torch.stack([item['sequence'] for item in batch])
    node_features = torch.stack([item['node_features'] for item in batch])
    labels = torch.cat([item['label'] for item in batch])
    monitor_indices = torch.LongTensor([item['monitor_idx'] for item in batch])
    
    return {
        'sequences': sequences,
        'node_features': node_features,
        'labels': labels,
        'monitor_indices': monitor_indices,
        'monitor_ids': [item['monitor_id'] for item in batch],
        'interims': [item['interim'] for item in batch]
    }

class TGNNTrainer:
    """
    Trainer class for T-GNN model
    """
    
    def __init__(self, config: Dict[str, Any] = None, device: str = None, run_name: str = None):
        self.config = config or DEFAULT_TGNN_CONFIG.copy()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize MLflow manager
        self.mlflow_manager = get_mlflow_manager()
        
        # Update logger for specific training run
        if run_name:
            global logger
            logger = get_training_logger(run_name)
        
        logger.info(f"Using device: {self.device}")
        
        # Log model configuration
        get_logger_manager().log_model_info(logger, self.config)
    
    def prepare_data(self, processed_data: Dict[str, Any], 
                    test_size: float = 0.2, val_size: float = 0.2,
                    random_state: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data loaders for training, validation, and testing
        
        Args:
            processed_data: Processed dataset from feature engineering
            test_size: Proportion of data for testing
            val_size: Proportion of remaining data for validation
            random_state: Random seed
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        samples = processed_data['samples']
        graph_structure = processed_data['graph_structure']
        
        # Split data
        train_val_samples, test_samples = train_test_split(
            samples, test_size=test_size, random_state=random_state,
            stratify=[s['label'] for s in samples]
        )
        
        train_samples, val_samples = train_test_split(
            train_val_samples, test_size=val_size, random_state=random_state,
            stratify=[s['label'] for s in train_val_samples]
        )
        
        # Create datasets
        train_dataset = StormResponseDataset(train_samples, graph_structure)
        val_dataset = StormResponseDataset(val_samples, graph_structure)
        test_dataset = StormResponseDataset(test_samples, graph_structure)
        
        # Create data loaders
        batch_size = self.config.get('batch_size', 32)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0
        )
        
        # Log dataset information
        dataset_info = {
            'num_monitors': len(processed_data['monitor_metadata']),
            'total_samples': len(samples),
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'test_samples': len(test_samples),
            'sequence_length': self.config['sequence_length'],
            'class_distribution': {}
        }
        
        # Calculate class distribution
        for label in [0, 1, 2]:  # good, partial, bad
            count = sum(1 for s in samples if s['label'] == label)
            class_names = ['good', 'partial', 'bad']
            dataset_info['class_distribution'][class_names[label]] = count
        
        self.dataset_info = dataset_info
        
        return train_loader, val_loader, test_loader
    
    def create_model(self) -> TGNNClassifier:
        """
        Create T-GNN model
        """
        from backend.machine_learning.models.tgnn_model import create_tgnn_model
        
        model = create_tgnn_model(self.config)
        model.to(self.device)
        
        return model
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """
        Create optimizer
        """
        optimizer_type = self.config.get('optimizer', 'adamw')
        learning_rate = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        if optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        return optimizer
    
    def create_scheduler(self, optimizer: optim.Optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler
        """
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=1e-6
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: optim.Optimizer, criterion: nn.Module,
                   all_node_features: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, float]:
        """
        Train for one epoch
        """
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move data to device
            sequences = batch['sequences'].to(self.device)
            labels = batch['labels'].to(self.device)
            monitor_indices = batch['monitor_indices'].to(self.device)
            
            # Forward pass
            logits = model(
                temporal_data=sequences,
                node_features=all_node_features,
                edge_index=edge_index,
                batch_node_indices=monitor_indices
            )
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{total_correct/total_samples:.4f}'
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': total_correct / total_samples
        }
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader,
                      criterion: nn.Module, all_node_features: torch.Tensor,
                      edge_index: torch.Tensor) -> Dict[str, float]:
        """
        Validate for one epoch
        """
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                sequences = batch['sequences'].to(self.device)
                labels = batch['labels'].to(self.device)
                monitor_indices = batch['monitor_indices'].to(self.device)
                
                # Forward pass
                logits = model(
                    temporal_data=sequences,
                    node_features=all_node_features,
                    edge_index=edge_index,
                    batch_node_indices=monitor_indices
                )
                
                # Compute loss
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
    
    def train(self, processed_data: Dict[str, Any], 
              run_name: Optional[str] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Complete training pipeline
        
        Args:
            processed_data: Processed dataset
            run_name: MLflow run name
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        # Start MLflow run
        run_id = self.mlflow_manager.start_run(run_name)
        
        try:
            # Log configuration and dataset info
            self.mlflow_manager.log_model_config(self.config)
            
            # Prepare data
            train_loader, val_loader, test_loader = self.prepare_data(processed_data)
            self.mlflow_manager.log_dataset_info(self.dataset_info)
            
            # Create model
            model = self.create_model()
            
            # Create loss function with class weights
            class_counts = np.array([
                self.dataset_info['class_distribution']['good'],
                self.dataset_info['class_distribution']['partial'],
                self.dataset_info['class_distribution']['bad']
            ])
            
            # Calculate class weights (inverse frequency)
            total_samples = np.sum(class_counts)
            class_weights = total_samples / (3.0 * class_counts)
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            
            criterion = TGNNLoss(class_weights=class_weights)
            
            # Create optimizer and scheduler
            optimizer = self.create_optimizer(model)
            scheduler = self.create_scheduler(optimizer)
            
            # Prepare graph data
            spatial_features = torch.FloatTensor(processed_data['spatial_features']).to(self.device)
            edge_index = torch.LongTensor(processed_data['graph_structure']['edge_list']).t().contiguous().to(self.device)
            
            # Training loop
            epochs = self.config.get('epochs', 100)
            best_val_f1 = 0.0
            best_model_state = None
            patience = self.config.get('patience', 20)
            patience_counter = 0
            
            training_history = {
                'train_loss': [],
                'train_accuracy': [],
                'val_loss': [],
                'val_accuracy': [],
                'val_f1_macro': [],
                'val_f1_weighted': []
            }
            
            logger.info(f"Starting training for {epochs} epochs")
            
            for epoch in range(epochs):
                # Training
                train_metrics = self.train_epoch(
                    model, train_loader, optimizer, criterion,
                    spatial_features, edge_index
                )
                
                # Validation
                val_metrics = self.validate_epoch(
                    model, val_loader, criterion,
                    spatial_features, edge_index
                )
                
                # Update learning rate
                if scheduler:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_metrics['f1_macro'])
                    else:
                        scheduler.step()
                
                # Log metrics
                self.mlflow_manager.log_training_metrics(epoch, train_metrics)
                self.mlflow_manager.log_validation_metrics(epoch, val_metrics)
                
                # Update history
                training_history['train_loss'].append(train_metrics['loss'])
                training_history['train_accuracy'].append(train_metrics['accuracy'])
                training_history['val_loss'].append(val_metrics['loss'])
                training_history['val_accuracy'].append(val_metrics['accuracy'])
                training_history['val_f1_macro'].append(val_metrics['f1_macro'])
                training_history['val_f1_weighted'].append(val_metrics['f1_weighted'])
                
                # Check for best model
                if val_metrics['f1_macro'] > best_val_f1:
                    best_val_f1 = val_metrics['f1_macro']
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Logging
                if epoch % 10 == 0 or epoch == epochs - 1:
                    logger.info(
                        f"Epoch {epoch}/{epochs-1} - "
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Train Acc: {train_metrics['accuracy']:.4f}, "
                        f"Val Loss: {val_metrics['loss']:.4f}, "
                        f"Val Acc: {val_metrics['accuracy']:.4f}, "
                        f"Val F1: {val_metrics['f1_macro']:.4f}"
                    )
                
                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Load best model
            if best_model_state:
                model.load_state_dict(best_model_state)
            
            # Final evaluation on test set
            test_metrics = self.validate_epoch(
                model, test_loader, criterion, spatial_features, edge_index
            )
            
            # Log test results
            self.mlflow_manager.log_test_results(
                np.array(test_metrics['labels']),
                np.array(test_metrics['predictions']),
                np.array(test_metrics['probabilities'])
            )
            
            # Log model
            self.mlflow_manager.log_model(model, self.config)
            
            logger.info(f"Training completed. Best Val F1: {best_val_f1:.4f}, "
                       f"Test F1: {test_metrics['f1_macro']:.4f}")
            
            return model, training_history
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self.mlflow_manager.end_run()
    
    def cross_validate(self, processed_data: Dict[str, Any], 
                      n_splits: int = 5) -> Dict[str, List[float]]:
        """
        Perform cross-validation
        
        Args:
            processed_data: Processed dataset
            n_splits: Number of CV folds
            
        Returns:
            Dictionary with CV results
        """
        samples = processed_data['samples']
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_results = {
            'val_accuracy': [],
            'val_f1_macro': [],
            'val_f1_weighted': []
        }
        
        X = np.arange(len(samples))
        y = np.array([s['label'] for s in samples])
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            logger.info(f"Training fold {fold + 1}/{n_splits}")
            
            # Split data
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]
            
            # Create datasets and loaders
            graph_structure = processed_data['graph_structure']
            train_dataset = StormResponseDataset(train_samples, graph_structure)
            val_dataset = StormResponseDataset(val_samples, graph_structure)
            
            batch_size = self.config.get('batch_size', 32)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                    shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                  shuffle=False, collate_fn=collate_fn)
            
            # Train model for this fold
            model = self.create_model()
            optimizer = self.create_optimizer(model)
            criterion = TGNNLoss()
            
            # Simplified training for CV
            spatial_features = torch.FloatTensor(processed_data['spatial_features']).to(self.device)
            edge_index = torch.LongTensor(processed_data['graph_structure']['edge_list']).t().contiguous().to(self.device)
            
            epochs = min(50, self.config.get('epochs', 100))  # Reduced epochs for CV
            
            for epoch in range(epochs):
                # Training
                self.train_epoch(model, train_loader, optimizer, criterion,
                               spatial_features, edge_index)
            
            # Validation
            val_metrics = self.validate_epoch(model, val_loader, criterion,
                                            spatial_features, edge_index)
            
            cv_results['val_accuracy'].append(val_metrics['accuracy'])
            cv_results['val_f1_macro'].append(val_metrics['f1_macro'])
            cv_results['val_f1_weighted'].append(val_metrics['f1_weighted'])
        
        # Log CV results
        for metric, values in cv_results.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"CV {metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        return cv_results