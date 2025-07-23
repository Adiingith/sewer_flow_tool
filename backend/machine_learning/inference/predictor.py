"""
Inference module for T-GNN storm response classification
Handles real-time prediction and model loading
"""
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import asyncio
from pathlib import Path
import json
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from backend.machine_learning.models.tgnn_model import TGNNClassifier, DEFAULT_TGNN_CONFIG
from backend.machine_learning.data.data_loader import StormResponseDataLoader
from backend.machine_learning.features.feature_engineering import TGNNDataProcessor
from backend.machine_learning.training.mlflow_manager import get_mlflow_manager
from backend.machine_learning.utils.logger_config import get_prediction_logger, log_function_call

logger = get_prediction_logger()

class TGNNPredictor:
    """
    T-GNN predictor for storm response classification
    """
    
    def __init__(self, model_name: str = "TGNNStormClassifier", 
                 model_version: Optional[str] = None,
                 model_stage: str = "Production",
                 device: str = None):
        """
        Initialize predictor
        
        Args:
            model_name: Name of the registered model
            model_version: Specific model version (optional)
            model_stage: Model stage to load from
            device: Computation device
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model_stage = model_stage
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model = None
        self.config = None
        self.data_processor = None
        self.mlflow_manager = get_mlflow_manager()
        
        # Label mapping for 7 action categories
        self.label_decoder = {
            0: 'no_action_continue_monitoring',
            1: 'investigate_storm_failure',
            2: 'investigate_dryday_failure',
            3: 'sensor_fault_or_ragging',
            4: 'partial_data_needs_review',
            5: 'partial_data_no_action',
            6: 'recommend_remove_or_relocate'
        }
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
        logger.info(f"Initialized TGNNPredictor with device: {self.device}")
    
    async def load_model(self) -> bool:
        """
        Load model from MLflow registry
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load model and config from MLflow
            model, config = self.mlflow_manager.load_model(
                model_name=self.model_name,
                version=self.model_version,
                stage=self.model_stage
            )
            
            self.model = model.to(self.device)
            self.config = config or DEFAULT_TGNN_CONFIG
            self.model.eval()
            
            # Initialize data processor
            self.data_processor = TGNNDataProcessor(
                sequence_length=self.config.get('sequence_length', 2016)
            )
            
            logger.info(f"Successfully loaded model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _load_fallback_model(self, model_path: str) -> bool:
        """
        Load model from local file as fallback
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not Path(model_path).exists():
                logger.warning(f"Model file not found: {model_path}")
                return False
            
            # Load model state dict
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load config
            config_path = Path(model_path).parent / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = DEFAULT_TGNN_CONFIG
            
            # Create model
            from backend.machine_learning.models.tgnn_model import create_tgnn_model
            self.model = create_tgnn_model(self.config)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize data processor
            self.data_processor = TGNNDataProcessor(
                sequence_length=self.config.get('sequence_length', 2016)
            )
            
            logger.info(f"Loaded fallback model from: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            return False
    
    @log_function_call(logger)
    async def predict_single(self, db_session: AsyncSession, 
                           monitor_id: str, interim: str) -> Dict[str, Any]:
        """
        Predict storm response for a single monitor
        
        Args:
            db_session: Database session
            monitor_id: Monitor business ID
            interim: Interim period
            
        Returns:
            Prediction results
        """
        if not self.model or not self.data_processor:
            return {
                'error': 'Model not loaded. Please load model first.',
                'monitor_id': monitor_id,
                'interim': interim
            }
        
        try:
            # Load data
            data_loader = StormResponseDataLoader(db_session)
            prediction_data = await data_loader.load_data_for_prediction(monitor_id, interim)
            
            if 'error' in prediction_data:
                return {
                    'error': prediction_data['error'],
                    'monitor_id': monitor_id,
                    'interim': interim
                }
            
            # Process data
            processed_data = self.data_processor.process_data_for_prediction(prediction_data)
            
            if 'error' in processed_data:
                return {
                    'error': processed_data['error'],
                    'monitor_id': monitor_id,
                    'interim': interim
                }
            
            # Prepare tensors
            sequence = torch.FloatTensor(processed_data['time_series_sequence']).unsqueeze(0).to(self.device)
            spatial_features = torch.FloatTensor(processed_data['spatial_features']).to(self.device)
            edge_index = torch.LongTensor(processed_data['graph_structure']['edge_list']).t().contiguous().to(self.device)
            monitor_indices = torch.LongTensor([processed_data['monitor_idx']]).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                logits = self.model(
                    temporal_data=sequence,
                    node_features=spatial_features,
                    edge_index=edge_index,
                    batch_node_indices=monitor_indices
                )
                
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(logits, dim=1)
            
            # Extract results
            prob_array = probabilities.cpu().numpy()[0]
            predicted_label = self.label_decoder[predicted_class.item()]
            confidence = float(prob_array[predicted_class.item()])
            
            # Determine confidence level
            if confidence >= self.confidence_thresholds['high']:
                confidence_level = 'high'
            elif confidence >= self.confidence_thresholds['medium']:
                confidence_level = 'medium'
            else:
                confidence_level = 'low'
            
            # Generate suggestion based on prediction
            suggestion = self._generate_suggestion(predicted_label, confidence_level, prob_array)
            
            return {
                'monitor_id': monitor_id,
                'interim': interim,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'confidence_level': confidence_level,
                'probabilities': {
                    'good': float(prob_array[0]),
                    'partial': float(prob_array[1]),
                    'bad': float(prob_array[2])
                },
                'suggestion': suggestion,
                'prediction_time': datetime.utcnow().isoformat(),
                'model_info': {
                    'model_name': self.model_name,
                    'model_version': self.model_version,
                    'model_stage': self.model_stage
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for monitor {monitor_id}: {e}")
            return {
                'error': f'Prediction failed: {str(e)}',
                'monitor_id': monitor_id,
                'interim': interim
            }
    
    async def predict_batch(self, db_session: AsyncSession,
                          monitor_ids: List[str], interim: str) -> List[Dict[str, Any]]:
        """
        Predict storm response for multiple monitors
        
        Args:
            db_session: Database session
            monitor_ids: List of monitor business IDs
            interim: Interim period
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Process each monitor individually for now
        # TODO: Implement true batch processing for efficiency
        for monitor_id in monitor_ids:
            result = await self.predict_single(db_session, monitor_id, interim)
            results.append(result)
        
        return results
    
    def _generate_suggestion(self, predicted_label: str, confidence_level: str, 
                           probabilities: np.ndarray) -> str:
        """
        Generate actionable suggestions based on prediction
        
        Args:
            predicted_label: Predicted class label
            confidence_level: Confidence level
            probabilities: Class probabilities
            
        Returns:
            Suggestion string
        """
        # Action-specific suggestions based on Action和Data_quality映射关系.md
        action_suggestions = {
            'no_action_continue_monitoring': {
                'high': "Data appears normal. Continue regular monitoring and data collection.",
                'medium': "Likely normal data. Continue monitoring with routine checks.",
                'low': "Uncertain classification. Manual review recommended before proceeding."
            },
            'investigate_storm_failure': {
                'high': "Storm response failure detected. Dispatch field team to check device position, orientation, and well height.",
                'medium': "Possible storm response issues. Schedule field investigation for device verification.",
                'low': "Uncertain storm response. Manual data review and field check recommended."
            },
            'investigate_dryday_failure': {
                'high': "Dry weather flow anomaly detected. Investigate potential pipe sedimentation or device obstruction.",
                'medium': "Possible dry day flow issues. Schedule inspection for sediment buildup or blockages.",
                'low': "Uncertain dry day data. Review baseline flow patterns and schedule field check."
            },
            'sensor_fault_or_ragging': {
                'high': "Device fault or ragging detected. Dispatch team for sensor cleaning or reinstallation.",
                'medium': "Likely sensor issues or debris. Schedule maintenance for cleaning/calibration.",
                'low': "Possible device problems. Manual data review and maintenance check needed."
            },
            'partial_data_needs_review': {
                'high': "Partial data quality issues identified. Add to review queue for manual assessment.",
                'medium': "Data partially usable but questionable. Review for storm response weakness or short duration.",
                'low': "Uncertain data quality. Detailed manual review required."
            },
            'partial_data_no_action': {
                'high': "Partial data acceptable for current use. Continue monitoring with noted limitations.",
                'medium': "Data partially usable. Continue collection while monitoring for improvements.",
                'low': "Uncertain data status. Review acceptability criteria and continue monitoring."
            },
            'recommend_remove_or_relocate': {
                'high': "Long-term data ineffectiveness. Consider device removal or relocation to more suitable site.",
                'medium': "Potential long-term issues. Evaluate for possible removal or relocation.",
                'low': "Uncertain long-term viability. Extended evaluation needed before removal decision."
            }
        }
        
        base_suggestion = action_suggestions.get(predicted_label, {}).get(
            confidence_level, 
            "Unable to determine appropriate action."
        )
        
        # Add probability context for close predictions
        max_prob = np.max(probabilities)
        second_max_idx = np.argsort(probabilities)[-2]
        second_max_prob = probabilities[second_max_idx]
        
        if max_prob - second_max_prob < 0.15:  # Close probabilities (tightened threshold for 7 classes)
            alternative_label = self.label_decoder[second_max_idx]
            base_suggestion += f" Note: Close probability with '{alternative_label}' ({second_max_prob:.2f})."
        
        return base_suggestion
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Model information dictionary
        """
        if not self.model:
            return {'error': 'No model loaded'}
        
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'model_stage': self.model_stage,
            'config': self.config,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def warm_up(self) -> bool:
        """
        Warm up the model with dummy data
        
        Returns:
            True if successful
        """
        if not self.model:
            return False
        
        try:
            # Create dummy data
            batch_size = 1
            sequence_length = self.config.get('sequence_length', 2016)
            input_channels = self.config.get('input_channels', 4)
            node_feature_dim = self.config.get('node_feature_dim', 100)
            
            dummy_sequence = torch.randn(batch_size, sequence_length, input_channels).to(self.device)
            dummy_spatial = torch.randn(10, node_feature_dim).to(self.device)  # 10 dummy nodes
            dummy_edge_index = torch.LongTensor([[0, 1], [1, 0]]).t().contiguous().to(self.device)
            dummy_indices = torch.LongTensor([0]).to(self.device)
            
            # Run forward pass
            with torch.no_grad():
                _ = self.model(
                    temporal_data=dummy_sequence,
                    node_features=dummy_spatial,
                    edge_index=dummy_edge_index,
                    batch_node_indices=dummy_indices
                )
            
            logger.info("Model warm-up completed")
            return True
            
        except Exception as e:
            logger.error(f"Model warm-up failed: {e}")
            return False

# Global predictor instance
_predictor_instance = None

async def get_predictor(model_name: str = "TGNNStormClassifier",
                       model_version: Optional[str] = None,
                       model_stage: str = "Production") -> TGNNPredictor:
    """
    Get or create global predictor instance
    
    Args:
        model_name: Name of the model
        model_version: Model version
        model_stage: Model stage
        
    Returns:
        TGNNPredictor instance
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = TGNNPredictor(
            model_name=model_name,
            model_version=model_version,
            model_stage=model_stage
        )
        
        # Try to load model
        success = await _predictor_instance.load_model()
        if not success:
            logger.warning("Failed to load model from MLflow, using fallback")
            # You can implement fallback loading here if needed
        else:
            # Warm up the model
            _predictor_instance.warm_up()
    
    return _predictor_instance

async def predict_storm_response(db_session: AsyncSession,
                                monitor_id: str, interim: str) -> Dict[str, Any]:
    """
    Convenience function for single prediction
    
    Args:
        db_session: Database session
        monitor_id: Monitor business ID
        interim: Interim period
        
    Returns:
        Prediction result
    """
    predictor = await get_predictor()
    return await predictor.predict_single(db_session, monitor_id, interim)

async def predict_storm_response_batch(db_session: AsyncSession,
                                     monitor_ids: List[str], interim: str) -> List[Dict[str, Any]]:
    """
    Convenience function for batch prediction
    
    Args:
        db_session: Database session
        monitor_ids: List of monitor business IDs
        interim: Interim period
        
    Returns:
        List of prediction results
    """
    predictor = await get_predictor()
    return await predictor.predict_batch(db_session, monitor_ids, interim)