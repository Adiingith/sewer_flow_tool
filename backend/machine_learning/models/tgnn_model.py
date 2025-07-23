"""
T-GNN (Temporal Graph Neural Network) model for storm response classification
Combines time series modeling with graph neural networks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class TemporalEncoder(nn.Module):
    """
    Temporal encoder for time series data using GRU or TCN
    """
    
    def __init__(self, input_channels: int = 4, hidden_dim: int = 64, 
                 num_layers: int = 2, encoder_type: str = 'gru'):
        super(TemporalEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder_type = encoder_type
        
        if encoder_type == 'gru':
            self.temporal_encoder = nn.GRU(
                input_size=input_channels,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0.0,
                bidirectional=True
            )
            self.output_dim = hidden_dim * 2  # Bidirectional
            
        elif encoder_type == 'tcn':
            # Temporal Convolutional Network
            self.tcn_layers = nn.ModuleList()
            dilation = 1
            for i in range(num_layers):
                self.tcn_layers.append(
                    nn.Conv1d(
                        in_channels=input_channels if i == 0 else hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=3,
                        dilation=dilation,
                        padding=dilation
                    )
                )
                dilation *= 2
            
            self.output_dim = hidden_dim
            
        elif encoder_type == 'cnn':
            # 1D CNN encoder
            self.conv_layers = nn.ModuleList([
                nn.Conv1d(input_channels, hidden_dim, kernel_size=7, padding=3),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            ])
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.output_dim = hidden_dim
            
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for temporal encoding
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_channels)
            
        Returns:
            Encoded temporal features (batch_size, output_dim)
        """
        if self.encoder_type == 'gru':
            # GRU encoding
            output, hidden = self.temporal_encoder(x)
            # Take the last output from both directions
            temporal_features = output[:, -1, :]  # (batch_size, hidden_dim * 2)
            
        elif self.encoder_type == 'tcn':
            # TCN encoding
            x = x.transpose(1, 2)  # (batch_size, input_channels, sequence_length)
            
            for layer in self.tcn_layers:
                x = F.relu(layer(x))
            
            # Global average pooling
            temporal_features = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
            
        elif self.encoder_type == 'cnn':
            # CNN encoding
            x = x.transpose(1, 2)  # (batch_size, input_channels, sequence_length)
            
            for conv in self.conv_layers:
                x = F.relu(conv(x))
            
            # Global average pooling
            temporal_features = self.pool(x).squeeze(-1)  # (batch_size, hidden_dim)
        
        return temporal_features

class SpatialEncoder(nn.Module):
    """
    Spatial encoder using Graph Neural Networks
    """
    
    def __init__(self, node_feature_dim: int, hidden_dim: int = 64, 
                 num_layers: int = 2, gnn_type: str = 'gcn', 
                 num_heads: int = 4, dropout: float = 0.2):
        super(SpatialEncoder, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        
        # Input projection
        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        if gnn_type == 'gcn':
            for i in range(num_layers):
                self.gnn_layers.append(
                    GCNConv(hidden_dim, hidden_dim)
                )
        elif gnn_type == 'gat':
            for i in range(num_layers):
                self.gnn_layers.append(
                    GATConv(
                        hidden_dim, 
                        hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True if i < num_layers - 1 else False
                    )
                )
        elif gnn_type == 'sage':
            for i in range(num_layers):
                self.gnn_layers.append(
                    GraphSAGE(
                        hidden_dim, hidden_dim, num_layers=1,
                        out_channels=hidden_dim
                    )
                )
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for spatial encoding
        
        Args:
            node_features: Node features (num_nodes, node_feature_dim)
            edge_index: Edge indices (2, num_edges)
            batch: Batch indices for multiple graphs (optional)
            
        Returns:
            Encoded spatial features (num_nodes, hidden_dim)
        """
        # Project node features
        x = self.node_proj(node_features)
        x = F.relu(x)
        
        # Apply GNN layers
        for i, layer in enumerate(self.gnn_layers):
            residual = x
            
            if self.gnn_type == 'sage':
                x = layer(x, edge_index)
            else:
                x = layer(x, edge_index)
            
            # Apply dropout and layer norm
            x = self.dropout_layer(x)
            
            # Residual connection (if dimensions match)
            if x.shape == residual.shape:
                x = x + residual
            
            x = self.layer_norm(x)
            x = F.relu(x)
        
        return x

class TGNNClassifier(nn.Module):
    """
    Complete T-GNN model for storm response classification
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(TGNNClassifier, self).__init__()
        
        # Model configuration
        self.sequence_length = config.get('sequence_length', 2016)
        self.input_channels = config.get('input_channels', 4)
        self.node_feature_dim = config.get('node_feature_dim', 100)
        self.temporal_hidden_dim = config.get('temporal_hidden_dim', 64)
        self.spatial_hidden_dim = config.get('spatial_hidden_dim', 64)
        self.num_classes = config.get('num_classes', 3)
        self.temporal_encoder_type = config.get('temporal_encoder_type', 'gru')
        self.spatial_encoder_type = config.get('spatial_encoder_type', 'gcn')
        self.temporal_layers = config.get('temporal_layers', 2)
        self.spatial_layers = config.get('spatial_layers', 2)
        self.dropout = config.get('dropout', 0.2)
        self.fusion_type = config.get('fusion_type', 'concat')  # concat, add, attention
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            input_channels=self.input_channels,
            hidden_dim=self.temporal_hidden_dim,
            num_layers=self.temporal_layers,
            encoder_type=self.temporal_encoder_type
        )
        
        # Spatial encoder
        self.spatial_encoder = SpatialEncoder(
            node_feature_dim=self.node_feature_dim,
            hidden_dim=self.spatial_hidden_dim,
            num_layers=self.spatial_layers,
            gnn_type=self.spatial_encoder_type,
            dropout=self.dropout
        )
        
        # Feature fusion
        if self.fusion_type == 'concat':
            fusion_dim = self.temporal_encoder.output_dim + self.spatial_hidden_dim
        elif self.fusion_type == 'add':
            # Ensure dimensions match
            assert self.temporal_encoder.output_dim == self.spatial_hidden_dim
            fusion_dim = self.spatial_hidden_dim
        elif self.fusion_type == 'attention':
            fusion_dim = max(self.temporal_encoder.output_dim, self.spatial_hidden_dim)
            self.attention_weights = nn.Linear(
                self.temporal_encoder.output_dim + self.spatial_hidden_dim, 2
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, self.num_classes)
        )
    
    def forward(self, temporal_data: torch.Tensor, node_features: torch.Tensor,
                edge_index: torch.Tensor, batch_node_indices: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of T-GNN model
        
        Args:
            temporal_data: Time series data (batch_size, sequence_length, input_channels)
            node_features: All node features (num_total_nodes, node_feature_dim)
            edge_index: Edge indices (2, num_edges)
            batch_node_indices: Indices of nodes in this batch (batch_size,)
            batch: Batch indices for spatial encoder (optional)
            
        Returns:
            Class logits (batch_size, num_classes)
        """
        # Temporal encoding
        temporal_features = self.temporal_encoder(temporal_data)  # (batch_size, temporal_dim)
        
        # Spatial encoding
        spatial_features_all = self.spatial_encoder(
            node_features, edge_index, batch
        )  # (num_total_nodes, spatial_dim)
        
        # Select spatial features for nodes in this batch
        spatial_features = spatial_features_all[batch_node_indices]  # (batch_size, spatial_dim)
        
        # Feature fusion
        if self.fusion_type == 'concat':
            fused_features = torch.cat([temporal_features, spatial_features], dim=1)
        elif self.fusion_type == 'add':
            fused_features = temporal_features + spatial_features
        elif self.fusion_type == 'attention':
            # Attention-based fusion
            combined = torch.cat([temporal_features, spatial_features], dim=1)
            attention_scores = F.softmax(self.attention_weights(combined), dim=1)
            
            temporal_weighted = attention_scores[:, 0:1] * temporal_features
            spatial_weighted = attention_scores[:, 1:2] * spatial_features
            fused_features = temporal_weighted + spatial_weighted
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits
    
    def predict_proba(self, temporal_data: torch.Tensor, node_features: torch.Tensor,
                     edge_index: torch.Tensor, batch_node_indices: torch.Tensor,
                     batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get prediction probabilities
        
        Returns:
            Class probabilities (batch_size, num_classes)
        """
        with torch.no_grad():
            logits = self.forward(temporal_data, node_features, edge_index, 
                                batch_node_indices, batch)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def predict(self, temporal_data: torch.Tensor, node_features: torch.Tensor,
               edge_index: torch.Tensor, batch_node_indices: torch.Tensor,
               batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get class predictions
        
        Returns:
            Predicted classes (batch_size,)
        """
        probabilities = self.predict_proba(temporal_data, node_features, edge_index,
                                         batch_node_indices, batch)
        predictions = torch.argmax(probabilities, dim=1)
        return predictions

class TGNNLoss(nn.Module):
    """
    Custom loss function for T-GNN with class balancing
    """
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None, 
                 label_smoothing: float = 0.0):
        super(TGNNLoss, self).__init__()
        
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, 
                                             label_smoothing=label_smoothing)
        else:
            self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss
        
        Args:
            logits: Model output logits (batch_size, num_classes)
            targets: Target labels (batch_size,)
            
        Returns:
            Loss value
        """
        return self.ce_loss(logits, targets)

def create_tgnn_model(config: Dict[str, Any]) -> TGNNClassifier:
    """
    Factory function to create T-GNN model
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        T-GNN model instance
    """
    model = TGNNClassifier(config)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param.data)
    
    model.apply(init_weights)
    return model

# Default model configuration
DEFAULT_TGNN_CONFIG = {
    'sequence_length': 2016,  # One week at 5-min intervals
    'input_channels': 4,  # depth, flow, velocity, rainfall
    'node_feature_dim': 100,  # Combined spatial + statistical features
    'temporal_hidden_dim': 64,
    'spatial_hidden_dim': 64,
    'num_classes': 7,  # 7 action categories
    'temporal_encoder_type': 'gru',  # gru, tcn, cnn
    'spatial_encoder_type': 'gcn',  # gcn, gat, sage
    'temporal_layers': 2,
    'spatial_layers': 2,
    'dropout': 0.2,
    'fusion_type': 'concat'  # concat, add, attention
}