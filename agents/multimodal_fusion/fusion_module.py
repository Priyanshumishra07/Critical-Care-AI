"""
Multimodal Fusion Module for combining imaging and clinical data features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ClinicalDataEncoder(nn.Module):
    """Encodes tabular clinical data (vital signs, lab results, demographics)."""
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, clinical_data: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(clinical_data))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x

class ImageFeatureExtractor(nn.Module):
    """Extracts features from medical imaging using pre-trained CNN."""
    
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        # Using a simple CNN for feature extraction
        # In practice, this would use a pre-trained ResNet/DenseNet backbone
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, feature_dim)
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(image)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MultimodalFusionModel(nn.Module):
    """Combines imaging and clinical features for enhanced predictions."""
    
    def __init__(self, image_feature_dim: int = 512, clinical_feature_dim: int = 128, 
                 num_classes: int = 2):
        super().__init__()
        self.image_encoder = ImageFeatureExtractor(image_feature_dim)
        self.clinical_encoder = ClinicalDataEncoder(hidden_dim=clinical_feature_dim)
        
        # Fusion layer
        fusion_dim = image_feature_dim + clinical_feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, image: torch.Tensor, clinical_data: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        clinical_features = self.clinical_encoder(clinical_data)
        
        # Concatenate features
        fused = torch.cat([image_features, clinical_features], dim=1)
        
        # Final prediction
        output = self.fusion(fused)
        return output

class MultimodalFusionAgent:
    """Agent for multimodal fusion analysis combining imaging and clinical data."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.model = self._load_model(model_path)
        else:
            logger.warning(f"Model not found at {model_path}. Using rule-based assessment.")
    
    def _load_model(self, model_path: str) -> MultimodalFusionModel:
        """Load trained multimodal fusion model."""
        model = MultimodalFusionModel()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def analyze(self, image_path: Optional[str], clinical_data: Dict) -> Dict:
        """
        Perform multimodal fusion analysis.
        
        Args:
            image_path: Path to medical image (optional)
            clinical_data: Dictionary with vital signs, lab results, demographics
            
        Returns:
            Dictionary with diagnosis, confidence, and risk assessment
        """
        if self.model is None:
            # Rule-based fallback
            return self._rule_based_assessment(clinical_data)
        
        # Model-based assessment would go here
        # For now, return rule-based
        return self._rule_based_assessment(clinical_data)
    
    def _rule_based_assessment(self, clinical_data: Dict) -> Dict:
        """Rule-based assessment when model is not available."""
        vital_signs = clinical_data.get('vital_signs', {})
        lab_results = clinical_data.get('lab_results', {})
        
        risk_score = 0.0
        
        # Temperature
        temp = vital_signs.get('temperature', 37.0)
        if temp > 38.5:
            risk_score += 0.2
        
        # Heart rate
        hr = vital_signs.get('heart_rate', 70)
        if hr > 100:
            risk_score += 0.15
        
        # WBC
        wbc = lab_results.get('wbc', 7.0)
        if wbc > 12.0:
            risk_score += 0.2
        
        # Lactate
        lactate = lab_results.get('lactate', 1.0)
        if lactate > 2.0:
            risk_score += 0.25
        
        risk_score = min(risk_score, 1.0)
        
        if risk_score < 0.3:
            risk_level = "LOW"
        elif risk_score < 0.6:
            risk_level = "MODERATE"
        else:
            risk_level = "HIGH"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'confidence': 0.7,
            'method': 'rule_based'
        }

import os

