"""
Pneumonia Detection Agent for Chest X-ray Analysis

This module provides pneumonia classification from chest X-ray images.
"""

import logging
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

class PneumoniaClassification:
    """
    Pneumonia detection agent using deep learning for chest X-ray classification.
    
    This agent classifies chest X-ray images as:
    - 'pneumonia': Positive for pneumonia
    - 'normal': Normal lung appearance
    """
    
    def __init__(self, model_path, device=None):
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.class_names = ['normal', 'pneumonia']
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._build_model(weights=None)
        if os.path.exists(model_path):
            self._load_model_weights(model_path)
        else:
            self.logger.warning(f"Model file not found at {model_path}. Using untrained model.")
        self.model.to(self.device)
        self.model.eval()
        
        # Image transformations
        self.mean_nums = [0.485, 0.456, 0.406]
        self.std_nums = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean_nums, std=self.std_nums)
        ])
    
    def _build_model(self, weights=None):
        """Initialize the DenseNet model with custom classification layer."""
        # Match the exact structure from training script (PneumoniaClassifier)
        class PneumoniaClassifier(nn.Module):
            def __init__(self, num_classes=2):
                super(PneumoniaClassifier, self).__init__()
                self.backbone = models.densenet121(weights=None)
                num_ftrs = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Linear(num_ftrs, num_classes)
            
            def forward(self, x):
                return self.backbone(x)
        
        return PneumoniaClassifier(num_classes=len(self.class_names))
    
    def _load_model_weights(self, model_path):
        """Load pre-trained model weights."""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise e
    
    def predict(self, img_path):
        """
        Predict pneumonia from a chest X-ray image.
        
        Args:
            img_path: Path to the chest X-ray image
            
        Returns:
            str: 'pneumonia' or 'normal'
        """
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)
            input_tensor = Variable(image_tensor).to(self.device)
            
            with torch.no_grad():
                out = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(out, dim=1)
                _, preds = torch.max(out, 1)
                idx = preds.cpu().numpy()[0]
                pred_class = self.class_names[idx]
                confidence = probabilities[0][idx].item()
                
            self.logger.info(f"Predicted Class: {pred_class} (Confidence: {confidence:.2%})")
            
            return pred_class
        except Exception as e:
            self.logger.error(f"Error during pneumonia prediction: {str(e)}")
            return None

