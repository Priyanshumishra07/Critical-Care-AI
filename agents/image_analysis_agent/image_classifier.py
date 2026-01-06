import os
import json
import base64
from mimetypes import guess_type

from typing import TypedDict
from langchain_core.output_parsers import JsonOutputParser

class ClassificationDecision(TypedDict):
    """Output structure for the decision agent."""
    image_type: str
    reasoning: str
    confidence: float

class ImageClassifier:
    """Uses GPT-4o Vision to analyze images and determine their type."""
    
    def __init__(self, vision_model):
        self.vision_model = vision_model
        self.json_parser = JsonOutputParser(pydantic_object=ClassificationDecision)
        
    def local_image_to_data_url(self, image_path: str) -> str:
        """
        Get the url of a local image
        """
        mime_type, _ = guess_type(image_path)

        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

        return f"data:{mime_type};base64,{base64_encoded_data}"
    
    def _fallback_classify_image(self, image_path: str) -> dict:
        """Fallback image classification using file analysis when LLM is unavailable."""
        import cv2
        import numpy as np
        
        # Simple heuristics-based classification
        filename_lower = os.path.basename(image_path).lower()
        
        # Check filename for keywords
        if any(keyword in filename_lower for keyword in ['chest', 'xray', 'x-ray', 'pneumonia', 'lung', 'cxr']):
            return {
                "image_type": "CHEST X-RAY",
                "reasoning": "Detected chest X-ray based on filename keywords",
                "confidence": 0.7
            }
        elif any(keyword in filename_lower for keyword in ['skin', 'lesion', 'dermatology', 'isic', 'melanoma']):
            return {
                "image_type": "SKIN LESION",
                "reasoning": "Detected skin lesion image based on filename keywords",
                "confidence": 0.7
            }
        
        # Try to analyze image dimensions and characteristics
        try:
            img = cv2.imread(image_path)
            if img is not None:
                height, width = img.shape[:2]
                # Chest X-rays are typically wider than tall
                if width > height * 1.2:
                    return {
                        "image_type": "CHEST X-RAY",
                        "reasoning": "Detected chest X-ray based on image dimensions (wide format)",
                        "confidence": 0.6
                    }
                # Skin images are often square or portrait
                elif abs(width - height) < width * 0.2:
                    return {
                        "image_type": "SKIN LESION",
                        "reasoning": "Detected skin lesion based on image dimensions (square/portrait)",
                        "confidence": 0.6
                    }
        except Exception as e:
            print(f"[ImageAnalyzer] Fallback analysis error: {e}")
        
        # Default fallback
        return {
            "image_type": "CLINICAL IMAGING",
            "reasoning": "Could not determine specific type, defaulting to clinical imaging",
            "confidence": 0.5
        }
    
    def classify_image(self, image_path: str) -> str:
        """Analyzes the image to classify it as a medical image and determine it's type."""
        print(f"[ImageAnalyzer] Analyzing image: {image_path}")

        # Check if vision model is available
        if self.vision_model is None:
            print("[ImageAnalyzer] Vision model not available, using fallback classification")
            return self._fallback_classify_image(image_path)

        try:
            vision_prompt = [
                {"role": "system", "content": "You are an expert in medical imaging. Analyze the uploaded image."},
                {"role": "user", "content": [
                    {"type": "text", "text": (
                        """
                        Determine if this is a medical image. If it is, classify it as:
                        'CHEST X-RAY', 'SKIN LESION', 'CT SCAN', 'CLINICAL IMAGING', or 'OTHER'. If it's not a medical image, return 'NON-MEDICAL'.
                        You must provide your answer in JSON format with the following structure:
                        {{
                        "image_type": "IMAGE TYPE",
                        "reasoning": "Your step-by-step reasoning for selecting this agent",
                        "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this classification task
                        }}
                        """
                    )},
                    {"type": "image_url", "image_url": {"url": self.local_image_to_data_url(image_path)}}  # Correct format
                ]}
            ]
            
            # Invoke LLM to classify the image
            response = self.vision_model.invoke(vision_prompt)

            try:
                # Ensure the response is parsed as JSON
                response_json = self.json_parser.parse(response.content)
                return response_json  # Returns a dictionary instead of a string
            except json.JSONDecodeError:
                print("[ImageAnalyzer] Warning: Response was not valid JSON, using fallback.")
                return self._fallback_classify_image(image_path)
        except Exception as e:
            print(f"[ImageAnalyzer] Error using vision model: {e}, using fallback")
            return self._fallback_classify_image(image_path)

        # return response.content.strip().lower()
