from .image_classifier import ImageClassifier
from .pneumonia_agent.pneumonia_inference import PneumoniaClassification
from .skin_lesion_agent.skin_lesion_inference import SkinLesionSegmentation

class ImageAnalysisAgent:
    """
    Agent responsible for processing image uploads and classifying them as medical or non-medical, 
    and determining their type. Also handles pneumonia detection and skin lesion segmentation.
    """
    
    def __init__(self, config):
        # Handle both full config object and medical_cv config
        if hasattr(config, 'medical_cv'):
            medical_cv_config = config.medical_cv
        else:
            medical_cv_config = config
        
        self.image_classifier = ImageClassifier(vision_model=medical_cv_config.llm)
        self.pneumonia_agent = PneumoniaClassification(model_path=medical_cv_config.pneumonia_model_path)
        self.skin_lesion_agent = SkinLesionSegmentation(model_path=medical_cv_config.skin_lesion_model_path)
    
    # classify image
    def analyze_image(self, image_path: str) -> str:
        """Classifies images as medical or non-medical and determines their type."""
        return self.image_classifier.classify_image(image_path)
    
    # pneumonia agent
    def classify_pneumonia(self, image_path: str) -> str:
        """Classify chest X-ray for pneumonia."""
        return self.pneumonia_agent.predict(image_path)
    
    # skin lesion agent
    def segment_skin_lesion(self, image_path: str, output_path: str) -> bool:
        """Segment skin lesion and create overlay visualization."""
        return self.skin_lesion_agent.predict(image_path, output_path)
