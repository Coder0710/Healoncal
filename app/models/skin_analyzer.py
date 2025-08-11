import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModelForImageSegmentation
import logging
import os
import cv2
import numpy as np
import datetime
from typing import Dict, Any, List, Tuple, Optional
import uuid
import random
from enum import Enum
from dataclasses import dataclass
from scipy import stats

# Enums for skin analysis
class SkinType(str, Enum):
    NORMAL = "Normal"
    DRY = "Dry"
    OILY = "Oily"
    COMBINATION = "Combination"
    SENSITIVE = "Sensitive"

class SkinConcern(str, Enum):
    ACNE = "Acne"
    PIGMENTATION = "Pigmentation"
    PORES = "Large Pores"
    WRINKLES = "Wrinkles"
    DULLNESS = "Dullness"
    REDNESS = "Redness"

@dataclass
class SkinAnalysisResult:
    skin_type: SkinType
    skin_tone: str
    perceived_age: int
    eye_age: int
    concerns: Dict[SkinConcern, float]  # concern type -> confidence
    moisture_level: float  # 0-1 scale
    oiliness: float  # 0-1 scale
    pores_visibility: float  # 0-1 scale
    redness: float  # 0-1 scale
    texture_score: float  # 0-1 scale (higher is rougher)
    analysis_timestamp: datetime.datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinAnalyzer:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Initialize models
        self.skin_type_model = None
        self.face_analysis_model = None
        self.skin_concern_model = None
        self.processor = None
        
        # Load all required models
        self._load_models()
    
    def _load_models(self):
        """Load all required models for skin analysis"""
        try:
            # Load face analysis model (age, gender, etc.)
            logger.info("Loading face analysis model...")
            self.face_processor = AutoImageProcessor.from_pretrained('microsoft/resnet-50', use_fast=True)
            self.face_analysis_model = AutoModelForImageClassification.from_pretrained('microsoft/resnet-50')
            self.face_analysis_model = self.face_analysis_model.to(self.device)
            self.face_analysis_model.eval()
            
            # Load skin type classifier
            logger.info("Loading skin type classifier...")
            self.skin_type_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224', use_fast=True)
            self.skin_type_model = AutoModelForImageClassification.from_pretrained('google/vit-base-patch16-224')
            self.skin_type_model = self.skin_type_model.to(self.device)
            self.skin_type_model.eval()
            
            # Load skin concern detector
            logger.info("Loading skin concern detector...")
            self.skin_concern_processor = AutoImageProcessor.from_pretrained('facebook/deit-base-patch16-224', use_fast=True)
            self.skin_concern_model = AutoModelForImageClassification.from_pretrained('facebook/deit-base-patch16-224')
            self.skin_concern_model = self.skin_concern_model.to(self.device)
            self.skin_concern_model.eval()
            
            logger.info("✅ All models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {str(e)}")
            raise RuntimeError("Failed to initialize skin analysis models. Please check your internet connection and try again.")
    
    def _detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face in the image and return bounding box"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            return faces[0]  # Return first face found
        return None

    def _detect_eyes(self, face_roi: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect eyes in the face region"""
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 4)
        return eyes

    def _analyze_skin_type(self, face_roi: np.ndarray) -> Dict[str, float]:
        """Analyze skin type based on texture, oiliness, and pores"""
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Analyze texture (variance of Laplacian)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_score = np.var(laplacian)
        
        # Convert to LAB color space for better skin analysis
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Analyze oiliness (brightness in LAB space)
        oiliness = np.mean(l) / 255.0
        
        # Analyze pores (local binary patterns)
        radius = 3
        n_points = 8 * radius
        lbp = np.zeros_like(gray)
        for i in range(radius, gray.shape[0] - radius):
            for j in range(radius, gray.shape[1] - radius):
                center = gray[i, j]
                code = 0
                for k in range(n_points):
                    x = i + int(radius * np.cos(2 * np.pi * k / n_points))
                    y = j - int(radius * np.sin(2 * np.pi * k / n_points))
                    code |= (gray[x, y] > center) << k
                lbp[i, j] = code
        
        # Calculate histogram of LBP
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype("float") / (hist.sum() + 1e-7)
        
        # Calculate energy (uniformity of the histogram)
        energy = np.sum(hist**2)
        
        # Calculate entropy (randomness in the texture)
        entropy = -np.sum([p * np.log2(p + 1e-10) for p in hist if p > 0])
        
        # Determine skin type based on features
        if oiliness > 0.7 and energy > 0.03:
            skin_type = SkinType.OILY
        elif oiliness < 0.4 and energy < 0.02:
            skin_type = SkinType.DRY
        elif 0.4 <= oiliness <= 0.6 and entropy > 5.0:
            skin_type = SkinType.COMBINATION
        elif entropy > 5.5:
            skin_type = SkinType.SENSITIVE
        else:
            skin_type = SkinType.NORMAL
        
        return {
            'skin_type': skin_type,
            'oiliness': float(oiliness),
            'texture_score': float(texture_score / 1000),  # Normalized
            'pores_visibility': float(energy * 2),  # Scale to 0-1
            'moisture_level': 1.0 - (oiliness * 0.5)  # Simple estimation
        }

    def _analyze_face(self, face_roi: np.ndarray) -> Dict[str, Any]:
        """Analyze face for age, gender, and other attributes"""
        # Convert to PIL Image for processing
        face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        
        # Preprocess for the model
        inputs = self.face_processor(images=face_pil, return_tensors="pt").to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.face_analysis_model(**inputs)
            logits = outputs.logits
            
        # Process predictions (this is a simplified example)
        # In a real implementation, you would use a proper age/gender estimation model
        predicted_age = int(torch.argmax(logits[0]).item() % 100)  # Example
        
        # For demo purposes, add some randomness
        predicted_age = max(18, min(80, predicted_age + random.randint(-5, 5)))
        
        # Estimate eye age (typically slightly younger than actual age)
        eye_age = max(18, predicted_age - random.randint(2, 5))
        
        return {
            'perceived_age': predicted_age,
            'eye_age': eye_age,
            'skin_tone': self._estimate_skin_tone(face_roi)
        }
    
    def _estimate_skin_tone(self, face_roi: np.ndarray) -> str:
        """Estimate skin tone from face ROI"""
        # Convert to YCrCb color space (better for skin tone analysis)
        ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
        
        # Define skin tone ranges in YCrCb
        # These values are approximate and may need adjustment
        skin_tones = {
            'Very Fair': ((0, 133, 77), (255, 173, 127)),
            'Fair': ((0, 128, 133), (255, 168, 183)),
            'Medium': ((0, 128, 143), (255, 168, 193)),
            'Olive': ((0, 128, 153), (255, 168, 213)),
            'Tan': ((0, 128, 163), (255, 168, 223)),
            'Brown': ((0, 128, 173), (255, 168, 233)),
            'Dark': ((0, 128, 183), (255, 168, 243))
        }
        
        # Count pixels in each skin tone range
        tone_counts = {}
        for tone, (lower, upper) in skin_tones.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(ycrcb, lower, upper)
            tone_counts[tone] = cv2.countNonZero(mask)
        
        # Return the most common skin tone
        return max(tone_counts.items(), key=lambda x: x[1])[0]
    
    def _detect_skin_concerns(self, face_roi: np.ndarray) -> Dict[SkinConcern, float]:
        """Detect various skin concerns"""
        # This is a simplified implementation
        # In a real system, you would use a more sophisticated model
        concerns = {}
        
        # Convert to grayscale for some analyses
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect acne (simple blob detection)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 100
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        acne_score = min(1.0, len(keypoints) / 20.0)  # Normalize to 0-1
        concerns[SkinConcern.ACNE] = acne_score
        
        # Detect pigmentation (color variance)
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        pigmentation = np.std(a) / 10.0  # Normalize
        concerns[SkinConcern.PIGMENTATION] = min(1.0, pigmentation)
        
        # Detect pores (high frequency components)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        pores_score = np.mean(magnitude_spectrum) / 100.0  # Normalize
        concerns[SkinConcern.PORES] = min(1.0, pores_score)
        
        # Detect wrinkles (edge detection)
        edges = cv2.Canny(gray, 50, 150)
        wrinkles_score = np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255) * 10
        concerns[SkinConcern.WRINKLES] = min(1.0, wrinkles_score)
        
        # Dullness (low contrast)
        contrast = gray.std() / 128.0  # Normalize
        concerns[SkinConcern.DULLNESS] = max(0, 1.0 - contrast)
        
        # Redness (from a channel in LAB space)
        redness = np.mean(a) / 128.0  # Normalize
        concerns[SkinConcern.REDNESS] = min(1.0, redness)
        
        return concerns
    
    def _preprocess_image(self, image_path: str) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """Preprocess the image and detect face"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read the image")
                
            # Convert to RGB for processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face
            face_bbox = self._detect_face(image)
            
            if face_bbox is None:
                logger.warning("No face detected in the image")
                return image_rgb, None
                
            return image_rgb, face_bbox
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def analyze_skin(self, image_path: str) -> SkinAnalysisResult:
        """
        Main method to analyze skin from an image
        
        Args:
            image_path: Path to the image file to analyze
            
        Returns:
            SkinAnalysisResult object containing all analysis results
        """
        try:
            # Preprocess image and detect face
            image, face_bbox = self._preprocess_image(image_path)
            
            if face_bbox is None:
                raise ValueError("No face detected in the image")
                
            # Extract face region
            x, y, w, h = face_bbox
            face_roi = image[y:y+h, x:x+w]
            
            # Perform all analyses
            skin_analysis = self._analyze_skin_type(face_roi)
            face_analysis = self._analyze_face(face_roi)
            concerns = self._detect_skin_concerns(face_roi)
            
            # Create result object
            result = SkinAnalysisResult(
                skin_type=skin_analysis['skin_type'],
                skin_tone=face_analysis['skin_tone'],
                perceived_age=face_analysis['perceived_age'],
                eye_age=face_analysis['eye_age'],
                concerns=concerns,
                moisture_level=skin_analysis['moisture_level'],
                oiliness=skin_analysis['oiliness'],
                pores_visibility=skin_analysis['pores_visibility'],
                redness=concerns.get(SkinConcern.REDNESS, 0.0),
                texture_score=skin_analysis['texture_score'],
                analysis_timestamp=datetime.datetime.now()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing skin: {str(e)}")
            raise
    
    def get_recommendations(self, analysis_result: SkinAnalysisResult) -> Dict[str, Any]:
        """
        Generate personalized recommendations based on skin analysis
        
        Args:
            analysis_result: Result from analyze_skin()
            
        Returns:
            Dictionary containing product and treatment recommendations
        """
        recommendations = {
            'professional_help': [],
            'products': [],
            'routines': [],
            'next_scan': (datetime.datetime.now() + datetime.timedelta(weeks=4)).strftime("%Y-%m-%d")
        }
        
        # Professional help recommendations
        if any(score > 0.7 for score in analysis_result.concerns.values()):
            recommendations['professional_help'].append({
                'type': 'Dermatologist',
                'priority': 'High',
                'reason': 'Significant skin concerns detected that may require professional attention'
            })
        
        # Product recommendations based on skin type and concerns
        if analysis_result.skin_type == SkinType.DRY:
            recommendations['products'].extend([
                {
                    'name': 'Hyaluronic Acid Serum',
                    'purpose': 'Hydration',
                    'description': 'Deeply hydrates and plumps the skin',
                    'key_ingredients': ['Hyaluronic Acid', 'Glycerin']
                },
                {
                    'name': 'Rich Moisturizing Cream',
                    'purpose': 'Moisture Barrier',
                    'description': 'Repairs and protects the skin barrier',
                    'key_ingredients': ['Ceramides', 'Shea Butter']
                }
            ])
        
        # Add more product recommendations based on other skin types and concerns...
        
        # Routine recommendations
        morning_routine = []
        if analysis_result.oiliness > 0.6:
            morning_routine.append('Use a gentle foaming cleanser')
        else:
            morning_routine.append('Use a hydrating cleanser')
            
        morning_routine.extend([
            'Apply vitamin C serum',
            'Apply moisturizer suitable for your skin type',
            'Apply broad-spectrum SPF 30+ sunscreen'
        ])
        
        recommendations['routines'] = {
            'morning': morning_routine,
            'evening': [
                'Double cleanse (oil-based cleanser followed by water-based)',
                'Apply treatment products (retinol, acids, etc.)',
                'Apply night cream or sleeping mask'
            ]
        }
        
        return recommendations

    def _map_to_skin_condition(self, predicted_label: str, confidence: float) -> Tuple[str, float]:
        """
        Map the model's predicted label to a standardized skin condition.
        
        Args:
            predicted_label: The label predicted by the model
            confidence: The confidence score (0-1)
            
        Returns:
            Tuple of (mapped_condition, adjusted_confidence)
        """
        if not predicted_label:
            return "unable to determine", 0.0
            
        label_lower = str(predicted_label).lower()
        logger.info(f"Mapping label: {label_lower} with confidence: {confidence}")
        
        # Define condition mapping with minimum confidence thresholds and alternative names
        condition_map = {
            # Format: 'key': ('display_name', min_confidence, [alternative_names])
            # Cancerous/Malignant conditions (high priority)
            'melanoma': ('suspicious lesion - consult doctor immediately', 0.7, ['malignant melanoma', 'skin cancer']),
            'basal cell carcinoma': ('suspicious lesion - consult doctor immediately', 0.75, ['bcc']),
            'squamous cell carcinoma': ('suspicious lesion - consult doctor immediately', 0.75, ['scc']),
            'actinic keratosis': ('pre-cancerous lesion - medical evaluation recommended', 0.65, ['solar keratosis']),
            'carcinoma': ('suspicious lesion - medical evaluation recommended', 0.7, []),
            
            # Common skin conditions
            'psoriasis': ('psoriasis', 0.65, ['psoriatic']),
            'eczema': ('eczema', 0.6, ['atopic dermatitis']),
            'acne': ('acne', 0.55, ['pimple', 'pustule', 'blackhead', 'whitehead']),
            'rosacea': ('rosacea', 0.6, []),
            'dermatitis': ('contact dermatitis', 0.6, ['irritant dermatitis', 'allergic contact dermatitis']),
            'seborrheic dermatitis': ('seborrheic dermatitis', 0.6, ['dandruff', 'cradle cap']),
            'tinea': ('fungal infection', 0.6, ['ringworm', 'athlete\'s foot', 'jock itch']),
            'impetigo': ('bacterial skin infection', 0.65, []),
            'herpes': ('viral skin infection', 0.7, ['cold sore', 'fever blister']),
            
            # Benign Growths
            'nevus': ('mole', 0.55, ['mole']),
            'dermatofibroma': ('dermatofibroma (benign skin growth)', 0.6, []),
            'sebaceous hyperplasia': ('sebaceous hyperplasia (enlarged oil glands)', 0.6, []),
            'skin tag': ('skin tag (benign growth)', 0.55, ['acrochordon']),
            
            # Vascular Lesions
            'hemangioma': ('hemangioma (benign blood vessel growth)', 0.6, ['strawberry mark']),
            'cherry angioma': ('cherry angioma (benign blood vessel growth)', 0.6, []),
            'spider angioma': ('spider angioma (benign blood vessel growth)', 0.6, ['spider nevus']),
            'pyogenic granuloma': ('pyogenic granuloma (benign vascular growth)', 0.65, []),
            
            # Normal Skin
            'normal': ('healthy skin', 0.5, ['clear skin', 'unremarkable']),
            'healthy': ('healthy skin', 0.5, ['clear skin', 'unremarkable'])
        }
        
        # Check for direct matches first with confidence threshold
        for key, value in condition_map.items():
            try:
                if len(value) == 3:
                    display_name, min_confidence, alternatives = value
                else:
                    display_name, min_confidence = value
                    alternatives = []
                
                # Check direct match
                if key in label_lower and confidence >= min_confidence:
                    return display_name, confidence
                    
                # Check alternative names
                if any(alt in label_lower for alt in alternatives) and confidence >= min_confidence:
                    return display_name, confidence * 0.95  # Slightly reduce confidence for alternative name matches
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid condition map entry for key '{key}': {value}")
                continue
        
        # If no direct match, try partial matching
        for key, value in condition_map.items():
            try:
                if len(value) == 3:
                    display_name, min_confidence, alternatives = value
                else:
                    display_name, min_confidence = value
                    alternatives = []
                
                key_clean = key.replace('_', ' ')
                search_terms = [key_clean] + alternatives
                
                # Check if any search term partially matches the label
                for term in search_terms:
                    if (term and 
                        (term in label_lower or 
                         any(word in label_lower.split() for word in term.split()))) and confidence >= min_confidence:
                        return display_name, confidence * 0.9  # Reduce confidence for partial matches
                        
            except (ValueError, TypeError) as e:
                continue
        
        # If still no match, return the original with reduced confidence
        return label_lower, confidence * 0.7
    
    def _estimate_skin_tone(self, image: Image.Image) -> str:
        """Estimate skin tone from the image using color analysis"""
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to LAB color space (better for skin tone analysis)
            lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Calculate mean values
            l_mean = np.mean(l)
            a_mean = np.mean(a)
            b_mean = np.mean(b)
            
            # Fitzpatrick scale classification (simplified)
            if l_mean > 180:
                return 'very light'
            elif l_mean > 160:
                return 'light'
            elif l_mean > 140:
                return 'medium'
            elif l_mean > 120:
                return 'olive/light brown'
            elif l_mean > 100:
                return 'brown'
            else:
                return 'dark brown/black'
                
        except Exception as e:
            logger.warning(f"Error estimating skin tone: {str(e)}")
            return 'medium'
    
    def _analyze_texture(self, image: Image.Image) -> dict:
        """Analyze skin texture using edge detection and contrast"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Calculate contrast (standard deviation of pixel intensities)
            contrast = np.std(gray)
            
            # Edge detection
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Texture classification
            if edge_density > 0.1 and contrast > 40:
                texture = 'rough/uneven'
            elif edge_density > 0.05 or contrast > 30:
                texture = 'slightly rough'
            else:
                texture = 'smooth'
                
            return {
                'texture': texture,
                'contrast': float(contrast),
                'edge_density': float(edge_density)
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing texture: {str(e)}")
            return {
                'texture': 'smooth',
                'contrast': 0,
                'edge_density': 0
            }
    
    def _get_suggested_actions(self, condition: str, analysis_quality: str, confidence: float) -> List[str]:
        """Generate suggested next steps based on analysis quality and confidence"""
        actions = []
        
        # Add quality-based suggestions
        if analysis_quality == 'low':
            actions.extend([
                "Consider better lighting for more accurate analysis",
                "Take a closer, well-focused photo of the area of concern",
                "Ensure the area is clean and free of makeup or lotions"
            ])
        
        # Add condition-specific suggestions
        condition = condition.lower()
        if any(term in condition for term in ['cancer', 'melanoma', 'carcinoma', 'suspicious']):
            actions.extend([
                "Consult a dermatologist immediately for evaluation",
                "Document the area with photos and notes for tracking",
                "Avoid sun exposure and use SPF 50+ sunscreen"
            ])
        elif confidence < 0.5:
            actions.append("The analysis confidence is low - consider professional evaluation")
        
        # Always include general skin care advice
        actions.append("Maintain a consistent skincare routine suitable for your skin type")
        
        return actions
    
    def _get_analysis_notes(self, condition: str, skin_tone: str, texture: dict) -> List[str]:
        """Generate additional analysis notes based on the findings"""
        notes = []
        
        # Add notes about skin tone
        if skin_tone:
            notes.append(f"Analysis suggests {skin_tone} skin tone")
        
        # Add notes about texture
        if texture.get('texture'):
            notes.append(f"Skin appears {texture['texture']} based on texture analysis")
        
        # Add condition-specific notes
        condition = condition.lower()
        if any(term in condition for term in ['dry', 'dehydrated']):
            notes.append("Consider using a more intensive moisturizer")
        elif 'oily' in condition:
            notes.append("Oil-balancing products may help regulate sebum production")
        
        return notes
    
    def _generate_recommendations(self, condition: str) -> List[str]:
        """Generate personalized recommendations based on the detected skin condition"""
        condition = condition.lower()
        recommendations = []
        
        # Serious conditions
        if any(term in condition for term in ['cancer', 'melanoma', 'carcinoma', 'suspicious']):
            recommendations.extend([
                "Medical attention required: Please consult a dermatologist immediately",
                "Document the area with photos and measurements for tracking",
                "Protect the area from sun exposure with SPF 50+ and clothing"
            ])
        # Common skin conditions
        elif 'acne' in condition:
            recommendations.extend([
                "Use a gentle, non-comedogenic cleanser twice daily",
                "Consider products with salicylic acid or benzoyl peroxide",
                "Avoid picking or squeezing blemishes to prevent scarring"
            ])
        elif any(term in condition for term in ['eczema', 'dermatitis']):
            recommendations.extend([
                "Use fragrance-free, hypoallergenic moisturizers regularly",
                "Avoid known irritants and harsh soaps",
                "Use lukewarm (not hot) water for cleansing"
            ])
        elif 'psoriasis' in condition:
            recommendations.extend([
                "Keep skin well-moisturized, especially after bathing",
                "Consider medicated shampoos for scalp involvement",
                "Manage stress as it can trigger flare-ups"
            ])
        # General skin care
        elif any(term in condition for term in ['mole', 'nevus']):
            recommendations.extend([
                "Monitor for changes in size, shape, or color (ABCDE rule)",
                "Use broad-spectrum SPF 30+ sunscreen daily",
                "Schedule annual skin checks with a dermatologist"
            ])
        elif any(term in condition for term in ['aging', 'wrinkle', 'fine line']):
            recommendations.extend([
                "Use a broad-spectrum SPF 30+ sunscreen daily",
                "Consider products with retinol or peptides",
                "Stay hydrated and maintain a diet rich in antioxidants"
            ])
        else:
            recommendations.extend([
                "Maintain a consistent skincare routine",
                "Stay hydrated and eat a balanced diet",
                "Use broad-spectrum sunscreen daily",
                "Consider consulting a dermatologist for personalized advice"
            ])
        
        return recommendations
        
    def analyze_skin(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a skin image and return the results.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing analysis results
        """
        start_time = datetime.datetime.now()
        try:
            logger.info(f"Processing image: {image_path}")
            
            # Check if image exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load and preprocess image
            try:
                # Convert to numpy array for OpenCV processing
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError("Could not read image file")
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect face
                face_bbox = self._detect_face(image_rgb)
                if face_bbox is None:
                    raise ValueError("No face detected in the image")
                
                # Extract face ROI
                x, y, w, h = face_bbox
                face_roi = image_rgb[y:y+h, x:x+w]
                
                # Convert to PIL Image for further processing
                face_pil = Image.fromarray(face_roi)
                
                # Analyze skin
                skin_type = self._analyze_skin_type(face_roi)
                skin_tone = self._estimate_skin_tone(face_roi)
                face_analysis = self._analyze_face(face_roi)
                concerns = self._detect_skin_concerns(face_roi)
                
                # Create result object
                result = SkinAnalysisResult(
                    skin_type=skin_type,
                    skin_tone=skin_tone,
                    perceived_age=face_analysis.get('age', 30),
                    eye_age=face_analysis.get('eye_age', 30),
                    concerns=concerns,
                    moisture_level=0.7,  # Placeholder values
                    oiliness=0.5,
                    pores_visibility=0.3,
                    redness=0.2,
                    texture_score=0.8,
                    analysis_timestamp=datetime.datetime.now()
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Error in analyze_skin: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f'Failed to analyze image: {str(e)}',
                'details': str(e)
            }
