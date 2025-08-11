import os
import cv2
import logging
from app.models.skin_analyzer import SkinAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_skin_analyzer(image_path: str):
    """
    Test the SkinAnalyzer with a sample image
    
    Args:
        image_path: Path to the image file to analyze
    """
    try:
        # Initialize the analyzer
        logger.info("Initializing Skin Analyzer...")
        analyzer = SkinAnalyzer()
        
        # Check if image exists
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return
        
        # Analyze the image
        logger.info(f"Analyzing image: {image_path}")
        result = analyzer.analyze_skin(image_path)
        
        # Print results
        print("\n=== SKIN ANALYSIS RESULTS ===")
        print(f"Image: {os.path.basename(image_path)}")
        print("-" * 50)
        
        if isinstance(result, dict):
            # Handle dictionary result (fallback)
            print("Analysis Results (Dictionary Format):")
            for key, value in result.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"{key}: {value}")
        else:
            # Handle SkinAnalysisResult dataclass
            print(f"Skin Type: {getattr(result, 'skin_type', 'N/A')}")
            print(f"Skin Tone: {getattr(result, 'skin_tone', 'N/A')}")
            print(f"Perceived Age: {getattr(result, 'perceived_age', 'N/A')}")
            print(f"Eye Age: {getattr(result, 'eye_age', 'N/A')}")
            
            # Print concerns if available
            concerns = getattr(result, 'concerns', {})
            if concerns:
                print("\nDetected Concerns:")
                for concern, confidence in concerns.items():
                    print(f"- {concern}: {confidence*100:.1f}%")
            
            # Get and print recommendations if available
            try:
                if hasattr(analyzer, 'get_recommendations'):
                    print("\n=== RECOMMENDATIONS ===")
                    recs = analyzer.get_recommendations(result)
                    
                    if isinstance(recs, dict):
                        if 'products' in recs and recs['products']:
                            print("\nRecommended Products:")
                            for product in recs['products']:
                                if isinstance(product, dict):
                                    print(f"\n- {product.get('name', 'Unknown')}")
                                    if 'purpose' in product:
                                        print(f"  Purpose: {product['purpose']}")
                                    if 'description' in product:
                                        print(f"  {product['description']}")
                                    if 'key_ingredients' in product:
                                        print(f"  Key Ingredients: {', '.join(product['key_ingredients'])}")
                        
                        if 'routines' in recs and isinstance(recs['routines'], dict):
                            print("\n=== DAILY ROUTINE ===")
                            if 'morning' in recs['routines']:
                                print("\nMorning:")
                                for step in recs['routines']['morning']:
                                    print(f"- {step}")
                            
                            if 'evening' in recs['routines']:
                                print("\nEvening:")
                                for step in recs['routines']['evening']:
                                    print(f"- {step}")
                        
                        if 'next_scan' in recs:
                            print(f"\nNext recommended scan: {recs['next_scan']}")
                            
            except Exception as e:
                print(f"\nCould not generate recommendations: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error during skin analysis: {str(e)}", exc_info=True)

def get_sample_images():
    """Get list of sample images from the images directory"""
    images_dir = os.path.join(os.path.dirname(__file__), 'images')
    if not os.path.exists(images_dir):
        print("Images directory not found. Creating it...")
        os.makedirs(images_dir, exist_ok=True)
        return []
    
    # Get all image files from the directory
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [
        os.path.join(images_dir, f) 
        for f in os.listdir(images_dir) 
        if f.lower().endswith(image_extensions)
    ]
    
    return image_files

if __name__ == "__main__":
    # Get all sample images
    sample_images = get_sample_images()
    
    if not sample_images:
        print("No images found in the 'images' directory.")
        print("Please add some test images to the 'images' folder and try again.")
    else:
        print(f"Found {len(sample_images)} images to process:\n")
        
        # Process each image
        for i, image_path in enumerate(sample_images, 1):
            print(f"\n{'='*50}")
            print(f"PROCESSING IMAGE {i}/{len(sample_images)}: {os.path.basename(image_path)}")
            print(f"{'='*50}")
            
            try:
                test_skin_analyzer(image_path)
            except Exception as e:
                print(f"\nError processing {image_path}: {str(e)}\n")
