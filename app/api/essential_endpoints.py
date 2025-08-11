from fastapi import APIRouter, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
import logging
from pathlib import Path
import uuid
import os
import base64
import json
import io
import numpy as np
from PIL import Image
import cv2
from typing import Dict, Any, List
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for client sessions
client_sessions = {}

def validate_image_quality(image: np.ndarray, angle: str) -> Dict[str, Any]:
    """
    Validate image quality for skin analysis.
    
    Args:
        image: Input image as numpy array
        angle: Capture angle (front, left, right, etc.)
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    
    # Check image dimensions
    height, width = image.shape[:2]
    if height < 480 or width < 640:
        issues.append(f"Image resolution too low: {width}x{height}")
    
    # Check image brightness (simple check using mean pixel value)
    if len(image.shape) == 3:  # Color image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    brightness = np.mean(gray)
    if brightness < 30:
        issues.append("Image too dark")
    elif brightness > 220:
        issues.append("Image too bright")
    
    # Check focus (using Laplacian variance)
    focus = cv2.Laplacian(gray, cv2.CV_64F).var()
    if focus < 100:  # Adjust threshold as needed
        issues.append("Image may be out of focus")
    
    # Angle-specific checks
    if angle == 'front':
        # Add front-specific checks
        pass
    elif angle in ['left', 'right']:
        # Add side-specific checks
        pass
    
    return {
        'is_valid': len(issues) == 0,
        'message': ", ".join(issues) if issues else "Image quality is good",
        'issues': issues
    }

async def process_analysis(client_id: str, captures: Dict[str, np.ndarray]):
    """Process captured images and perform analysis."""
    try:
        # In a real implementation, you would process the images here
        # For now, simulate processing
        await asyncio.sleep(2)  # Simulate processing time
        
        # Mock analysis results
        analysis_results = {
            'skin_type': 'Oily',
            'concerns': ['acne', 'blackheads'],
            'hydration_level': 75,
            'oiliness_level': 65,
            'pores_visibility': 'enlarged',
            'sensitivity_level': 'low',
            'capture_angles': list(captures.keys())
        }
        
        # Update the client session with analysis results
        if client_id in client_sessions:
            client_sessions[client_id]['analysis_results'] = analysis_results
            client_sessions[client_id]['status'] = 'completed'
            
    except Exception as e:
        logger.error(f"Error in analysis processing: {str(e)}")
        if client_id in client_sessions:
            client_sessions[client_id]['status'] = 'failed'
            client_sessions[client_id]['error'] = str(e)

@router.post("/analyze")
async def analyze_skin(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze skin from an uploaded image
    
    Args:
        file: Image file for skin analysis (JPEG, PNG)
        
    Returns:
        JSON with analysis results and recommendations
    """
    try:
        # Generate a unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Save the uploaded file temporarily
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_extension = os.path.splitext(file.filename)[1]
        file_path = upload_dir / f"{analysis_id}{file_extension}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # In a real implementation, you would process the image here
        # For now, return a mock response
        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "results": {
                "skin_type": "Oily",
                "concerns": ["acne", "blackheads"],
                "hydration_level": 75,
                "oiliness_level": 65,
                "pores_visibility": "enlarged",
                "sensitivity_level": "low"
            },
            "recommendations": {
                "cleanser": "Use a gentle foaming cleanser with salicylic acid",
                "moisturizer": "Oil-free, non-comedogenic moisturizer",
                "treatment": "Niacinamide serum for oil control",
                "sunscreen": "Oil-free SPF 50+"
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing skin analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/capture")
async def capture_image(
    request: Request,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Handle image capture for skin analysis.
    
    This endpoint accepts base64-encoded images and processes them for analysis.
    
    Expected JSON format:
    {
        "session_id": "unique_session_id",
        "client_id": "unique_client_id",
        "capture_id": "unique_capture_id",
        "angle": "front|left|right|chin_up|chin_down",
        "image_data": "base64_encoded_image"
    }
    """
    try:
        # Parse request data
        data = await request.json()
        
        # Extract required fields
        session_id = data.get('session_id')
        client_id = data.get('client_id')
        capture_id = data.get('capture_id')
        angle = data.get('angle')
        image_data = data.get('image_data')
        
        # Validate required fields
        if not all([session_id, client_id, capture_id, angle, image_data]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        # Validate angle
        valid_angles = ['front', 'left', 'right', 'chin_up', 'chin_down']
        if angle not in valid_angles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid angle. Must be one of: {', '.join(valid_angles)}"
            )
            
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_np = np.array(image)
            
            # Validate image quality
            validation = validate_image_quality(image_np, angle)
            if not validation['is_valid']:
                return {
                    'success': False,
                    'message': validation['message'],
                    'capture_id': capture_id,
                    'angle': angle,
                    'issues': validation['issues']
                }
            
            # Initialize or update client session
            if client_id not in client_sessions:
                client_sessions[client_id] = {
                    'captures': {},
                    'analysis_started': False,
                    'angles_remaining': ['front', 'left', 'right'],
                    'analysis_results': {},
                    'status': 'capturing'
                }
            
            session = client_sessions[client_id]
            session['captures'][angle] = image_np
            
            # Remove this angle from remaining angles
            if angle in session['angles_remaining']:
                session['angles_remaining'].remove(angle)
            
            # If all angles are captured, start analysis in background
            if not session['angles_remaining'] and not session['analysis_started']:
                session['analysis_started'] = True
                session['status'] = 'processing'
                
                # Start analysis in background
                background_tasks.add_task(
                    process_analysis,
                    client_id=client_id,
                    captures=session['captures'].copy()
                )
            
            return {
                'success': True,
                'message': 'Capture received successfully',
                'capture_id': capture_id,
                'angle': angle,
                'remaining_angles': session['angles_remaining'],
                'status': session['status']
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    except Exception as e:
        logger.error(f"Error in capture endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capture/status/{client_id}")
async def get_capture_status(client_id: str) -> Dict[str, Any]:
    """
    Get the status of captures and analysis for a client.
    
    Args:
        client_id: The client's unique identifier
        
    Returns:
        JSON with capture and analysis status
    """
    if client_id not in client_sessions:
        return {
            'success': False,
            'message': 'No session found for this client',
            'status': 'not_found'
        }
    
    session = client_sessions[client_id]
    return {
        'success': True,
        'status': session.get('status', 'unknown'),
        'captured_angles': list(session.get('captures', {}).keys()),
        'remaining_angles': session.get('angles_remaining', []),
        'analysis_completed': session.get('analysis_started', False) and 'analysis_results' in session,
        'analysis_results': session.get('analysis_results')
    }

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy", "service": "skin-analysis-api"}
