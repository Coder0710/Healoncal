from fastapi import APIRouter, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, Any, List, Optional, Generator
import cv2
import numpy as np
import logging
import asyncio
import base64
from enum import Enum, auto
from pathlib import Path
import uuid
import json
import time
from pydantic import BaseModel
from datetime import datetime

from app.models.skin_analyzer import SkinAnalyzer, SkinConcern, SkinType

class CaptureAngle(str, Enum):
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    CHIN_UP = "chin_up"
    CHIN_DOWN = "chin_down"

class ImageQualityError(Exception):
    def __init__(self, message: str, issues: List[str]):
        self.message = message
        self.issues = issues
        super().__init__(self.message)

class AnalysisSeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    CAPTURING = "capturing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class AnalysisRequest(BaseModel):
    user_id: str
    capture_angles: List[str] = ["front", "left", "right"]
    
class AnalysisResponse(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    progress: int
    message: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

router = APIRouter()
session_data = {}
analysis_tasks = {}  # Store ongoing analysis tasks

# Initialize the skin analyzer
skin_analyzer = SkinAnalyzer()

# Create scans directory if it doesn't exist
SCAN_DIR = Path("scans")
SCAN_DIR.mkdir(exist_ok=True)

def validate_image_quality(image: np.ndarray) -> None:
    """Validate image quality for skin analysis."""
    issues = []
    
    # Check image dimensions
    height, width = image.shape[:2]
    if width < 640 or height < 480:
        issues.append(f"Image resolution too low: {width}x{height}. Minimum required: 640x480")
    
    # Check brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < 50:
        issues.append(f"Image too dark (brightness: {brightness:.1f})")
    elif brightness > 200:
        issues.append(f"Image overexposed (brightness: {brightness:.1f})")
    
    # Check focus (using Laplacian variance)
    focus = cv2.Laplacian(gray, cv2.CV_64F).var()
    if focus < 100:
        issues.append(f"Image appears blurry (focus score: {focus:.1f})")
    
    if issues:
        raise ImageQualityError("Image quality issues detected", issues)

def get_severity_level(analysis: Dict[str, Any]) -> AnalysisSeverity:
    """Determine severity level based on analysis results."""
    # This is a simplified example - adjust thresholds based on your requirements
    if analysis.get('concerns', {}).get(SkinConcern.ACNE, 0) > 0.7:
        return AnalysisSeverity.SEVERE
    if analysis.get('concerns', {}).get(SkinConcern.REDNESS, 0) > 0.8:
        return AnalysisSeverity.SEVERE
    return AnalysisSeverity.MODERATE

@router.websocket("/ws/live-capture")
async def websocket_live_capture(websocket: WebSocket):
    """WebSocket endpoint for live skin analysis with multi-angle capture."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session_data[session_id] = {
        'captures': {},
        'analysis': {}
    }
    
    try:
        # Get capture configuration
        config = await websocket.receive_json()
        required_angles = [a.value for a in CaptureAngle]
        
        for angle in required_angles:
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                await websocket.send_json({
                    'type': 'instruction',
                    'message': f'Position your face for {angle.replace("_", " ")} view',
                    'next_step': 'capture',
                    'angle': angle,
                    'retry_count': retry_count
                })
            
                # Wait for image data
                data = await websocket.receive()
                if 'bytes' not in data:
                    continue
                    
                # Process image
                image_data = data['bytes']
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                try:
                    # Validate image quality
                    validate_image_quality(image)
                    
                    # Save the captured image temporarily
                    img_path = SCAN_DIR / f"{session_id}_{angle}.jpg"
                    cv2.imwrite(str(img_path), image)
                    
                    # Analyze the image
                    result = skin_analyzer.analyze(image)
                    severity = get_severity_level(result)
                    
                    # Store results
                    session_data[session_id]['captures'][angle] = {
                        'image_path': str(img_path),
                        'analysis': result,
                        'severity': severity.value
                    }
                    
                    await websocket.send_json({
                        'type': 'success',
                        'angle': angle,
                        'severity': severity.value,
                        'message': f'Successfully captured {angle} view',
                        'next_angle': angle if retry_count < max_retries - 1 else None
                    })
                    break  # Move to next angle after successful capture
                
                except ImageQualityError as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        await websocket.send_json({
                            'type': 'error',
                            'angle': angle,
                            'message': 'Max retries reached. Moving to next angle.',
                            'issues': e.issues,
                            'retry': False
                        })
                    else:
                        await websocket.send_json({
                            'type': 'retry',
                            'angle': angle,
                            'message': 'Image quality issues detected',
                            'issues': e.issues,
                            'retry_count': retry_count
                        })
                
        # Generate final analysis and recommendations
        analysis_results = list(session_data[session_id]['captures'].values())
        if analysis_results:
            final_analysis = await generate_final_analysis([r['analysis'] for r in analysis_results])
            recommendations = await generate_recommendations(final_analysis)
            
            # Save the final analysis
            session_data[session_id]['analysis'] = {
                'final_analysis': final_analysis,
                'recommendations': recommendations
            }
            
            await websocket.send_json({
                'type': 'analysis_complete',
                'analysis': final_analysis,
                'recommendations': recommendations,
                'session_id': session_id
            })
        
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed by client")
    except Exception as e:
        logger.error(f"Error in WebSocket: {str(e)}")
        await websocket.send_json({
            'type': 'error',
            'message': f'Analysis failed: {str(e)}'
        })

@router.post("/analyze-skin")
async def analyze_skin(
    file: UploadFile = File(...)
):
    """Analyze skin from an uploaded image
    
    Args:
        file: Image file for skin analysis (JPEG, PNG)
        
    Returns:
        JSON with analysis results and recommendations
    """
    try:
        # Read and validate image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        # Validate image quality
        try:
            validate_image_quality(image)
        except ImageQualityError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Image quality issues detected",
                    "issues": e.issues
                }
            )
        
        # Analyze the image
        result = skin_analyzer.analyze(image)
        
        # Generate recommendations
        recommendations = await generate_recommendations(result)
        
        # Save the analysis
        scan_id = str(uuid.uuid4())
        img_path = SCAN_DIR / f"{scan_id}.jpg"
        cv2.imwrite(str(img_path), image)
        
        return {
            "status": "success",
            "scan_id": scan_id,
            "analysis": result,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_skin: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only image files are allowed"
            )
            
        # Generate a unique filename
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in ['jpg', 'jpeg', 'png']:
            file_ext = 'jpg'
            
        filename = f"{uuid.uuid4()}.{file_ext}"
        file_path = UPLOAD_DIR / filename
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Analyze the skin
        analysis = skin_analyzer.analyze_skin(str(file_path.absolute()))
        
        if analysis['status'] != 'success':
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to analyze the image"
            )
        
        # Store the scan in history
        scan_id = str(uuid.uuid4())
        scan_data = {
            "scan_id": scan_id,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "image_path": str(file_path.absolute()),
            "analysis": analysis,
            "gender": gender,
            "mood": mood
        }
        scan_history[scan_id] = scan_data
        
        # Prepare response
        response = {
            "status": "success",
            "scan_id": scan_id,
            "analysis": analysis
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )
    finally:
        await file.close()

@router.get("/scan-history", response_model=Dict[str, Any])
@router.get("/scan-history/{user_id}", response_model=Dict[str, Any])
async def get_scan_history(user_id: str = None):
    """
    Get scan history
    
    Args:
        user_id: Optional user ID to filter scans. If not provided, returns all scans.
        
    Returns:
        List of previous scans with analysis results
    """
    try:
        # Always return all scans, but filter by user_id if provided
        all_scans = []
        for scan_id, scan_data in scan_history.items():
            # Skip if user_id is provided and doesn't match
            if user_id is not None and scan_data.get("user_id") != user_id:
                continue
                
            # Get the condition and confidence from the analysis
            analysis = scan_data.get("analysis", {})
            condition = "unknown"
            confidence = 0.0
            
            # Handle different analysis result formats
            if isinstance(analysis, dict):
                if "analysis" in analysis and isinstance(analysis["analysis"], dict):
                    condition = analysis["analysis"].get("condition", "unknown")
                    confidence = analysis["analysis"].get("confidence", 0.0)
                else:
                    # Handle direct condition in analysis
                    condition = analysis.get("condition", "unknown")
                    confidence = analysis.get("confidence", 0.0)
            
            # Get image URL if available
            image_path = scan_data.get("image_path")
            image_url = f"http://127.0.0.1:8000/uploads/{os.path.basename(image_path)}" if image_path and os.path.exists(image_path) else None
            
            all_scans.append({
                "scan_id": scan_id,
                "user_id": scan_data.get("user_id"),
                "timestamp": scan_data.get("timestamp"),
                "condition": condition,
                "confidence": round(float(confidence), 4),
                "image_url": image_url,
                "full_analysis": analysis  # Include full analysis in the response
            })
        
        # Sort scans by timestamp (newest first)
        all_scans.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return {
            "status": "success",
            "message": f"Showing {len(all_scans)} scans" + (f" for user {user_id}" if user_id else ""),
            "count": len(all_scans),
            "scans": all_scans
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching scan history: {str(e)}"
        )

@router.get("/analysis/{session_id}")
async def get_analysis(session_id: str):
    """Get analysis results for a session"""
    if session_id not in session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = session_data[session_id]
    if not session.get('analysis'):
        raise HTTPException(status_code=404, detail="Analysis not completed")
        
    return {
        "status": "success",
        "session_id": session_id,
        **session['analysis']
    }

@router.get("/recommendations/{scan_id}", response_model=Dict[str, Any])
async def get_recommendations(scan_id: str):
    """
    Get personalized recommendations based on a previous scan
    
    Args:
        scan_id: The scan ID to get recommendations for
        
    Returns:
        Personalized product and treatment recommendations
    """
    try:
        if scan_id not in scan_history:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Scan not found"
            )
        
        scan_data = scan_history[scan_id]
        
        # Extract relevant data for recommendations
        analysis = scan_data.get("analysis", {})
        conditions = analysis.get("conditions", [])
        tone = analysis.get("tone", {})
        
        # Generate recommendations
        recommendations = {
            "products": [],
            "treatments": []
        }
        
        # Add basic recommendations based on conditions
        condition_names = {cond["condition"] for cond in conditions}
        
        if any(c in condition_names for c in ["acne", "oiliness"]):
            recommendations["products"].extend([
                {"name": "Salicylic Acid Cleanser", "brand": "CeraVe", "type": "cleanser"},
                {"name": "Niacinamide Serum", "brand": "The Ordinary", "type": "serum"}
            ])
            recommendations["treatments"].append("Salicylic Acid Peel")
            
        if "dryness" in condition_names:
            recommendations["products"].extend([
                {"name": "Hyaluronic Acid Serum", "brand": "The Ordinary", "type": "serum"},
                {"name": "Ceramide Moisturizer", "brand": "CeraVe", "type": "moisturizer"}
            ])
            recommendations["treatments"].append("HydraFacial")
        
        # Add tone-specific recommendations
        if tone and 'tone_level' in tone and 'undertone' in tone:
            foundation_shade = f"{tone['tone_level']} {tone['undertone']}"
            recommendations["products"].append({
                'name': f'Foundation - Shade: {foundation_shade}',
                'type': 'foundation',
                'brand': 'Maybelline Fit Me'
            })
        
        # Add default recommendations if none were added
        if not recommendations["products"]:
            recommendations["products"].extend([
                {"name": "Gentle Cleanser", "brand": "CeraVe", "type": "cleanser"},
                {"name": "Daily Moisturizing Lotion", "brand": "CeraVe", "type": "moisturizer"},
                {"name": "Sunscreen SPF 30+", "brand": "Neutrogena", "type": "sunscreen"}
            ])
            recommendations["treatments"].append("Basic Facial")
        
        return {
            "status": "success",
            "recommendations": recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {str(e)}"
        )
