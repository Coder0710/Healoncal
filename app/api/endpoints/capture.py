"""
HTTP endpoints for image capture fallback.
"""
import base64
import json
import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
import numpy as np
from PIL import Image
import io
import cv2
import asyncio
import logging

from app.services.skin_analysis_service import SkinAnalysisService
from app.api.endpoints.websocket import manager, validate_image_quality
from app.services.gemini_service import gemini_service

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)
router = APIRouter()
service = SkinAnalysisService()

async def process_analysis(client_id: str, captures: Dict[str, np.ndarray]):
    """Process captured images and perform analysis."""
    try:
        service = SkinAnalysisService()
        
        # Save images to storage and get their paths
        image_paths = {}
        for angle, image_np in captures.items():
            # Convert numpy array to bytes
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            image_bytes = buffer.tobytes()
            
            # Save to storage
            file_path = await service.store_user_image(client_id, angle, image_bytes)
            image_paths[angle] = file_path
        
        # Perform analysis on all images in parallel
        analysis_results = await service.analyze_multiple_angles(captures)
        
        # Update the client session with analysis results
        if client_id in manager.client_sessions:
            manager.client_sessions[client_id]['analysis_results'] = analysis_results
            
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error in process_analysis: {str(e)}", exc_info=True)
        raise

@router.post("/capture")
async def capture_image(request: Request, background_tasks: BackgroundTasks):
    """
    Handle image capture via HTTP POST request.
    This is a fallback when WebSocket is not available.
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
                    'angle': angle
                }
            
            # Store the capture in the client's session
            if client_id not in manager.client_sessions:
                manager.client_sessions[client_id] = {
                    'captures': {},
                    'analysis_started': False,
                    'angles_remaining': ['front', 'left', 'right'],
                    'analysis_results': {}
                }
            
            session = manager.client_sessions[client_id]
            session['captures'][angle] = image_np
            
            # Remove this angle from remaining angles
            if angle in session['angles_remaining']:
                session['angles_remaining'].remove(angle)
            
            # If all angles are captured, start analysis in background
            if not session['angles_remaining'] and not session['analysis_started']:
                session['analysis_started'] = True
                
                # Start analysis in background
                background_tasks.add_task(
                    process_analysis,
                    client_id=client_id,
                    captures=session['captures'].copy()  # Pass a copy to avoid race conditions
                )
                
            return {
                'success': True,
                'message': 'Capture received successfully',
                'capture_id': capture_id,
                'angle': angle,
                'remaining_angles': session['angles_remaining']
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    except Exception as e:
        logger.error(f"Error in capture endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
