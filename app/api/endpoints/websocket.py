"""
WebSocket endpoints for real-time communication with Supabase integration.
"""
import asyncio
import base64
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import WebSocket, WebSocketDisconnect, APIRouter, HTTPException
import cv2
import numpy as np
from PIL import Image
import io
import random

from app.services.skin_analysis_service import SkinAnalysisService
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()
service = SkinAnalysisService()

class ConnectionManager:
    """Manages active WebSocket connections."""
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.analysis_tasks: Dict[str, asyncio.Task] = {}
        self.client_sessions: Dict[str, Dict] = {}  # Store client session data
        self.instructions = {
            'front': 'Face the camera directly with your head straight. Ensure your entire face is visible.',
            'left': 'Slowly turn your head 45 degrees to the left, keeping your face in the frame.',
            'right': 'Now turn your head 45 degrees to the right, keeping your face in the frame.'
        }

    async def connect(self, websocket: WebSocket, client_id: str):
        """Register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_sessions[client_id] = {
            'captures': {},
            'analysis_started': False,
            'angles_remaining': ['front', 'left', 'right'],
            'analysis_results': {}
        }
        logger.info(f"🔗 [CONNECTION] Client {client_id} connected and session initialized")
        logger.info(f"📊 [CONNECTION] Active connections: {len(self.active_connections)}, Sessions: {len(self.client_sessions)}")

    def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.client_sessions:
            del self.client_sessions[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def send_message(self, client_id: str, message: dict):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            try:
                msg_type = message.get('type', 'unknown_type')
                logger.debug(f"📤 [SEND] Sending {msg_type} message to {client_id}")
                await self.active_connections[client_id].send_json(message)
                logger.debug(f"✅ [SEND] Successfully sent {msg_type} message to {client_id}")
            except Exception as e:
                logger.error(f"💥 [SEND] Error sending message to {client_id}: {str(e)}", exc_info=True)
                # Remove broken connection
                if client_id in self.active_connections:
                    del self.active_connections[client_id]
                    logger.warning(f"🗑️ [SEND] Removed broken connection for {client_id}")
        else:
            logger.warning(f"❌ [SEND] Client {client_id} not found in active connections when trying to send {message.get('type', 'unknown_type')} message")
            logger.info(f"📊 [SEND] Current active connections: {list(self.active_connections.keys())}")

async def send_next_instruction(client_id: str):
    """Send instructions for the next angle to capture."""
    if client_id not in manager.client_sessions:
        return
        
    session = manager.client_sessions[client_id]
    if not session['angles_remaining']:
        return
        
    next_angle = session['angles_remaining'][0]
    await manager.send_message(client_id, {
        "type": "capture_instruction",
        "angle": next_angle,
        "instruction": manager.instructions.get(next_angle, "Please position your face as shown."),
        "progress": {
            "current": 3 - len(session['angles_remaining']),
            "total": 3
        }
    })

def validate_image_quality_fast(image: np.ndarray, angle: str) -> Dict:
    """Fast image validation with minimal processing."""
    issues = []
    
    # Check image size (quick)
    height, width = image.shape[:2]
    if height < 240 or width < 240:  # Even more lenient for speed
        issues.append("Image resolution is too low.")
    
    # Quick brightness check (sample-based for speed)
    # Sample pixels instead of full image for faster processing
    sample_size = min(50, height//4, width//4)
    center_y, center_x = height//2, width//2
    sample_region = image[center_y-sample_size:center_y+sample_size, 
                         center_x-sample_size:center_x+sample_size]
    
    if sample_region.size > 0:
        brightness = np.mean(cv2.cvtColor(sample_region, cv2.COLOR_BGR2GRAY))
        if brightness < 30:  # Very permissive
            issues.append("Image is too dark.")
        elif brightness > 250:
            issues.append("Image is overexposed.")
    
    # Skip focus check for speed - analysis will handle blur detection
    # Skip face detection for speed - analysis will handle face detection
    
    # Always return valid for maximum speed - let analysis handle quality issues
    is_valid = len(issues) == 0
    
    logger.debug(f"Fast validation for {angle}: {'PASSED' if is_valid else 'FAILED'}")
    
    return {
        "valid": is_valid,
        "message": ", ".join(issues) if issues else "Image accepted for analysis.",
        "issues": issues
    }

def validate_image_quality(image: np.ndarray, angle: str) -> Dict:
    """Validate if the captured image is suitable for analysis."""
    issues = []
    
    # Check image size
    height, width = image.shape[:2]
    logger.debug(f"Image dimensions for {angle}: {width}x{height}")
    
    if height < 320 or width < 320:  # More lenient resolution requirement
        issues.append("Image resolution is too low. Please move closer to the camera.")
    
    # Check brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    logger.debug(f"Image brightness for {angle}: {brightness:.2f}")
    
    if brightness < 50:  # More lenient brightness range
        issues.append("Image is too dark. Please improve lighting.")
    elif brightness > 240:
        issues.append("Image is overexposed. Please reduce lighting.")
    
    # Check focus (using Laplacian variance) - Very lenient threshold
    focus = cv2.Laplacian(gray, cv2.CV_64F).var()
    logger.debug(f"Image focus score for {angle}: {focus:.2f}")
    
    if focus < 10:  # Very lenient focus requirement (was 30, originally 100)
        issues.append("Image is extremely blurry. Please hold the camera steady and try again.")
    
    # Check face position/angle - More lenient face detection
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)  # More lenient face detection
        
        logger.debug(f"Faces detected for {angle}: {len(faces)}")
        
        if len(faces) == 0:
            # Don't fail for no face - just warn
            logger.warning(f"No face detected for {angle}, but allowing image")
            # issues.append("No face detected. Please ensure your face is clearly visible in the frame.")
        else:
            # Check if face is centered (more lenient)
            x, y, w, h = faces[0]
            img_center_x = width // 2
            img_center_y = height // 2
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            logger.debug(f"Face position for {angle}: center=({face_center_x}, {face_center_y}), image_center=({img_center_x}, {img_center_y})")
            
            if abs(face_center_x - img_center_x) > width * 0.3:  # More lenient centering
                logger.warning(f"Face not centered for {angle}, but allowing image")
                # issues.append("Please center your face in the frame.")
            
            # Check face size (more lenient range)
            face_ratio = h / height
            logger.debug(f"Face ratio for {angle}: {face_ratio:.2f}")
            
            if face_ratio < 0.15:  # More lenient face size requirements
                logger.warning(f"Face too small for {angle}, but allowing image")
                # issues.append("Please move closer to the camera.")
            elif face_ratio > 0.85:
                logger.warning(f"Face too large for {angle}, but allowing image")
                # issues.append("Please move slightly further from the camera.")
                
    except Exception as e:
        logger.warning(f"Face detection failed for {angle}: {str(e)}, but allowing image")
    
    # Log final validation result
    is_valid = len(issues) == 0
    logger.info(f"Image validation for {angle}: {'PASSED' if is_valid else 'FAILED'} - Issues: {issues}")
    
    return {
        "valid": is_valid,
        "message": ", ".join(issues) if issues else "Image quality is good.",
        "issues": issues
    }

manager = ConnectionManager()

@router.websocket("/ws/live-capture")
@router.websocket("/ws/live-capture/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str = None):
    """
    Handle WebSocket connections for live image capture and analysis.
    
    Args:
        websocket: The WebSocket connection
        client_id: Unique client identifier
    """
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get('type', 'unknown')
            logger.info(f"📨 [WEBSOCKET MESSAGE] Received from {client_id}: {msg_type}")
            if msg_type != 'capture':  # Don't log full capture data (too large)
                logger.debug(f"📨 [WEBSOCKET MESSAGE] Full data: {data}")
            
            if data['type'] == 'capture':
                try:
                    logger.info(f"Processing capture for angle: {data.get('angle')}")
                    
                    # Store the image data for user-specific folder
                    angle = data.get('angle', 'front')
                    image_data = data['image_data']
                    user_id = client_id  # Use client_id as user_id
                    
                    logger.debug(f"Received image data for angle {angle}, size: {len(image_data) if image_data else 0}")
                    
                    # Decode and validate the image first
                    if isinstance(image_data, str):
                        if ',' in image_data:
                            image_data = image_data.split(',')[1]
                        image_bytes = base64.b64decode(image_data)
                    else:
                        image_bytes = image_data
                    
                    logger.debug(f"Decoded image bytes for angle {angle}, size: {len(image_bytes)}")
                    
                    # Convert to PIL Image for validation
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    logger.debug(f"Converted image for validation, shape: {frame.shape}")
                    
                    # Use fast validation for better performance
                    validation_result = validate_image_quality_fast(frame, angle)
                    logger.debug(f"Validation result for angle {angle}: {validation_result}")
                    
                    if not validation_result["valid"]:
                        logger.warning(f"Image validation failed for angle {angle}: {validation_result['message']}")
                        await manager.send_message(client_id, {
                            "type": "validation_error",
                            "message": validation_result["message"],
                            "angle": angle,
                            "issues": validation_result.get("issues", []),
                            "suggestion": "Please adjust your position and lighting, then try again."
                        })
                        continue
                    
                    # Store image in user-specific folder and database
                    logger.info(f"Storing {angle} image for user {user_id}")
                    storage_result = await service.capture_and_store_image(
                        user_id=user_id,
                        angle=angle,
                        image_data=image_bytes
                    )
                    
                    if not storage_result.get('success', False):
                        logger.error(f"Failed to store {angle} image: {storage_result.get('error')}")
                        await manager.send_message(client_id, {
                            'type': 'capture_error',
                            'angle': angle,
                            'error': f"Failed to store image: {storage_result.get('error')}"
                        })
                        continue
                    
                    # Update session with storage information
                    session = manager.client_sessions[client_id]
                    logger.debug(f"Current session state before storing: angles_remaining={session['angles_remaining']}")
                    
                    session['captures'][angle] = {
                        "record_id": storage_result['record_id'],
                        "file_path": storage_result['file_path'],
                        "capture_time": datetime.now().isoformat(),
                        "angle": angle,
                        "status": "stored"
                    }
                    
                    # Remove captured angle from remaining
                    if angle in session['angles_remaining']:
                        session['angles_remaining'].remove(angle)
                        logger.info(f"Removed {angle} from remaining angles. Remaining: {session['angles_remaining']}")
                    else:
                        logger.warning(f"Angle {angle} was not in remaining angles list: {session['angles_remaining']}")
                    
                    # Send success response
                    success_message = f"✅ Successfully stored {angle} view. " + \
                                    (f"Now capture {session['angles_remaining'][0]} view." if session['angles_remaining'] else "All angles captured! Ready for analysis.")
                    
                    logger.info(f"Sending capture success for {angle}. Message: {success_message}")
                    
                    await manager.send_message(client_id, {
                        'type': 'capture_success',
                        'angle': angle,
                        'message': success_message,
                        'angles_remaining': session['angles_remaining'],
                        'all_captures_complete': not bool(session['angles_remaining'])
                    })
                    
                    # Don't automatically start analysis - wait for explicit request
                    if not session['angles_remaining']:
                        # All captures complete, but don't start analysis yet
                        logger.info(f"🎉 [CAPTURE COMPLETE] All captures complete for client {client_id}")
                        logger.info(f"📤 [CAPTURE COMPLETE] Sending all_captures_complete message to client {client_id}")
                        
                        try:
                            await manager.send_message(client_id, {
                                "type": "all_captures_complete",
                                "message": "🎉 All images captured and stored! Click 'Start Analysis' to begin processing.",
                                "angles_captured": list(session['captures'].keys()),
                                "total_captures": len(session['captures']),
                                "ready_for_analysis": True
                            })
                            logger.info(f"✅ [CAPTURE COMPLETE] Successfully sent all_captures_complete message to client {client_id}")
                            logger.info(f"🔍 [CAPTURE COMPLETE] Frontend should now show 'Start Analysis' button")
                        except Exception as e:
                            logger.error(f"💥 [CAPTURE COMPLETE] Failed to send all_captures_complete message to client {client_id}: {str(e)}")
                    else:
                        # Send instructions for next angle
                        logger.info(f"Sending instructions for next angle: {session['angles_remaining'][0]}")
                        await send_next_instruction(client_id)
                    
                except Exception as e:
                    error_msg = f"Error processing capture: {str(e)}"
                    logger.error(error_msg)
                    await manager.send_message(client_id, {
                        'type': 'capture_error',
                        'angle': data.get('angle', 'unknown'),
                        'error': error_msg
                    })
                    
            elif data.get("type") == "register":
                # Client is registering/identifying itself
                logger.info(f"📝 [REGISTER] Client {client_id} registering")
                await manager.send_message(client_id, {
                    "type": "status",
                    "status": "connected",
                    "client_id": client_id,
                    "message": "Successfully connected to analysis server"
                })
                
                # Send initial instructions for first angle
                await send_next_instruction(client_id)
                logger.info(f"✅ [REGISTER] Client {client_id} registration complete")
                
            elif data.get("type") == "ping":
                # Connectivity test
                logger.info(f"🏓 [PING] Received ping from {client_id}")
                await manager.send_message(client_id, {
                    "type": "pong", 
                    "message": "Server is alive",
                    "timestamp": datetime.now().isoformat()
                })
                
            elif data.get("type") == "image":
                # Process the image
                try:
                    angle = data.get("angle", "front").lower()
                    if angle not in ['front', 'left', 'right']:
                        raise ValueError("Invalid angle. Must be 'front', 'left', or 'right'")
                    
                    session = manager.client_sessions[client_id]
                    if angle not in session['angles_remaining']:
                        await manager.send_message(client_id, {
                            "type": "error",
                            "message": f"Angle {angle} already captured. Please capture remaining angles.",
                            "angles_remaining": session['angles_remaining']
                        })
                        continue
                    
                    # Extract base64 image data
                    if "," in data.get("data", ""):
                        image_data = base64.b64decode(data["data"].split(",")[1])
                    else:
                        image_data = base64.b64decode(data["data"])
                        
                    # Process the image
                    image = Image.open(io.BytesIO(image_data))
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Validate image quality
                    validation_result = validate_image_quality(frame, angle)
                    if not validation_result["valid"]:
                        await manager.send_message(client_id, {
                            "type": "validation_error",
                            "message": validation_result["message"],
                            "angle": angle,
                            "issues": validation_result.get("issues", []),
                            "suggestion": "Please adjust your position and lighting, then try again."
                        })
                        continue
                    
                    # Store the capture
                    capture_id = f"{client_id}_{angle}_{int(time.time())}"
                    session['captures'][angle] = {
                        "image_data": image_data,
                        "capture_time": datetime.now().isoformat(),
                        "angle": angle,
                        "status": "captured"
                    }
                    
                    # Remove captured angle from remaining
                    if angle in session['angles_remaining']:
                        session['angles_remaining'].remove(angle)
                    
                    # Notify client of successful capture
                    await manager.send_message(client_id, {
                        "type": "capture_success",
                        "message": f"✅ Successfully captured {angle} view. " + 
                                  (f"Now capture {session['angles_remaining'][0]} view." if session['angles_remaining'] else "All angles captured!"),
                        "angle": angle,
                        "capture_id": capture_id,
                        "angles_remaining": session['angles_remaining'],
                        "all_captures_complete": not bool(session['angles_remaining'])
                    })
                    
                    # If all angles captured, notify completion but don't auto-start analysis
                    if not session['angles_remaining']:
                        logger.info(f"🎉 [IMAGE COMPLETE] All angles captured via image path for client {client_id}")
                        await manager.send_message(client_id, {
                            "type": "all_captures_complete",
                            "message": "🎉 All images captured! Click 'Start Analysis' to begin processing.",
                            "angles_captured": list(session['captures'].keys()),
                            "total_captures": len(session['captures']),
                            "ready_for_analysis": True
                        })
                        
                        # Don't auto-start analysis - wait for explicit start_analysis message
                    else:
                        # Send instructions for next angle
                        await send_next_instruction(client_id)
                
                except Exception as e:
                    logger.error(f"Error processing image: {str(e)}")
                    await manager.send_message(client_id, {
                        "type": "error",
                        "message": f"Error processing image: {str(e)}"
                    })
            
            elif data.get("type") == "retry_capture":
                # Handle retry request
                angle = data.get("angle", "front").lower()
                session = manager.client_sessions[client_id]
                if angle in session['captures']:
                    del session['captures'][angle]
                if angle not in session['angles_remaining']:
                    session['angles_remaining'].append(angle)
                
                await send_next_instruction(client_id)
            
            elif data.get("type") == "start_analysis":
                # Handle explicit analysis start request
                try:
                    import time
                    start_time = time.time()
                    logger.info(f"🚀 [ANALYSIS START] ✅ SUCCESS! Received start_analysis request from client {client_id} at {start_time}")
                    logger.info(f"🎯 [ANALYSIS START] User clicked 'Start Analysis' button - this means the fix worked!")
                    session = manager.client_sessions[client_id]
                    
                    if session.get('analysis_started', False):
                        logger.warning(f"❌ [ANALYSIS ERROR] Analysis already in progress for client {client_id}")
                        await manager.send_message(client_id, {
                            "type": "error",
                            "message": "Analysis already in progress"
                        })
                        continue
                    
                    if session['angles_remaining']:
                        logger.warning(f"❌ [ANALYSIS ERROR] Missing angles for client {client_id}: {session['angles_remaining']}")
                        await manager.send_message(client_id, {
                            "type": "error", 
                            "message": f"Please capture all angles first. Missing: {', '.join(session['angles_remaining'])}"
                        })
                        continue
                    
                    logger.info(f"✅ [ANALYSIS VALIDATION] Starting analysis for client {client_id} with captures: {list(session['captures'].keys())}")
                    logger.info(f"📊 [ANALYSIS DATA] Session captures: {session['captures']}")
                    
                    session['analysis_started'] = True
                    session['analysis_start_time'] = start_time
                    
                    await manager.send_message(client_id, {
                        "type": "analysis_started", 
                        "message": "🔍 Starting analysis..."
                    })
                    
                    # Use optimized parallel processing for maximum speed
                    logger.info(f"🎯 [ANALYSIS TASK] Using FAST PARALLEL method for maximum speed")
                    try:
                        analysis_task = asyncio.create_task(analyze_images_with_debug(client_id, session))
                        logger.info(f"✅ [ANALYSIS TASK] Fast parallel analysis task created and started for client {client_id}")
                    except Exception as task_error:
                        logger.error(f"💥 [ANALYSIS TASK] Fast method failed, using fallback: {str(task_error)}")
                        analysis_task = asyncio.create_task(analyze_images_simple_fallback(client_id, session))
                    
                except Exception as e:
                    logger.error(f"💥 [ANALYSIS ERROR] Error starting analysis for client {client_id}: {str(e)}", exc_info=True)
                    await manager.send_message(client_id, {
                        "type": "error",
                        "message": f"Error starting analysis: {str(e)}"
                    })
            
            else:
                # Unknown message type
                await manager.send_message(client_id, {
                    "type": "error",
                    "message": f"Unknown message type: {data.get('type')}"
                })
                
    except WebSocketDisconnect:
        logger.info(f"🔌 [WEBSOCKET] Client {client_id} disconnected")
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"💥 [WEBSOCKET] WebSocket error for {client_id}: {str(e)}", exc_info=True)
        try:
            await manager.send_message(client_id, {
                "type": "error", 
                "message": f"Server error: {str(e)}"
            })
        except:
            pass  # Client may already be disconnected
        manager.disconnect(client_id)

async def analyze_images_simple_fallback(client_id: str, session: dict):
    """Simple fallback analysis function with basic functionality."""
    import time
    start_time = time.time()
    
    try:
        logger.info(f"🔄 [FALLBACK] Starting simple fallback analysis for client {client_id}")
        
        # Get the WebSocket connection
        if client_id not in manager.active_connections:
            logger.error(f"❌ [FALLBACK] Client {client_id} not found for analysis")
            return
        
        ws = manager.active_connections[client_id]
        user_id = client_id
        
        # Send progress update
        await ws.send_json({
            "type": "analysis_progress",
            "message": "🚀 Starting simple analysis (fallback mode)...",
            "progress": 0.1
        })
        
        # Prepare image records
        image_records = []
        for angle, capture_info in session['captures'].items():
            image_records.append({
                'angle': angle,
                'file_path': capture_info['file_path'],
                'record_id': capture_info['record_id']
            })
        
        logger.info(f"📋 [FALLBACK] Prepared {len(image_records)} image records")
        
        # Call the original analysis service (not the fast one)
        await ws.send_json({
            "type": "analysis_progress",
            "message": "🔍 Processing images with fallback method...",
            "progress": 0.5
        })
        
        # Use the original method instead of the optimized one
        analysis_result = await service.analyze_stored_images(user_id, image_records)
        
        if analysis_result.get('success', False):
            # Store combined results in session
            session['analysis_results'] = analysis_result['individual_results']
            combined_results = analysis_result['combined_results']
            
            duration = time.time() - start_time
            await ws.send_json({
                "type": "analysis_complete",
                "message": f"✅ Fallback analysis complete in {duration:.1f}s!",
                "results": combined_results,
                "angle_results": analysis_result['individual_results']
            })
            
            logger.info(f"✅ [FALLBACK] Analysis complete for user {user_id} in {duration:.2f}s")
        else:
            error_msg = analysis_result.get('error', 'Analysis failed')
            logger.error(f"❌ [FALLBACK] Analysis failed for user {user_id}: {error_msg}")
            await ws.send_json({
                "type": "error",
                "message": f"Fallback analysis failed: {error_msg}"
            })
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"💥 [FALLBACK] Error in fallback analysis after {duration:.2f}s: {str(e)}", exc_info=True)
        if client_id in manager.active_connections:
            await manager.send_message(client_id, {
                "type": "error",
                "message": f"Fallback analysis error: {str(e)}"
            })

async def analyze_images_with_debug(client_id: str, session: dict):
    """Analyze images with comprehensive debug logging and timing."""
    import time
    overall_start = time.time()
    
    try:
        logger.info(f"🚀 [DEBUG] Starting analyze_images_with_debug for client {client_id}")
        logger.info(f"📊 [DEBUG] Overall start time: {overall_start}")
        
        # Get the WebSocket connection
        if client_id not in manager.active_connections:
            logger.error(f"❌ [DEBUG] Client {client_id} not found for analysis")
            return
        
        ws = manager.active_connections[client_id]
        user_id = client_id
        
        logger.info(f"✅ [DEBUG] WebSocket connection found for {client_id}")
        logger.info(f"👤 [DEBUG] User ID: {user_id}")
        
        # Send initial progress
        step_start = time.time()
        await ws.send_json({
            "type": "analysis_progress",
            "message": "🚀 Initializing fast parallel analysis...",
            "progress": 0.1,
            "debug_info": f"Step: Initialize (t={step_start - overall_start:.2f}s)"
        })
        logger.info(f"📤 [DEBUG] Sent initialization message ({time.time() - step_start:.3f}s)")
        
        # Prepare image records
        step_start = time.time()
        image_records = []
        for angle, capture_info in session['captures'].items():
            record = {
                'angle': angle,
                'file_path': capture_info['file_path'],
                'record_id': capture_info['record_id']
            }
            image_records.append(record)
            logger.info(f"📋 [DEBUG] Prepared record for {angle}: {record}")
        
        logger.info(f"📊 [DEBUG] Prepared {len(image_records)} image records ({time.time() - step_start:.3f}s)")
        
        await ws.send_json({
            "type": "analysis_progress",
            "message": f"📋 Prepared {len(image_records)} images for analysis...",
            "progress": 0.2,
            "debug_info": f"Records prepared (t={time.time() - overall_start:.2f}s)"
        })
        
        # Progress callback with debug info
        async def debug_progress_callback(angle: str, current: int, total: int):
            progress_time = time.time()
            try:
                progress_pct = 0.2 + (0.7 * current / total)  # 20% to 90%
                await ws.send_json({
                    "type": "analysis_progress",
                    "message": f"🔍 Analyzing {angle} view... ({current}/{total})",
                    "angle": angle,
                    "progress": progress_pct,
                    "debug_info": f"Processing {angle} #{current} (t={progress_time - overall_start:.2f}s)"
                })
                logger.info(f"🔄 [DEBUG] Progress update: {angle} {current}/{total} at {progress_time - overall_start:.2f}s")
            except Exception as e:
                logger.error(f"💥 [DEBUG] Error sending progress update: {str(e)}")
        
        # Call the fast analysis service
        logger.info(f"⚡ [DEBUG] Calling analyze_stored_images_fast at {time.time() - overall_start:.2f}s")
        analysis_start = time.time()
        
        try:
            analysis_result = await service.analyze_stored_images_fast(user_id, image_records, debug_progress_callback)
            logger.info(f"⚡ [DEBUG] analyze_stored_images_fast returned: success={analysis_result.get('success', False)}")
        except Exception as service_error:
            logger.error(f"💥 [DEBUG] Error calling analyze_stored_images_fast: {str(service_error)}", exc_info=True)
            raise service_error
        
        analysis_end = time.time()
        analysis_duration = analysis_end - analysis_start
        logger.info(f"⚡ [DEBUG] analyze_stored_images_fast completed in {analysis_duration:.2f}s")
        
        # Send completion update
        await ws.send_json({
            "type": "analysis_progress",
            "message": "✨ Finalizing results...",
            "progress": 0.95,
            "debug_info": f"Analysis done, finalizing (t={time.time() - overall_start:.2f}s)"
        })
        
        if analysis_result.get('success', False):
            # Store combined results in session
            session['analysis_results'] = analysis_result['individual_results']
            combined_results = analysis_result['combined_results']
            
            logger.info(f"✅ [DEBUG] Analysis successful, sending final results")
            logger.info(f"📊 [DEBUG] Individual results keys: {list(analysis_result['individual_results'].keys())}")
            logger.info(f"🎯 [DEBUG] Combined results keys: {list(combined_results.keys())}")
            
            # Send final results
            final_time = time.time()
            total_duration = final_time - overall_start
            
            await ws.send_json({
                "type": "analysis_complete",
                "message": f"🎉 Analysis complete in {total_duration:.1f}s!",
                "results": combined_results,
                "angle_results": analysis_result['individual_results'],
                "debug_info": {
                    "total_time": f"{total_duration:.2f}s",
                    "analysis_time": f"{analysis_duration:.2f}s",
                    "overhead_time": f"{total_duration - analysis_duration:.2f}s"
                }
            })
            
            logger.info(f"🎉 [DEBUG] Analysis complete for user {user_id} in {total_duration:.2f}s")
            logger.info(f"⚡ [DEBUG] Core analysis: {analysis_duration:.2f}s, Total overhead: {total_duration - analysis_duration:.2f}s")
        else:
            # Analysis failed
            error_msg = analysis_result.get('error', 'Analysis failed')
            logger.error(f"❌ [DEBUG] Analysis failed for user {user_id}: {error_msg}")
            await ws.send_json({
                "type": "error",
                "message": f"Analysis failed: {error_msg}",
                "debug_info": f"Failed at {time.time() - overall_start:.2f}s"
            })
        
    except Exception as e:
        error_time = time.time()
        duration = error_time - overall_start
        logger.error(f"💥 [DEBUG] Error in analyze_images_with_debug after {duration:.2f}s: {str(e)}", exc_info=True)
        if client_id in manager.active_connections:
            await manager.send_message(client_id, {
                "type": "error",
                "message": f"Error during analysis: {str(e)}",
                "debug_info": f"Failed at {duration:.2f}s"
            })

async def analyze_images(client_id: str, session: dict):
    """Analyze images stored in user-specific folder."""
    try:
        # Get the WebSocket connection
        if client_id not in manager.active_connections:
            logger.error(f"Client {client_id} not found for analysis")
            return
        
        ws = manager.active_connections[client_id]
        user_id = client_id  # Use client_id as user_id
        
        # First, notify that we're starting the analysis process
        await ws.send_json({
            "type": "analysis_started",
            "message": f"Starting analysis of {len(session['captures'])} stored images..."
        })
        
        # Prepare image records for analysis
        image_records = []
        for angle, capture_info in session['captures'].items():
            image_records.append({
                'angle': angle,
                'file_path': capture_info['file_path'],
                'record_id': capture_info['record_id']
            })
        
        logger.info(f"Starting analysis of stored images for user {user_id}: {[r['angle'] for r in image_records]}")
        
        # Call the service to analyze all stored images with progress callbacks
        async def progress_callback(angle: str, current: int, total: int):
            """Send progress updates to the client"""
            try:
                await ws.send_json({
                    "type": "analysis_progress",
                    "message": f"🔍 Analyzing {angle} view... ({current}/{total})",
                    "angle": angle,
                    "progress": current / total
                })
            except Exception as e:
                logger.error(f"Error sending progress update: {str(e)}")
        
        analysis_result = await service.analyze_stored_images_fast(user_id, image_records, progress_callback)
        
        if analysis_result.get('success', False):
            # Store combined results in session
            session['analysis_results'] = analysis_result['individual_results']
            combined_results = analysis_result['combined_results']
            
            # Send final results
            await ws.send_json({
                "type": "analysis_complete",
                "message": "🎉 Analysis complete!",
                "results": combined_results,
                "angle_results": analysis_result['individual_results']
            })
            
            logger.info(f"Analysis complete for user {user_id}")
        else:
            # Analysis failed
            error_msg = analysis_result.get('error', 'Analysis failed')
            logger.error(f"Analysis failed for user {user_id}: {error_msg}")
            await ws.send_json({
                "type": "error",
                "message": f"Analysis failed: {error_msg}"
            })
        
    except Exception as e:
        logger.error(f"Error in analyze_images: {str(e)}")
        if client_id in manager.active_connections:
            await manager.send_message(client_id, {
                "type": "error",
                "message": f"Error during analysis: {str(e)}"
            })

def combine_analysis_results(angle_results: Dict) -> Dict:
    """Combine analysis results from multiple angles."""
    combined = {
        "skin_type": max(
            [(angle, res["skin_type"]) for angle, res in angle_results.items() if "skin_type" in res],
            key=lambda x: len([k for k in angle_results.keys() if angle_results[k].get("skin_type") == x[1]]),
            default=(None, "Normal")
        )[1],
        "conditions": {},
        "average_moisture": 0.0,
        "average_oiliness": 0.0,
        "average_pores": 0.0,
        "average_texture": 0.0,
        "analyzed_angles": list(angle_results.keys()),
        "analysis_time": datetime.now().isoformat(),
        "recommendations": []
    }
    
    # Calculate averages
    valid_results = [r for r in angle_results.values() if not r.get("error")]
    if valid_results:
        combined["average_moisture"] = round(sum(r.get("moisture_level", 0) for r in valid_results) / len(valid_results), 2)
        combined["average_oiliness"] = round(sum(r.get("oiliness", 0) for r in valid_results) / len(valid_results), 2)
        combined["average_pores"] = round(sum(r.get("pores_visibility", 0) for r in valid_results) / len(valid_results), 2)
        combined["average_texture"] = round(sum(r.get("texture_score", 0) for r in valid_results) / len(valid_results), 2)
    
    # Combine conditions
    for angle, result in angle_results.items():
        if "conditions" in result:
            for condition in result["conditions"]:
                name = condition["name"]
                if name not in combined["conditions"]:
                    combined["conditions"][name] = {
                        "count": 0,
                        "total_confidence": 0,
                        "severities": {}
                    }
                combined["conditions"][name]["count"] += 1
                combined["conditions"][name]["total_confidence"] += condition["confidence"]
                combined["conditions"][name]["severities"][condition["severity"]] = combined["conditions"][name]["severities"].get(condition["severity"], 0) + 1
    
    # Calculate average confidence and most common severity for each condition
    for name, data in combined["conditions"].items():
        data["average_confidence"] = round(data["total_confidence"] / data["count"], 2)
        data["severity"] = max(data["severities"].items(), key=lambda x: x[1])[0]
        data["prevalence"] = round(data["count"] / len(angle_results), 2)
    
    # Generate recommendations based on combined results
    combined["recommendations"] = generate_recommendations(combined)
    
    return combined

def generate_recommendations(analysis: Dict):
    """Generate personalized recommendations based on analysis results."""
    recommendations = {
        'cleanser': "Use a gentle, pH-balanced cleanser twice daily.",
        'moisturizer': "Apply a non-comedogenic moisturizer after cleansing.",
        'sunscreen': "Use a broad-spectrum SPF 30+ sunscreen daily.",
        'treatments': []
    }
    
    # Add specific recommendations based on skin type
    skin_type = analysis.get('skin_type', 'combination').lower()
    if 'dry' in skin_type:
        recommendations['moisturizer'] = "Use a rich, hydrating moisturizer with ceramides and hyaluronic acid."
    elif 'oily' in skin_type:
        recommendations['moisturizer'] = "Use an oil-free, non-comedogenic moisturizer."
    
    # Add recommendations based on concerns
    concerns = analysis.get('concerns', {})
    for concern, severity in concerns.items():
        if severity > 0.5:  # Only address significant concerns
            if 'acne' in concern.lower():
                recommendations['treatments'].append("Consider using a salicylic acid or benzoyl peroxide treatment for acne.")
            elif 'wrinkle' in concern.lower() or 'aging' in concern.lower():
                recommendations['treatments'].append("Consider using a retinol or peptide serum to address signs of aging.")
            elif 'dark spot' in concern.lower() or 'hyperpigmentation' in concern.lower():
                recommendations['treatments'].append("Consider using vitamin C or niacinamide to help with hyperpigmentation.")
    
    if not recommendations['treatments']:
        recommendations['treatments'].append("Your skin appears to be in good condition. Maintain your current routine.")
    
    return recommendations

# Add WebSocket endpoint to the router with the correct path
router.websocket("/ws/live-capture")(websocket_endpoint)
router.websocket("/ws/live-capture/{client_id}")(websocket_endpoint)
