"""
Analysis API endpoints for skin analysis functionality.
"""
import uuid
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, status, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from enum import Enum

from app.services.skin_analysis_service import SkinAnalysisService
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["analysis"])
service = SkinAnalysisService()

# In-memory storage for analysis tasks (in production, use a database)
analysis_tasks: Dict[str, Dict] = {}

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    CAPTURING = "capturing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class AnalysisRequest(BaseModel):
    """Request model for starting a new skin analysis."""
    user_id: str = Field(..., description="Unique identifier for the user")
    angles: List[str] = Field(
        default=settings.CAPTURE_ANGLES,
        description="List of angles to capture (e.g., ['front', 'left', 'right'])"
    )

class AnalysisResponse(BaseModel):
    """Response model for analysis status."""
    analysis_id: str = Field(..., description="Unique identifier for the analysis task")
    status: AnalysisStatus = Field(..., description="Current status of the analysis")
    progress: int = Field(0, description="Progress percentage (0-100)")
    message: str = Field(..., description="Status message")
    results: Optional[Dict] = Field(None, description="Analysis results when completed")
    error: Optional[str] = Field(None, description="Error message if analysis failed")

@router.post("/start", response_model=AnalysisResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
) -> AnalysisResponse:
    """
    Start a new skin analysis task.
    
    This endpoint initiates an asynchronous analysis task and returns immediately
    with a task ID that can be used to check the status.
    """
    analysis_id = f"analysis_{uuid.uuid4().hex}"
    
    # Create initial task entry
    analysis_tasks[analysis_id] = {
        "status": AnalysisStatus.PENDING,
        "progress": 0,
        "message": "Analysis initialized",
        "results": None,
        "error": None,
        "start_time": time.time(),
        "end_time": None
    }
    
    # Start background task
    background_tasks.add_task(
        process_analysis,
        analysis_id=analysis_id,
        user_id=request.user_id,
        angles=request.angles
    )
    
    return AnalysisResponse(
        analysis_id=analysis_id,
        status=AnalysisStatus.PENDING,
        progress=0,
        message="Analysis started"
    )

@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_status(analysis_id: str) -> AnalysisResponse:
    """
    Get the status of an analysis task.
    
    Use this endpoint to poll for updates on an ongoing analysis.
    """
    if analysis_id not in analysis_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with ID {analysis_id} not found"
        )
    
    task = analysis_tasks[analysis_id]
    
    return AnalysisResponse(
        analysis_id=analysis_id,
        status=task["status"],
        progress=task["progress"],
        message=task["message"],
        results=task.get("results"),
        error=task.get("error")
    )

async def process_analysis(analysis_id: str, user_id: str, angles: List[str]):
    """Background task to process the skin analysis."""
    try:
        task = analysis_tasks[analysis_id]
        
        # Update status to capturing
        task.update({
            "status": AnalysisStatus.CAPTURING,
            "progress": 10,
            "message": "Starting image capture"
        })
        
        # Capture and analyze images from each angle
        analysis_results = []
        
        for i, angle in enumerate(angles):
            # Update progress
            progress = 10 + int(60 * (i / len(angles)))
            task.update({
                "progress": progress,
                "message": f"Capturing {angle} view"
            })
            
            # Capture image
            success, image_path, error = await service.capture_image(angle)
            if not success:
                raise Exception(f"Failed to capture {angle} view: {error}")
            
            # Analyze image
            task.update({
                "progress": 10 + int(60 * ((i + 0.5) / len(angles))),
                "message": f"Analyzing {angle} view"
            })
            
            result = await service.analyze_image(image_path, angle)
            analysis_results.append(result)
        
        # Generate final report
        task.update({
            "status": AnalysisStatus.PROCESSING,
            "progress": 75,
            "message": "Generating final report"
        })
        
        final_report = service.generate_final_report(analysis_results)
        
        # Update task with results
        task.update({
            "status": AnalysisStatus.COMPLETED,
            "progress": 100,
            "message": "Analysis complete",
            "results": final_report,
            "end_time": time.time()
        })
        
        # Clean up after some time (1 hour)
        await asyncio.sleep(3600)
        if analysis_id in analysis_tasks:
            del analysis_tasks[analysis_id]
            
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        if analysis_id in analysis_tasks:
            analysis_tasks[analysis_id].update({
                "status": AnalysisStatus.FAILED,
                "error": error_msg,
                "end_time": time.time()
            })
