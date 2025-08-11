from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import os
import uvicorn
import logging
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

from app.core.config import settings
from app.api.essential_endpoints import router as essential_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

logger = logging.getLogger(__name__)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

app = FastAPI(
    title="Skin Analysis API",
    description="Essential endpoints for skin analysis functionality",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include essential endpoints
app.include_router(
    essential_router,
    prefix="/api",
    tags=["skin-analysis"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "Not found"}},
)

# Serve static files from uploads directory
@app.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "Skin Analysis API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": [
            "POST /api/analyze - Upload an image for skin analysis",
            "GET /api/health - Health check endpoint"
        ]
    }

# Error handlers
@app.exception_handler(404)
async def not_found_exception_handler(request, exc):
    return {
        "error": "Not Found",
        "message": f"The requested URL {request.url} was not found"
    }

@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    return {
        "error": "Validation Error",
        "message": str(exc)
    }

@app.exception_handler(500)
async def internal_exception_handler(request, exc):
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST or "0.0.0.0",
        port=int(settings.PORT or 8000),
        reload=settings.DEBUG if hasattr(settings, 'DEBUG') else True,
        log_level="info" if not (hasattr(settings, 'DEBUG') and settings.DEBUG) else "debug",
        workers=1,
    )
