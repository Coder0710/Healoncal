"""
API Endpoints Package

This package contains all API endpoint modules for the application.
"""

# Import routers to make them available when importing from this package
from .analysis import router as analysis_router
from .websocket import router as websocket_router
from .capture import router as capture_router

__all__ = ["analysis_router", "websocket_router", "capture_router"]
