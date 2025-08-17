"""
Module for initializing database and initial data.
"""
import logging
from pathlib import Path
from typing import Any, Dict

from app.core.config import get_settings

logger = logging.getLogger(__name__)

def init_db() -> None:
    """Initialize database and create required directories."""
    settings = get_settings()
    
    # Create required directories
    required_dirs = [
        settings.STORAGE_DIR,
        settings.MODEL_PATH.parent,
        Path("logs"),
    ]
    
    for directory in required_dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
    
    logger.info("Database initialization complete")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run initialization
    init_db()
