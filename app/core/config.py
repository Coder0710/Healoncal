"""
Application configuration settings.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import List, Optional
import os

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = Field(default="Skin Analysis API", description="Name of the application")
    DEBUG: bool = Field(default=True, description="Enable debug mode")
    
    # Server settings
    HOST: str = Field(default="0.0.0.0", description="Host to bind the server to")
    PORT: int = Field(default=8000, description="Port to run the server on")
    
    # Supabase settings
    SUPABASE_URL: str = Field(..., description="Supabase project URL")
    SUPABASE_KEY: str = Field(..., description="Supabase API key")
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = Field(
        None,
        description="Supabase service role key for admin operations. If not provided, will fall back to SUPABASE_KEY"
    )
    
    # Storage settings
    STORAGE_DIR: str = Field(
        default=str(Path("data").absolute()),
        description="Directory to store uploaded files and analysis results"
    )
    
    # Analysis settings
    CAPTURE_ANGLES: List[str] = Field(
        default=["front", "left", "right"],
        description="List of angles to capture for analysis"
    )
    MAX_RETRIES: int = Field(
        default=3,
        description="Maximum number of retry attempts for image capture"
    )
    
    # Model settings
    MODEL_PATH: str = Field(
        default="models/skin_analysis_model.pth",
        description="Path to the skin analysis model"
    )
    
    # Gemini API settings
    GEMINI_API_KEY: str = Field(
        ...,
        description="Google Gemini API key for analysis summarization"
    )
    GEMINI_MODEL: str = Field(
        default="gemini-1.5-pro-latest",
        description="Gemini model to use for analysis summarization"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields in .env file

# Create instance
settings = Settings()

# Create storage directories
Path(settings.STORAGE_DIR).mkdir(parents=True, exist_ok=True)
