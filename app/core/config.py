"""
Application configuration settings with production-ready defaults.
"""
from pydantic_settings import BaseSettings
from pydantic import Field, HttpUrl, AnyHttpUrl
from pathlib import Path
from typing import List, Optional, Union
import os
from functools import lru_cache

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = Field("Skin Analysis API", description="Name of the application")
    DEBUG: bool = Field(False, description="Enable debug mode (disable in production!)")
    ENVIRONMENT: str = Field("production", description="Runtime environment (e.g., development, staging, production)")
    
    # Server settings
    HOST: str = Field("0.0.0.0", description="Host to bind the server to")
    PORT: int = Field(8000, description="Port to run the server on")
    WORKERS: int = Field(1, description="Number of worker processes")
    
    # Security
    SECRET_KEY: str = Field(..., description="Secret key for cryptographic operations")
    API_PREFIX: str = Field("/api", description="API prefix for all routes")
    BACKEND_CORS_ORIGINS: List[Union[str, AnyHttpUrl]] = Field(
        ["*"],
        description="List of origins allowed to make cross-origin requests"
    )
    
    # Rate limiting
    RATE_LIMIT: int = Field(100, description="Requests per minute per IP address")
    RATE_LIMIT_WINDOW: int = Field(60, description="Rate limit window in seconds")
    
    # Supabase settings
    SUPABASE_URL: HttpUrl = Field(..., description="Supabase project URL")
    SUPABASE_KEY: str = Field(..., description="Supabase API key")
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = Field(
        None,
        description="Supabase service role key for admin operations"
    )
    
    # Storage settings
    STORAGE_DIR: Path = Field(
        default_factory=lambda: Path("data").absolute(),
        description="Directory to store uploaded files and analysis results"
    )
    MAX_UPLOAD_SIZE: int = Field(
        5 * 1024 * 1024,  # 5MB
        description="Maximum file upload size in bytes"
    )
    
    # Analysis settings
    CAPTURE_ANGLES: List[str] = Field(
        default_factory=lambda: ["front", "left", "right"],
        description="List of angles to capture for analysis"
    )
    MAX_RETRIES: int = Field(
        3,
        description="Maximum number of retry attempts for image capture"
    )
    
    # Model settings
    MODEL_PATH: Path = Field(
        default_factory=lambda: Path("models/skin_analysis_model.pth"),
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

@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings, cached for performance.
    """
    settings = Settings()
    
    # Create storage directories
    settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create models directory if it doesn't exist
    settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    return settings


# For backward compatibility
settings = get_settings()
