#!/bin/bash
# This script is used to start the FastAPI application in production

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads

# Start the application
exec uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 4
