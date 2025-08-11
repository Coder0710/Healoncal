"""
Gemini Service for analysis summarization and combination.
"""
import os
import logging
from typing import List, Dict, Any
import google.generativeai as genai
from app.core.config import settings

logger = logging.getLogger(__name__)

class GeminiService:
    """Service for interacting with Google Gemini API."""
    
    def __init__(self):
        """Initialize the Gemini service with API key."""
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
            self.generation_config = {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            logger.info("✅ Gemini service initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini service: {str(e)}")
            raise
    
    async def combine_analysis_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine multiple analysis results using Gemini.
        
        Args:
            results: List of analysis results from different angles
            
        Returns:
            Combined analysis result
        """
        try:
            # Prepare the prompt
            prompt = """You are a professional dermatologist assistant. Please analyze the following 
            skin analysis results from different angles and provide a comprehensive combined analysis.
            
            Analysis Results from Different Angles:
            """
            
            for i, result in enumerate(results, 1):
                prompt += f"\n--- Angle {i} ---\n"
                for key, value in result.items():
                    prompt += f"{key}: {value}\n"
            
            prompt += """
            
            Please provide a comprehensive analysis that includes:
            1. Overall skin type assessment
            2. Main skin concerns and their severity
            3. Any inconsistencies between different angles
            4. General recommendations
            5. Confidence level of the combined analysis
            """
            
            # Generate the combined analysis
            response = await self.model.generate_content_async(
                prompt,
                generation_config=self.generation_config
            )
            
            # Process the response
            combined_analysis = {
                "combined_analysis": response.text,
                "individual_results": results,
                "confidence": 0.9  # This could be calculated based on response
            }
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Error in combine_analysis_results: {str(e)}")
            raise

# Singleton instance
gemini_service = GeminiService()
