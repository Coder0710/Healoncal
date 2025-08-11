"""
Skin Analysis Service

This module provides the core functionality for skin analysis,
including image capture, processing, and analysis with Supabase integration.
"""
import os
import uuid
import time
import logging
import asyncio
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import cv2
import numpy as np
import asyncio
from PIL import Image
from supabase import create_client, Client
import concurrent.futures

from app.models.skin_analyzer import SkinAnalyzer
from app.core.config import settings
from app.services.gemini_service import gemini_service

logger = logging.getLogger(__name__)

class SkinAnalysisService:
    """Service class for handling skin analysis operations with Supabase integration."""
    
    def __init__(self):
        """Initialize the skin analysis service with Supabase client."""
        self.analyzer = SkinAnalyzer()
        
        # Initialize Supabase client with regular key
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY are required in the configuration")
            
        self.supabase: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )
        
        # Create a separate client with service role for admin operations if available
        self.admin_supabase = None
        if settings.SUPABASE_SERVICE_ROLE_KEY:
            self.admin_supabase = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_SERVICE_ROLE_KEY
            )
        self.bucket_name = "skin-scans"
        
        # Initialize the bucket - don't try to check if it exists first
        # as this often causes 400 errors with Supabase
        self._initialize_bucket()
    
    def _initialize_bucket(self):
        """Initialize the storage bucket - assume it already exists."""
        # Since bucket already exists, just log that we're using it
        logger.info(f"✅ Using existing bucket: {self.bucket_name}")
        # No need to try creating bucket since it clearly already exists
    
    async def _upload_to_supabase(self, image_data: bytes, filename: str) -> str:
        """
        Upload image to Supabase Storage.
        
        Args:
            image_data: Raw image bytes
            filename: Name to save the file as
            
        Returns:
            Public URL of the uploaded file
        """
        path = f"scans/{filename}"
        
        # Try with admin client first if available
        if self.admin_supabase:
            try:
                # Upload the file with bytes directly
                result = self.admin_supabase.storage.from_(self.bucket_name).upload(
                    path=path,
                    file=image_data,
                    file_options={"content-type": "image/jpeg"}
                )
                
                if hasattr(result, 'error') and result.error:
                    raise Exception(f"Admin upload failed: {result.error}")
            
                # Get public URL
                url = self.admin_supabase.storage.from_(self.bucket_name).get_public_url(path)
                logger.info(f"Successfully uploaded {filename} using admin client")
                return url
                
            except Exception as admin_error:
                logger.warning(f"Admin upload failed, will try regular client: {admin_error}")
        
        # If admin client failed or not available, try with regular client
        try:
            # Upload the file with bytes directly
            result = self.supabase.storage.from_(self.bucket_name).upload(
                path=path,
                file=image_data,
                file_options={"content-type": "image/jpeg"}
            )
            
            if hasattr(result, 'error') and result.error:
                raise Exception(f"Regular client upload failed: {result.error}")
        
            # Get public URL
            url = self.supabase.storage.from_(self.bucket_name).get_public_url(path)
            logger.info(f"Successfully uploaded {filename} using regular client")
            return url
            
        except Exception as e:
            error_msg = f"Failed to upload {filename} to Supabase: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    async def save_analysis(self, client_id: str, angle: str, image_data: bytes, analysis: dict) -> str:
        """
        Save analysis to Supabase.
        
        Args:
            client_id: Unique client identifier
            angle: The angle of the capture
            image_data: Raw image bytes
            analysis: Analysis results
            
        Returns:
            Analysis ID from the database
        """
        try:
            # Upload image
            filename = f"{client_id}_{angle}_{int(time.time())}.jpg"
            image_url = await self._upload_to_supabase(image_data, filename)
            
            # Use admin client with service role to bypass RLS
            if self.admin_supabase:
                result = self.admin_supabase.table("skin_analyses").insert({
                    "client_id": client_id,
                    "angle": angle,
                    "image_url": image_url,
                    "analysis": analysis,
                    "created_at": datetime.utcnow().isoformat()
                }).execute()
            else:
                # Fallback to regular client
                result = self.supabase.table("skin_analyses").insert({
                    "client_id": client_id,
                    "angle": angle,
                    "image_url": image_url,
                    "analysis": analysis,
                    "created_at": datetime.utcnow().isoformat()
                }).execute()
            
            if not result.data:
                raise ValueError("Failed to save analysis to database")
                
            return result.data[0]['id']
            
        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
            raise
    
    async def create_user_folder(self, user_id: str) -> bool:
        """
        Create a user-specific folder in the bucket.
        Note: Supabase creates folders automatically when files are uploaded,
        so we don't need to create a placeholder file.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            True (folder will be created when first file is uploaded)
        """
        try:
            # Supabase automatically creates folder structure when files are uploaded
            # No need for placeholder files
            logger.info(f"✅ User folder for {user_id} will be created automatically on first upload")
            return True
            
        except Exception as e:
            logger.error(f"Error in create_user_folder for {user_id}: {str(e)}")
            return False

    async def store_user_image(self, user_id: str, angle: str, image_data: bytes) -> str:
        """
        Store user image in their specific folder.
        
        Args:
            user_id: Unique user identifier
            angle: The angle of the capture (front, left, right)
            image_data: Raw image bytes
            
        Returns:
            Storage path of the uploaded image
        """
        try:
            # Generate unique filename with timestamp
            timestamp = int(time.time())
            filename = f"{angle}_{timestamp}.jpg"
            file_path = f"users/{user_id}/{filename}"
            
            # Upload the image
            if self.admin_supabase:
                try:
                    result = self.admin_supabase.storage.from_(self.bucket_name).upload(
                        path=file_path,
                        file=image_data,
                        file_options={"content-type": "image/jpeg"}
                    )
                    logger.info(f"✅ Stored {angle} image for user {user_id} at {file_path}")
                    return file_path
                except Exception as admin_error:
                    logger.warning(f"Admin image upload failed: {admin_error}")
            
            # Fallback to regular client
            result = self.supabase.storage.from_(self.bucket_name).upload(
                path=file_path,
                file=image_data,
                file_options={"content-type": "image/jpeg"}
            )
            logger.info(f"✅ Stored {angle} image for user {user_id} at {file_path}")
            return file_path
            
        except Exception as e:
            error_msg = f"Failed to store {angle} image for user {user_id}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    async def retrieve_user_image(self, user_id: str, file_path: str) -> bytes:
        """
        Retrieve user image from their folder.
        
        Args:
            user_id: Unique user identifier
            file_path: Path to the image file in storage
            
        Returns:
            Image bytes
        """
        try:
            # Download the image
            if self.admin_supabase:
                try:
                    result = self.admin_supabase.storage.from_(self.bucket_name).download(file_path)
                    logger.info(f"✅ Retrieved image for user {user_id} from {file_path}")
                    return result
                except Exception as admin_error:
                    logger.warning(f"Admin image download failed: {admin_error}")
            
            # Fallback to regular client
            result = self.supabase.storage.from_(self.bucket_name).download(file_path)
            logger.info(f"✅ Retrieved image for user {user_id} from {file_path}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to retrieve image for user {user_id} from {file_path}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    async def capture_and_store_image(self, user_id: str, angle: str, image_data: bytes) -> Dict[str, Any]:
        """
        Capture and store image in user-specific folder without analysis.
        
        Args:
            user_id: Unique user identifier
            angle: The angle of the capture
            image_data: Raw image bytes or base64 string
            
        Returns:
            Dictionary with storage results
        """
        try:
            logger.info(f"Storing {angle} image for user {user_id}")
            
            # Ensure user folder exists
            await self.create_user_folder(user_id)
            
            # Convert base64 to bytes if needed
            if isinstance(image_data, str):
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data
            
            # Store the image in user folder
            file_path = await self.store_user_image(user_id, angle, image_bytes)
            
            # Save record to database with file path
            analysis_record = {
                "client_id": user_id,
                "angle": angle,
                "image_url": file_path,  # Store the path for later retrieval
                "analysis": {"status": "stored", "message": "Image stored, awaiting analysis"},
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Use admin client to save to database
            if self.admin_supabase:
                db_result = self.admin_supabase.table("skin_analyses").insert(analysis_record).execute()
            else:
                db_result = self.supabase.table("skin_analyses").insert(analysis_record).execute()
            
            if not db_result.data:
                raise ValueError("Failed to save storage record to database")
            
            record_id = db_result.data[0]['id']
            logger.info(f"✅ Successfully stored {angle} image for user {user_id}, record ID: {record_id}")
            
            return {
                'success': True,
                'record_id': record_id,
                'file_path': file_path,
                'angle': angle,
                'message': f'Image stored successfully for {angle} view'
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in capture_and_store_image: {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'angle': angle
            }

    async def analyze_stored_images_fast(self, user_id: str, image_records: List[Dict], progress_callback=None) -> Dict[str, Any]:
        """
        Analyze images in parallel for much faster processing.
        
        Args:
            user_id: Unique user identifier
            image_records: List of image records with file paths and record IDs
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with combined analysis results
        """
        import time
        overall_start = time.time()
        
        try:
            logger.info(f"🚀 [FAST ANALYSIS] Starting for user {user_id} at {overall_start}")
            logger.info(f"📊 [FAST ANALYSIS] Processing {len(image_records)} images: {[r['angle'] for r in image_records]}")
            
            # Create or get analysis session
            session_start = time.time()
            session_id = await self._create_or_get_analysis_session(user_id, image_records)
            session_duration = time.time() - session_start
            logger.info(f"✅ [FAST ANALYSIS] Analysis session ready: {session_id} ({session_duration:.3f}s)")
            
            # Process all images in parallel
            async def analyze_single_image(record: Dict, index: int) -> tuple:
                """Analyze a single image and return (angle, result)"""
                try:
                    image_start = time.time()
                    angle = record['angle']
                    file_path = record['file_path'] 
                    record_id = record['record_id']
                    
                    logger.info(f"🔍 [SINGLE IMAGE] Starting {angle} analysis for user {user_id}")
                    
                    # Send progress update
                    if progress_callback:
                        await progress_callback(angle, index + 1, len(image_records))
                    
                    # Retrieve and analyze image
                    retrieve_start = time.time()
                    image_bytes = await self.retrieve_user_image(user_id, file_path)
                    retrieve_duration = time.time() - retrieve_start
                    logger.info(f"📥 [SINGLE IMAGE] Retrieved {angle} image: {len(image_bytes)} bytes ({retrieve_duration:.3f}s)")
                    
                    analyze_start = time.time()
                    analysis_result = await self.analyze_image_only(image_bytes, angle)
                    analyze_duration = time.time() - analyze_start
                    logger.info(f"🧠 [SINGLE IMAGE] Analyzed {angle} image ({analyze_duration:.3f}s): {analysis_result.get('success', False)}")
                    
                    if analysis_result.get('success', False):
                        analysis_data = analysis_result.get('analysis', {})
                        
                        # Prepare update data
                        prep_start = time.time()
                        update_data = {
                            "image_url": self._get_public_image_url(file_path),
                            "analysis": analysis_data,
                            "session_id": session_id,
                            "analysis_status": "completed",
                            "updated_at": datetime.utcnow().isoformat(),
                            "file_size": len(image_bytes),
                            "processing_time_ms": int((time.time() - image_start) * 1000)
                        }
                        prep_duration = time.time() - prep_start
                        
                        total_image_time = time.time() - image_start
                        logger.info(f"✅ [SINGLE IMAGE] {angle} completed successfully in {total_image_time:.3f}s")
                        logger.info(f"⏱️ [SINGLE IMAGE] {angle} breakdown - Retrieve: {retrieve_duration:.3f}s, Analyze: {analyze_duration:.3f}s, Prep: {prep_duration:.3f}s")
                        
                        return (angle, analysis_data, record_id, update_data, None)
                    else:
                        error_msg = analysis_result.get('error', 'Analysis failed')
                        logger.error(f"❌ [SINGLE IMAGE] {angle} analysis failed: {error_msg}")
                        error_update = {
                            "analysis": {"status": "failed", "error": error_msg},
                            "analysis_status": "failed", 
                            "updated_at": datetime.utcnow().isoformat()
                        }
                        return (angle, None, record_id, error_update, error_msg)
                        
                except Exception as e:
                    total_time = time.time() - image_start
                    logger.error(f"💥 [SINGLE IMAGE] Error analyzing {record.get('angle', 'unknown')} after {total_time:.3f}s: {str(e)}")
                    error_update = {
                        "analysis": {"status": "failed", "error": str(e)},
                        "analysis_status": "failed",
                        "updated_at": datetime.utcnow().isoformat()
                    }
                    return (record.get('angle', 'unknown'), None, record.get('record_id'), error_update, str(e))
            
            # Process all images in parallel with optimized concurrency for speed
            semaphore = asyncio.Semaphore(5)  # Increased to 5 concurrent analyses for better performance
            
            async def bounded_analyze(record, index):
                async with semaphore:
                    return await analyze_single_image(record, index)
            
            # Execute all analyses in parallel
            parallel_start = time.time()
            logger.info(f"⚡ [PARALLEL START] Starting parallel analysis of {len(image_records)} images")
            results = await asyncio.gather(*[
                bounded_analyze(record, i) for i, record in enumerate(image_records)
            ], return_exceptions=True)
            parallel_duration = time.time() - parallel_start
            logger.info(f"⚡ [PARALLEL DONE] Parallel analysis completed in {parallel_duration:.3f}s")
            
            # Process results and batch database updates
            process_start = time.time()
            analysis_results = {}
            batch_updates = []
            individual_analyses = []
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"💥 [RESULT PROCESSING] Analysis task failed: {str(result)}")
                    continue
                    
                angle, analysis_data, record_id, update_data, error = result
                
                if error:
                    analysis_results[angle] = {"error": error, "status": "error"}
                    logger.warning(f"❌ [RESULT PROCESSING] {angle} had error: {error}")
                else:
                    analysis_results[angle] = analysis_data
                    individual_analyses.append({
                        'angle': angle,
                        'analysis': analysis_data,
                        'record_id': record_id
                    })
                    logger.info(f"✅ [RESULT PROCESSING] {angle} processed successfully")
                
                # Add to batch update
                batch_updates.append((record_id, update_data))
            
            process_duration = time.time() - process_start
            logger.info(f"📊 [RESULT PROCESSING] Processed {len(results)} results in {process_duration:.3f}s")
            
            # Perform batch database updates
            batch_start = time.time()
            await self._batch_update_records(batch_updates)
            batch_duration = time.time() - batch_start
            logger.info(f"💾 [BATCH UPDATE] Updated {len(batch_updates)} database records in {batch_duration:.3f}s")
            
            # Generate combined results using local fallback (faster than Gemini)
            combine_start = time.time()
            combined_results = self._combine_analysis_results_fast(analysis_results)
            combine_duration = time.time() - combine_start
            logger.info(f"🔄 [COMBINE] Generated combined results in {combine_duration:.3f}s")
            
            # Update analysis session with final results
            session_update_start = time.time()
            await self._update_analysis_session(session_id, combined_results, analysis_results)
            session_update_duration = time.time() - session_update_start
            logger.info(f"📝 [SESSION UPDATE] Updated analysis session in {session_update_duration:.3f}s")
            
            # Create analysis history record
            history_start = time.time()
            await self._create_analysis_history_record(user_id, session_id, combined_results, analysis_results)
            history_duration = time.time() - history_start
            logger.info(f"📚 [HISTORY] Created history record in {history_duration:.3f}s")
            
            total_duration = time.time() - overall_start
            logger.info(f"🎉 [FAST ANALYSIS COMPLETE] User {user_id} completed in {total_duration:.3f}s")
            logger.info(f"⏱️ [TIMING BREAKDOWN] Session: {session_duration:.3f}s, Parallel: {parallel_duration:.3f}s, Process: {process_duration:.3f}s, Batch: {batch_duration:.3f}s, Combine: {combine_duration:.3f}s, Session Update: {session_update_duration:.3f}s, History: {history_duration:.3f}s")
            
            return {
                'success': True,
                'user_id': user_id,
                'session_id': session_id,
                'individual_results': analysis_results,
                'combined_results': combined_results,
                'timing': {
                    'total': total_duration,
                    'session_setup': session_duration,
                    'parallel_analysis': parallel_duration,
                    'result_processing': process_duration,
                    'batch_updates': batch_duration,
                    'combine_results': combine_duration,
                    'session_update': session_update_duration,
                    'history_creation': history_duration
                }
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in analyze_stored_images_fast for user {user_id}: {error_msg}", exc_info=True)
            return {
                'success': False,
                'error': error_msg,
                'user_id': user_id
            }

    async def analyze_stored_images(self, user_id: str, image_records: List[Dict]) -> Dict[str, Any]:
        """
        Analyze images that are already stored in user folder and save results to database.
        
        Args:
            user_id: Unique user identifier
            image_records: List of image records with file paths and record IDs
            
        Returns:
            Dictionary with combined analysis results
        """
        try:
            logger.info(f"Starting analysis of stored images for user {user_id}")
            logger.info(f"Image records to analyze: {image_records}")
            
            # Create or get analysis session
            session_id = await self._create_or_get_analysis_session(user_id, image_records)
            logger.info(f"Using analysis session: {session_id}")
            
            analysis_results = {}
            individual_analyses = []
            
            for record in image_records:
                try:
                    angle = record['angle']
                    file_path = record['file_path']
                    record_id = record['record_id']
                    
                    logger.info(f"Analyzing {angle} image for user {user_id} from {file_path}")
                    
                    # Retrieve image from storage
                    logger.debug(f"Retrieving image from storage for {angle}")
                    image_bytes = await self.retrieve_user_image(user_id, file_path)
                    logger.debug(f"Retrieved {len(image_bytes)} bytes for {angle} image")
                    
                    # Perform analysis using the analyze_image_only method
                    logger.debug(f"Starting image analysis for {angle}")
                    analysis_result = await self.analyze_image_only(image_bytes, angle)
                    logger.debug(f"Analysis result for {angle}: {analysis_result.get('success', False)}")
                    
                    if analysis_result.get('success', False):
                        analysis_data = analysis_result.get('analysis', {})
                        analysis_results[angle] = analysis_data
                        individual_analyses.append({
                            'angle': angle,
                            'analysis': analysis_data,
                            'record_id': record_id
                        })
                        
                        # Update the database record with comprehensive analysis results
                        logger.debug(f"Updating database record for {angle}")
                        
                        # Get public URL for the image
                        public_url = self._get_public_image_url(file_path)
                        
                        # Prepare comprehensive update data according to schema
                        update_data = {
                            "image_url": public_url,
                            "analysis": analysis_data,
                            "session_id": session_id,
                            "analysis_status": "completed",
                            "updated_at": datetime.utcnow().isoformat(),
                            # Extract key analysis metrics for easier querying
                            "file_size": len(image_bytes),
                            "processing_time_ms": int(time.time() * 1000) % 10000  # Approximate processing time
                        }
                        
                        # Update skin_analyses table
                        if self.admin_supabase:
                            db_result = self.admin_supabase.table("skin_analyses").update(update_data).eq("id", record_id).execute()
                        else:
                            db_result = self.supabase.table("skin_analyses").update(update_data).eq("id", record_id).execute()
                        
                        if db_result.data:
                            logger.info(f"✅ Successfully analyzed and updated {angle} image for user {user_id}")
                        else:
                            logger.warning(f"Database update returned no data for {angle} record {record_id}")
                            
                    else:
                        error_msg = analysis_result.get('error', 'Analysis failed')
                        logger.error(f"Analysis failed for {angle} image: {error_msg}")
                        analysis_results[angle] = {
                            "error": error_msg,
                            "status": "error"
                        }
                        
                        # Update record with error status
                        error_update = {
                            "analysis": {"status": "failed", "error": error_msg},
                            "analysis_status": "failed",
                            "updated_at": datetime.utcnow().isoformat()
                        }
                        
                        if self.admin_supabase:
                            self.admin_supabase.table("skin_analyses").update(error_update).eq("id", record_id).execute()
                        else:
                            self.supabase.table("skin_analyses").update(error_update).eq("id", record_id).execute()
                        
                except Exception as e:
                    logger.error(f"Error analyzing {record.get('angle', 'unknown')} image: {str(e)}", exc_info=True)
                    analysis_results[record.get('angle', 'unknown')] = {
                        "error": str(e),
                        "status": "error"
                    }
            
            # Generate combined results
            logger.info(f"Generating combined results for user {user_id}")
            combined_results = self._combine_analysis_results(analysis_results)
            
            # Update analysis session with final results
            await self._update_analysis_session(session_id, combined_results, analysis_results)
            
            # Create analysis history record for quick lookups
            await self._create_analysis_history_record(user_id, session_id, combined_results, analysis_results)
            
            logger.info(f"✅ Completed analysis of all stored images for user {user_id}")
            return {
                'success': True,
                'user_id': user_id,
                'session_id': session_id,
                'individual_results': analysis_results,
                'combined_results': combined_results
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in analyze_stored_images for user {user_id}: {error_msg}", exc_info=True)
            return {
                'success': False,
                'error': error_msg,
                'user_id': user_id
            }

    async def _create_or_get_analysis_session(self, user_id: str, image_records: List[Dict]) -> str:
        """Create or get existing analysis session for the user."""
        try:
            # Look for existing active session for this user
            if self.admin_supabase:
                existing_sessions = self.admin_supabase.table("analysis_sessions").select("*").eq("client_id", user_id).eq("session_status", "active").execute()
            else:
                existing_sessions = self.supabase.table("analysis_sessions").select("*").eq("client_id", user_id).eq("session_status", "active").execute()
            
            if existing_sessions.data:
                session_id = existing_sessions.data[0]['id']
                logger.info(f"Using existing session {session_id} for user {user_id}")
                return str(session_id)
            
            # Create new session
            angles = [record['angle'] for record in image_records]
            session_data = {
                "client_id": user_id,
                "session_status": "active",
                "total_captures": len(image_records),
                "completed_angles": angles,
                "remaining_angles": [],
                "created_at": datetime.utcnow().isoformat()
            }
            
            if self.admin_supabase:
                result = self.admin_supabase.table("analysis_sessions").insert(session_data).execute()
            else:
                result = self.supabase.table("analysis_sessions").insert(session_data).execute()
            
            session_id = result.data[0]['id']
            logger.info(f"Created new session {session_id} for user {user_id}")
            return str(session_id)
            
        except Exception as e:
            logger.error(f"Error creating/getting analysis session: {str(e)}")
            # Return a fallback session ID
            return f"session_{user_id}_{int(time.time())}"

    async def _update_analysis_session(self, session_id: str, combined_results: Dict, individual_results: Dict):
        """Update the analysis session with final results."""
        try:
            update_data = {
                "session_status": "completed",
                "final_analysis": combined_results,
                "completed_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if self.admin_supabase:
                result = self.admin_supabase.table("analysis_sessions").update(update_data).eq("id", session_id).execute()
            else:
                result = self.supabase.table("analysis_sessions").update(update_data).eq("id", session_id).execute()
            
            logger.info(f"Updated analysis session {session_id} with final results")
            
        except Exception as e:
            logger.error(f"Error updating analysis session {session_id}: {str(e)}")

    async def _create_analysis_history_record(self, user_id: str, session_id: str, combined_results: Dict, individual_results: Dict):
        """Create a summary record in analysis_history table."""
        try:
            # Extract dominant concerns from individual results
            all_concerns = []
            for angle_result in individual_results.values():
                if isinstance(angle_result, dict) and 'concerns' in angle_result:
                    concerns_data = angle_result['concerns']
                    if isinstance(concerns_data, dict):
                        all_concerns.extend(concerns_data.keys())
                    elif isinstance(concerns_data, list):
                        all_concerns.extend([c.get('name', str(c)) if isinstance(c, dict) else str(c) for c in concerns_data])
            
            # Get most common concerns
            concern_counts = {}
            for concern in all_concerns:
                concern_counts[concern] = concern_counts.get(concern, 0) + 1
            
            dominant_concerns = sorted(concern_counts.keys(), key=lambda x: concern_counts[x], reverse=True)[:3]
            
            # Calculate overall score (average of key metrics)
            overall_score = (
                combined_results.get('average_moisture', 0) +
                combined_results.get('average_texture', 0) + 
                (100 - combined_results.get('average_oiliness', 0)) +  # Lower oiliness is better
                (100 - combined_results.get('average_pores', 0))      # Smaller pores are better
            ) / 4 / 100  # Normalize to 0-1 scale
            
            history_data = {
                "session_id": session_id,
                "analysis_date": datetime.utcnow().date().isoformat(),
                "skin_type": combined_results.get('skin_type', 'Normal'),
                "dominant_concerns": dominant_concerns,
                "overall_score": round(overall_score, 2),
                "created_at": datetime.utcnow().isoformat()
            }
            
            if self.admin_supabase:
                result = self.admin_supabase.table("analysis_history").insert(history_data).execute()
            else:
                result = self.supabase.table("analysis_history").insert(history_data).execute()
            
            logger.info(f"Created analysis history record for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error creating analysis history record: {str(e)}")

    def _get_public_image_url(self, file_path: str) -> str:
        """Get public URL for stored image."""
        try:
            if self.admin_supabase:
                return self.admin_supabase.storage.from_(self.bucket_name).get_public_url(file_path)
            else:
                return self.supabase.storage.from_(self.bucket_name).get_public_url(file_path)
        except Exception as e:
            logger.error(f"Error getting public URL for {file_path}: {str(e)}")
            return file_path  # Return file path as fallback

    def _combine_analysis_results(self, angle_results: Dict) -> Dict:
        """Combine analysis results from multiple angles (internal method)."""
        combined = {
            "skin_type": "Normal",
            "skin_tone": "Unknown",
            "perceived_age": 0,
            "eye_age": 0,
            "conditions": {},
            "concerns": [],
            "average_moisture": 0.0,
            "average_oiliness": 0.0,
            "average_pores": 0.0,
            "average_texture": 0.0,
            "average_redness": 0.0,
            "analyzed_angles": list(angle_results.keys()),
            "analysis_time": datetime.now().isoformat(),
            "recommendations": {},
            "overall_score": 0.0,
            "confidence_score": 0.0
        }
        
        # Filter out error results and handle invalid data
        valid_results = []
        for angle, result in angle_results.items():
            if not isinstance(result, dict):
                logger.warning(f"Skipping invalid result for {angle}: not a dictionary")
                continue
                
            if result.get("error") or result.get("status") == "error":
                logger.warning(f"Skipping result for {angle} due to error: {result.get('error')}")
                continue
                
            # Ensure skin_type is a string and not a dictionary
            if 'skin_type' in result and isinstance(result['skin_type'], dict):
                result['skin_type'] = result['skin_type'].get('skin_type', 'Normal')
                
            valid_results.append(result)
        
        if not valid_results:
            logger.warning("No valid analysis results to combine")
            return combined
        
        # Determine most common skin type with safe handling
        skin_types = []
        for r in valid_results:
            skin_type = r.get("skin_type")
            if isinstance(skin_type, str):
                skin_types.append(skin_type)
            elif isinstance(skin_type, dict):
                skin_types.append(skin_type.get('skin_type', 'Normal'))
            else:
                skin_types.append('Normal')
                
        if skin_types:
            skin_type_counts = {}
            for st in skin_types:
                if isinstance(st, str):  # Ensure we only count string skin types
                    skin_type_counts[st] = skin_type_counts.get(st, 0) + 1
            
            if skin_type_counts:  # Only try to get max if we have valid counts
                combined["skin_type"] = max(skin_type_counts.items(), key=lambda x: x[1])[0]
            else:
                combined["skin_type"] = 'Normal'
        
        # Determine most common skin tone with safe handling
        skin_tones = []
        for r in valid_results:
            skin_tone = r.get("skin_tone")
            if isinstance(skin_tone, str):
                skin_tones.append(skin_tone)
            elif isinstance(skin_tone, dict):
                skin_tones.append(skin_tone.get('skin_tone', 'Unknown'))
            else:
                skin_tones.append('Unknown')
                
        if skin_tones:
            tone_counts = {}
            for tone in skin_tones:
                if isinstance(tone, str):  # Ensure we only count string tones
                    tone_counts[tone] = tone_counts.get(tone, 0) + 1
            
            if tone_counts:  # Only try to get max if we have valid counts
                combined["skin_tone"] = max(tone_counts.items(), key=lambda x: x[1])[0]
            else:
                combined["skin_tone"] = 'Unknown'
        
        # Calculate average ages
        ages = [r.get("perceived_age", 0) for r in valid_results if r.get("perceived_age", 0) > 0]
        if ages:
            combined["perceived_age"] = round(sum(ages) / len(ages))
        
        eye_ages = [r.get("eye_age", 0) for r in valid_results if r.get("eye_age", 0) > 0]
        if eye_ages:
            combined["eye_age"] = round(sum(eye_ages) / len(eye_ages))
        
        # Safely calculate average metrics with type checking
        def safe_float(value, default=0.0):
            try:
                return float(value) if value is not None else default
            except (ValueError, TypeError):
                return default
                
        # Initialize sums and counts for each metric
        metrics = {
            'moisture_level': 0.0,
            'oiliness': 0.0,
            'pores_visibility': 0.0,
            'texture_score': 0.0,
            'redness': 0.0
        }
        
        # Calculate sums
        for r in valid_results:
            for metric in metrics.keys():
                value = r.get(metric)
                metrics[metric] += safe_float(value)
        
        # Calculate and store averages
        count = max(1, len(valid_results))  # Avoid division by zero
        combined["average_moisture"] = round(metrics['moisture_level'] / count, 2)
        combined["average_oiliness"] = round(metrics['oiliness'] / count, 2)
        combined["average_pores"] = round(metrics['pores_visibility'] / count, 2)
        combined["average_texture"] = round(metrics['texture_score'] / count, 2)
        combined["average_redness"] = round(metrics['redness'] / count, 2)
        
        # Combine concerns/conditions from all angles
        all_concerns = []
        condition_data = {}
        
        for result in valid_results:
            # Handle different concern formats
            concerns = result.get("concerns", {})
            if isinstance(concerns, dict):
                for concern_name, severity in concerns.items():
                    all_concerns.append(concern_name)
                    if concern_name not in condition_data:
                        condition_data[concern_name] = {
                            "count": 0,
                            "total_severity": 0,
                            "max_severity": 0,
                            "angles": []
                        }
                    condition_data[concern_name]["count"] += 1
                    condition_data[concern_name]["total_severity"] += float(severity) if isinstance(severity, (int, float)) else 0.5
                    condition_data[concern_name]["max_severity"] = max(condition_data[concern_name]["max_severity"], float(severity) if isinstance(severity, (int, float)) else 0.5)
                    condition_data[concern_name]["angles"].append(angle_results)
            elif isinstance(concerns, list):
                for concern in concerns:
                    if isinstance(concern, dict):
                        concern_name = concern.get("name", str(concern))
                        severity = concern.get("severity", 0.5)
                    else:
                        concern_name = str(concern)
                        severity = 0.5
                    
                    all_concerns.append(concern_name)
                    if concern_name not in condition_data:
                        condition_data[concern_name] = {
                            "count": 0,
                            "total_severity": 0,
                            "max_severity": 0,
                            "angles": []
                        }
                    condition_data[concern_name]["count"] += 1
                    condition_data[concern_name]["total_severity"] += float(severity)
                    condition_data[concern_name]["max_severity"] = max(condition_data[concern_name]["max_severity"], float(severity))
        
        # Process combined conditions
        for concern_name, data in condition_data.items():
            combined["conditions"][concern_name] = {
                "average_severity": round(data["total_severity"] / data["count"], 2),
                "max_severity": round(data["max_severity"], 2),
                "prevalence": round(data["count"] / len(valid_results), 2),
                "affected_angles": data["count"]
            }
        
        # Create simplified concerns list for easy access
        combined["concerns"] = list(set(all_concerns))
        
        # Calculate overall confidence score
        confidence_scores = [r.get("confidence", 0.8) for r in valid_results if "confidence" in r]
        if confidence_scores:
            combined["confidence_score"] = round(sum(confidence_scores) / len(confidence_scores), 2)
        else:
            combined["confidence_score"] = 0.8  # Default confidence
        
        # Calculate overall score
        combined["overall_score"] = self._calculate_overall_score(combined)
        
        # Generate recommendations
        combined["recommendations"] = self._generate_recommendations(combined)
        
        return combined
    
    def _calculate_overall_score(self, combined_results: Dict) -> float:
        """Calculate an overall skin health score (0-1 scale)."""
        try:
            # Base score from skin metrics (higher moisture and texture are better, lower oiliness and pores are better)
            moisture_score = min(combined_results.get('average_moisture', 0) / 100, 1.0)
            texture_score = min(combined_results.get('average_texture', 0) / 100, 1.0)
            oiliness_penalty = min(combined_results.get('average_oiliness', 0) / 100, 1.0)
            pores_penalty = min(combined_results.get('average_pores', 0) / 100, 1.0)
            redness_penalty = min(combined_results.get('average_redness', 0) / 100, 1.0)
            
            # Calculate base score
            base_score = (moisture_score + texture_score + (1 - oiliness_penalty) + (1 - pores_penalty) + (1 - redness_penalty)) / 5
            
            # Apply concern penalties
            concern_penalty = 0
            conditions = combined_results.get('conditions', {})
            for condition, data in conditions.items():
                severity = data.get('max_severity', 0)
                prevalence = data.get('prevalence', 0)
                concern_penalty += (severity * prevalence) * 0.1  # Max 10% penalty per significant concern
            
            final_score = max(0, base_score - concern_penalty)
            return round(final_score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating overall score: {str(e)}")
            return 0.5  # Default neutral score

    def _generate_recommendations(self, analysis: Dict) -> Dict:
        """Generate personalized recommendations based on analysis results."""
        recommendations = {
            'daily_routine': [],
            'treatments': [],
            'lifestyle': [],
            'products': {
                'cleanser': None,
                'moisturizer': None,
                'sunscreen': "Use a broad-spectrum SPF 30+ sunscreen daily",
                'treatments': []
            },
            'priority_concerns': [],
            'next_steps': []
        }
        
        skin_type = analysis.get('skin_type', 'Normal').lower()
        moisture_level = analysis.get('average_moisture', 50)
        oiliness = analysis.get('average_oiliness', 50)
        conditions = analysis.get('conditions', {})
        
        # Handle case where skin_type might be 'unknown' or empty
        if skin_type in ['unknown', 'none', '']:
            skin_type = 'normal'
        
        # Basic daily routine based on skin type
        if 'dry' in skin_type or moisture_level < 40:
            recommendations['products']['cleanser'] = "Use a gentle, hydrating cleanser with ceramides"
            recommendations['products']['moisturizer'] = "Apply a rich, hydrating moisturizer with hyaluronic acid twice daily"
            recommendations['daily_routine'].append("Apply moisturizer while skin is still damp to lock in hydration")
        elif 'oily' in skin_type or oiliness > 70:
            recommendations['products']['cleanser'] = "Use a gentle foaming cleanser with salicylic acid"
            recommendations['products']['moisturizer'] = "Use an oil-free, non-comedogenic moisturizer"
            recommendations['daily_routine'].append("Cleanse twice daily but avoid over-washing")
        elif 'combination' in skin_type:
            recommendations['products']['cleanser'] = "Use a balanced, pH-neutral cleanser"
            recommendations['products']['moisturizer'] = "Use different moisturizers for T-zone and dry areas"
        else:
            recommendations['products']['cleanser'] = "Use a gentle, pH-balanced cleanser twice daily"
            recommendations['products']['moisturizer'] = "Apply a lightweight, non-comedogenic moisturizer"
        
        # Address specific conditions
        priority_concerns = []
        for condition, data in conditions.items():
            severity = data.get('max_severity', 0) if isinstance(data, dict) else 0.5
            prevalence = data.get('prevalence', 0) if isinstance(data, dict) else 0.5
            
            if severity > 0.5 and prevalence > 0.5:  # Significant concern
                priority_concerns.append(condition)
                
                if 'acne' in condition.lower():
                    recommendations['treatments'].append("Consider using a salicylic acid or benzoyl peroxide treatment")
                    recommendations['products']['treatments'].append("Spot treatment with 2% salicylic acid")
                elif 'wrinkle' in condition.lower() or 'aging' in condition.lower() or 'fine lines' in condition.lower():
                    recommendations['treatments'].append("Consider incorporating a retinol serum (start slowly)")
                    recommendations['products']['treatments'].append("Anti-aging serum with retinol or peptides")
                elif 'dark spot' in condition.lower() or 'hyperpigmentation' in condition.lower():
                    recommendations['treatments'].append("Use vitamin C serum in the morning")
                    recommendations['products']['treatments'].append("Vitamin C serum for hyperpigmentation")
                elif 'redness' in condition.lower() or 'irritation' in condition.lower():
                    recommendations['treatments'].append("Use products with niacinamide to reduce redness")
                    recommendations['lifestyle'].append("Identify and avoid potential triggers")
                elif 'pores' in condition.lower():
                    recommendations['treatments'].append("Use a BHA exfoliant 2-3 times per week")
                    recommendations['products']['treatments'].append("Pore-minimizing serum with niacinamide")
        
        recommendations['priority_concerns'] = priority_concerns[:3]  # Top 3 concerns
        
        # General lifestyle recommendations
        recommendations['lifestyle'].extend([
            "Stay hydrated by drinking 8+ glasses of water daily",
            "Get 7-8 hours of quality sleep each night",
            "Eat a balanced diet rich in antioxidants",
            "Manage stress through exercise or meditation"
        ])
        
        # Next steps based on analysis
        overall_score = analysis.get('overall_score', 0.5)
        if overall_score < 0.4:
            recommendations['next_steps'].append("Consider consulting a dermatologist for professional advice")
        elif overall_score < 0.7:
            recommendations['next_steps'].append("Focus on consistent daily skincare routine")
        else:
            recommendations['next_steps'].append("Maintain current routine and monitor for changes")
        
        if len(priority_concerns) > 2:
            recommendations['next_steps'].append("Address one concern at a time to avoid skin irritation")
        
        recommendations['next_steps'].append("Take progress photos and reassess skin in 4-6 weeks")
        
        # Ensure we always return a dictionary
        return recommendations

    async def analyze_image_only(self, image_data: bytes, angle: str) -> Dict[str, Any]:
        """
        Analyze an image without saving to Supabase (for batch processing).
        
        Args:
            image_data: Raw image bytes
            angle: The angle of the capture
            
        Returns:
            Dictionary with analysis results only
        """
        import time
        method_start = time.time()
        
        try:
            logger.info(f"🔬 [ANALYZE ONLY] Starting analysis-only for angle: {angle}")
            
            # Convert base64 to bytes if needed
            decode_start = time.time()
            try:
                if isinstance(image_data, str):
                    logger.debug(f"🔄 [ANALYZE ONLY] Converting base64 string to bytes for {angle}")
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                else:
                    image_bytes = image_data
                
                if not image_bytes:
                    raise ValueError("No image data provided")
                    
            except Exception as e:
                logger.error(f"❌ [ANALYZE ONLY] Error decoding image data for {angle}: {str(e)}")
                raise ValueError(f"Invalid image data: {str(e)}")
            
            decode_duration = time.time() - decode_start
            logger.info(f"✅ [ANALYZE ONLY] Decoded {len(image_bytes)} bytes for {angle} in {decode_duration:.3f}s")
            
            # Initialize variables to avoid UnboundLocalError
            convert_duration = 0.0
            analysis_duration = 0.0
            process_duration = 0.0
            temp_path = None
            
            try:
                # Convert to PIL Image and save temporarily for analysis
                convert_start = time.time()
                logger.debug(f"🖼️ [ANALYZE ONLY] Converting image for analysis: {angle}")
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                # Save to temporary file for analysis
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                temp_path = temp_file.name
                temp_file.close()
                
                image.save(temp_path, 'JPEG', quality=95)
                convert_duration = time.time() - convert_start
                logger.info(f"💾 [ANALYZE ONLY] Saved temporary image for {angle} to: {temp_path} ({convert_duration:.3f}s)")
                    
            except Exception as e:
                logger.error(f"💥 [ANALYZE ONLY] Error processing image for {angle}: {str(e)}")
                raise ValueError(f"Image processing failed: {str(e)}")
            
            try:
                # Perform analysis - THIS IS LIKELY THE BOTTLENECK
                analysis_start = time.time()
                logger.info(f"🧠 [ANALYZE ONLY] Starting CORE ANALYSIS for {angle} (this may take time...)")
                
                # Run the CPU-intensive analysis in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    analysis_result = await loop.run_in_executor(
                        pool,
                        lambda: self.analyzer.analyze_skin(temp_path)
                    )
                
                analysis_duration = time.time() - analysis_start
                logger.info(f"🧠 [ANALYZE ONLY] CORE ANALYSIS completed for {angle} in {analysis_duration:.3f}s")
                
                # Clean up temporary file
                cleanup_start = time.time()
                try:
                    os.unlink(temp_path)
                    logger.debug(f"🗑️ [ANALYZE ONLY] Cleaned up temp file for {angle}")
                except:
                    pass  # Ignore cleanup errors
                cleanup_duration = time.time() - cleanup_start
                
                # Handle case where no face is detected
                process_start = time.time()
                if not analysis_result or (isinstance(analysis_result, dict) and analysis_result.get('status') == 'error'):
                    # Create a basic analysis result for no face detected
                    logger.warning(f"⚠️ [ANALYZE ONLY] No face detected for {angle}, creating basic analysis result")
                    analysis = {
                        'status': 'no_face_detected',
                        'message': 'No face detected in the image. Please ensure good lighting and that your face is clearly visible.',
                        'skin_type': 'unknown',
                        'skin_tone': 'unknown',
                        'perceived_age': 0,
                        'eye_age': 0,
                        'concerns': {},
                        'moisture_level': 0.0,
                        'oiliness': 0.0,
                        'pores_visibility': 0.0,
                        'redness': 0.0,
                        'texture_score': 0.0,
                        'analysis_timestamp': datetime.now().isoformat(),
                        'recommendations': ['Please retake the photo with better lighting and face positioning']
                    }
                else:
                    # Convert SkinAnalysisResult to dictionary if it's not already a dict
                    logger.debug(f"🔄 [ANALYZE ONLY] Converting analysis result to dict for {angle}")
                    if hasattr(analysis_result, '__dict__'):
                        # It's a dataclass/object, convert to dict
                        analysis = {'status': 'success'}
                        for field_name, field_value in analysis_result.__dict__.items():
                            if hasattr(field_value, '__dict__'):  # Handle nested objects
                                analysis[field_name] = field_value.__dict__
                            elif hasattr(field_value, 'value'):  # Handle enums
                                analysis[field_name] = field_value.value
                            elif isinstance(field_value, datetime):  # Handle datetime
                                analysis[field_name] = field_value.isoformat()
                            else:
                                analysis[field_name] = field_value
                    else:
                        # It's already a dict
                        analysis = analysis_result
                        analysis['status'] = 'success'
                
                process_duration = time.time() - process_start
                logger.info(f"📋 [ANALYZE ONLY] Processed analysis result for {angle} in {process_duration:.3f}s")
                    
            except Exception as e:
                analysis_error_time = time.time()
                logger.error(f"💥 [ANALYZE ONLY] Analysis failed for {angle} after {analysis_error_time - analysis_start:.3f}s: {str(e)}")
                # Clean up temporary file on error
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
                # Create error analysis result instead of raising exception
                logger.warning(f"⚠️ [ANALYZE ONLY] Creating error result for {angle}")
                analysis = {
                    'status': 'error',
                    'message': f'Analysis failed: {str(e)}',
                    'skin_type': 'unknown',
                    'skin_tone': 'unknown',
                    'perceived_age': 0,
                    'eye_age': 0,
                    'concerns': {},
                    'moisture_level': 0.0,
                    'oiliness': 0.0,
                    'pores_visibility': 0.0,
                    'redness': 0.0,
                    'texture_score': 0.0,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'recommendations': ['Please retake the photo and ensure good image quality']
                }
            
            total_method_time = time.time() - method_start
            logger.info(f"✅ [ANALYZE ONLY] Successfully completed analysis-only for {angle} in {total_method_time:.3f}s")
            logger.info(f"⏱️ [ANALYZE ONLY] {angle} timing - Decode: {decode_duration:.3f}s, Convert: {convert_duration:.3f}s, Analysis: {analysis_duration:.3f}s, Process: {process_duration:.3f}s")
            
            return {
                'success': True,
                'angle': angle,
                'analysis': analysis,
                'timing': {
                    'total': total_method_time,
                    'decode': decode_duration,
                    'convert': convert_duration,
                    'core_analysis': analysis_duration,
                    'process_result': process_duration
                }
            }
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in analyze_image_only: {error_msg}", exc_info=True)
            return {
                'success': False,
                'error': error_msg,
                'angle': angle
            }
    
    async def capture_image(self, image_data: bytes, angle: str, client_id: str) -> Dict[str, Any]:
        """
        Process and store a captured image.
        
        Args:
            image_data: Base64 encoded image data or bytes
            angle: The angle of the capture
            client_id: Unique client identifier
            
        Returns:
            Dictionary with capture results
        """
        try:
            logger.info(f"Starting image processing for angle: {angle}")
            
            # Convert base64 to bytes if needed
            try:
                if isinstance(image_data, str):
                    logger.debug("Converting base64 string to bytes")
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                else:
                    image_bytes = image_data
                
                if not image_bytes:
                    raise ValueError("No image data provided")
                    
            except Exception as e:
                logger.error(f"Error decoding image data: {str(e)}")
                raise ValueError(f"Invalid image data: {str(e)}")
            
            try:
                # Convert to PIL Image and save temporarily for analysis
                logger.debug("Converting image for analysis")
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                # Save to temporary file for analysis
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                temp_path = temp_file.name
                temp_file.close()
                
                image.save(temp_path, 'JPEG', quality=95)
                logger.debug(f"Saved temporary image to: {temp_path}")
                    
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                raise ValueError(f"Image processing failed: {str(e)}")
            
            try:
                # Perform analysis
                logger.debug("Performing skin analysis")
                analysis_result = self.analyzer.analyze_skin(temp_path)
                
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass  # Ignore cleanup errors
                
                # Handle case where no face is detected
                if not analysis_result or (isinstance(analysis_result, dict) and analysis_result.get('status') == 'error'):
                    # Create a basic analysis result for no face detected
                    logger.warning("No face detected, creating basic analysis result")
                    analysis = {
                        'status': 'no_face_detected',
                        'message': 'No face detected in the image. Please ensure good lighting and that your face is clearly visible.',
                        'skin_type': 'unknown',
                        'skin_tone': 'unknown',
                        'perceived_age': 0,
                        'eye_age': 0,
                        'concerns': {},
                        'moisture_level': 0.0,
                        'oiliness': 0.0,
                        'pores_visibility': 0.0,
                        'redness': 0.0,
                        'texture_score': 0.0,
                        'analysis_timestamp': datetime.now().isoformat(),
                        'recommendations': ['Please retake the photo with better lighting and face positioning']
                    }
                else:
                    # Convert SkinAnalysisResult to dictionary if it's not already a dict
                    if hasattr(analysis_result, '__dict__'):
                        # It's a dataclass/object, convert to dict
                        analysis = {'status': 'success'}
                        for field_name, field_value in analysis_result.__dict__.items():
                            if hasattr(field_value, '__dict__'):  # Handle nested objects
                                analysis[field_name] = field_value.__dict__
                            elif hasattr(field_value, 'value'):  # Handle enums
                                analysis[field_name] = field_value.value
                            elif isinstance(field_value, datetime):  # Handle datetime
                                analysis[field_name] = field_value.isoformat()
                            else:
                                analysis[field_name] = field_value
                    else:
                        # It's already a dict
                        analysis = analysis_result
                        analysis['status'] = 'success'
                    
            except Exception as e:
                logger.error(f"Analysis failed: {str(e)}")
                # Clean up temporary file on error
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
                # Create error analysis result instead of raising exception
                logger.warning("Analysis failed, creating error result")
                analysis = {
                    'status': 'error',
                    'message': f'Analysis failed: {str(e)}',
                    'skin_type': 'unknown',
                    'skin_tone': 'unknown',
                    'perceived_age': 0,
                    'eye_age': 0,
                    'concerns': {},
                    'moisture_level': 0.0,
                    'oiliness': 0.0,
                    'pores_visibility': 0.0,
                    'redness': 0.0,
                    'texture_score': 0.0,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'recommendations': ['Please retake the photo and ensure good image quality']
                }
            
            try:
                # Save to Supabase
                logger.debug("Saving analysis to database")
                analysis_id = await self.save_analysis(
                    client_id=client_id,
                    angle=angle,
                    image_data=image_bytes,
                    analysis=analysis
                )
                
                if not analysis_id:
                    raise ValueError("Failed to save analysis to database")
                
                logger.info(f"Successfully processed image for angle: {angle}")
                return {
                    'success': True,
                    'analysis_id': analysis_id,
                    'angle': angle,
                    'analysis': analysis
                }
                
            except Exception as e:
                logger.error(f"Error saving analysis: {str(e)}")
                raise ValueError(f"Failed to save analysis: {str(e)}")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in capture_image: {error_msg}", exc_info=True)
            return {
                'success': False,
                'error': error_msg,
                'angle': angle
            }
    
    async def analyze_skin(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze skin from an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Perform analysis (CPU-bound operation, run in thread pool)
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(
                    pool,
                    lambda: self.analyzer.analyze_skin(image_rgb)
                )
            
            return {
                "status": "success",
                "analysis": result.dict() if hasattr(result, 'dict') else result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in skin analysis: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            
    async def analyze_multiple_angles(self, images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze skin from multiple angles in parallel.
        
        Args:
            images: Dictionary mapping angle names to image arrays
            
        Returns:
            Combined analysis results
        """
        try:
            # Process all images in parallel
            tasks = [self.analyze_skin(image) for angle, image in images.items()]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            valid_results = []
            for angle, result in zip(images.keys(), results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing {angle} angle: {str(result)}")
                elif result.get("status") == "success":
                    valid_results.append({"angle": angle, **result["analysis"]})
            
            if not valid_results:
                raise ValueError("No valid analysis results obtained from any angle")
                
            # Combine results using Gemini
            combined_analysis = await gemini_service.combine_analysis_results(valid_results)
            
            # Store all results in Supabase
            analysis_id = str(uuid.uuid4())
            storage_path = f"analyses/{analysis_id}"
            
            # Store individual results
            for i, result in enumerate(valid_results):
                angle = result.pop('angle')
                file_path = f"{storage_path}/{angle}.json"
                self._store_analysis_result(file_path, result)
            
            # Store combined analysis
            combined_path = f"{storage_path}/combined.json"
            self._store_analysis_result(combined_path, combined_analysis)
            
            return {
                "status": "success",
                "analysis_id": analysis_id,
                "individual_results": [r["analysis"] for r in valid_results],
                "combined_analysis": combined_analysis,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in multi-angle analysis: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _batch_update_records(self, batch_updates: List[tuple]) -> None:
        """Batch update multiple database records for better performance."""
        try:
            if not batch_updates:
                return
                
            logger.info(f"💾 [BATCH UPDATE] Performing batch update of {len(batch_updates)} records")
            
            # Execute all updates concurrently for maximum speed
            async def update_single_record(record_id, update_data):
                try:
                    if self.admin_supabase:
                        result = self.admin_supabase.table("skin_analyses").update(update_data).eq("id", record_id).execute()
                    else:
                        result = self.supabase.table("skin_analyses").update(update_data).eq("id", record_id).execute()
                        
                    if not result.data:
                        logger.warning(f"⚠️ [BATCH UPDATE] No data returned for update of record {record_id}")
                        
                    return True
                except Exception as e:
                    logger.error(f"❌ [BATCH UPDATE] Failed to update record {record_id}: {str(e)}")
                    return False
            
            # Execute all updates in parallel for maximum speed
            update_tasks = [update_single_record(record_id, update_data) for record_id, update_data in batch_updates]
            results = await asyncio.gather(*update_tasks, return_exceptions=True)
            
            successful_updates = sum(1 for result in results if result is True)
            logger.info(f"✅ [BATCH UPDATE] Completed {successful_updates}/{len(batch_updates)} record updates")
            
        except Exception as e:
            logger.error(f"💥 [BATCH UPDATE] Error in batch update: {str(e)}")

    def _combine_analysis_results_fast(self, angle_results: Dict) -> Dict:
        """Fast local analysis combination without external API calls."""
        combined = {
            "skin_type": "Normal",
            "skin_tone": "Unknown", 
            "perceived_age": 0,
            "eye_age": 0,
            "conditions": {},
            "concerns": [],
            "average_moisture": 0.0,
            "average_oiliness": 0.0,
            "average_pores": 0.0,
            "average_texture": 0.0,
            "average_redness": 0.0,
            "analyzed_angles": list(angle_results.keys()),
            "analysis_time": datetime.now().isoformat(),
            "recommendations": {},
            "overall_score": 0.0,
            "confidence_score": 0.85,  # High confidence for fast processing
            "processing_method": "fast_local"
        }
        
        # Filter valid results
        valid_results = []
        for angle, result in angle_results.items():
            if (isinstance(result, dict) and 
                not result.get("error") and 
                result.get("status") != "error"):
                
                # Ensure skin_type is a string
                if 'skin_type' in result and isinstance(result['skin_type'], dict):
                    result['skin_type'] = result['skin_type'].get('skin_type', 'Normal')
                    
                valid_results.append(result)
        
        if not valid_results:
            logger.warning("No valid analysis results to combine")
            return combined
        
        # Quick skin type determination
        skin_types = [r.get("skin_type", "Normal") for r in valid_results if r.get("skin_type")]
        if skin_types:
            # Use most common skin type
            combined["skin_type"] = max(set(skin_types), key=skin_types.count)
        
        # Fast average calculations
        def safe_avg(key, default=0.0):
            values = [float(r.get(key, 0)) for r in valid_results if r.get(key) is not None]
            return round(sum(values) / len(values), 2) if values else default
        
        combined["average_moisture"] = safe_avg("moisture_level")
        combined["average_oiliness"] = safe_avg("oiliness") 
        combined["average_pores"] = safe_avg("pores_visibility")
        combined["average_texture"] = safe_avg("texture_score")
        combined["average_redness"] = safe_avg("redness")
        combined["perceived_age"] = int(safe_avg("perceived_age"))
        combined["eye_age"] = int(safe_avg("eye_age"))
        
        # Calculate overall score
        scores = [
            combined["average_moisture"],
            100 - combined["average_oiliness"],  # Invert oiliness
            100 - combined["average_pores"],     # Invert pores
            combined["average_texture"],
            100 - combined["average_redness"]    # Invert redness
        ]
        combined["overall_score"] = round(sum(scores) / len(scores), 1)
        
        # Simple recommendations based on averages
        recs = []
        if combined["average_moisture"] < 40:
            recs.append("Use a hydrating moisturizer to improve skin moisture")
        if combined["average_oiliness"] > 60:
            recs.append("Consider oil-control products for balanced skin")
        if combined["average_pores"] > 50:
            recs.append("Use pore-minimizing treatments")
        if not recs:
            recs.append("Your skin appears healthy - maintain your current routine")
            
        combined["recommendations"] = {"general": recs}
        
        logger.info(f"Fast analysis combination completed for {len(valid_results)} results")
        return combined

    def _store_analysis_result(self, path: str, data: Dict[str, Any]) -> None:
        """Store analysis result in Supabase."""
        try:
            # Convert data to JSON string
            import json
            data_str = json.dumps(data, indent=2)
            
            # Upload to Supabase storage
            result = self.supabase.storage.from_(self.bucket_name).upload(
                path=path,
                file=data_str,
                file_options={"content-type": "application/json"}
            )
            
            if hasattr(result, 'error') and result.error:
                logger.error(f"Failed to store analysis result: {result.error}")
                
        except Exception as e:
            logger.error(f"Error storing analysis result: {str(e)}")

    def generate_final_report(self, analysis_results: List[Dict]) -> Dict:
        """
        Generate a final analysis report from multiple angle analyses.
        
        Args:
            analysis_results: List of analysis results from different angles
            
        Returns:
            Combined analysis report
        """
        if not analysis_results:
            return {"error": "No analysis results provided"}
            
        # Combine results (simplified - in reality, you'd want more sophisticated logic)
        combined = {
            "analysis_id": f"analysis_{uuid.uuid4().hex}",
            "timestamp": time.time(),
            "angles_analyzed": [r.get('angle', 'unknown') for r in analysis_results],
            "overall_analysis": {},
            "recommendations": [],
            "severity": "mild"  # Default
        }
        
        # Aggregate analysis (simplified)
        for result in analysis_results:
            if 'analysis' in result:
                for k, v in result['analysis'].items():
                    if k not in combined['overall_analysis']:
                        combined['overall_analysis'][k] = v
        
        # Generate recommendations
        combined['recommendations'] = self._generate_recommendations(combined['overall_analysis'])
        
        return combined
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Example recommendations (customize based on your analysis)
        if analysis.get('acne_severity') in ['moderate', 'severe']:
            recommendations.append("Consider using a salicylic acid cleanser.")
            
        if analysis.get('wrinkles_present', False):
            recommendations.append("Use sunscreen daily to prevent further sun damage.")
            
        if analysis.get('hydration_level', 0) < 0.5:
            recommendations.append("Your skin appears dry. Consider using a more hydrating moisturizer.")
        
        if not recommendations:
            recommendations.append("Your skin appears healthy. Maintain your current routine.")
            
        return recommendations 