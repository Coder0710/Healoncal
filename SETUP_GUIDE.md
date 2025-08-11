# HealOnCal Setup Guide 🚀

## ✅ Prerequisites Completed
- [x] Python dependencies installed
- [x] Virtual environment activated
- [x] Code issues fixed (PIL import, method calls, data conversion)

## 🔧 Final Setup Steps

### 1. **Supabase Database Setup**

#### Create the Database Table
1. Go to your Supabase dashboard
2. Navigate to **SQL Editor**
3. Copy and paste the contents of `supabase_setup.sql`
4. Click **Run** to execute the SQL

#### Create Storage Bucket
1. Go to **Storage** in your Supabase dashboard
2. Click **Create new bucket**
3. Name it: `skin-scans`
4. Enable **Public access**
5. Click **Create bucket**

### 2. **Environment Configuration**

Create a `.env` file in your project root:

```env
# Supabase Configuration - REPLACE WITH YOUR ACTUAL VALUES
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_KEY=your-anon-public-api-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Application Configuration
DEBUG=True
HOST=0.0.0.0
PORT=8000
```

**Where to find your Supabase credentials:**
- Go to **Settings > API** in your Supabase dashboard
- `SUPABASE_URL`: Project URL
- `SUPABASE_KEY`: anon public key  
- `SUPABASE_SERVICE_ROLE_KEY`: service_role secret key

### 3. **Running the Application**

```powershell
# Activate virtual environment
.\.venv\Scripts\activate.ps1

# Start the application
python app/main.py
```

### 4. **Access Your Application**

- **Main Application**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs  
- **Live Capture Interface**: http://localhost:8000/static/live_capture.html

## 📱 Using the Application

### Face Detection Tips
The AI models work best when:
- ✅ Face is well-lit and clearly visible
- ✅ Face takes up 30-50% of the image
- ✅ No heavy makeup or filters
- ✅ Direct, frontal view for best results
- ✅ Clean background (not required but helps)

### Multi-Angle Capture
The app captures 3 angles:
1. **Front**: Direct frontal view
2. **Left**: Left profile (90-degree turn)  
3. **Right**: Right profile (90-degree turn)

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. "No face detected"
- **Solution**: Improve lighting and ensure face is clearly visible
- **Fallback**: The app now gracefully handles this and still saves the image

#### 2. Supabase connection errors
- **Check**: Environment variables are set correctly
- **Check**: Supabase URL and keys are valid
- **Check**: Database table exists (run the SQL setup)

#### 3. Storage upload fails
- **Check**: `skin-scans` bucket exists and has public access
- **Check**: Service role key has storage permissions

#### 4. WebSocket connection issues
- **Solution**: Refresh the page and ensure good internet connection

## 🎯 Application Features

### ✅ Working Features
- Real-time camera capture via WebSocket
- Multi-angle skin scanning (front, left, right)
- AI-powered skin analysis using:
  - ResNet-50 for face analysis
  - Vision Transformer (ViT) for skin type classification  
  - DeiT for skin concern detection
- Cloud storage integration with Supabase
- Graceful error handling for failed face detection
- Automatic image uploads and database storage

### 🔬 Analysis Capabilities
- **Skin Type Detection**: Normal, Dry, Oily, Combination, Sensitive
- **Age Estimation**: Perceived age and eye age analysis
- **Skin Tone Classification**: Automated skin tone categorization
- **Concern Detection**: Acne, pigmentation, pores, wrinkles, etc.
- **Texture Analysis**: Moisture level, oiliness, pore visibility

## 📊 Database Schema

The `skin_analyses` table stores:
```sql
- id (UUID): Unique identifier
- client_id (TEXT): Client session identifier  
- angle (TEXT): Capture angle (front/left/right)
- image_url (TEXT): Supabase storage URL
- analysis (JSONB): Complete analysis results
- created_at (TIMESTAMPTZ): Timestamp
- updated_at (TIMESTAMPTZ): Last updated
```

## 🚀 Next Steps

1. **Run the SQL setup** in Supabase
2. **Add your environment variables**
3. **Start the application**
4. **Test with good lighting and clear face images**

## 💡 Tips for Best Results

- Use consistent lighting across all three angles
- Keep the same distance from camera for all shots
- Allow the AI models time to load on first startup (2-3 minutes)
- Check the browser console for any JavaScript errors during capture

Your HealOnCal skin analysis application is now ready! 🎉 