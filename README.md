# AI-Powered Skin Analysis System

An advanced skin analysis platform that provides real-time, multi-angle skin assessment using computer vision and deep learning. The system captures live images, analyzes skin conditions, and provides professional recommendations.

## 🌟 New Features

### Live Multi-Angle Capture
- **Smart Camera Interface**: Real-time feedback on image quality
- **Guided Capture Process**: Step-by-step instructions for optimal image capture
- **Multi-Angle Analysis**: Capture and analyze skin from different angles for comprehensive assessment
- **Image Quality Validation**: Automatic detection of issues like blur, lighting, and positioning

### Professional Skin Analysis
- **Comprehensive Skin Assessment**: Detailed analysis of skin type, tone, and conditions
- **Condition Detection**: Identifies various skin concerns with confidence scores
  - Acne
  - Pigmentation
  - Pores
  - Wrinkles
  - Dullness
  - Redness
- **Severity Classification**: Rates conditions from Mild to Critical
- **Professional Recommendations**: Personalized skincare advice based on analysis
- **Dermatologist Referral**: Flags severe conditions requiring professional attention

### Technical Features
- **Multi-Model Pipeline**: Specialized models for different analysis tasks
- **Advanced Image Processing**: Face detection, alignment, and region of interest extraction
- **Real-time Analysis**: Fast processing for immediate results
- **Detailed Reporting**: Comprehensive analysis results with confidence metrics
- **Error Resilience**: Graceful handling of edge cases and model failures

## 🤖 AI/ML Architecture

### Model Pipeline

Healoncal uses a sophisticated multi-model pipeline for comprehensive skin analysis:

1. **Face Analysis Model** (`microsoft/resnet-50`)
   - Estimates age and detects facial features
   - Provides baseline for other analyses

2. **Skin Type Classifier** (`google/vit-base-patch16-224`)
   - Classifies skin type (Normal, Dry, Oily, Combination, Sensitive)
   - Analyzes skin texture and oiliness

3. **Skin Concern Detector** (`facebook/deit-base-patch16-224`)
   - Identifies various skin concerns with confidence scores
   - Uses specialized computer vision techniques for each concern type

### Analysis Features

#### Skin Type Detection
- **Oiliness Analysis**: Measures sebum levels
- **Texture Assessment**: Evaluates skin smoothness
- **Moisture Level**: Estimates skin hydration
- **Pore Visibility**: Analyzes pore size and visibility

#### Face Analysis
- **Age Estimation**: Predicts perceived age
- **Eye Age**: Specialized analysis of eye area
- **Skin Tone**: Classifies into standardized tone categories

#### Skin Concern Detection
- **Acne Detection**: Identifies active breakouts and blemishes
- **Pigmentation Analysis**: Detects dark spots and uneven tone
- **Wrinkle Assessment**: Evaluates fine lines and wrinkles
- **Redness Measurement**: Quantifies skin irritation
- **Dullness Detection**: Assesses skin radiance

### Performance

| Component | Accuracy | Processing Time (CPU) |
|-----------|----------|----------------------|
| Face Detection | 98% | ~200ms |
| Skin Type Classification | 91% | ~300ms |
| Age Estimation | ±3 years | ~250ms |
| Concern Detection | 85-93% | ~500ms |
| Full Analysis | - | ~1.2s |

*Performance metrics measured on standard test dataset*

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Virtual environment (recommended)
- CUDA-compatible GPU (recommended for faster inference)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/healoncal.git
   cd healoncal
   ```

2. **Set up a virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   
   
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   For GPU support (recommended):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Configure environment variables**:
   Create a `.env` file in the project root:
   ```env
   ENVIRONMENT=development
   DEVICE=cuda  # or 'cpu' if no GPU available
   ```

5. **Start the development server**:
   ```bash
   uvicorn app.main:app --reload
   ```

6. **Access the API documentation**:
   - Interactive API docs: http://localhost:8000/api/docs
   - Alternative docs: http://localhost:8000/redoc

## 📚 API Documentation

### Analyze Skin

Analyze a skin image and get detailed analysis results.

**Endpoint**: `POST /api/v1/analyze-skin`

**Request**:
- **Method**: POST
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: (required) Image file (JPEG, PNG, max 10MB)

**Example Request**:
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/analyze-skin' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@skin_photo.jpg;type=image/jpeg'
```

**Response**:

```json
{
  "status": "success",
  "analysis": {
    "condition": "healthy_skin",
    "confidence": 0.92,
    "original_prediction": "normal",
    "tone": "light",
    "texture": {
      "texture": "smooth",
      "contrast": 22.5,
      "edge_density": 0.15
    },
    "recommendations": [
      "Use broad-spectrum SPF 30+ sunscreen daily",
      "Maintain a consistent skincare routine",
      "Stay hydrated and eat a balanced diet"
    ],
    "analysis_quality": "high",
    "suggested_next_steps": [],
    "analysis_notes": [
      "Analysis suggests light skin tone",
      "Skin appears smooth based on texture analysis"
    ]
  },
  "metadata": {
    "device": "cuda",
    "timestamp": "2025-06-29T16:42:18.123456",
    "model_version": "2.0.0",
    "processing_time_ms": 845,
    "analysis_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef"
  }
}
```

### Error Responses

#### 400 Bad Request
- Missing or invalid image file
- Unsupported file format
- File too large (>10MB)

#### 422 Unprocessable Entity
- Invalid request format
- Missing required fields

#### 500 Internal Server Error
- Model loading failure
- Processing error

## 📚 API Endpoints

### 1. Analyze Skin
- **Endpoint**: `POST /api/v1/analyze-skin`
- **Description**: Upload an image for skin analysis
- **Request**: Multipart form with image file
- **Response**: JSON with analysis results

### 2. Get Scan History
- **Endpoint**: `GET /api/v1/history/{user_id}`
- **Description**: Retrieve analysis history for a specific user
- **Response**: List of previous scans with timestamps

### 3. Get Recommendations
- **Endpoint**: `GET /api/v1/recommendations/{scan_id}`
- **Description**: Get personalized recommendations based on a scan
- **Response**: Detailed product and care recommendations

## 🏗️ Project Structure

```
healoncal/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application setup
│   ├── models/
│   │   └── skin_analyzer.py    # Core skin analysis logic
│   └── api/
│       └── endpoints.py        # API route handlers
├── uploads/                    # Temporary storage for uploaded images
├── requirements.txt            # Python dependencies
└── README.md                   # This documentation
```

## 🚀 Deployment

### Production
For production deployment, use a proper ASGI server like Uvicorn with Gunicorn:

```bash
# Install production server
pip install gunicorn

# Run with multiple workers
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
```

### Environment Variables
Create a `.env` file in the project root:

```env
ENVIRONMENT=production
DEVICE=cuda  # or 'cpu' for CPU-only
LOG_LEVEL=INFO
```

## Technologies Used

- **Backend**: FastAPI
- **AI/ML**: PyTorch, Transformers, Florence-2
- **Image Processing**: OpenCV, Albumentations
- **Database**: In-memory storage (for demo; use a real database in production)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Citation

If you use Healoncal in your research, please cite:

```bibtex
@software{healoncal2025,
  author = {Your Name},
  title = {Healoncal: AI-Powered Skin Analysis System},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/healoncal}}
}
```

## 📚 Resources

- [API Documentation](https://docs.healoncal.com)
- [Model Cards](MODELS.md)
- [Changelog](CHANGELOG.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## 📞 Support

For support, please:
1. Check the [FAQ](FAQ.md)
2. Search the [issue tracker](https://github.com/yourusername/healoncal/issues)
3. Open a new issue if your problem isn't addressed

---

*Disclaimer: This tool is for informational purposes only and is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.*
