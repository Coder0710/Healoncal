# Skin Analysis API Documentation

## Base URL
`https://your-render-app-url.onrender.com`

## Authentication
All endpoints require an API key in the header:
```
X-API-Key: your_api_key_here
```

## Endpoints

### 1. Health Check
Check if the API is running.

**Endpoint:** `GET /api/health`

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "timestamp": "2023-08-12T16:47:00Z"
}
```

### 2. Analyze Skin
Upload an image for skin analysis.

**Endpoint:** `POST /api/analyze`

**Request Headers:**
- `Content-Type: multipart/form-data`
- `X-API-Key: your_api_key_here`

**Request Body (form-data):**
- `file`: Image file (required)
- `angle`: String indicating the angle of the image (e.g., "front", "left", "right") (required)
- `client_id`: Unique client identifier (required)

**Response (Success - 200 OK):**
```json
{
  "status": "success",
  "analysis_id": "unique_analysis_id",
  "angle": "front",
  "analysis": {
    "skin_type": "oily",
    "conditions": [
      {
        "name": "acne",
        "severity": "moderate",
        "confidence": 0.87
      }
    ],
    "recommendations": ["Use oil-free moisturizer", "Consider seeing a dermatologist"]
  }
}
```

### 3. Get Capture Status
Get the status of a capture session.

**Endpoint:** `GET /api/capture/status/{client_id}`

**Response (Success - 200 OK):**
```json
{
  "client_id": "client123",
  "status": "completed",
  "captures": ["front", "left", "right"],
  "analysis_complete": true,
  "analysis_result": {
    "overall_skin_condition": "good",
    "recommendations": ["Use sunscreen daily", "Drink more water"]
  }
}
```

## Error Responses

### 400 Bad Request
```json
{
  "error": "Validation Error",
  "message": "Missing required field: angle"
}
```

### 401 Unauthorized
```json
{
  "error": "Unauthorized",
  "message": "Invalid or missing API key"
}
```

### 404 Not Found
```json
{
  "error": "Not Found",
  "message": "The requested resource was not found"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal Server Error",
  "message": "An unexpected error occurred"
}
```

## Rate Limiting
- 100 requests per minute per IP address
- 1000 requests per day per API key

## Best Practices
1. Always check the response status code before processing the response
2. Implement proper error handling for API failures
3. Cache responses when possible to reduce API calls
4. Use the `client_id` to track user sessions across multiple requests
5. Handle image compression on the client side before uploading (recommended max size: 2MB)
