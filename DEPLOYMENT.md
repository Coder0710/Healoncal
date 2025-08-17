# Deployment Guide for Skin Analysis API

This guide provides step-by-step instructions for deploying the Skin Analysis API on Render.

## Prerequisites

1. A Render.com account
2. GitHub/GitLab/Bitbucket account (for repository connection)
3. Required API keys and credentials (Supabase, Gemini, etc.)

## Deployment Steps

### 1. Fork the Repository

Fork the repository to your GitHub/GitLab/Bitbucket account.

### 2. Set Up on Render

1. Log in to your [Render Dashboard](https://dashboard.render.com/)
2. Click "New" and select "Web Service"
3. Connect your repository (GitHub/GitLab/Bitbucket)
4. Select the forked repository

### 3. Configure Environment Variables

Add the following required environment variables in the Render dashboard:

```
# Application
ENVIRONMENT=production
DEBUG=False
SECRET_KEY=your-secret-key-here
API_PREFIX=/api

# Supabase
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-key
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key

# Gemini AI
GEMINI_API_KEY=your-gemini-api-key

# CORS (adjust as needed)
BACKEND_CORS_ORIGINS='["https://your-frontend-domain.com"]'
```

### 4. Configure Build & Deploy

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT app.main:app`
- **Python Version**: 3.10.13 (as specified in runtime.txt)

### 5. Set Up Custom Domain (Optional)

1. Go to your Render service
2. Click on "Settings" tab
3. Under "Custom Domains", add your domain
4. Follow the DNS configuration instructions

### 6. Deploy

Click "Save" and Render will automatically start the deployment process.

## Post-Deployment

1. **Verify Health Check**: Visit `https://your-render-url/api/health` to ensure the API is running
2. **Test Endpoints**: Use the API documentation at `https://your-render-url/docs`
3. **Set Up Monitoring**: Consider adding monitoring (e.g., Sentry)

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `ENVIRONMENT` | Yes | Runtime environment (production/staging/development) |
| `DEBUG` | No | Debug mode (False in production) |
| `SECRET_KEY` | Yes | Secret key for cryptographic operations |
| `API_PREFIX` | No | API prefix (default: /api) |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_KEY` | Yes | Supabase API key |
| `SUPABASE_SERVICE_ROLE_KEY` | No | Supabase service role key |
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `BACKEND_CORS_ORIGINS` | No | Allowed CORS origins (JSON array) |

## Troubleshooting

- **Deployment Fails**: Check the build logs in the Render dashboard
- **API Not Responding**: Verify the PORT environment variable is set correctly
- **Database Connection Issues**: Check Supabase credentials and network access

## Maintenance

- Regularly update dependencies
- Monitor resource usage in the Render dashboard
- Set up proper backup strategies for your database

## Security Considerations

- Never commit sensitive information to version control
- Use environment variables for all secrets
- Enable HTTPS and enforce it in production
- Set appropriate CORS policies
- Regularly rotate API keys and secrets
