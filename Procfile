web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT app.main:app
release: python -c "from app.core.initial_data import init_db; init_db()"
