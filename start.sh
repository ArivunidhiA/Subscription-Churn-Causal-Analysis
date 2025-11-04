#!/bin/bash

# Ensure streamlit_app.py exists
python -c "from app import create_streamlit_app; create_streamlit_app()" 2>/dev/null || true

# Start FastAPI server in background
python -m uvicorn app:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Wait for FastAPI to start
sleep 3

# Start Streamlit in foreground
python -m streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# If Streamlit exits, kill FastAPI
kill $FASTAPI_PID 2>/dev/null || true
wait

