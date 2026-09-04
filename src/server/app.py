"""
FastAPI Server Application for Smartphone Dead Reckoning.
"""
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.server.routes import router as api_router

app = FastAPI(
    title="Intelligent Smartphone Dead Reckoning Navigation",
    description="Multi-Sensor Fusion, AI SpeedNet, ES-EKF, and HMM Lane Snapping Navigation System for GNSS-Denied Environments.",
    version="1.0.0"
)

# Enable CORS for local development and mobile devices on LAN
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(api_router)

# Mount static files
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Dead Reckoning Server Running. static/index.html not found."}
