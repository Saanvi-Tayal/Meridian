"""
REST Endpoints for Dead Reckoning Navigation Server.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from src.server.session_manager import SessionManager

router = APIRouter(prefix="/api")
session_manager = SessionManager()


class StartSessionRequest(BaseModel):
    trip_id: str


class ToggleBlackoutRequest(BaseModel):
    forced: Optional[bool] = None


class StepRequest(BaseModel):
    step_size: int = 10  # 10 frames = 0.1s at 100Hz


@router.get("/trips")
def list_trips():
    """List all available trips in the dataset."""
    trips = session_manager.list_available_trips()
    return {"trips": trips, "count": len(trips)}


@router.post("/session/start")
def start_session(req: StartSessionRequest):
    """Start a simulation session for the chosen trip."""
    try:
        res = session_manager.start_session(req.trip_id)
        return {"status": "success", "session": res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/engine/step")
def step_simulation(req: StepRequest = StepRequest()):
    """Advance simulation by step_size frames and return fused telemetry."""
    try:
        sess = session_manager.get_session()
        state = sess.step(step_size=req.step_size)
        return state
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/engine/toggle_blackout")
def toggle_blackout(req: ToggleBlackoutRequest = ToggleBlackoutRequest()):
    """Toggle or set GPS blackout state."""
    try:
        sess = session_manager.get_session()
        is_blackout = sess.toggle_blackout(forced=req.forced)
        return {"status": "success", "blackout_active": is_blackout}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/engine/reset")
def reset_session():
    """Reset the current session back to frame 0."""
    try:
        sess = session_manager.get_session()
        sess.reset()
        return {"status": "success", "frame": 0}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
