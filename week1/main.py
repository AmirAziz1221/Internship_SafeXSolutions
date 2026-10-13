"""
main.py
-------
FastAPI backend for the TestTrick face-presence monitor (hybrid version:
YOLO-face for detection + MediaPipe Face Mesh for the full/partial check).

Run with:
    uvicorn main:app --reload --port 8000

Endpoints:
    WS  /ws/proctor/{session_id}  - real-time frame stream + warnings
    GET /report/{session_id}      - end-of-test integrity report
    GET /                         - health check
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from face_monitor import FaceMonitor

app = FastAPI(title="TestTrick Face Presence Monitor (Hybrid: YOLO + Face Mesh)")

# Allow the demo frontend (opened as a local file or on another port) to connect.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Configurable thresholds (seconds) ----
ABSENT_WARNING_SECONDS = 5.0    # no face at all
PARTIAL_WARNING_SECONDS = 2.5   # face detected but part of it hidden/out of frame

# Path to the YOLO-face weights - see README for where to download them.
monitor = FaceMonitor(yolo_weights_path="models/yolo26s-face.pt")


@dataclass
class SessionState:
    absent_since: float = None
    partial_since: float = None
    warned_absent: bool = False
    warned_partial: bool = False
    events: List[dict] = field(default_factory=list)


sessions: Dict[str, SessionState] = {}


def log_event(state: SessionState, event_type: str, extra: dict = None):
    entry = {"type": event_type, "timestamp": time.time()}
    if extra:
        entry.update(extra)
    state.events.append(entry)


@app.websocket("/ws/proctor/{session_id}")
async def proctor_socket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    state = sessions.setdefault(session_id, SessionState())

    try:
        while True:
            data = await websocket.receive_json()
            frame_b64 = data.get("frame")
            if not frame_b64:
                continue

            try:
                frame = FaceMonitor.decode_base64_frame(frame_b64)
                result = monitor.analyze(frame)
            except Exception:
                traceback.print_exc()
                await websocket.send_json({"status": "error"})
                continue
            now = time.time()

            response = {"status": result.status, "face_count": result.face_count}

            if result.status == "full":
                # Fully compliant frame -> reset everything
                state.absent_since = None
                state.partial_since = None
                state.warned_absent = False
                state.warned_partial = False

            elif result.status == "absent":
                state.partial_since = None
                if state.absent_since is None:
                    state.absent_since = now
                elapsed = now - state.absent_since
                response["absent_seconds"] = round(elapsed, 1)
                if elapsed >= ABSENT_WARNING_SECONDS and not state.warned_absent:
                    response["warning"] = "FACE_NOT_VISIBLE"
                    response["message"] = "Your face is not visible. Please stay in front of the camera."
                    state.warned_absent = True
                    log_event(state, "warning_absent", {"duration": round(elapsed, 1)})

            elif result.status == "partial":
                state.absent_since = None
                if state.partial_since is None:
                    state.partial_since = now
                elapsed = now - state.partial_since
                response["partial_seconds"] = round(elapsed, 1)
                response["missing_regions"] = result.missing_regions
                if elapsed >= PARTIAL_WARNING_SECONDS and not state.warned_partial:
                    response["warning"] = "FACE_PARTIALLY_HIDDEN"
                    response["message"] = "Please show your full face to the camera."
                    state.warned_partial = True
                    log_event(state, "warning_partial", {
                        "duration": round(elapsed, 1),
                        "missing_regions": result.missing_regions,
                    })

            if result.face_count > 1:
                response["multi_face_warning"] = True
                log_event(state, "multi_face_detected", {"face_count": result.face_count})

            await websocket.send_json(response)

    except WebSocketDisconnect:
        pass


@app.get("/report/{session_id}")
async def get_report(session_id: str):
    state = sessions.get(session_id)
    if not state:
        return {"error": "session not found"}

    absent_events = [e for e in state.events if e["type"] == "warning_absent"]
    partial_events = [e for e in state.events if e["type"] == "warning_partial"]
    multi_face_events = [e for e in state.events if e["type"] == "multi_face_detected"]

    return {
        "session_id": session_id,
        "total_warnings": len(absent_events) + len(partial_events),
        "absent_warnings": len(absent_events),
        "partial_face_warnings": len(partial_events),
        "multi_face_flags": len(multi_face_events),
        "events": state.events,
    }


@app.get("/")
async def root():
    return {"status": "TestTrick face-presence monitor (hybrid) is running"}
