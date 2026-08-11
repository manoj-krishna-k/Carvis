"""
CARVIS Phase 2 — intruder alert flow.

Mount this router in your existing main.py:

    from alerts import router as alerts_router
    app.include_router(alerts_router)

Single-owner system: no auth, no multi-user handling. Device token and
phone number are just stored as config/state — this matches your setup
(no login system, phone-number stored directly in backend).
"""

import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from fcm import send_intruder_alert

router = APIRouter(prefix="/carvis", tags=["intruder-alerts"])

# --- In-memory state (swap for SQLite/JSON file if you want persistence
#     across backend restarts — not required for a demo/hackathon build) ---

_device_token: Optional[str] = None
_owner_phone_number: Optional[str] = None

_alerts: dict[str, dict] = {}
_latest_alert_id: Optional[str] = None
_stats = {"total_alerts": 0, "allowed": 0, "denied": 0, "pending": 0}


# ---------- Schemas ----------

class DeviceRegistration(BaseModel):
    fcm_token: str
    phone_number: str


class Decision(BaseModel):
    decision: str  # "allow" or "deny"


# ---------- Device registration (called once from the phone app) ----------

@router.post("/device/register")
def register_device(payload: DeviceRegistration):
    global _device_token, _owner_phone_number
    _device_token = payload.fcm_token
    _owner_phone_number = payload.phone_number
    return {"status": "registered"}


# ---------- Called internally by your ML pipeline when it flags an anomaly ----------

def trigger_intruder_alert() -> str:
    """
    Call this from wherever your model currently produces an anomaly
    verdict (e.g. right after Isolation Forest/XGBoost flags
    'not owner'). Creates a pending alert and pushes to the phone.
    """
    global _latest_alert_id

    if not _device_token:
        raise RuntimeError(
            "No device registered yet — open the app once so it can "
            "register its FCM token before alerts can be sent."
        )

    alert_id = str(uuid.uuid4())
    _alerts[alert_id] = {
        "id": alert_id,
        "status": "pending",  # pending | allowed | denied
        "created_at": time.time(),
    }
    _latest_alert_id = alert_id
    _stats["total_alerts"] += 1
    _stats["pending"] += 1

    send_intruder_alert(_device_token, alert_id)
    return alert_id


# ---------- Called from the phone app after the owner taps Allow/Deny ----------

@router.post("/alert/{alert_id}/decision")
def submit_decision(alert_id: str, payload: Decision):
    if alert_id not in _alerts:
        raise HTTPException(status_code=404, detail="Unknown alert_id")

    if payload.decision not in ("allow", "deny"):
        raise HTTPException(status_code=400, detail="decision must be 'allow' or 'deny'")

    # Note: "allow" does NOT retrain the model or mark this pattern as the
    # owner's own — it just clears this specific alert, per design.
    _alerts[alert_id]["status"] = "allowed" if payload.decision == "allow" else "denied"
    _alerts[alert_id]["decided_at"] = time.time()
    _stats["pending"] = max(0, _stats["pending"] - 1)
    if payload.decision == "allow":
        _stats["allowed"] += 1
    else:
        _stats["denied"] += 1
    return {"status": "ok", "alert": _alerts[alert_id]}


# ---------- Polled by the website AND the app's dashboard screen ----------

@router.get("/stats")
def get_stats():
    return _stats


# ---------- Polled by the website to reflect current status ----------

@router.get("/alert/latest")
def get_latest_alert():
    if not _latest_alert_id:
        return {"status": "none"}
    return _alerts[_latest_alert_id]
