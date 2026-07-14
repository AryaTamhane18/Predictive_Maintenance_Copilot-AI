# api.py
# FastAPI backend
# Receives machine_id + file_index + optional image from dashboard
# Runs the pipeline and returns FaultReport

import os
import sys
import shutil
import uvicorn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pipeline import run_pipeline
from tools.history import log_fault
from tools.anomaly import get_machine_info, get_sensor_files, MACHINES

app = FastAPI(
    title       = "Predictive Maintenance Copilot API",
    description = "AI-powered bearing fault detection and diagnosis",
    version     = "3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"]
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Health check ──────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":  "running",
        "message": "Predictive Maintenance Copilot API v3.0"
    }


# ── Machine info endpoint — used by dashboard for dropdown + slider ───
@app.get("/machines")
def get_machines():
    """
    Returns all available machines with their file counts.
    Dashboard uses this to populate the dropdown and slider.
    """
    result = {}
    for machine_id, info in MACHINES.items():
        files = get_sensor_files(machine_id)
        result[machine_id] = {
            "label":       info["label"],
            "description": info["failed"],
            "total_files": len(files),
            "first_file":  os.path.basename(files[0])  if files else "",
            "last_file":   os.path.basename(files[-1]) if files else ""
        }
    return result


# ── Main analysis endpoint ────────────────────────────────────────────
@app.post("/analyse-sensor")
async def analyse_sensor(
    machine_id:  str           = Form(...),
    file_index:  int           = Form(...),
    asset_id:    str           = Form(default="BEARING_01"),
    image_file:  Optional[UploadFile] = File(default=None)
):
    """
    Main endpoint — runs the full 3-agent pipeline.

    machine_id:  "bearing_1" / "bearing_2" / "bearing_4"
    file_index:  which file to analyse (from slider, 1-based)
    asset_id:    engineer-facing asset name
    image_file:  optional machine photo for LLaVA visual inspection
    """
    print(f"\n{'='*50}")
    print(f"  Analysis request")
    print(f"  Machine:    {machine_id}")
    print(f"  File index: {file_index}")
    print(f"  Asset ID:   {asset_id}")
    print(f"{'='*50}")

    # validate machine_id
    if machine_id not in MACHINES:
        return {
            "status":  "error",
            "message": f"Unknown machine: {machine_id}. "
                       f"Valid: {list(MACHINES.keys())}"
        }

    # handle optional image upload
    image_path = None
    if image_file and image_file.filename:
        image_path = os.path.join(UPLOAD_DIR, image_file.filename)
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image_file.file, f)
        print(f"  Image: {image_file.filename}")
    else:
        print("  Image: none")

    # run full pipeline
    try:
        result = run_pipeline(
            machine_id = machine_id,
            file_index = file_index,
            asset_id   = asset_id,
            image_path = image_path
        )
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        # clean up uploaded image
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

    fault_report = result.get("fault_report")

    if not fault_report:
        return {
            "status":       "normal",
            "machine_id":   machine_id,
            "file_index":   file_index,
            "message":      "No fault detected — machine operating normally.",
            "fault_report": None
        }

    # add machine label to report for dashboard display
    fault_report["machine_label"] = MACHINES[machine_id]["label"]

    return {
        "status":       "anomaly_detected",
        "machine_id":   machine_id,
        "file_index":   file_index,
        "message":      "Fault detected and diagnosed.",
        "fault_report": fault_report
    }


# ── Approve endpoint ──────────────────────────────────────────────────
@app.post("/approve-fault")
def approve_fault(
    asset_id:           str = Form(...),
    fault_type:         str = Form(...),
    severity:           str = Form(...),
    recommended_action: str = Form(...),
    sensor_summary:     str = Form(...)
):
    """
    Called when engineer clicks Approve in dashboard.
    Logs fault to SQLite history database.
    """
    result = log_fault(
        asset_id           = asset_id,
        fault_type         = fault_type,
        severity           = severity,
        recommended_action = recommended_action,
        sensor_summary     = sensor_summary,
        approved           = True
    )
    return {
        "status":  "logged",
        "message": f"Fault approved and logged for {asset_id}",
        "result":  result
    }


# ── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)