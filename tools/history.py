# tools/history.py
# Stores and retrieves past fault records in SQLite
# Agent calls get_history() to check if this asset has failed before

import sqlite3
import os
# import json
from datetime import datetime

# DB lives in project root
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fault_history.db"
)

def init_db():
    """Create the faults table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS faults (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id        TEXT NOT NULL,
            fault_type      TEXT,
            severity        TEXT,
            recommended_action TEXT,
            sensor_summary  TEXT,
            approved        INTEGER DEFAULT 0,
            created_at      TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_fault(asset_id: str, fault_type: str, severity: str,
              recommended_action: str, sensor_summary: str,
              approved: bool = False) -> dict:
    """
    Save a fault record to SQLite.
    Called by dashboard when engineer clicks Approve.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO faults
        (asset_id, fault_type, severity, recommended_action,
         sensor_summary, approved, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        asset_id, fault_type, severity,
        recommended_action, sensor_summary,
        1 if approved else 0,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()
    return {"logged": True, "asset_id": asset_id, "fault_type": fault_type}


def get_history(asset_id: str, limit: int = 5) -> dict:
    """
    Retrieve past fault records for a given asset.
    Called by Diagnosis agent via MCP to check fault history.
    """
    init_db()
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        SELECT fault_type, severity, recommended_action,
               sensor_summary, created_at
        FROM   faults
        WHERE  asset_id = ?
        ORDER  BY created_at DESC
        LIMIT  ?
    """, (asset_id, limit))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return {
            "asset_id":    asset_id,
            "found":       False,
            "record_count": 0,
            "history":     [],
            "summary":     f"No previous faults recorded for asset {asset_id}."
        }

    history = []
    for row in rows:
        history.append({
            "fault_type":          row[0],
            "severity":            row[1],
            "recommended_action":  row[2],
            "sensor_summary":      row[3],
            "recorded_at":         row[4]
        })

    # plain English summary for the agent
    fault_types = list(set(r["fault_type"] for r in history))
    summary = (
        f"Asset {asset_id} has {len(rows)} previous fault(s). "
        f"Fault types seen: {', '.join(fault_types)}. "
        f"Most recent: {history[0]['fault_type']} "
        f"({history[0]['severity']}) on {history[0]['recorded_at'][:10]}."
    )

    return {
        "asset_id":     asset_id,
        "found":        True,
        "record_count": len(rows),
        "history":      history,
        "summary":      summary
    }


# ── Quick test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Testing history tool ===\n")

    # Log 2 fake faults
    print("1. Logging test faults...")
    log_fault(
        asset_id           = "BEARING_01",
        fault_type         = "Inner race wear",
        severity           = "major",
        recommended_action = "Inspect bearing 3 and schedule replacement "
                             "within 48 hours",
        sensor_summary     = "ch5 RMS 62% above baseline — major threshold "
                             "is 50%",
        approved           = True
    )
    log_fault(
        asset_id           = "BEARING_01",
        fault_type         = "Early lubrication degradation",
        severity           = "minor",
        recommended_action = "Re-grease at next planned stop and re-check "
                             "vibration",
        sensor_summary     = "ch6 RMS 34% above baseline — minor threshold "
                             "is 30%",
        approved           = True
    )
    print("   ✅ 2 faults logged\n")

    # Retrieve history
    print("2. Retrieving history for BEARING_01...")
    result = get_history("BEARING_01")
    print(f"   Found:   {result['found']}")
    print(f"   Records: {result['record_count']}")
    print(f"   Summary: {result['summary']}")

    # Test asset with no history
    print("\n3. Retrieving history for unknown asset...")
    result2 = get_history("BEARING_99")
    print(f"   Found:   {result2['found']}")
    print(f"   Summary: {result2['summary']}")

    print("\n✅ history.py working correctly")
