import sqlite3
import json
from datetime import datetime
from models import FaultReport, SeverityLevel

DB_PATH = "data/maintenance_history.db"


def init_db():
    """Create the database and table if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fault_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            fault_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            recommended_action TEXT NOT NULL,
            cited_manual_section TEXT,
            approved INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def save_fault(report: FaultReport):
    """Save a fault report to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO fault_history 
        (machine_id, timestamp, fault_type, severity, 
         recommended_action, cited_manual_section, approved)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        report.machine_id,
        report.timestamp,
        report.fault_type,
        report.severity.value,
        report.recommended_action,
        report.cited_manual_section,
        int(report.approved)
    ))
    conn.commit()
    conn.close()


def get_history(machine_id: str, limit: int = 5) -> list:
    """
    Get past fault reports for a machine.
    Returns the most recent faults first.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT machine_id, timestamp, fault_type, severity,
               recommended_action, cited_manual_section, approved
        FROM fault_history
        WHERE machine_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (machine_id, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "machine_id": row[0],
            "timestamp": row[1],
            "fault_type": row[2],
            "severity": row[3],
            "recommended_action": row[4],
            "cited_manual_section": row[5],
            "approved": bool(row[6])
        })
    
    return history


def get_history_summary(machine_id: str) -> str:
    """
    Return a plain-English summary of past faults.
    This is what the agent will read.
    """
    history = get_history(machine_id)
    
    if not history:
        return f"No previous faults recorded for {machine_id}."
    
    summary = f"Past {len(history)} fault(s) for {machine_id}:\n"
    for i, fault in enumerate(history, 1):
        summary += f"{i}. [{fault['severity']}] {fault['fault_type']} "
        summary += f"on {fault['timestamp'][:10]} — "
        summary += f"Action: {fault['recommended_action']}\n"
    
    return summary


if __name__ == "__main__":
    print("Initialising database...")
    init_db()
    
    # Save a sample fault to test
    sample = FaultReport(
        machine_id="bearing_1",
        timestamp=datetime.now().isoformat(),
        fault_type="Bearing vibration anomaly",
        severity=SeverityLevel.HIGH,
        recommended_action="Inspect bearing 3, check lubrication",
        cited_manual_section="Page 224 — Vibration root cause analysis",
        approved=False
    )
    save_fault(sample)
    print("Sample fault saved.")
    
    # Retrieve history
    summary = get_history_summary("bearing_1")
    print(f"\nHistory summary:\n{summary}")