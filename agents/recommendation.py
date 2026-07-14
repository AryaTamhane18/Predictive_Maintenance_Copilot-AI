# agents/recommendation.py
# Recommendation agent — third and final agent
# Takes diagnosis output and produces clean FaultReport
# Validated by Pydantic

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from models import FaultReport


def run_recommendation_agent(diagnosis_output: dict) -> FaultReport:
    """
    Recommendation agent:
    1. Takes full diagnosis output
    2. Asks Mistral to write a clean engineer-facing report
    3. Returns validated FaultReport (Pydantic)
    """
    print(f"\n📝 Recommendation agent starting...")

    asset_id           = diagnosis_output.get("asset_id",           "BEARING_01")
    machine_id         = diagnosis_output.get("machine_id",         "bearing_1")
    file_index         = diagnosis_output.get("file_index",         0)
    fault_type         = diagnosis_output.get("fault_type",         "Unknown fault")
    root_cause         = diagnosis_output.get("root_cause",         "Unknown.")
    evidence           = diagnosis_output.get("evidence",           "")
    manual_reference   = diagnosis_output.get("manual_reference",   "SKF Handbook")
    recommended_action = diagnosis_output.get("recommended_action", "Inspect immediately.")
    urgency            = diagnosis_output.get("urgency",            "within 48 hours")
    confidence         = diagnosis_output.get("confidence",         "medium")
    caveat             = diagnosis_output.get("caveat",             "")
    anomaly_raw        = diagnosis_output.get("anomaly_raw",        "")
    analysis_mode      = diagnosis_output.get("analysis_mode",      "sensor_only")
    visual_findings    = diagnosis_output.get("visual_findings",    "No image.")
    component_type     = diagnosis_output.get("component_type",     "unknown")

    llm = ChatOllama(model="mistral", temperature=0)

    prompt = f"""You are a senior maintenance engineer writing a fault report
for a factory engineer who needs to take action immediately.

DIAGNOSIS SUMMARY:
Asset ID: {asset_id}
Machine: {machine_id}
Fault type: {fault_type}
Root cause: {root_cause}
Evidence: {evidence}
Manual reference: {manual_reference}
Recommended action: {recommended_action}
Urgency: {urgency}
Confidence: {confidence}
Visual findings: {visual_findings}

Write a professional fault report in EXACT format — no extra text:

SEVERITY: minor/major/critical
FAULT_TYPE: <concise fault name — max 6 words>
ROOT_CAUSE: <one clear sentence in plain English>
RECOMMENDED_ACTION: <specific step-by-step action>
MANUAL_REFERENCE: <exact manual section>
SENSOR_SUMMARY: <one sentence summarising sensor findings>"""

    print("   → Asking Mistral to write fault report...")
    print("   → (waiting 20–40 seconds...)")

    response   = llm.invoke([HumanMessage(content=prompt)])
    llm_output = response.content.strip()
    print("   → Report generated ✅")

    # parse output
    parsed = {}
    for line in llm_output.split('\n'):
        line = line.strip()
        if ':' in line:
            key, _, value = line.partition(':')
            parsed[key.strip()] = value.strip().strip('"')

    # get timestamp from anomaly_raw
    timestamp = ""
    for line in anomaly_raw.split('\n'):
        if "File:" in line and "(" in line:
            try:
                timestamp = line.split("(")[1].split(")")[0]
            except Exception:
                pass

    # build validated FaultReport
    report = FaultReport(
        asset_id           = asset_id,
        machine_id         = machine_id,
        file_index         = file_index,
        timestamp_name     = timestamp,
        severity           = parsed.get("SEVERITY",
                                        "major"),
        fault_type         = parsed.get("FAULT_TYPE",
                                        fault_type),
        root_cause         = parsed.get("ROOT_CAUSE",
                                        root_cause),
        recommended_action = parsed.get("RECOMMENDED_ACTION",
                                        recommended_action),
        manual_reference   = parsed.get("MANUAL_REFERENCE",
                                        manual_reference),
        sensor_summary     = parsed.get("SENSOR_SUMMARY",
                                        anomaly_raw[:150]),
        analysis_mode      = analysis_mode,
        confidence         = confidence,
        caveat             = caveat,
        approved           = None
    )

    print(f"\n   📋 Fault Report:")
    print(f"      Asset:     {report.asset_id}")
    print(f"      Machine:   {report.machine_id}")
    print(f"      Severity:  {report.severity}")
    print(f"      Fault:     {report.fault_type}")
    print(f"      Action:    {report.recommended_action}")
    print(f"      Urgency:   {urgency}")
    print(f"      Confidence:{report.confidence}")
    print(f"      Approved:  {report.approved} (pending engineer)")

    return report


# ── Test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  RECOMMENDATION AGENT TEST")
    print("=" * 55)

    fake_diagnosis = {
        "asset_id":           "BEARING_01",
        "machine_id":         "bearing_2",
        "file_index":         934,
        "fault_type":         "Critical bearing fault",
        "root_cause":         (
            "Potential outer race failure due to 156.85% "
            "deviation from baseline."
        ),
        "evidence":           "Sensor ch1 deviation 156.85%",
        "manual_reference":   "SKF Handbook page 324",
        "recommended_action": "Inspect and replace bearing immediately.",
        "urgency":            "immediate",
        "confidence":         "high",
        "caveat":             (
            "Diagnosis based on sensor data only — "
            "visual inspection recommended."
        ),
        "anomaly_raw":        (
            "Machine: bearing_2\n"
            "File: 934 of 984 (2004.02.18.22.02.39)\n"
            "Anomaly detected: True\n"
            "Worst channel: ch1\n"
            "Deviation from baseline: 156.85%\n"
            "Severity: critical"
        ),
        "analysis_mode":      "sensor_only",
        "visual_findings":    "No image provided.",
        "component_type":     "unknown"
    }

    report = run_recommendation_agent(fake_diagnosis)

    print("\n" + "=" * 55)
    print("  FINAL FAULT REPORT (Pydantic validated)")
    print("=" * 55)
    print(f"  Asset ID:   {report.asset_id}")
    print(f"  Machine:    {report.machine_id}")
    print(f"  Severity:   {report.severity}")
    print(f"  Fault:      {report.fault_type}")
    print(f"  Root cause: {report.root_cause}")
    print(f"  Action:     {report.recommended_action}")
    print(f"  Manual ref: {report.manual_reference}")
    print(f"  Sensor:     {report.sensor_summary}")
    print(f"  Confidence: {report.confidence}")
    print(f"  Caveat:     {report.caveat}")
    print(f"  Approved:   {report.approved}")
    print("\n✅ Recommendation agent working correctly")