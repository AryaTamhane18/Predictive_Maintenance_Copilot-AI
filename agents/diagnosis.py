# agents/diagnosis.py
# Diagnosis agent — second agent in the pipeline
# Searches SKF manual via RAG
# Optionally analyses machine photo with LLaVA
# Asks Mistral to produce grounded diagnosis

import os
import sys
import base64
import requests
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from mcp_server import tool_search_manuals, tool_get_history


def encode_image(image_path: str) -> str:
    """Convert image to base64 for LLaVA."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyse_image_with_llava(image_path: str) -> dict:
    """
    Send machine photo to LLaVA.
    Returns structured visual assessment.
    """
    print("   → Sending image to LLaVA...")
    image_b64 = encode_image(image_path)

    prompt = """You are a strict industrial quality inspector.
Analyse this machine component image carefully.
Answer ONLY based on what you can actually see.

Reply in EXACT format — no extra text:
COMPONENT: <what type of part is this>
CONDITION: <new/good/worn/damaged/severely_damaged>
VISIBLE_DAMAGE: <yes/no>
DAMAGE_TYPE: <describe damage if visible, or "none">
SEVERITY: <none/minor/major/critical>
CONFIDENCE: <high/medium/low>
SUMMARY: <one sentence describing what you see>"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model":  "llava",
            "prompt": prompt,
            "images": [image_b64],
            "stream": False
        }
    )
    raw = response.json().get("response", "")
    print(f"   → LLaVA complete ✅")

    # parse structured response
    result = {
        "component":      "machine component",
        "condition":      "unknown",
        "visible_damage": False,
        "damage_type":    "none",
        "severity":       "none",
        "confidence":     "low",
        "summary":        raw[:200],
        "raw":            raw
    }

    for line in raw.strip().split('\n'):
        if ':' not in line:
            continue
        key, _, value = line.partition(':')
        key   = key.strip().upper()
        value = value.strip().lower()

        if key == "COMPONENT":
            result["component"] = value
        elif key == "CONDITION":
            result["condition"] = value
        elif key == "VISIBLE_DAMAGE":
            result["visible_damage"] = value == "yes"
        elif key == "DAMAGE_TYPE":
            result["damage_type"] = value
        elif key == "SEVERITY":
            result["severity"] = value
        elif key == "CONFIDENCE":
            result["confidence"] = value
        elif key == "SUMMARY":
            result["summary"] = line.partition(':')[2].strip()

    print(f"   → Component: {result['component']}")
    print(f"   → Condition: {result['condition']}")
    print(f"   → Visible damage: {result['visible_damage']}")
    print(f"   → Severity: {result['severity']}")

    return result


def run_diagnosis_agent(monitor_output: dict,
                        image_path:     str = None) -> dict:
    """
    Diagnosis agent:
    1. Searches SKF manual for relevant content (RAG)
    2. Optionally analyses machine photo with LLaVA
    3. Checks asset fault history
    4. Asks Mistral to produce grounded diagnosis
    """
    print(f"\n🔬 Diagnosis agent starting...")

    asset_id       = monitor_output.get("asset_id",       "BEARING_01")
    machine_id     = monitor_output.get("machine_id",     "bearing_1")
    anomaly_raw    = monitor_output.get("anomaly_raw",    "")
    interpretation = monitor_output.get("interpretation", "")
    severity       = monitor_output.get("severity",       "unknown")
    deviation      = monitor_output.get("deviation",      0.0)

    # Step 1 — image analysis (if provided)
    visual = None
    if image_path and os.path.exists(image_path):
        visual = analyse_image_with_llava(image_path)
    else:
        print("   → No image — sensor-only mode")

    # Step 2 — RAG search based on anomaly + visual findings
    if visual and visual["visible_damage"]:
        search_query = (
            f"{visual['component']} fault "
            f"{visual['damage_type']} "
            f"{visual['condition']} bearing maintenance"
        )
    else:
        search_query = (
            f"bearing anomaly {severity} vibration deviation "
            f"{interpretation}"
        )

    print("   → Searching SKF manual via RAG...")
    manual_result = tool_search_manuals.invoke({"query": search_query})
    print("   → Manual search complete ✅")

    # Step 3 — fault history
    print("   → Checking fault history...")
    history_result = tool_get_history.invoke({"asset_id": asset_id})
    print("   → History check complete ✅")

    # Step 4 — build grounded prompt
    # sensor evidence section
    sensor_section = f"""
SENSOR DATA EVIDENCE:
Machine ID: {machine_id}
{anomaly_raw}
Monitor interpretation: {interpretation}
"""

    # visual evidence section
    visual_section = ""
    if visual:
        if visual["visible_damage"]:
            visual_section = f"""
VISUAL INSPECTION (uploaded image):
Component identified: {visual['component']}
Condition: {visual['condition']}
Visible damage: YES
Damage type: {visual['damage_type']}
Severity: {visual['severity']}
LLaVA confidence: {visual['confidence']}
Summary: {visual['summary']}
"""
        else:
            visual_section = f"""
VISUAL INSPECTION (uploaded image):
Component: {visual['component']}
Condition: {visual['condition']}
Visible damage: NO
Summary: {visual['summary']}
Note: No visible external damage detected.
"""

    # determine what drives diagnosis
    if visual and visual["visible_damage"]:
        mode        = "both_confirm"
        instruction = ("Both sensor data AND visual inspection show "
                       "evidence of fault. Base diagnosis on both.")
    elif visual and not visual["visible_damage"]:
        mode        = "sensor_primary"
        instruction = ("Sensor data shows anomaly but visual inspection "
                       "shows no visible damage. Likely internal fault "
                       "not visible externally. State this clearly.")
    else:
        mode        = "sensor_only"
        instruction = ("No image provided. Base diagnosis solely on "
                       "sensor data.")

    llm = ChatOllama(model="mistral", temperature=0)

    prompt = f"""You are an expert bearing maintenance engineer.
Give a grounded diagnosis based ONLY on the evidence below.
Do not invent faults not supported by evidence.

ASSET: {asset_id}
INSTRUCTION: {instruction}

{sensor_section}
{visual_section}
RELEVANT SKF MANUAL CONTENT:
{manual_result}

ASSET FAULT HISTORY:
{history_result}

Reply in EXACT format — no extra text:
FAULT_TYPE: <specific fault type>
ROOT_CAUSE: <one clear sentence — evidence based>
EVIDENCE_USED: <what evidence drove this diagnosis>
MANUAL_REFERENCE: <relevant manual section>
RECOMMENDED_ACTION: <specific action for engineer>
URGENCY: <immediate/within 48 hours/within 1 week/monitor>
CONFIDENCE: <high/medium/low>
CAVEAT: <any important limitation of this diagnosis>"""

    print("\n   → Asking Mistral for diagnosis...")
    print("   → (waiting 20–40 seconds...)")

    response   = llm.invoke([HumanMessage(content=prompt)])
    llm_output = response.content.strip()
    print("   → Mistral diagnosis complete ✅")

    # parse output
    parsed = {}
    for line in llm_output.split('\n'):
        line = line.strip()
        if ':' in line:
            key, _, value = line.partition(':')
            parsed[key.strip()] = value.strip().strip('"')

    output = {
        "asset_id":           asset_id,
        "machine_id":         machine_id,
        "file_index":         monitor_output.get("file_index", 0),
        "fault_type":         parsed.get("FAULT_TYPE",
                                         "Bearing anomaly"),
        "root_cause":         parsed.get("ROOT_CAUSE",
                                         "Unknown root cause."),
        "evidence":           parsed.get("EVIDENCE_USED",
                                         anomaly_raw[:100]),
        "manual_reference":   parsed.get("MANUAL_REFERENCE",
                                         "SKF Maintenance Handbook"),
        "recommended_action": parsed.get("RECOMMENDED_ACTION",
                                         "Inspect immediately."),
        "urgency":            parsed.get("URGENCY",
                                         "within 48 hours"),
        "confidence":         parsed.get("CONFIDENCE",    "medium"),
        "caveat":             parsed.get("CAVEAT",         ""),
        "visual_findings":    visual["raw"] if visual else "No image.",
        "component_type":     visual["component"] if visual
                              else "unknown",
        "anomaly_raw":        anomaly_raw,
        "analysis_mode":      mode
    }

    print(f"\n   📋 Diagnosis result:")
    print(f"      Fault:      {output['fault_type']}")
    print(f"      Root cause: {output['root_cause']}")
    print(f"      Action:     {output['recommended_action']}")
    print(f"      Urgency:    {output['urgency']}")
    print(f"      Confidence: {output['confidence']}")
    print(f"      Mode:       {output['analysis_mode']}")

    return output


# ── Test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  DIAGNOSIS AGENT TEST")
    print("=" * 55)

    # simulate monitor output for a critical fault
    fake_monitor = {
        "asset_id":       "BEARING_01",
        "machine_id":     "bearing_2",
        "file_index":     934,
        "anomaly_raw": (
            "Machine: bearing_2\n"
            "File: 934 of 984\n"
            "Anomaly detected: True\n"
            "Worst channel: ch1\n"
            "Deviation from baseline: 156.85%\n"
            "Severity: critical\n"
            "Anomalous readings: 148 of 20480"
        ),
        "interpretation": (
            "Critical bearing fault — 156.85% deviation from baseline "
            "indicating potential outer race failure."
        ),
        "severity":   "critical",
        "deviation":  156.85,
        "escalate":   True
    }

    result = run_diagnosis_agent(
        monitor_output = fake_monitor,
        image_path     = None  # no image for test
    )

    print("\n" + "=" * 55)
    print("  FINAL DIAGNOSIS")
    print("=" * 55)
    print(f"  Fault:      {result['fault_type']}")
    print(f"  Root cause: {result['root_cause']}")
    print(f"  Action:     {result['recommended_action']}")
    print(f"  Urgency:    {result['urgency']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Caveat:     {result['caveat']}")
    print("\n✅ Diagnosis agent working correctly")

    print('hi')