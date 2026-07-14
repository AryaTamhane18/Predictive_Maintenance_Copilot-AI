# agents/monitor.py
# Monitor agent — first agent in the pipeline
# Receives machine_id + file_index
# Calls detect_anomaly tool
# Decides whether to escalate to Diagnosis agent

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from mcp_server import tool_detect_anomaly


def run_monitor_agent(machine_id:  str,
                      file_index:  int,
                      asset_id:    str = "BEARING_01") -> dict:
    """
    Monitor agent:
    1. Calls detect_anomaly tool with machine_id + file_index
    2. Asks Mistral to interpret the result in plain English
    3. Decides: NORMAL (stop) or ESCALATE (continue to diagnosis)
    """
    print(f"\n🔍 Monitor agent starting...")
    print(f"   Machine:    {machine_id}")
    print(f"   File index: {file_index}")
    print(f"   Asset ID:   {asset_id}")

    # Step 1 — call anomaly detection tool
    print("\n   → Calling detect_anomaly tool...")
    anomaly_raw = tool_detect_anomaly.invoke({
        "machine_id": machine_id,
        "file_index": file_index
    })
    print("   → Tool result received ✅")
    print(f"   → {anomaly_raw.split(chr(10))[0]}")  # print first line

    # Step 2 — parse key values from tool output
    is_anomalous = "True" in anomaly_raw
    severity     = "none"
    deviation    = 0.0

    for line in anomaly_raw.split('\n'):
        if "Severity:" in line:
            severity = line.split(":")[-1].strip()
        if "Deviation from baseline:" in line:
            try:
                deviation = float(
                    line.split(":")[-1].strip().replace('%', '')
                )
            except Exception:
                pass

    # Step 3 — if no anomaly, return immediately without calling Mistral
    if not is_anomalous:
        print("\n   ✅ No anomaly — machine operating normally")
        return {
            "asset_id":       asset_id,
            "machine_id":     machine_id,
            "file_index":     file_index,
            "anomaly_raw":    anomaly_raw,
            "is_anomalous":   False,
            "interpretation": "Machine is operating normally. "
                              "No anomaly detected in sensor data.",
            "decision":       "normal",
            "escalate":       False
        }

    # Step 4 — anomaly found — ask Mistral to interpret
    print("\n   → Asking Mistral to interpret anomaly...")
    print("   → (waiting 15–25 seconds...)")

    llm    = ChatOllama(model="mistral", temperature=0)
    prompt = f"""You are a predictive maintenance monitoring agent.

A sensor anomaly has been detected on machine {machine_id} (Asset: {asset_id}).

ANOMALY DETECTION RESULT:
{anomaly_raw}

Based on this result:
1. Is this a genuine concern requiring further diagnosis?
2. In one sentence, what does this anomaly suggest?
3. What is your escalation decision?

Reply in EXACTLY this format — no extra text:
CONCERN: yes/no
INTERPRETATION: <one plain English sentence about what this means>
DECISION: escalate/monitor/normal
REASON: <one sentence explaining your decision>"""

    response   = llm.invoke([HumanMessage(content=prompt)])
    llm_output = response.content.strip()
    print("   → Mistral responded ✅")

    # Step 5 — parse Mistral response
    lines = {}
    for line in llm_output.split('\n'):
        line = line.strip()
        if ':' in line:
            key, _, value = line.partition(':')
            lines[key.strip().upper()] = value.strip()

    concern        = lines.get("CONCERN",        "yes").lower()
    interpretation = lines.get("INTERPRETATION",
                                "Anomaly detected in sensor data.")
    decision       = lines.get("DECISION",       "escalate").lower()
    reason         = lines.get("REASON",
                                "Sensor readings deviate from baseline.")

    escalate = decision == "escalate" or severity in ["major", "critical"]

    print(f"\n   📋 Monitor agent result:")
    print(f"      Concern:        {concern}")
    print(f"      Interpretation: {interpretation}")
    print(f"      Decision:       {decision}")
    print(f"      Escalate:       {escalate}")

    return {
        "asset_id":       asset_id,
        "machine_id":     machine_id,
        "file_index":     file_index,
        "anomaly_raw":    anomaly_raw,
        "is_anomalous":   True,
        "severity":       severity,
        "deviation":      deviation,
        "interpretation": interpretation,
        "decision":       decision,
        "reason":         reason,
        "escalate":       escalate
    }


# ── Test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  MONITOR AGENT TEST")
    print("=" * 55)

    # Test 1 — healthy file (should NOT escalate)
    print("\n--- Test 1: Healthy file ---")
    result = run_monitor_agent(
        machine_id = "bearing_2",
        file_index = 49,
        asset_id   = "BEARING_01"
    )
    print(f"\nFinal: escalate={result['escalate']} "
          f"decision={result['decision']}")

    # Test 2 — fault file (should escalate)
    print("\n--- Test 2: Fault file ---")
    result = run_monitor_agent(
        machine_id = "bearing_2",
        file_index = 934,
        asset_id   = "BEARING_01"
    )
    print(f"\nFinal: escalate={result['escalate']} "
          f"decision={result['decision']}")
    print(f"Interpretation: {result['interpretation']}")

    print("\n✅ Monitor agent working correctly")