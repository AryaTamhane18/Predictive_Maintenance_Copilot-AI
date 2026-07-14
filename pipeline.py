# pipeline.py
# LangGraph pipeline — wires all 3 agents in sequence
# Input: machine_id + file_index + optional image_path
# Output: FaultReport or None (if no fault)

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from agents.monitor        import run_monitor_agent
from agents.diagnosis      import run_diagnosis_agent
from agents.recommendation import run_recommendation_agent


# ── Shared state flowing between all agents ───────────────────────────
class PipelineState(TypedDict):
    # inputs
    machine_id:  str
    file_index:  int
    asset_id:    str
    image_path:  Optional[str]

    # monitor output
    monitor_output: Optional[dict]
    escalate:       Optional[bool]

    # diagnosis output
    diagnosis_output: Optional[dict]

    # final output
    fault_report: Optional[dict]
    completed:    Optional[bool]


# ── Node functions ────────────────────────────────────────────────────

def monitor_node(state: PipelineState) -> PipelineState:
    """Node 1 — Monitor agent."""
    print("\n" + "="*55)
    print("  STEP 1 / 3 — MONITOR AGENT")
    print("="*55)

    result = run_monitor_agent(
        machine_id = state["machine_id"],
        file_index = state["file_index"],
        asset_id   = state["asset_id"]
    )

    return {
        **state,
        "monitor_output": result,
        "escalate":       result["escalate"]
    }


def diagnosis_node(state: PipelineState) -> PipelineState:
    """Node 2 — Diagnosis agent."""
    print("\n" + "="*55)
    print("  STEP 2 / 3 — DIAGNOSIS AGENT")
    print("="*55)

    result = run_diagnosis_agent(
        monitor_output = state["monitor_output"],
        image_path     = state.get("image_path")
    )

    return {
        **state,
        "diagnosis_output": result
    }


def recommendation_node(state: PipelineState) -> PipelineState:
    """Node 3 — Recommendation agent."""
    print("\n" + "="*55)
    print("  STEP 3 / 3 — RECOMMENDATION AGENT")
    print("="*55)

    report = run_recommendation_agent(state["diagnosis_output"])

    return {
        **state,
        "fault_report": report.model_dump(),
        "completed":    True
    }


def normal_node(state: PipelineState) -> PipelineState:
    """Node — runs when monitor says no fault."""
    print("\n✅ Monitor agent: no fault detected — machine healthy.")
    return {
        **state,
        "fault_report": None,
        "completed":    True
    }


# ── Routing ───────────────────────────────────────────────────────────

def route_after_monitor(state: PipelineState) -> str:
    """Escalate to diagnosis or end with normal status."""
    if state.get("escalate"):
        return "diagnosis"
    return "normal"


# ── Build pipeline ────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("monitor",        monitor_node)
    graph.add_node("diagnosis",      diagnosis_node)
    graph.add_node("recommendation", recommendation_node)
    graph.add_node("normal",         normal_node)

    graph.set_entry_point("monitor")

    graph.add_conditional_edges(
        "monitor",
        route_after_monitor,
        {
            "diagnosis": "diagnosis",
            "normal":    "normal"
        }
    )

    graph.add_edge("diagnosis",      "recommendation")
    graph.add_edge("recommendation", END)
    graph.add_edge("normal",         END)

    return graph.compile()


# ── Main entry point ──────────────────────────────────────────────────

def run_pipeline(machine_id: str,
                 file_index:  int,
                 asset_id:    str = "BEARING_01",
                 image_path:  str = None) -> dict:
    """
    Run the full 3-agent pipeline.

    machine_id: "bearing_1" / "bearing_2" / "bearing_4"
    file_index: which sensor file to analyse (1-based)
    asset_id:   engineer-facing asset identifier
    image_path: optional machine photo for LLaVA
    """
    pipeline = build_pipeline()

    initial_state: PipelineState = {
        "machine_id":      machine_id,
        "file_index":      file_index,
        "asset_id":        asset_id,
        "image_path":      image_path,
        "monitor_output":  None,
        "escalate":        None,
        "diagnosis_output":None,
        "fault_report":    None,
        "completed":       False
    }

    from tools.anomaly import get_machine_info
    info = get_machine_info(machine_id)

    print(f"\n🚀 Starting Predictive Maintenance Pipeline...")
    print(f"   Machine:    {machine_id} — {info.get('label', '')}")
    print(f"   File index: {file_index} of "
          f"{info.get('total_files', '?')}")
    print(f"   Asset ID:   {asset_id}")
    print(f"   Image:      {os.path.basename(image_path) if image_path else 'None'}")

    return pipeline.invoke(initial_state)


# ── Test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("  PIPELINE TEST 1 — Healthy machine")
    print("=" * 55)

    result = run_pipeline(
        machine_id = "bearing_2",
        file_index = 49,        # healthy file
        asset_id   = "BEARING_01"
    )

    report = result.get("fault_report")
    if report:
        print(f"\n⚠️ Fault: {report['fault_type']} "
              f"({report['severity']})")
    else:
        print("\n✅ Result: No fault — machine healthy")

    print("\n" + "=" * 55)
    print("  PIPELINE TEST 2 — Fault scenario")
    print("=" * 55)

    result = run_pipeline(
        machine_id = "bearing_2",
        file_index = 934,       # critical fault file
        asset_id   = "BEARING_01"
    )

    report = result.get("fault_report")
    if report:
        print(f"\n🚨 PIPELINE COMPLETE — FAULT REPORT")
        print(f"   Asset:      {report['asset_id']}")
        print(f"   Machine:    {report['machine_id']}")
        print(f"   Severity:   {report['severity']}")
        print(f"   Fault:      {report['fault_type']}")
        print(f"   Root cause: {report['root_cause']}")
        print(f"   Action:     {report['recommended_action']}")
        print(f"   Confidence: {report['confidence']}")
        print(f"   Approved:   {report['approved']}")
    else:
        print("\n✅ No fault detected")

    print("\n✅ Full pipeline working end to end")
    print("Done")