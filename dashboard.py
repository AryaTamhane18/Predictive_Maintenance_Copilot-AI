# dashboard.py
# Streamlit dashboard — complete UI for Predictive Maintenance Copilot
# Machine dropdown + file slider + optional image upload + HITL

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import io
import os

st.set_page_config(
    page_title = "Predictive Maintenance Copilot",
    page_icon  = "🔧",
    layout     = "wide"
)

API_URL = "http://localhost:8000"

# ── Load machine info from API ────────────────────────────────────────
@st.cache_data(ttl=60)
def load_machines():
    try:
        resp = requests.get(f"{API_URL}/machines", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    # fallback if API not running
    return {
        "bearing_1": {
            "label": "Motor line bearing",
            "total_files": 2156,
            "description": "Bearing 3 & 4 fail"
        },
        "bearing_2": {
            "label": "Pump station bearing",
            "total_files": 984,
            "description": "Bearing 1 fails"
        },
        "bearing_4": {
            "label": "Conveyor system bearing",
            "total_files": 6324,
            "description": "Bearing 3 fails"
        }
    }


machines = load_machines()

# ── Header ────────────────────────────────────────────────────────────
st.title("🔧 Predictive Maintenance Copilot")
st.markdown(
    "*AI-powered bearing fault detection and diagnosis — "
    "Ollama · LangGraph · ChromaDB · FastAPI*"
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # Asset ID
    asset_id = st.text_input(
        "Asset ID",
        value = "BEARING_01",
        help  = "Identifier for the asset being monitored"
    )

    st.divider()
    st.header("🏭 Select Machine")

    # Machine dropdown
    machine_options = {
        mid: f"{mid} — {info['label']}"
        for mid, info in machines.items()
    }
    selected_machine = st.selectbox(
        "Machine",
        options = list(machine_options.keys()),
        format_func = lambda x: machine_options[x],
        help = "Select which machine to analyse"
    )

    # get total files for selected machine
    total_files = machines[selected_machine]["total_files"]
    machine_label = machines[selected_machine]["label"]

    st.divider()
    st.header("📅 Select Time Point")

    # file index slider
    file_index = st.slider(
        "Sensor file (time point)",
        min_value = 1,
        max_value = total_files,
        value     = total_files,  # default = latest = near failure
        step      = 1,
        help      = (
            f"1 = experiment start (healthy bearing)  |  "
            f"{total_files} = experiment end (bearing failed)  |  "
            f"Total files: {total_files}"
        )
    )

    # show what percentage through the experiment this is
    pct = round((file_index / total_files) * 100, 1)
    if pct < 20:
        st.success(f"📍 {pct}% through experiment — likely healthy")
    elif pct < 60:
        st.info(f"📍 {pct}% through experiment — mid-life")
    elif pct < 85:
        st.warning(f"📍 {pct}% through experiment — degradation possible")
    else:
        st.error(f"📍 {pct}% through experiment — fault likely present")

    st.divider()
    st.header("📸 Visual Inspection")

    image_file = st.file_uploader(
        "Machine photo (optional)",
        type = ["jpg", "jpeg", "png"],
        help = (
            "Upload a photo of the machine component. "
            "LLaVA will visually inspect it for damage, "
            "wear, corrosion, or other issues."
        )
    )

    st.divider()

    analyse_btn = st.button(
        "🚀 Run Analysis",
        type                = "primary",
        use_container_width = True
    )

# ── Main area — two columns ───────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Sensor Data Info")

    # show machine info card
    st.info(
        f"**Machine:** {machine_label}\n\n"
        f"**Total sensor files:** {total_files:,}\n\n"
        f"**Selected file:** {file_index} of {total_files} "
        f"({pct}% through experiment)\n\n"
        f"**Failed bearing:** "
        f"{machines[selected_machine].get('description', 'N/A')}"
    )

    # health indicator bar
    st.markdown("**Estimated health at selected time point:**")
    health_pct = max(0, 100 - pct)
    if health_pct > 80:
        st.progress(health_pct / 100,
                    text=f"Health score: ~{int(health_pct)}% (healthy)")
    elif health_pct > 40:
        st.progress(health_pct / 100,
                    text=f"Health score: ~{int(health_pct)}% (degrading)")
    else:
        st.progress(health_pct / 100,
                    text=f"Health score: ~{int(health_pct)}% (critical range)")

with col2:
    st.subheader("📸 Machine Photo")
    if image_file is not None:
        st.image(
            image_file,
            caption             = "Uploaded for LLaVA visual inspection",
            use_container_width = True
        )
        image_file.seek(0)
    else:
        st.info(
            "No photo uploaded.\n\n"
            "System will run in sensor-only mode.\n\n"
            "Upload a photo for combined sensor + visual inspection."
        )

st.divider()

# ── Run Analysis ──────────────────────────────────────────────────────
if analyse_btn:

    st.subheader("🤖 Agent Pipeline Running...")

    progress = st.progress(0, text="Starting...")
    status   = st.empty()

    c1, c2, c3 = st.columns(3)
    with c1:
        s1 = st.empty()
        s1.info("⏳ Monitor agent")
    with c2:
        s2 = st.empty()
        s2.info("⏳ Diagnosis agent")
    with c3:
        s3 = st.empty()
        s3.info("⏳ Recommendation agent")

    try:
        # update UI
        progress.progress(10, text="Monitor agent running...")
        s1.warning("🔍 Monitor agent running...")
        status.info(
            f"🔄 Analysing {machine_label} — "
            f"file {file_index}/{total_files} "
            f"({pct}% through experiment)..."
        )

        # prepare request
        data  = {
            "machine_id": selected_machine,
            "file_index": str(file_index),
            "asset_id":   asset_id
        }
        files = {}

        if image_file is not None:
            image_file.seek(0)
            files["image_file"] = (
                image_file.name,
                image_file,
                "image/jpeg"
            )

        # call API
        response = requests.post(
            f"{API_URL}/analyse-sensor",
            data    = data,
            files   = files if files else None,
            timeout = 300
        )

        # update progress
        progress.progress(60, text="Diagnosis agent running...")
        s1.success("✅ Monitor agent done")
        s2.warning("🔬 Diagnosis agent running...")

        progress.progress(85, text="Recommendation agent running...")
        s2.success("✅ Diagnosis agent done")
        s3.warning("📝 Recommendation agent running...")

        if response.status_code != 200:
            st.error(f"API error {response.status_code}: "
                     f"{response.text}")
        else:
            result = response.json()
            progress.progress(100, text="Complete!")
            s3.success("✅ Recommendation agent done")
            status.empty()

            st.divider()

            # ── No fault ─────────────────────────────────────────
            if result["status"] == "normal":
                st.success(
                    f"✅ No fault detected — "
                    f"{machine_label} operating normally at "
                    f"file {file_index}/{total_files} "
                    f"({pct}% through experiment)."
                )
                st.balloons()

            # ── Error ────────────────────────────────────────────
            elif result["status"] == "error":
                st.error(f"❌ Error: {result['message']}")

            # ── Fault detected ───────────────────────────────────
            else:
                report = result["fault_report"]
                sev    = report.get("severity", "").lower()

                if "critical" in sev:
                    st.error(
                        f"🚨 CRITICAL FAULT DETECTED — "
                        f"{report['fault_type']}"
                    )
                elif "major" in sev:
                    st.warning(
                        f"⚠️ MAJOR FAULT DETECTED — "
                        f"{report['fault_type']}"
                    )
                else:
                    st.info(
                        f"ℹ️ MINOR FAULT DETECTED — "
                        f"{report['fault_type']}"
                    )

                # ── Fault report card ─────────────────────────────
                st.subheader("📋 AI-Generated Fault Report")

                r1, r2 = st.columns(2)
                with r1:
                    st.metric("Asset ID",   report["asset_id"])
                    st.metric("Machine",
                              report.get("machine_label",
                                         report["machine_id"]))
                    st.metric("Severity",   report["severity"])
                    st.metric("Fault Type", report["fault_type"])
                    st.metric("File",
                              f"{report['file_index']} / "
                              f"{total_files} "
                              f"({pct}% through experiment)")

                with r2:
                    st.markdown("**Root Cause**")
                    st.write(report["root_cause"])

                    st.markdown("**Manual Reference**")
                    st.write(report["manual_reference"])

                    if report.get("timestamp_name"):
                        st.markdown("**Sensor Timestamp**")
                        st.write(report["timestamp_name"])

                st.markdown("**Recommended Action**")
                st.info(report["recommended_action"])

                st.markdown("**Sensor Summary**")
                st.write(report["sensor_summary"])

                # ── Confidence + mode ─────────────────────────────
                st.divider()
                conf_col, mode_col = st.columns(2)

                with conf_col:
                    confidence = report.get("confidence", "medium").lower()
                    if confidence == "high":
                        st.success("🟢 High confidence diagnosis")
                    elif confidence == "medium":
                        st.warning("🟡 Medium confidence — review recommended")
                    else:
                        st.error("🔴 Low confidence — manual inspection required")

                with mode_col:
                    mode = report.get("analysis_mode", "")
                    mode_labels = {
                        "both_confirm":   "✅ Visual + Sensor both confirm",
                        "visual_primary": "📸 Visual inspection led diagnosis",
                        "sensor_primary": "📊 Sensor data led — no visible damage",
                        "sensor_only":    "📊 Sensor data only — no image",
                        "no_fault":       "✅ No fault detected"
                    }
                    st.info(mode_labels.get(mode, f"Mode: {mode}"))

                caveat = report.get("caveat", "")
                if caveat:
                    st.warning(f"⚠️ Important caveat: {caveat}")

                # ── HITL ─────────────────────────────────────────
                st.divider()
                st.subheader("👷 Engineer Review — Human in the Loop")
                st.write(
                    "Review the AI diagnosis above "
                    "and make your decision:"
                )

                h1, h2 = st.columns(2)

                with h1:
                    if st.button(
                        "✅ Approve & Log Fault",
                        type                = "primary",
                        use_container_width = True
                    ):
                        approve = requests.post(
                            f"{API_URL}/approve-fault",
                            data = {
                                "asset_id":           report["asset_id"],
                                "fault_type":         report["fault_type"],
                                "severity":           report["severity"],
                                "recommended_action": report["recommended_action"],
                                "sensor_summary":     report["sensor_summary"]
                            }
                        )
                        if approve.status_code == 200:
                            st.success(
                                "✅ Fault approved and logged. "
                                "Action required: "
                                f"{report['recommended_action']}"
                            )
                        else:
                            st.error("Failed to log fault.")

                with h2:
                    if st.button(
                        "❌ Override — No Action Needed",
                        use_container_width = True
                    ):
                        st.warning(
                            "⚠️ Fault overridden by engineer. "
                            "No action logged. "
                            "Continue monitoring."
                        )

                # store in session
                st.session_state["last_report"] = report

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to API. "
            "Make sure api.py is running:\n"
            "`python api.py`"
        )
    except requests.exceptions.Timeout:
        st.error(
            "⏱️ Request timed out. "
            "Pipeline is taking longer than 5 minutes. "
            "Try again."
        )
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")

# ── Footer ────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Predictive Maintenance Copilot v3.0 — "
    "Ollama (Mistral + LLaVA) · "
    "LangGraph · ChromaDB · FastAPI · Streamlit"
)