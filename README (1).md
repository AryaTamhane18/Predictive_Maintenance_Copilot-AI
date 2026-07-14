# 🔧 Predictive Maintenance Copilot

> **An agentic AI system that monitors industrial bearing sensor data, detects faults autonomously, and generates plain-English diagnostic reports — running fully locally with zero cloud dependency.**

---

## 📌 What This Project Does

Machines in factories contain **bearings** — small but critical components that fail without warning, causing expensive unplanned downtime. This project builds an AI copilot that:

- **Watches** vibration sensor readings from industrial bearings
- **Detects** when readings deviate abnormally from a healthy baseline
- **Diagnoses** the fault by searching a real maintenance manual via RAG
- **Inspects** uploaded machine photos using a vision model
- **Reports** to the engineer in plain English with severity, root cause, recommended action, and manual citation
- **Waits** for the engineer to approve or override before logging anything

All models run locally via Ollama — no OpenAI, no cloud API, no data leaves the machine.

---

## 📸 Dashboard

**Main dashboard** — machine selector, chronological file slider, health indicator, and optional machine photo uploaded for LLaVA visual inspection. All 3 agents completed in sequence.

**Fault report** — AI-generated critical fault diagnosis with root cause, SKF manual reference, sensor timestamp, and recommended action.

> Screenshots: `dashboard_overview.png` and `fault_report.png` — place both in the repo root alongside this README and GitHub will render them automatically.

---

## 🏗️ Architecture

```
Engineer (Browser)
        ↓
Streamlit Dashboard
  - Machine dropdown (bearing_1 / bearing_2 / bearing_4)
  - Chronological file slider (1 = healthy → max = near failure)
  - Optional machine photo upload
  - Human-in-the-loop Approve / Override buttons
        ↓
FastAPI REST Backend  (port 8000)
  POST /analyse-sensor
  POST /approve-fault
  GET  /machines
        ↓
LangGraph Pipeline — 3 agents in sequence
        ↓
┌─────────────────────────────────────────────────────┐
│  Agent 1 — Monitor                                  │
│  Calls detect_anomaly() MCP tool                    │
│  Computes RMS deviation from healthy baseline       │
│  Mistral interprets result                          │
│  Decision: NORMAL (stop) or ESCALATE (continue)     │
└──────────────────────┬──────────────────────────────┘
                       ↓ (if anomaly found)
┌─────────────────────────────────────────────────────┐
│  Agent 2 — Diagnosis                                │
│  Calls search_manuals() → ChromaDB RAG              │
│  Calls get_history()    → SQLite                    │
│  Optionally: LLaVA reads machine photo              │
│  Analysis mode determined by evidence               │
│  Mistral produces grounded diagnosis                │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  Agent 3 — Recommendation                           │
│  Mistral writes final engineer-facing report        │
│  Pydantic validates structured output               │
│  Returns FaultReport with severity + action         │
└──────────────────────┬──────────────────────────────┘
                       ↓
        Dashboard shows fault report
        Engineer clicks Approve → SQLite log
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Agent orchestration | LangGraph | Multi-agent pipeline with shared state |
| LLM — text | Mistral-7B via Ollama | Interpretation, diagnosis, report writing |
| LLM — vision | LLaVA via Ollama | Visual inspection of machine photos |
| Embeddings | nomic-embed-text via Ollama | Embedding manual chunks for RAG |
| Vector database | ChromaDB | Semantic search over SKF manual |
| Tool interface | LangChain tools | MCP tool server callable by agents |
| REST API | FastAPI | Backend serving the pipeline |
| Dashboard UI | Streamlit | Engineer-facing interface |
| Structured output | Pydantic | Validated fault report schema |
| Data analysis | Pandas + NumPy | Sensor file loading, RMS computation |
| Fault storage | SQLite | Approved fault history |
| PDF extraction | pypdf | Extracting SKF manual text |
| Dataset | NASA IMS Bearing Dataset | 9,464 real sensor files from 3 run-to-failure experiments |
| Knowledge base | SKF Bearing Maintenance Handbook | 1,721 embedded chunks in ChromaDB |

---

## 📂 Project Structure

```
predictive-maintenance-copilot/
│
├── agents/
│   ├── monitor.py          # Agent 1 — anomaly detection + escalation
│   ├── diagnosis.py        # Agent 2 — RAG + LLaVA + Mistral diagnosis
│   └── recommendation.py   # Agent 3 — Pydantic fault report
│
├── tools/
│   ├── anomaly.py          # RMS baseline + percentage deviation detection
│   ├── rag.py              # PDF loading, chunking, ChromaDB indexing
│   └── history.py          # SQLite fault history read/write
│
├── data/
│   └── NASA_Bearing_dataset/
│       ├── 1st_test/       # 2,156 files — Bearing 3 & 4 fail
│       ├── 2nd_test/       # 984 files  — Bearing 1 fails
│       └── 3rd_test/       # 6,324 files — Bearing 3 fails
│
├── manuals/
│   └── SKF-bearing-maintenance-handbook.pdf
│
├── chroma_db/              # Auto-created — ChromaDB vector store
│
├── models.py               # Pydantic schemas (AnomalyResult, FaultReport, etc.)
├── mcp_server.py           # MCP tool server — exposes 3 tools to agents
├── pipeline.py             # LangGraph graph — wires all 3 agents
├── api.py                  # FastAPI app
├── dashboard.py            # Streamlit UI
├── fault_history.db        # SQLite — auto-created on first approval
└── requirements.txt
```

---

## 🔬 How Anomaly Detection Works

Each bearing has its own **per-channel baseline** computed from the first 20 sensor files (confirmed healthy across all experiments).

```
For each sensor file:
  RMS = √( mean(readings²) )          ← Root Mean Square per channel
  deviation = (RMS - baseline_mean)
              ─────────────────────  × 100%
                  baseline_mean

Severity thresholds (tuned per machine from data analysis):

  bearing_1 (Motor line):      minor=30%  major=50%  critical=75%
  bearing_2 (Pump station):    minor=20%  major=50%  critical=100%
  bearing_4 (Conveyor system): minor=15%  major=30%  critical=60%
```

Validated manually at 1%, 5%, 20%, 50%, 80%, 95%, and 99% of each experiment's timeline — healthy files correctly return no anomaly, fault files correctly escalate.

---

## 🤖 Multimodal Analysis

When a machine photo is uploaded:

1. LLaVA analyses it with a structured prompt and returns:
   - Component type (bearing, motor, gear, shaft, etc.)
   - Condition (new / good / worn / damaged / severely damaged)
   - Visible damage (yes/no), damage type, severity, confidence

2. `determine_analysis_mode()` decides what drives the diagnosis:

| Sensor | Image | Mode | Meaning |
|---|---|---|---|
| Anomaly ✓ | Damage ✓ | `both_confirm` | Strongest case — both confirm fault |
| Anomaly ✓ | No damage ✓ | `sensor_primary` | Likely internal fault not visible externally |
| Normal ✓ | Damage ✓ | `visual_primary` | Surface damage — sensor not yet affected |
| Normal ✓ | No damage ✓ | `no_fault` | Machine healthy — no action needed |
| — | No image | `sensor_only` | Sensor data drives diagnosis |

This prevents hallucinations — the system only reports faults supported by actual evidence.

---

## 📊 Dataset

**NASA IMS Bearing Dataset** — University of Cincinnati / NASA Ames

Three real run-to-failure experiments recorded at 2,000 RPM under 6,000 lbs load:

| Experiment | Files | Duration | Failure |
|---|---|---|---|
| 1st test (bearing_1) | 2,156 | Oct–Nov 2003 (34 days) | Bearing 3 inner race + Bearing 4 roller |
| 2nd test (bearing_2) | 984 | Feb–Mar 2004 (44 days) | Bearing 1 outer race |
| 3rd test (bearing_4) | 6,324 | Mar–Apr 2004 (45 days) | Bearing 3 outer race |

Each file: 1-second vibration snapshot, 20,480 readings at 20 kHz, recorded every 10 minutes.

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10 or 3.11
- [Ollama](https://ollama.com) installed on your machine
- ~10 GB free disk space for models

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/predictive-maintenance-copilot.git
cd predictive-maintenance-copilot
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

All required packages with exact versions are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Pull Ollama models

```bash
ollama pull mistral
ollama pull llava
ollama pull nomic-embed-text
```

> ⚠️ Total download: ~8.5 GB. Ensure Ollama is running before pulling.

### 5. Download the dataset

Download the **NASA IMS Bearing Dataset** from Kaggle:
```
https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset
```
Extract and place in:
```
data/NASA_Bearing_dataset/
  ├── 1st_test/1st_test/
  ├── 2nd_test/2nd_test/
  └── 3rd_test/4th_test/txt/
```

### 6. Download the SKF manual

Download the **SKF Bearing Maintenance Handbook** (free PDF):
```
https://web.mit.edu/2.70/Reading%20Materials/SKF-bearing-maintenance-handbook.pdf
```
Save as:
```
manuals/SKF-bearing-maintenance-handbook.pdf
```

### 7. Load manual into ChromaDB (first time only)

```bash
python tools/rag.py
```

This chunks the PDF, embeds it with nomic-embed-text, and stores 1,721 chunks in ChromaDB. Takes 3–8 minutes. Only needs to run once.

---

## 🚀 Running the Project

Open **two terminals**, both with the virtual environment active:

**Terminal 1 — API backend:**
```bash
python api.py
```
API runs at `http://localhost:8000`  
Swagger docs at `http://localhost:8000/docs`

**Terminal 2 — Streamlit dashboard:**
```bash
streamlit run dashboard.py
```
Dashboard opens at `http://localhost:8501`

---

## 🎮 How to Use

1. **Select a machine** from the dropdown (Motor line / Pump station / Conveyor system)
2. **Drag the slider** to any point in the experiment timeline
   - Left (low numbers) → early in experiment → bearing healthy
   - Right (high numbers) → late in experiment → bearing degrading or failed
3. **Optionally upload** a machine component photo for visual inspection
4. **Click Run Analysis** — watch all 3 agents complete in sequence
5. **Read the fault report** — severity, root cause, recommended action, manual reference
6. **Click Approve & Log Fault** to save to database, or **Override** if you disagree

---

## 📋 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/machines` | Returns all machines with file counts |
| `POST` | `/analyse-sensor` | Runs full pipeline — returns FaultReport |
| `POST` | `/approve-fault` | Logs approved fault to SQLite |

**POST /analyse-sensor parameters:**

```
machine_id   string  "bearing_1" / "bearing_2" / "bearing_4"
file_index   int     which sensor file to analyse (1-based)
asset_id     string  engineer-facing asset name (e.g. "BEARING_01")
image_file   file    optional machine photo (jpg/png)
```

---

## ⚙️ Components

| File | Role |
|---|---|
| `tools/anomaly.py` | RMS % deviation detection — 3 machines, 9,464 sensor files |
| `tools/rag.py` | ChromaDB RAG over SKF bearing manual — 1,721 chunks |
| `tools/history.py` | SQLite fault history — save and retrieve past faults |
| `mcp_server.py` | MCP tool server — 3 tools exposed to LangGraph agents |
| `agents/monitor.py` | Anomaly check + ESCALATE / NORMAL decision |
| `agents/diagnosis.py` | Root cause via RAG + history + optional LLaVA vision |
| `agents/recommendation.py` | Structured Pydantic fault report generation |
| `pipeline.py` | LangGraph graph — wires all 3 agents with shared state |
| `api.py` | FastAPI REST backend |
| `dashboard.py` | Streamlit HITL dashboard |
| `models.py` | Pydantic schemas — AnomalyResult, FaultReport |

---

## 🛠️ MCP Tools

| Tool | Description |
|---|---|
| `tool_detect_anomaly` | RMS % deviation analysis on bearing sensor data |
| `tool_search_manuals` | RAG search over SKF bearing maintenance manual |
| `tool_get_history` | Retrieve past fault records for a given asset |

---

## 🧪 Example Output

**Fault detected:**

```json
{
  "status": "anomaly_detected",
  "machine_id": "bearing_1",
  "fault_report": {
    "asset_id": "BEARING_01",
    "machine_label": "Motor line bearing",
    "fault_type": "Fretting Corrosion and Forced Fracture",
    "severity": "Critical",
    "recommended_action": "Inspect and replace the damaged motor line bearing immediately. Investigate the cause of fretting corrosion to prevent future occurrences.",
    "manual_reference": "SKF-bearing-maintenance-handbook.pdf, Fig. 22",
    "confidence": "high",
    "analysis_mode": "both_confirm"
  }
}
```

**No fault:**

```json
{
  "status": "normal",
  "machine_id": "bearing_2",
  "message": "No anomaly detected. Machine is operating normally."
}
```

---

## 🤖 Agent Behaviour

| Scenario | Behaviour |
|---|---|
| Healthy bearing (early files) | Monitor Agent → NORMAL → pipeline stops |
| Degrading bearing (mid files) | Monitor Agent → ESCALATE → full diagnosis runs |
| Failed bearing (late files) | Monitor Agent → ESCALATE → CRITICAL report generated |
| Image uploaded, damage visible | LLaVA inspects → `both_confirm` mode → stronger diagnosis |
| Image uploaded, no damage visible | `sensor_primary` mode → internal fault flagged |
| No image uploaded | Sensor data + manual RAG drives diagnosis |
| Repeated fault on same asset | History tool surfaces past faults for context |

---

## ⚠️ Honest Limitations

- Sensor data comes from pre-recorded NASA files — not a live IoT stream
- No model training or fine-tuning — uses pre-trained Mistral and LLaVA
- LLM responses take 15–40 seconds on CPU (no GPU required)
- Single user — no authentication or multi-user support
- No Docker containerisation or cloud deployment

In a production setting, the CSV file reader would be replaced by a live MQTT/IoT sensor stream. The agent pipeline, RAG system, and HITL workflow require no architectural changes.

---

## 👥 Authors

**Arya Tamhane**  
M.Sc. Big Data and Artificial Intelligence — SRH University of Applied Sciences Heidelberg (Leipzig Campus)  
[GitHub](https://github.com/AryaTamhane18) · [LinkedIn](https://www.linkedin.com/in/arya-tamhane-543a21231/)

**Rudra Pingale**  
M.Sc. Big Data and Artificial Intelligence — SRH University of Applied Sciences Heidelberg (Leipzig Campus)

---

## 📄 Licence

MIT

The NASA IMS Bearing Dataset is publicly available via the NASA Prognostics Data Repository.  
The SKF Bearing Maintenance Handbook is publicly available on the MIT server.
