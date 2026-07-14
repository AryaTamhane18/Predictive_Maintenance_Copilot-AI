# tools/anomaly.py
# RMS-based anomaly detection
# Uses percentage deviation from baseline instead of Z-score
# Z-score fails when std is extremely small (bearing_1, bearing_4)
# bearing_2 uses Z-score because it has a clear dramatic spike

import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import AnomalyResult

BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"
)

MACHINES = {
    "bearing_1": {
        "folder":           os.path.join(BASE, "1st_test", "1st_test"),
        "label":            "Motor line bearing",
        "failed":           "Bearing 3 (inner race) + Bearing 4 (roller)",
        "n_channels":       8,
        "monitor_channels": ["ch5", "ch6", "ch8"],
        "minor_pct":    30,
        "major_pct":    50,
        "critical_pct": 75
    },
    "bearing_2": {
        "folder":           os.path.join(BASE, "2nd_test", "2nd_test"),
        "label":            "Pump station bearing",
        "failed":           "Bearing 1 (outer race)",
        "n_channels":       4,
        "monitor_channels": ["ch1", "ch2", "ch3", "ch4"],
        "minor_pct":    20,
        "major_pct":    50,
        "critical_pct": 100
    },
    "bearing_4": {
        "folder":           os.path.join(BASE, "3rd_test", "4th_test", "txt"),
        "label":            "Conveyor system bearing",
        "failed":           "Bearing 3 (outer race)",
        "n_channels":       4,
        "monitor_channels": ["ch3", "ch4"],
        "minor_pct":    15,
        "major_pct":    30,
        "critical_pct": 60
    }
}

# Cache baseline per machine — computed once per session
_baseline_cache = {}
N_BASELINE = 20  # first 20 files — confirmed healthy for all machines


def get_sensor_files(machine_id: str) -> list:
    """All valid sensor files sorted chronologically."""
    if machine_id not in MACHINES:
        raise ValueError(f"Unknown machine: {machine_id}. "
                         f"Valid: {list(MACHINES.keys())}")
    folder = MACHINES[machine_id]["folder"]
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Not found: {folder}")
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.getsize(os.path.join(folder, f)) > 1000
        and "readme" not in f.lower()
        and not f.endswith('.pdf')
    ])


def load_file(file_path: str) -> pd.DataFrame:
    """Load a NASA sensor file."""
    for enc in ['latin-1', 'utf-8', 'cp1252']:
        try:
            df = pd.read_csv(
                file_path, sep='\t', header=None,
                encoding=enc, engine='python'
            )
            if df.shape[1] >= 1:
                df.columns = [f'ch{i+1}' for i in range(df.shape[1])]
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df.dropna()
        except Exception:
            continue
    raise ValueError(f"Cannot read: {file_path}")


def compute_rms(series: pd.Series) -> float:
    """Root Mean Square of a signal."""
    arr = series.dropna().values.astype(float)
    return float(np.sqrt(np.mean(arr ** 2))) if len(arr) > 0 else 0.0


def get_baseline(machine_id: str, files: list) -> dict:
    """
    Compute baseline RMS per channel from first N_BASELINE files.
    Returns {channel: baseline_rms}
    """
    if machine_id in _baseline_cache:
        return _baseline_cache[machine_id]

    monitor_chs    = MACHINES[machine_id]["monitor_channels"]
    baseline_files = files[:N_BASELINE]
    channel_rms    = {ch: [] for ch in monitor_chs}

    for f in baseline_files:
        try:
            df = load_file(f)
            for ch in monitor_chs:
                if ch in df.columns:
                    rms = compute_rms(df[ch])
                    if rms > 0:
                        channel_rms[ch].append(rms)
        except Exception:
            continue

    baseline = {}
    for ch, vals in channel_rms.items():
        if vals:
            baseline[ch] = float(np.mean(vals))

    _baseline_cache[machine_id] = baseline
    return baseline


def detect_anomaly(machine_id: str,
                   file_index:  int) -> AnomalyResult:
    """
    Detect anomaly using percentage deviation from baseline RMS.

    For each monitored channel:
        pct_deviation = (current_RMS - baseline_RMS) / baseline_RMS * 100

    Severity based on machine-specific thresholds:
        none     → pct < minor_pct
        minor    → minor_pct  <= pct < major_pct
        major    → major_pct  <= pct < critical_pct
        critical → pct >= critical_pct
    """
    files = get_sensor_files(machine_id)
    total = len(files)

    file_index = max(1, min(file_index, total))
    idx        = file_index - 1

    config   = MACHINES[machine_id]
    baseline = get_baseline(machine_id, files)

    target_df    = load_file(files[idx])
    monitor_chs  = [c for c in config["monitor_channels"]
                    if c in target_df.columns and c in baseline]

    if not monitor_chs:
        monitor_chs = list(target_df.columns)[:1]

    # find worst channel by percentage deviation
    worst_channel  = None
    worst_pct      = 0.0
    channel_pcts   = {}

    for ch in monitor_chs:
        rms      = compute_rms(target_df[ch])
        base_rms = baseline.get(ch, rms)
        if base_rms == 0:
            continue
        pct = ((rms - base_rms) / base_rms) * 100
        channel_pcts[ch] = round(pct, 2)

        if pct > worst_pct:
            worst_pct     = pct
            worst_channel = ch

    if worst_channel is None:
        worst_channel = monitor_chs[0]
        worst_pct     = 0.0

    # severity from machine-specific thresholds
    minor_pct    = config["minor_pct"]
    major_pct    = config["major_pct"]
    critical_pct = config["critical_pct"]

    is_anomalous = worst_pct >= minor_pct

    if worst_pct >= critical_pct:
        severity = "critical"
    elif worst_pct >= major_pct:
        severity = "major"
    elif worst_pct >= minor_pct:
        severity = "minor"
    else:
        severity = "none"

    # anomaly count in worst channel
    ch_data       = target_df[worst_channel].dropna()
    ch_mean_raw   = ch_data.mean()
    ch_std_raw    = ch_data.std() + 1e-9
    anomaly_count = int(
        (ch_data.abs() > abs(ch_mean_raw) + 3 * ch_std_raw).sum()
    )

    timestamp = os.path.basename(files[idx])
    label     = config["label"]

    return AnomalyResult(
        machine_id     = machine_id,
        file_index     = file_index,
        total_files    = total,
        is_anomalous   = is_anomalous,
        channel        = worst_channel,
        max_z_score    = round(worst_pct, 2),
        anomaly_count  = anomaly_count,
        total_readings = len(target_df),
        severity       = severity,
        timestamp_name = timestamp,
        message        = (
            f"{'⚠️ Anomaly' if is_anomalous else '✅ Normal'} — "
            f"{label} | "
            f"File {file_index}/{total} ({timestamp}) | "
            f"Worst channel: {worst_channel} | "
            f"Deviation: {round(worst_pct, 2)}% above baseline | "
            f"Severity: {severity}"
        )
    )


def get_machine_info(machine_id: str) -> dict:
    """Return metadata for dashboard."""
    if machine_id not in MACHINES:
        return {}
    files = get_sensor_files(machine_id)
    info  = MACHINES[machine_id].copy()
    info["total_files"] = len(files)
    info["first_file"]  = os.path.basename(files[0])  if files else ""
    info["last_file"]   = os.path.basename(files[-1]) if files else ""
    return info


def get_all_machines() -> dict:
    """Return all machine configs — used by dashboard dropdown."""
    return MACHINES


if __name__ == "__main__":
    print("=" * 70)
    print("  Percentage Deviation Anomaly Detection")
    print("=" * 70)

    for machine_id in ["bearing_1", "bearing_2", "bearing_4"]:
        config = MACHINES[machine_id]
        files  = get_sensor_files(machine_id)
        total  = len(files)

        print(f"\n{machine_id} — {config['label']}")
        print(f"  Total files: {total} | "
              f"Monitoring: {config['monitor_channels']}")
        print(f"  Thresholds: "
              f"minor={config['minor_pct']}% | "
              f"major={config['major_pct']}% | "
              f"critical={config['critical_pct']}%")

        baseline = get_baseline(machine_id, files)
        print(f"  Baseline RMS:")
        for ch, val in baseline.items():
            print(f"    {ch}: {val:.6f}")

        test_points = [
            ("Healthy  (1%)",  max(1, int(total * 0.01))),
            ("Healthy  (5%)",  max(1, int(total * 0.05))),
            ("Early   (20%)",  max(1, int(total * 0.20))),
            ("Mid     (50%)",  max(1, int(total * 0.50))),
            ("Late    (80%)",  max(1, int(total * 0.80))),
            ("Fault   (95%)",  max(1, int(total * 0.95))),
            ("Near end (99%)", max(1, int(total * 0.99))),
        ]

        print(f"\n  {'Point':<20} {'File':>6} "
              f"{'Deviation%':>12} {'Severity':>10} "
              f"{'Channel':>8} {'Anomaly':>8}")
        print(f"  {'-'*70}")

        for label, fidx in test_points:
            r = detect_anomaly(machine_id, fidx)
            print(f"  {label:<20} {fidx:>6} "
                  f"{r.max_z_score:>11.2f}% "
                  f"{r.severity:>10} "
                  f"{r.channel:>8} "
                  f"{str(r.is_anomalous):>8}")

    print("\n" + "=" * 70)
    print("  Expected:")
    print("  bearing_1: healthy=False, fault at 95%+ = True")
    print("  bearing_2: healthy=False, fault at 80%+ = True")
    print("  bearing_4: healthy=False, fault at 95%+ = True")
    print("=" * 70)

    print("Hi")