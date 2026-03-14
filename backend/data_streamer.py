"""
Project Chronos - MIMIC Patient Streamer
=========================================
Simulates a live ICU bedside monitor by replaying MIMIC-III and MIMIC-IV
Demo patient records and streaming them to the backend API.

This mimics the actual workflow of an HL7 FHIR feed or a real-time
hospital integration engine (like Mirth Connect) sending continuous
patient vitals updates to a clinical decision support endpoint.

Usage:
  python backend/data_streamer.py [--speed 5] [--patients 20]
  
  --speed:    Seconds between each vital sign update (default: 5s)
  --patients: Max concurrent patients to simulate (default: 20)
"""

import sys
import time
import json
import httpx
import asyncio
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from loguru import logger

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "data"
API_BASE   = "http://localhost:8000"

# MIMIC item IDs for vital sign chart events
# MIMIC-IV MetaVision IDs (220xxx) + MIMIC-III CareVue legacy IDs
# CareVue system was used for MIMIC-III patients with subject_id < 100000
# MetaVision was used for MIMIC-III patients with subject_id >= 200000 and all MIMIC-IV
MIMIC_ITEM_VITALS = {
    # ── Heart Rate ──────────────────────────────────────────
    220045: "heart_rate",           # MIMIC-IV / MetaVision
    211:    "heart_rate",           # MIMIC-III CareVue
    # ── Systolic BP ─────────────────────────────────────────
    220050: "systolic_bp",          # MIMIC-IV MetaVision (arterial line)
    51:     "systolic_bp",          # MIMIC-III CareVue arterial
    455:    "systolic_bp",          # MIMIC-III CareVue NBP
    # ── Diastolic BP ────────────────────────────────────────
    220051: "diastolic_bp",         # MIMIC-IV MetaVision (arterial)
    8368:   "diastolic_bp",         # MIMIC-III CareVue arterial
    8441:   "diastolic_bp",         # MIMIC-III CareVue NBP
    # ── Mean Arterial Pressure ───────────────────────────────
    220052: "mean_arterial_pressure", # MIMIC-IV MetaVision
    52:     "mean_arterial_pressure", # MIMIC-III CareVue arterial
    456:    "mean_arterial_pressure", # MIMIC-III CareVue NBP
    # ── Respiratory Rate ─────────────────────────────────────
    220210: "respiratory_rate",     # MIMIC-IV MetaVision
    618:    "respiratory_rate",     # MIMIC-III CareVue
    615:    "respiratory_rate",     # MIMIC-III CareVue (alt)
    # ── Temperature ─────────────────────────────────────────
    223762: "temperature",          # MIMIC-IV/MetaVision (Celsius)
    678:    "temperature",          # MIMIC-III CareVue (Fahrenheit — converted below)
    # ── SpO2 ────────────────────────────────────────────────
    220277: "spo2",                 # MIMIC-IV MetaVision
    646:    "spo2",                 # MIMIC-III CareVue
    50816:  "spo2",                 # Lab: O2 Saturation
    # ── Lab Values (shared MIMIC-III and IV) ─────────────────
    51006:  "lactate",              # Blood Lactate
    50010:  "lactate",              # MIMIC-III labevents lactate
    51300:  "wbc",                  # White Blood Cells
    50912:  "creatinine",           # Creatinine
    50885:  "bilirubin",            # Bilirubin, Total
    51265:  "platelets",            # Platelet Count
    50821:  "pao2",                 # pO2
}

# MIMIC-III / IV field name mapping
MIMIC4_CHART_COLS  = ["subject_id", "stay_id",  "charttime", "itemid", "valuenum"]
MIMIC3_CHART_COLS  = ["subject_id", "icustay_id", "charttime", "itemid", "valuenum"]


# ─────────────────────────────────────────────
# MIMIC Data Loader
# ─────────────────────────────────────────────

def load_mimic_patients(max_patients: int = 100) -> list[dict]:
    """
    Loads patient time-series from MIMIC-III and/or MIMIC-IV Demo.
    Returns a list of patient dicts, each containing a sorted timeline of vitals.
    """
    patients = []
    
    for mimic_name, mimic_dir, chart_cols in [
        ("MIMIC-IV",  DATA_DIR / "mimic4_demo",  MIMIC4_CHART_COLS),
        ("MIMIC-III", DATA_DIR / "mimic3_demo", MIMIC3_CHART_COLS),
    ]:
        # MIMIC-III: CHARTEVENTS.csv (uppercase) in versioned subdir
        # MIMIC-IV:  chartevents.csv in icu/ subdir
        # Use rglob + case-insensitive match to handle both.
        chart_candidates = list(mimic_dir.rglob("chartevents.csv")) + \
                           list(mimic_dir.rglob("CHARTEVENTS.csv"))
        lab_candidates   = list(mimic_dir.rglob("labevents.csv"))   + \
                           list(mimic_dir.rglob("LABEVENTS.csv"))
        chart_path = chart_candidates[0] if chart_candidates else None
        lab_path   = lab_candidates[0]   if lab_candidates   else None

        if chart_path is None:
            logger.warning(f"  {mimic_name}: chartevents.csv not found in {mimic_dir}. Skipping.")
            continue
        
        logger.info(f"Loading {mimic_name} chartevents...")
        
        try:
            # Load only the relevant item IDs to reduce memory usage
            chart_df = pd.read_csv(chart_path, usecols=lambda c: c in chart_cols, low_memory=False)
            chart_df.columns = chart_df.columns.str.lower()
            chart_df["charttime"] = pd.to_datetime(chart_df["charttime"], errors="coerce")
            chart_df = chart_df.dropna(subset=["charttime", "valuenum"])
            
            # Filter to only the vital sign item IDs we care about
            chart_df = chart_df[chart_df["itemid"].isin(MIMIC_ITEM_VITALS)]
            chart_df["vital_name"] = chart_df["itemid"].map(MIMIC_ITEM_VITALS)
            
            # Merge lab events if available
            if lab_path is not None and lab_path.exists():
                try:
                    lab_df = pd.read_csv(lab_path, low_memory=False)
                    lab_df.columns = lab_df.columns.str.lower()
                    lab_df["charttime"] = pd.to_datetime(lab_df.get("charttime", lab_df.get("labevents_charttime", pd.NaT)), errors="coerce")
                    lab_df = lab_df[lab_df["itemid"].isin(MIMIC_ITEM_VITALS)]
                    lab_df["vital_name"] = lab_df["itemid"].map(MIMIC_ITEM_VITALS)
                    chart_df = pd.concat([chart_df, lab_df[chart_df.columns.intersection(lab_df.columns)]], ignore_index=True)
                except Exception as e:
                    logger.warning(f"  Lab events merge failed: {e}")
            
            # Build patient-level vital sign timeline
            id_col = "stay_id" if "stay_id" in chart_df.columns else "icustay_id"
            if id_col not in chart_df.columns:
                id_col = "subject_id"
            
            subject_ids = chart_df["subject_id"].dropna().unique()[:max_patients]
            
            for sid in subject_ids:
                pat_df = chart_df[chart_df["subject_id"] == sid].copy()
                pat_df = pat_df.sort_values("charttime")
                
                # Pivot to wide format: one row per timestamp per vital
                timeline = (
                    pat_df.pivot_table(
                        index="charttime",
                        columns="vital_name",
                        values="valuenum",
                        aggfunc="mean"
                    )
                    .reset_index()
                    .rename(columns={"charttime": "timestamp"})
                )
                
                if len(timeline) < 3:
                    continue  # Skip patients with too little data

                # Convert Fahrenheit → Celsius for MIMIC-III CareVue item 678 (BT in °F).
                # After pivot, temperature column may contain values from item 678 (°F),
                # item 223762 (°C MetaVision), or both averaged. If the value exceeds 45°C
                # (physiologically impossible), it must be in Fahrenheit — convert.
                if "temperature" in timeline.columns:
                    timeline["temperature"] = timeline["temperature"].apply(
                        lambda v: (v - 32) / 1.8 if (pd.notna(v) and v > 45) else v
                    )
                
                patient = {
                    "patient_id": str(int(sid)),
                    "source": mimic_name,
                    "timeline": timeline,
                    "total_rows": len(timeline),
                    "current_row": 0,
                }
                patients.append(patient)
            
            logger.success(f"  {mimic_name}: Loaded {len([p for p in patients if p['source'] == mimic_name])} patients")
        
        except Exception as e:
            logger.error(f"  Failed to load {mimic_name}: {e}")
    
    if not patients:
        logger.warning("No MIMIC data loaded. Generating synthetic fallback patients for testing...")
        patients = generate_synthetic_patients(n=max_patients)
    
    return patients[:max_patients]


def generate_synthetic_patients(n: int = 10) -> list[dict]:
    """
    Generates entirely synthetic patient vitals for testing the pipeline
    when MIMIC demo data is not yet downloaded.
    
    Simulates realistic ICU vital progressions with some patients
    deteriorating toward sepsis/hypotension/cardiac arrest.
    """
    logger.info(f"Generating {n} synthetic test patients...")
    patients = []
    np.random.seed(42)
    
    for i in range(n):
        n_hours = np.random.randint(24, 72)
        t_range = pd.date_range("2024-01-01", periods=n_hours * 12, freq="5min")  # Every 5 min
        
        # Is this patient deteriorating? (30% chance)
        is_deteriorating = np.random.random() < 0.30
        deterioration_start = int(len(t_range) * 0.60) if is_deteriorating else len(t_range)
        
        def vital_trajectory(normal, disturbed, noise_std, start):
            vals = np.full(len(t_range), normal, dtype=float)
            if is_deteriorating:
                for j in range(start, len(t_range)):
                    progress = (j - start) / max(len(t_range) - start, 1)
                    vals[j] = normal + (disturbed - normal) * progress
            vals += np.random.normal(0, noise_std, len(t_range))
            return vals
        
        # Deteriorating: HR rises, MAP falls, Lactate spikes, SpO2 drops
        timeline = pd.DataFrame({
            "timestamp":              t_range,
            "heart_rate":             vital_trajectory(75, 115, 3, deterioration_start),
            "systolic_bp":            vital_trajectory(125, 85, 4, deterioration_start),
            "mean_arterial_pressure": vital_trajectory(93, 55, 3, deterioration_start),
            "diastolic_bp":           vital_trajectory(75, 45, 3, deterioration_start),
            "spo2":                   vital_trajectory(98, 89, 1, deterioration_start),
            "respiratory_rate":       vital_trajectory(15, 26, 1, deterioration_start),
            "temperature":            vital_trajectory(37.0, 38.8, 0.2, deterioration_start),
            "lactate":                vital_trajectory(1.2, 4.5, 0.2, deterioration_start),
            "creatinine":             vital_trajectory(1.0, 2.5, 0.1, deterioration_start),
            "wbc":                    vital_trajectory(9.0, 18.0, 0.5, deterioration_start),
            "gcs":                    np.clip(vital_trajectory(15, 10, 0.3, deterioration_start), 3, 15).astype(int),
            # BUG-G fix: Add vasopressor_dose (NE-equivalent, mcg/kg/min)
            "vasopressor_dose":       np.clip(
                vital_trajectory(0.0, 0.15, 0.005, deterioration_start), 0.0, None
            ),
            # BUG-G fix: Add FiO2 and PaO2 so PF ratio / A-a gradient can be computed.
            "fio2":                   np.clip(
                vital_trajectory(0.21, 0.80, 0.02, deterioration_start), 0.21, 1.0
            ),
            "pao2":                   np.clip(
                vital_trajectory(95.0, 58.0, 3.0, deterioration_start), 40.0, 150.0
            ),
            # BUG-CLAUDE-6-1 fix: Add bilirubin trajectory.
            # Without this, SOFA hepatic sub-score was always 0 and bilirubin_measured=0
            # for all synthetic patients. Deteriorating: liver congestion → high bilirubin.
            "bilirubin":              np.clip(
                vital_trajectory(0.8, 4.5, 0.1, deterioration_start), 0.3, 30.0
            ),
            # BUG-CLAUDE-6-2 fix: Add platelets trajectory.
            # Without this, SOFA coagulation sub-score was always 0 and platelets_measured=0.
            # Deteriorating: DIC / sepsis induces thrombocytopenia.
            "platelets":              np.clip(
                vital_trajectory(250.0, 60.0, 5.0, deterioration_start), 10.0, 500.0
            ),
        })

        # Clip to physiological ranges
        timeline["heart_rate"]   = timeline["heart_rate"].clip(30, 200)
        timeline["spo2"]         = timeline["spo2"].clip(60, 100)
        timeline["systolic_bp"]  = timeline["systolic_bp"].clip(50, 240)
        timeline["mean_arterial_pressure"] = timeline["mean_arterial_pressure"].clip(30, 150)
        timeline["lactate"]      = timeline["lactate"].clip(0.5, 15.0)

        
        patients.append({
            "patient_id":    f"SYN_{i+1:04d}",
            "source":        "Synthetic",
            "timeline":      timeline,
            "total_rows":    len(timeline),
            "current_row":   0,
            "is_deteriorating": is_deteriorating,
        })
    
    logger.success(f"Generated {n} synthetic patients ({sum(p['is_deteriorating'] for p in patients)} deteriorating)")
    return patients


# ─────────────────────────────────────────────
# Ground Truth Computation
# ─────────────────────────────────────────────

def compute_ground_truth(patient: dict) -> dict:
    """
    Computes ground truth labels for a patient from their FULL timeline.
    Since we have the entire ICU stay, we know what actually happened.
    
    This is the "answer key" that lets viewers validate whether the
    ML predictions are correct. It uses simple clinical thresholds
    applied to FUTURE data that the model has NOT yet seen.
    
    Returns dict with:
      - sepsis_occurred: bool  (did patient develop sepsis indicators?)
      - bp_collapse_occurred: bool  (did MAP drop critically?)
      - cardiac_event_occurred: bool  (did patient have cardiac instability?)
      - overall_deteriorated: bool  (any of the above?)
      - max_severity: str  (STABLE/MILD/SEVERE/CRITICAL)
      - timeline_progress_pct: float  (how far through the stay we are)
      - events_detail: list  (what happened and when)
    """
    timeline = patient["timeline"]
    idx = patient["current_row"]
    total = len(timeline)
    source = patient.get("source", "Unknown")
    
    # For synthetic patients, ground truth is known from generation
    if source == "Synthetic":
        is_det = patient.get("is_deteriorating", False)
        return {
            "sepsis_occurred": is_det,
            "bp_collapse_occurred": is_det,
            "cardiac_event_occurred": is_det,
            "overall_deteriorated": is_det,
            "max_severity": "SEVERE" if is_det else "STABLE",
            "timeline_progress_pct": round((idx / max(total, 1)) * 100, 1),
            "hours_remaining": round((total - idx) * 5 / 60, 1),  # 5-min intervals
            "events_detail": [
                {"event": "Synthetic deterioration", "severity": "SEVERE"}
            ] if is_det else [],
        }
    
    # For MIMIC patients, analyze the FUTURE portion of the timeline
    future = timeline.iloc[idx:] if idx < total else timeline.iloc[-10:]
    events = []
    
    # ── Sepsis: Sepsis-3 aligned — SOFA delta ≥ 2 from baseline ──
    # Old definition (Lactate > 4 + Temp/WBC) labeled chronic elevated lactate
    # patients (liver failure, malnutrition) as sepsis-positive — inflating FN rate.
    # Sepsis-3 (Singer et al., JAMA 2016): sepsis = life-threatening organ dysfunction
    # from infection, identified clinically by an ACUTE SOFA change ≥ 2 points.
    # Reference: Singer M. et al. JAMA. 2016;315(8):801-810.
    sepsis = False
    if "sofa_score" in future.columns and "sofa_score" in timeline.columns:
        # Baseline SOFA = median of first obs window (pre-current, up to 4 rows)
        baseline_window = timeline.iloc[max(0, idx - 4):idx]
        baseline_sofa = baseline_window["sofa_score"].dropna()
        baseline_val = float(baseline_sofa.median()) if len(baseline_sofa) > 0 else 0.0
        future_sofa = future["sofa_score"].dropna()
        if len(future_sofa) > 0:
            sofa_delta = float(future_sofa.max()) - baseline_val
            if sofa_delta >= 2.0:
                # Also require at least one infection-suggestive biomarker (not just any organ dysfunction)
                infection_signal = False
                if "lactate" in future.columns and future["lactate"].dropna().max() > 2.0:
                    infection_signal = True
                if "wbc" in future.columns:
                    wbcs = future["wbc"].dropna()
                    if len(wbcs) > 0 and (wbcs.max() > 12.0 or wbcs.min() < 4.0):
                        infection_signal = True
                if "temperature" in future.columns:
                    temps = future["temperature"].dropna()
                    if len(temps) > 0 and (temps.max() > 38.3 or temps.min() < 36.0):
                        infection_signal = True
                if infection_signal:
                    sepsis = True
                    events.append({
                        "event": f"Sepsis-3: SOFA delta={sofa_delta:.1f} + infection biomarkers",
                        "severity": "SEVERE"
                    })
    elif "lactate" in future.columns:
        # Fallback: SOFA not in dataset — use old lactate+biomarker heuristic
        high_lactate = future["lactate"].dropna()
        if len(high_lactate) > 0 and high_lactate.max() > 4.0:
            temp_check = False
            wbc_check = False
            if "temperature" in future.columns:
                temps = future["temperature"].dropna()
                temp_check = len(temps) > 0 and temps.max() > 38.5
            if "wbc" in future.columns:
                wbcs = future["wbc"].dropna()
                wbc_check = len(wbcs) > 0 and wbcs.max() > 12.0
            if temp_check or wbc_check:
                sepsis = True
                events.append({"event": "Sepsis indicators (Lactate + Temp/WBC, fallback)", "severity": "SEVERE"})
    
    # ── BP Collapse: MAP <60 for any reading ──
    bp_collapse = False
    if "mean_arterial_pressure" in future.columns:
        maps = future["mean_arterial_pressure"].dropna()
        if len(maps) > 0 and maps.min() < 60.0:
            bp_collapse = True
            events.append({"event": f"MAP dropped to {maps.min():.0f} mmHg", "severity": "SEVERE"})
    elif "systolic_bp" in future.columns:
        sbps = future["systolic_bp"].dropna()
        if len(sbps) > 0 and sbps.min() < 80.0:
            bp_collapse = True
            events.append({"event": f"SBP dropped to {sbps.min():.0f} mmHg", "severity": "SEVERE"})
    
    # ── Cardiac event: HR >150 or <40, or MAP <50 ──
    cardiac = False
    if "heart_rate" in future.columns:
        hrs = future["heart_rate"].dropna()
        if len(hrs) > 0:
            if hrs.max() > 150:
                cardiac = True
                events.append({"event": f"Tachycardia: HR peaked at {hrs.max():.0f} bpm", "severity": "CRITICAL"})
            if hrs.min() < 40:
                cardiac = True
                events.append({"event": f"Bradycardia: HR dropped to {hrs.min():.0f} bpm", "severity": "CRITICAL"})
    if "mean_arterial_pressure" in future.columns:
        maps = future["mean_arterial_pressure"].dropna()
        if len(maps) > 0 and maps.min() < 50.0:
            cardiac = True
            if not any("MAP" in e["event"] for e in events):
                events.append({"event": f"Severe hypotension: MAP {maps.min():.0f} mmHg", "severity": "CRITICAL"})
    
    # Overall severity
    overall = sepsis or bp_collapse or cardiac
    if cardiac:
        severity = "CRITICAL"
    elif sepsis or bp_collapse:
        severity = "SEVERE"
    elif overall:
        severity = "MILD"
    else:
        severity = "STABLE"
    
    return {
        "sepsis_occurred": sepsis,
        "bp_collapse_occurred": bp_collapse,
        "cardiac_event_occurred": cardiac,
        "overall_deteriorated": overall,
        "max_severity": severity,
        "timeline_progress_pct": round((idx / max(total, 1)) * 100, 1),
        "hours_remaining": round((total - idx) * 5 / 60, 1),  # approximate assuming ~5min intervals
        "events_detail": events,
    }


# ─────────────────────────────────────────────
# Streaming Loop
# ─────────────────────────────────────────────

def get_next_vitals(patient: dict) -> Optional[dict]:
    """Returns the next row of vitals for a patient, cycling back to start."""
    idx = patient["current_row"]
    timeline = patient["timeline"]
    
    if idx >= len(timeline):
        patient["current_row"] = 0  # Loop back (ICU shift simulation)
        idx = 0
    
    row = timeline.iloc[idx]
    patient["current_row"] += 1
    
    vitals = {"patient_id": patient["patient_id"], "timestamp": datetime.now(timezone.utc).isoformat()}
    
    for col in timeline.columns:
        if col == "timestamp":
            continue
        val = row.get(col, np.nan)
        if not (isinstance(val, float) and np.isnan(val)):
            vitals[col] = float(val) if isinstance(val, (int, float, np.integer, np.floating)) else None
    
    # R17: Include ground truth for validation overlay
    vitals["ground_truth_labels"] = compute_ground_truth(patient)
    
    return vitals


async def stream_patient_to_api(
    patient: dict,
    client: httpx.AsyncClient,
    interval_seconds: float
):
    """Continuously streams one patient's vitals to the API."""
    while True:
        vitals = get_next_vitals(patient)
        if vitals:
            try:
                resp = await client.post(
                    f"{API_BASE}/predict",
                    json=vitals,
                    timeout=10.0
                )
                if resp.status_code == 200:
                    result = resp.json()
                    score = result.get("crash_probability_score", 0)
                    level = result.get("crash_risk_level", "LOW")
                    pid   = patient["patient_id"]
                    
                    # Bug 28 fix: API returns crash_probability_score as 0-100 percentage,
                    # NOT as a 0-1 fraction. Threshold should be 60.0 (60%), not 0.60.
                    if score > 60.0:
                        logger.warning(f"🚨 [{pid}] Crash Score: {score:.1f}% | {level}")
                    else:
                        logger.debug(f"✅ [{pid}] Crash Score: {score:.1f}% | {level}")
                else:
                    logger.error(f"API error {resp.status_code} for patient {patient['patient_id']}")
            except Exception as e:
                logger.error(f"Stream error for {patient['patient_id']}: {e}")
        
        await asyncio.sleep(interval_seconds)


async def run_streamer(patients: list[dict], interval: float):
    """Runs all patient streams concurrently."""
    logger.info(f"\n🏥 PROJECT CHRONOS STREAMER ONLINE")
    logger.info(f"   Streaming {len(patients)} patients every {interval}s → {API_BASE}/predict\n")
    
    async with httpx.AsyncClient() as client:
        # Verify the API is running
        try:
            resp = await client.get(f"{API_BASE}/health", timeout=5.0)
            logger.success(f"✅ API connected ({API_BASE}). Status: {resp.json()}")
        except Exception:
            logger.error(f"❌ Cannot reach API at {API_BASE}. Is the server running?")
            logger.info("   Run: uvicorn backend.api:app --host 0.0.0.0 --port 8000")
            return
        
        # Launch all concurrent patient streams
        tasks = [
            asyncio.create_task(stream_patient_to_api(p, client, interval))
            for p in patients
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Streamer stopped.")
        except KeyboardInterrupt:
            logger.info("Shutting down streamer...")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Project Chronos - Patient Streamer")
    parser.add_argument("--speed",    type=float, default=5.0,
                        help="Seconds between vital updates (default: 5)")
    parser.add_argument("--patients", type=int, default=100,
                        help="Max concurrent patients (default: 100)")
    parser.add_argument("--api",      type=str, default="http://localhost:8000",
                        help="Backend API URL (default: http://localhost:8000)")
    args = parser.parse_args()
    
    global API_BASE
    API_BASE = args.api
    
    patients = load_mimic_patients(max_patients=args.patients)
    
    if not patients:
        logger.error("No patients to stream. Exiting.")
        sys.exit(1)
    
    asyncio.run(run_streamer(patients, args.speed))


if __name__ == "__main__":
    main()
