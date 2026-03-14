"""
Project Chronos - ML Ensemble Training Pipeline (God-Mode v2)
===============================================================
Trains a 5-engine ensemble for three prediction targets:
  1. Septic Shock          → CinC 2019 + eICU (40k+ patients, SepsisLabel)
  2. Acute Hypotension     → eICU (1.6M vital rows, MAP-based AHE label)
                           + CinC 2009 AHE Challenge (if downloaded)
  3. Hemodynamic Collapse  → VitalDB (6388 surgical cases) + Zenodo (112 real patients)
     (Cardiac Arrest Risk)   + CUDB (35 VT/VF recordings) + SDDB (23 Holter recordings)
                           + I-CARE (PhysioNet 2023, 1020+ post-arrest patients)
  NOTE: 'hemodynamic_collapse' is the clinical term for the circulatory failure cascade
  that precedes cardiac arrest. Predicting this 2-6h ahead allows prevention, not
  just detection. Proxy labels: MAP<50 for 5+ min OR SpO2<85 for 3+ min (VitalDB),
  supplemented by real arrest events from I-CARE, Zenodo, CUDB, SDDB.

ARCHITECTURE — 4 engines → LightGBM meta-stacker → isotonic calibration:
  Tabular:    LightGBM + XGBoost              (5-fold GroupKFold, SMOTE-ENN)
  Sequential: GRU-D + TCN                    (irregular time-series)
  Stacking:   LightGBM meta-learner on OOF predictions
  Final:      Isotonic calibration + F-beta threshold optimization

SAFETY:
  - Per-epoch checkpoints: models/{target}/checkpoints/
  - Per-fold save: models/{target}/fold_{k}/*.pkl
  - --resume flag: skips already-completed folds
  - Data validation: --validate flag checks all sources before training

USAGE:
  python backend/train_models.py --target sepsis
  python backend/train_models.py --target hemodynamic_collapse --resume
  python backend/train_models.py --all
  python backend/train_models.py --validate
"""

import os
import sys
import time
import random
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger

# ── File logger: persists across 20h+ training runs, survives terminal crashes ──
_log_dir = Path(__file__).parent / "logs"
_log_dir.mkdir(exist_ok=True)
logger.add(
    _log_dir / "training_{time:YYYY-MM-DD_HH-mm}.log",
    rotation="50 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
    backtrace=True,
    diagnose=False,
)


def _fmt_duration(seconds: float) -> str:
    """Format seconds into human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m:02d}m"

import lightgbm as lgb
import xgboost as xgb
import shap
import optuna
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    fbeta_score, classification_report,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# CatBoost removed: scored 0.60-0.62 AUROC (driven by int(pos_weight) bug producing
# zero positive-class weight). Removed completely rather than leaving disabled dead code.
# The 2-engine tabular ensemble (LightGBM + XGBoost) is faster, simpler, and honest.

try:
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import SMOTE, ADASYN
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    logger.warning("imbalanced-learn not installed — SMOTE/ADASYN disabled. pip install imbalanced-learn")

try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False
    logger.warning("wfdb not installed — ECG loaders disabled. pip install wfdb")

try:
    import psutil as _psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def log_ram(label: str = "") -> None:
    """Logs current RAM usage — both process-specific RSS and system-wide.

    OOM FIX: Previously only showed system-wide psutil.virtual_memory() which
    uses a different formula than Activity Monitor on macOS (vm.used excludes
    compressed + wired memory, making the percentage misleadingly low).
    Now also shows this process's RSS (Resident Set Size) — the actual RAM
    footprint visible in Activity Monitor's "Memory" column.
    """
    if not HAS_PSUTIL:
        return
    # Process-level: this is what Activity Monitor shows per-process
    proc = _psutil.Process()
    rss_gb = proc.memory_info().rss / (1024 ** 3)
    # System-level: overall pressure
    vm = _psutil.virtual_memory()
    total_gb = vm.total / (1024 ** 3)
    avail_gb = vm.available / (1024 ** 3)
    swap = _psutil.swap_memory()
    swap_gb = swap.used / (1024 ** 3)
    swap_flag = f" | Swap: {swap_gb:.1f} GB ⚠️" if swap_gb > 0.5 else ""
    label_str = f" [{label}]" if label else ""
    logger.info(
        f"  💾 RAM{label_str}: Python RSS: {rss_gb:.1f} GB | "
        f"System: {total_gb - avail_gb:.1f} / {total_gb:.1f} GB "
        f"(avail: {avail_gb:.1f} GB){swap_flag}"
    )

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from features import engineer_features, get_feature_columns, impute_vitals

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Global Reproducibility Seed
# ─────────────────────────────────────────────
SEED = 42

def set_global_seeds(seed: int = SEED) -> None:
    """Sets ALL random seeds for full reproducibility across numpy/torch/python/optuna."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # disables autotuner → reproducible
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Global seeds set to {seed}")

set_global_seeds(SEED)

# ─────────────────────────────────────────────
# DEVICE: Apple Silicon MPS or CPU
# ─────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    logger.info("Apple Silicon MPS detected. Neural model will use MPS acceleration.")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    logger.info("CUDA GPU detected.")
else:
    DEVICE = torch.device("cpu")
    logger.info("Running on CPU.")

# How many hours BEFORE the event we want to predict
PREDICTION_HORIZON_HOURS      = 4   # Mid of the 2-6h window
SEQUENCE_LENGTH               = 12  # Last 12 hours of vitals for GRU-D input (default)

# Per-target sequence lengths (hours of history for GRU-D / TCN)
# Sepsis / Hypotension: ICU patients, 12-24h lookback appropriate
# Hemodynamic Collapse: VitalDB surgical cases avg 2-4h → shorter lookback
TARGET_SEQ_LEN = {
    "sepsis":                12,   # 12h history for sepsis trajectory
    "hypotension":           12,   # 12h history for blood pressure trends
    "hemodynamic_collapse":   8,   # 2h lookback — matches VitalDB surgical duration
}


###############################################################################
# ── FOCAL LOSS FOR IMBALANCED SEQUENTIAL TRAINING ───────────────────────────
###############################################################################

class FocalBCE(nn.Module):
    """Focal Binary Cross-Entropy Loss (Lin et al., 2017).

    Addresses severe class imbalance (0.5% positive rate for hemodynamic_collapse)
    WITHOUT generating synthetic temporal sequences (SMOTE is inappropriate here).

    Key hyperparameters:
      gamma: focussing parameter. Higher = more focus on hard, misclassified examples.
             gamma=0 → standard BCE. gamma=2 is the standard recommendation.
      alpha: positive-class weight (set to 1 - positive_rate for calibration).

    Used for: GRU-D and TCN on hemodynamic_collapse target.
    GBTs (LightGBM/XGBoost/CatBoost) use their native pos_weight instead.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.95):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # weight on positive class

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: probabilities (0,1); target: binary labels
        eps = 1e-7
        pred = pred.clamp(eps, 1 - eps)
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        # Focal weight: (1 - p_t)^gamma where p_t is the model's probability for the true class
        p_t = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - p_t) ** self.gamma
        # Alpha weight: up-weight positives, down-weight negatives
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


###############################################################################
# ── SECTION 1: DATA LOADERS ────────────────────────────────────────────────
###############################################################################

def load_cinc2019(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Loads & preprocesses the PhysioNet CinC 2019 Challenge dataset.
    
    Data format: pipe-separated .psv files, one file per patient.
    Each row = one hour of vitals+labs.
    Columns include: HR, O2Sat, Temp, SBP, MAP, DBP, Resp, Lactate, ...
    Label: SepsisLabel (0 = no sepsis, 1 = sepsis onset this hour)
    
    Target construction for 2-6h prediction:
      We label each hour T as positive (1) if SepsisLabel=1 appears within
      the next PREDICTION_HORIZON_HOURS hours.
    """
    cinc_dir = data_dir / "cinc2019"
    psv_files = list(cinc_dir.rglob("*.psv"))
    
    if len(psv_files) == 0:
        logger.error(f"No CinC 2019 .psv files found in {cinc_dir}")
        logger.info("Run: python scripts/download_datasets.py --cinc")
        return None
    
    logger.info(f"Loading CinC 2019: {len(psv_files)} patient files...")
    frames = []
    
    # CinC 2019 column mapping to our internal naming
    # Previously only 15 of 40 columns were loaded. Now loading ALL 40.
    # Source: CinC 2019 challenge dataset header (physionet.org)
    COL_MAP = {
        # ── Monitor vitals (8) ──
        "HR": "heart_rate", "O2Sat": "spo2", "Temp": "temperature",
        "SBP": "systolic_bp", "MAP": "mean_arterial_pressure",
        "DBP": "diastolic_bp", "Resp": "respiratory_rate",
        "EtCO2": "etco2",
        # ── Lab panels (26) ──
        "BaseExcess": "base_excess", "HCO3": "hco3", "FiO2": "fio2",
        "pH": "ph", "PaCO2": "paco2", "SaO2": "sao2",
        "AST": "ast", "BUN": "bun", "Alkalinephos": "alkaline_phosphatase",
        "Calcium": "calcium", "Chloride": "chloride",
        "Creatinine": "creatinine", "Bilirubin_direct": "bilirubin_direct",
        "Glucose": "glucose", "Lactate": "lactate",
        "Magnesium": "magnesium", "Phosphate": "phosphate",
        "Potassium": "potassium", "Bilirubin_total": "bilirubin",
        "TroponinI": "troponin_i", "Hct": "hematocrit", "Hgb": "hemoglobin",
        "PTT": "ptt", "WBC": "wbc", "Fibrinogen": "fibrinogen",
        "Platelets": "platelets", "PaO2": "pao2", "GCS": "gcs",
        # ── Demographics (6) ──
        "Age": "age", "Gender": "gender",
        "Unit1": "unit1", "Unit2": "unit2",
        "HospAdmTime": "hosp_adm_time", "ICULOS": "iculos",
    }
    
    for i, psv_path in enumerate(psv_files):
        try:
            df = pd.read_csv(psv_path, sep="|")
            patient_id = psv_path.stem  # filename = patient ID
            
            df.rename(columns=COL_MAP, inplace=True)
            df["patient_id"] = patient_id
            df["hour"] = range(len(df))
            df["timestamp"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(df["hour"], unit="h")
            
            # Forward target: did sepsis onset occur in next N hours?
            if "SepsisLabel" in df.columns:
                label_col = df["SepsisLabel"].values
                future_labels = np.zeros(len(df), dtype=int)
                for t in range(len(df)):
                    # BUG-R8-FIX: Start at t+2, NOT t+0. Previously included the
                    # current hour, making this partially a detector rather than a
                    # predictor. Now consistent with eICU (t+2h) and VitalDB (t+120min).
                    # CinC SepsisLabel already has 6h built-in lookahead, so t+2 gives
                    # an effective 2-10h prediction horizon — clinically appropriate.
                    window_start = min(t + 2, len(df))
                    window_end = min(t + PREDICTION_HORIZON_HOURS + 1, len(df))
                    future_labels[t] = int(label_col[window_start:window_end].max()) if window_start < window_end else 0
                df["target_sepsis"] = future_labels
            else:
                df["target_sepsis"] = 0
            
            frames.append(df)
        except Exception as e:
            logger.warning(f"  Skipping {psv_path.name}: {e}")
    
    if not frames:
        logger.error("No valid CinC files parsed.")
        return None
    
    combined = pd.concat(frames, ignore_index=True)
    logger.success(f"CinC 2019: {len(combined):,} rows, {combined['patient_id'].nunique():,} patients")
    logger.info(f"  Sepsis rate: {combined['target_sepsis'].mean():.2%}")
    return combined


def load_eicu_hypotension(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Loads eICU Collaborative Research Database Demo for hypotension prediction.

    eICU Demo provides 2,520 ICU patients across multiple US hospitals with
    continuous vital sign monitoring at variable intervals.

    Signals used:
      vitalPeriodic.csv  → systemicmean (MAP), heartrate, sao2, respiration, temperature
      apachePatientResult.csv → actualicumortality (outcome label)
      patient.csv        → patientunitstayid, unitdischargestatus (discharge alive/dead)

    Target construction for 2-6h BP collapse prediction:
      Within a patient's stay, label hour T as positive (1) if:
        - MAP drops below 65 mmHg AND stays below for at least 2 consecutive observations
        within the next PREDICTION_HORIZON_HOURS hours.
      This matches the Surviving Sepsis Campaign definition of MAP-based shock.

    Note: eICU records time as 'observationoffset' (minutes from ICU admission).
          We resample to hourly intervals for consistency with CinC 2019.
    """
    eicu_dir = data_dir / "eicu_demo" / "eicu-collaborative-research-database-demo-2.0.1"

    vitals_path  = eicu_dir / "vitalPeriodic.csv"
    patient_path = eicu_dir / "patient.csv"

    if not vitals_path.exists():
        logger.error(f"eICU vitalPeriodic.csv not found at {vitals_path}")
        logger.info("Run: python scripts/download_datasets.py --eicu")
        return None

    logger.info("Loading eICU Demo for hypotension target...")

    # Load vitals — columns: patientunitstayid, observationoffset, systemicmean,
    #                         heartrate, sao2, respiration, temperature
    vitals = pd.read_csv(vitals_path, usecols=[
        "patientunitstayid", "observationoffset",
        "systemicmean", "heartrate", "sao2", "respiration", "temperature",
        "systemicsystolic", "systemicdiastolic",
    ])

    # Rename to internal schema
    vitals.rename(columns={
        "patientunitstayid":  "patient_id",
        "observationoffset":  "offset_min",
        "systemicmean":       "mean_arterial_pressure",
        "systemicsystolic":   "systolic_bp",
        "systemicdiastolic":  "diastolic_bp",
        "heartrate":          "heart_rate",
        "sao2":               "spo2",
        "respiration":        "respiratory_rate",
        "temperature":        "temperature",
    }, inplace=True)

    # Convert offset (minutes) to hours and bin to 1-hour intervals
    vitals["hour_bin"] = (vitals["offset_min"] / 60).astype(int)

    # Aggregate to hourly resolution (mean per hour per patient)
    hourly = (
        vitals
        .groupby(["patient_id", "hour_bin"])
        .agg({
            "mean_arterial_pressure": "mean",
            "systolic_bp":            "mean",
            "diastolic_bp":           "mean",
            "heart_rate":             "mean",
            "spo2":                   "mean",
            "respiratory_rate":       "mean",
            "temperature":            "mean",
        })
        .reset_index()
    )

    # ── NEE (Norepinephrine-Equivalent) vasopressor dose ─────────────────────
    # Join infusiondrug.csv to get real vasopressor dose per patient per hour.
    # eICU infusiondrug columns:
    #   patientunitstayid, infusionoffset (mins), drugname, drugrate, patientweight
    # patient.csv has admissionweight as fallback for body weight.
    # NEE conversion factors (Goradia et al., J Crit Care 2021; Russell et al., CCM 2018):
    #   Norepinephrine / Epinephrine: 1.0x (reference)
    #   Dopamine:     0.01x  (100 mcg/kg/min dopa ≈ 1 mcg/kg/min NE)
    #   Phenylephrine: 0.1x
    #   Vasopressin:  fixed ~0.04 mcg/kg/min NE proxy when infused at any rate
    NEE_FACTORS = {
        "norepinephrine": 1.0, "levophed": 1.0,
        "epinephrine":    1.0, "adrenaline": 1.0,
        "dopamine":       0.01,
        "phenylephrine":  0.1,  "neosynephrine": 0.1,
        "vasopressin":    None,  # fixed dose, handled separately
    }
    infusion_path = eicu_dir / "infusiondrug.csv"
    if infusion_path.exists():
        try:
            inf = pd.read_csv(infusion_path, usecols=[
                "patientunitstayid", "infusionoffset", "drugname",
                "drugrate", "patientweight",
            ], low_memory=False)
            pat = pd.read_csv(patient_path, usecols=[
                "patientunitstayid", "admissionweight"
            ])
            inf["drugname_lower"] = inf["drugname"].str.lower().fillna("")
            inf["drugrate_num"]   = pd.to_numeric(inf["drugrate"], errors="coerce")

            # Map drug to NEE factor
            def _nee_factor(name):
                for key, factor in NEE_FACTORS.items():
                    if key in name:
                        return factor  # None = vasopressin fixed
                return None

            inf["nee_factor"] = inf["drugname_lower"].apply(_nee_factor)
            inf = inf[inf["nee_factor"].notna() | inf["drugname_lower"].str.contains("vasopressin")]

            # Merge patient weight (prefer per-row weight, fallback to admission weight)
            inf = inf.merge(pat, on="patientunitstayid", how="left")
            inf["weight_kg"] = (
                pd.to_numeric(inf["patientweight"], errors="coerce")
                .fillna(pd.to_numeric(inf["admissionweight"], errors="coerce"))
                .fillna(80.0)
            )

            # Convert drugrate to NEE mcg/kg/min
            def _to_nee(row):
                rate = row["drugrate_num"]
                if pd.isna(rate) or rate <= 0:
                    return 0.0
                factor = row["nee_factor"]
                if factor is None:   # vasopressin — fixed NE equivalent
                    return 0.04
                wt = row["weight_kg"]
                return float(rate * factor / max(wt, 1.0))

            inf["nee_mcg_kg_min"] = inf.apply(_to_nee, axis=1)

            # Aggregate to patient + hour_bin (sum NEE over all infused drugs per hour)
            inf["hour_bin"] = (inf["infusionoffset"] / 60).astype(int)
            nee_hourly = (
                inf.groupby(["patientunitstayid", "hour_bin"])["nee_mcg_kg_min"]
                .sum()
                .reset_index()
                .rename(columns={"patientunitstayid": "patient_id", "nee_mcg_kg_min": "nee_dose"})
            )
            hourly = hourly.merge(nee_hourly, on=["patient_id", "hour_bin"], how="left")
            hourly["nee_dose"] = hourly["nee_dose"].fillna(0.0)
            hourly["vasopressor_active"] = (hourly["nee_dose"] > 0).astype(bool)
            logger.info(f"  NEE vasopressor feature added: {(hourly['nee_dose'] > 0).mean():.1%} of hours on pressors")
        except Exception as e:
            logger.warning(f"  NEE join failed ({e}) — defaulting nee_dose=0 for all rows")
            hourly["nee_dose"] = 0.0
            hourly["vasopressor_active"] = False
    else:
        logger.warning(f"  infusiondrug.csv not found at {infusion_path} — nee_dose=0")
        hourly["nee_dose"] = 0.0
        hourly["vasopressor_active"] = False
    # ─────────────────────────────────────────────────────────────────────────

    # Assign synthetic timestamps for feature engineering compatibility
    hourly["timestamp"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(hourly["hour_bin"], unit="h")

    # Build forward-looking hypotension label:
    # At each hour T, label=1 if MAP < 65 mmHg in at least 2 of the next N hours.
    frames = []
    for pid, group in hourly.groupby("patient_id"):
        group = group.sort_values("hour_bin").reset_index(drop=True)
        map_vals = group["mean_arterial_pressure"].fillna(90.0).values
        targets  = np.zeros(len(group), dtype=int)

        for t in range(len(group)):
            # NEW-BUG-FIX (A2.3): Start prediction window at t+2, not t.
            # Previous code included the current hour (t+0) making it a detector
            # rather than an early-warning predictor. We want to predict hypotension
            # 2–6 hours ahead, consistent with the project's stated clinical goal.
            window_start = min(t + 2, len(group))
            window_end   = min(t + PREDICTION_HORIZON_HOURS + 1, len(group))
            future_map   = map_vals[window_start:window_end]
            # At least 2 consecutive hours with MAP < 65 = sustained hypotension.
            # Guard: np.convolve raises ValueError if future_map has < 2 elements
            # (happens for short stays or rows near end of stay where window is empty).
            if len(future_map) < 2:
                targets[t] = 0
            else:
                low_map_run  = np.convolve((future_map < 65).astype(int), np.ones(2), "valid")
                targets[t]   = int(len(low_map_run) > 0 and low_map_run.max() >= 2)

        group["target_hypotension"] = targets
        frames.append(group)

    combined = pd.concat(frames, ignore_index=True)
    logger.success(f"eICU Hypotension: {len(combined):,} hourly rows, {combined['patient_id'].nunique():,} patients")
    logger.info(f"  Hypotension rate: {combined['target_hypotension'].mean():.2%}")
    return combined


def load_mimic3c_enrichment(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Loads MIMIC-3c aggregated dataset for ADMINISTRATIVE ENRICHMENT ONLY.

    IMPORTANT: This dataset contains ONLY static per-admission summaries:
      hadm_id, gender, age, LOSdays, NumDiagnosis, NumProcs, ExpiredHospital, etc.
    It has NO time-series vitals and CANNOT be used as a primary training source.

    Use case: Join on hadm_id to enrich MIMIC-III Demo records with:
      - Mortality label (ExpiredHospital)
      - Clinical complexity proxy (NumDiagnosis, NumChartEvents)
      - Length of stay (LOSdays)
    """
    mimic3c_path = data_dir / "kaggle_supplements" / "mimic3c_aggregated" / "mimic3c.csv"

    if not mimic3c_path.exists():
        logger.warning(f"mimic3c.csv not found at {mimic3c_path} — skipping enrichment")
        return None

    logger.info("Loading MIMIC-3c admin enrichment (join table only, no vitals)...")
    # Only load the columns useful for enrichment
    df = pd.read_csv(mimic3c_path, usecols=[
        "hadm_id", "age", "LOSdays", "NumDiagnosis",
        "NumChartEvents", "ExpiredHospital", "TotalNumInteract",
    ])
    logger.info(f"  Loaded {len(df):,} MIMIC-3c admission records for enrichment")
    return df


def load_vitaldb(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Loads VitalDB high-resolution patient data.
    
    VitalDB provides 1-7 second resolution vitals.
    We resample to 1-minute intervals and create cardiac arrest proxies:
      - Sustained MAP < 50 mmHg for > 5 minutes (hemodynamic collapse)
      - SpO2 < 85% for > 3 minutes (critical hypoxemia)
      - These are strong proxies for cardiac arrest onset
    
    Target: Any of the above proxy events within the next PREDICTION_HORIZON_HOURS.
    """
    vdb_dir = data_dir / "vitaldb"
    case_files = list(vdb_dir.glob("case_*.csv"))
    
    if not case_files:
        logger.error(f"No VitalDB case files found in {vdb_dir}")
        logger.info("Run: python scripts/download_datasets.py --vitaldb")
        return None
    
    logger.info(f"Loading VitalDB: {len(case_files)} case files...")
    frames = []
    
    for cf in case_files:
        try:
            df = pd.read_csv(cf)
            if df.empty:
                continue
            
            case_id = cf.stem.replace("case_", "")
            df["patient_id"] = f"vdb_{case_id}"
            
            # Standardize track names — VitalDB uses NIBP_MBP for non-invasive MAP
            # ABP_MEAN (invasive arterial line) is present on arterial monitoring cases,
            # NIBP_MBP (non-invasive BP mean) is present on standard surgical cases.
            # Both must map to mean_arterial_pressure.
            col_map = {}
            for c in df.columns:
                cl = c.lower()
                if "abp_mean" in cl or "abpm" in cl:          col_map[c] = "mean_arterial_pressure"
                elif "nibp_mbp" in cl or "nibp_m" in cl:      col_map[c] = "mean_arterial_pressure"  # ← FIXED
                elif "nibp_sbp" in cl or "nibp_s" in cl:      col_map[c] = "systolic_bp"
                elif "nibp_dbp" in cl or "nibp_d" in cl:      col_map[c] = "diastolic_bp"
                elif "abp_s" in cl or "abpsys" in cl:         col_map[c] = "systolic_bp"
                elif "abp_d" in cl or "abpdia" in cl:         col_map[c] = "diastolic_bp"
                elif "hr" == cl:                              col_map[c] = "heart_rate"
                elif "spo2" in cl:                            col_map[c] = "spo2"
                elif "etco2" in cl:                           col_map[c] = "etco2"
                elif "rr" == cl or "resp" in cl:              col_map[c] = "respiratory_rate"
                elif "bt" == cl or "temp" in cl:              col_map[c] = "temperature"
            df.rename(columns=col_map, inplace=True)

            # ── VitalDB time handling ───────────────────────────────────────────
            # VitalDB case CSVs downloaded via the web API have NO explicit time column.
            # The row index IS the second-offset (row 0 = second 0, row N = second N).
            # Files from the original VitalDB bulk download include a 'Time' column.
            # We handle both cases.
            if "time" in df.columns or "Time" in df.columns:
                time_col = "time" if "time" in df.columns else "Time"
                df["secs"] = pd.to_numeric(df[time_col], errors="coerce")
            else:
                # Row index = seconds from start of surgery (VitalDB API download format)
                df["secs"] = df.index.astype(float)
            df["timestamp"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(df["secs"], unit="s")

            # Resample to 1-minute intervals (VitalDB is 1-7 sec resolution).
            # IMPORTANT: must drop non-numeric columns before resample().mean(),
            # then re-attach patient_id after the resample.
            patient_id_val = df["patient_id"].iloc[0] if "patient_id" in df.columns else f"vdb_{case_id}"
            numeric_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
            df = (
                df[["timestamp"] + numeric_cols]
                .set_index("timestamp")
                .resample("1min")
                .mean()
                .reset_index()
            )
            df["patient_id"] = patient_id_val

            # Create hemodynamic collapse proxy target
            # MAP < 50 mmHg for ≥ 5 consecutive minutes → circulatory failure
            # SpO2 < 85% for ≥ 3 consecutive minutes → critical hypoxemia
            map_col = "mean_arterial_pressure"
            if map_col in df.columns:
                map_vals  = df[map_col].fillna(93.0).values
                spo2_vals = df.get("spo2", pd.Series([99.0]*len(df))).fillna(99.0).values

                targets = np.zeros(len(df), dtype=int)
                # AUDIT-FIX-2: Start prediction window 2 hours (120 min) ahead,
                # NOT at the current minute. Previously `map_vals[t:window_end]`
                # included the current time step — the model was partially
                # trained to DETECT current collapse rather than PREDICT it.
                # This matches eICU's t+2 offset for consistency across targets.
                _offset_min = 2 * 60  # 2-hour prediction horizon start
                for t in range(len(df) - 1):
                    window_start = min(t + _offset_min, len(df))
                    window_end   = min(t + PREDICTION_HORIZON_HOURS * 60, len(df))
                    future_map   = map_vals[window_start:window_end]
                    future_spo2  = spo2_vals[window_start:window_end]
                    # Guard: need at least 5 points for MAP convolution, 3 for SpO2
                    map_crash    = len(future_map) >= 5 and any(np.convolve((future_map  < 50).astype(int), np.ones(5), "valid") >= 5)
                    spo2_crash   = len(future_spo2) >= 3 and any(np.convolve((future_spo2 < 85).astype(int), np.ones(3), "valid") >= 3)
                    targets[t]   = int(map_crash or spo2_crash)

                df["target_hemodynamic_collapse"] = targets
            else:
                logger.warning(f"  {cf.name}: no MAP column found after renaming — labels not set")
                
            frames.append(df)
        except Exception as e:
            logger.warning(f"  Skipping {cf.name}: {e}")
    
    if not frames:
        logger.error("No valid VitalDB cases parsed.")
        return None
    
    combined = pd.concat(frames, ignore_index=True)
    logger.success(f"VitalDB: {len(combined):,} rows, {combined['patient_id'].nunique():,} patients")
    if "target_hemodynamic_collapse" in combined.columns:
        logger.info(f"  Hemodynamic collapse proxy rate: {combined['target_hemodynamic_collapse'].mean():.2%}")
    return combined


def load_zenodo_cardiac_arrest(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Loads the Zenodo Cardiac Patient Bed Head Ticket Dataset.

    Source: Teaching Hospital Karapitiya Galle, Sri Lanka.
    Published: Zenodo 7603772. CC BY 4.0. No login required.
    Paper: https://zenodo.org/records/7603772

    Contains 112 patients who were admitted to the cardiac ward from the Emergency
    Treatment Unit (ETU) after a cardiac event. Features are manually extracted
    from Bed Head Tickets (clinical paper records).

    19 features:
      SBP, DBP, HR, RR, BT (temp), SpO2, Age, Gender, GCS,
      Na, K, Cl (electrolytes), Urea, Creatinine,
      Alcohol, Smoking, Family history (FHIHD), Triage score, Outcome

    IMPORTANT: This is a STATIC dataset (one snapshot per patient, no time-series).
    It is used to:
      1. Provide real cardiac arrest patient labels for oversampling/SMOTE
      2. Validate that proxy labels from VitalDB match real clinical patterns
      3. Calibrate feature thresholds (e.g., GCS < 14 = high risk)

    Setup: Download from https://zenodo.org/records/7603772/files/CardiacPatientData.csv
           Place at data/zenodo_cardiac/CardiacPatientData.csv
    """
    zenodo_path = data_dir / "zenodo_cardiac" / "CardiacPatientData.csv"

    if not zenodo_path.exists():
        logger.warning(f"Zenodo cardiac dataset not found at {zenodo_path}")
        logger.info("Download: curl -L 'https://zenodo.org/records/7603772/files/CardiacPatientData.csv' -o data/zenodo_cardiac/CardiacPatientData.csv")
        return None

    logger.info("Loading Zenodo Cardiac Patient Dataset (Sri Lanka ETU)...")
    df = pd.read_csv(zenodo_path)

    # Normalise column names to lowercase with underscores
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Map to internal vital sign naming convention
    col_map = {
        "sbp":          "systolic_bp",
        "dbp":          "diastolic_bp",
        "hr":           "heart_rate",
        "rr":           "respiratory_rate",
        "bt":           "temperature",
        "spo2":         "spo2",
        "gcs":          "gcs",
        "na":           "sodium",
        "k":            "potassium",
        "cl":           "chloride",
        "urea":         "bun",              # BUN ≈ Urea (clinically equivalent)
        "creatinine":   "creatinine",
        "ceratinine":   "creatinine",       # Bug 24 fix: original CSV has typo 'Ceratinine'
        "triage_score": "triage_score",
        "age":          "age",
        "gender":       "gender",
        "fhcd":         "family_hx_cardiac_disease",  # FHCD = Family Hx Cardiac Disease
        "alcoholic":    "alcohol_use",
        "smoke":        "smoking",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # Outcome column encoding (VERIFIED against actual CSV data):
    #   Outcome = 1 → patient SURVIVED (93 patients, 83%) — NEGATIVE class
    #   Outcome = 0 → patient DIED / cardiac arrest (19 patients, 17%) — POSITIVE class
    # NEW-BUG-FIX: previous code used outcome directly (Outcome=1 → positive),
    # which INVERTED the labels — training model to predict survival as arrest.
    # Fix: invert the encoding so that death/arrest = 1 (positive), survival = 0.
    if "outcome" in df.columns:
        df["target_hemodynamic_collapse"] = (df["outcome"].astype(int) == 0).astype(int)
    elif "result" in df.columns:
        df["target_hemodynamic_collapse"] = (df["result"].astype(int) == 0).astype(int)
    else:
        # Fall back: label all as positive (all admitted with cardiac event)
        df["target_hemodynamic_collapse"] = 1
        logger.warning("  No outcome column found — labelling all 112 patients as positive class")

    # Assign patient IDs from the 'id' column (112 unique patients, multi-visit).
    # Using per-row IDs (e.g. zenodo_ca_0000) defeats GroupKFold patient-level splits.
    if "id" in df.columns:
        df["patient_id"] = df["id"].apply(lambda x: f"zenodo_ca_{int(x):04d}")
    else:
        df["patient_id"] = [f"zenodo_ca_{i:04d}" for i in range(len(df))]

    # Assign timestamps: each row = one visit/measurement. Use row number within patient.
    df["timestamp"] = df.groupby("patient_id").cumcount().apply(
        lambda n: pd.to_datetime("2020-01-01") + pd.to_timedelta(n, unit="h")
    )
    df["hour_bin"]  = df.groupby("patient_id").cumcount()

    pos_rate = df["target_hemodynamic_collapse"].mean()
    logger.success(f"Zenodo Cardiac: {len(df):,} patients | Positive rate: {pos_rate:.2%}")
    logger.info("  NOTE: Static dataset — real cardiac events used for label enrichment")
    return df


def load_icare(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Loads the I-CARE (International Cardiac Arrest REsearch) Dataset.
    PhysioNet/CinC 2023 Challenge. Open access, no credentialing required.
    https://physionet.org/content/i-care/2.1/

    1,020+ patients, multi-center, post-ROSC (Return Of Spontaneous Circulation).

    REAL SCHEMA (verified from PhysioNet I-CARE v2.1 docs):
    The .tsv files are per-patient clinical SUMMARIES with one row per patient:
        Age, Sex, Hospital, ROSC (minutes), OHCA, Shockable Rhythm, TTM (°C), Outcome, CPC
    They are NOT hourly vital sign time series. EEG/ECG is in binary .mat files.

    What we extract:
      - Label:    CPC score ≥ 3 → poor neuro outcome → target_hemodynamic_collapse = 1
      - Features: age, ROSC time (arrest severity), TTM temperature, shockable rhythm type
      - Each patient contributes ONE row to the tabular training set.

    Setup: Download from https://physionet.org/files/i-care/2.1/ — place at data/icare/
    """
    icare_dir = data_dir / "icare"
    tsv_files = list(icare_dir.rglob("*.tsv"))

    if not tsv_files:
        logger.warning(f"I-CARE .tsv files not found in {icare_dir}")
        logger.info("Download: wget -r -N -c -np --user USERNAME https://physionet.org/files/i-care/2.1/ -P data/icare/")
        return None

    logger.info(f"Loading I-CARE: {len(tsv_files)} patient clinical summary files...")
    rows = []

    for tsv_path in tsv_files:
        try:
            patient_id = tsv_path.stem
            df = pd.read_csv(tsv_path, sep="\t")
            if df.empty:
                continue
            df.columns = df.columns.str.strip()

            for _, row in df.iterrows():
                cpc     = row.get("CPC", np.nan)
                outcome = row.get("Outcome", "")
                age     = row.get("Age", np.nan)
                rosc    = row.get("ROSC", np.nan)   # minutes from arrest to ROSC
                ttm     = row.get("TTM", np.nan)    # target temp management (°C)
                shock   = row.get("Shockable Rhythm", False)

                # CPC 3-5 = severe disability or dead = positive class
                if not pd.isna(cpc):
                    target = int(float(cpc) >= 3)
                elif isinstance(outcome, str):
                    target = 1 if "poor" in outcome.lower() else 0
                else:
                    target = 1   # post-arrest default positive

                rows.append({
                    "patient_id":                 f"icare_{patient_id}",
                    "target_hemodynamic_collapse": target,
                    "timestamp":                  pd.to_datetime("2020-01-01"),
                    "hour_bin":                   0,
                    "age":          float(age)  if not pd.isna(age)  else np.nan,
                    "temperature":  float(ttm)  if not pd.isna(ttm)  else 36.0,
                    "rosc_minutes": float(rosc) if not pd.isna(rosc) else np.nan,
                    "shockable_rhythm": float(bool(shock)),
                })
        except Exception as e:
            logger.warning(f"  I-CARE {tsv_path.name}: {e}")

    if not rows:
        logger.warning("No valid I-CARE files parsed.")
        return None

    combined = pd.DataFrame(rows)
    pos_rate = combined["target_hemodynamic_collapse"].mean()
    logger.success(f"I-CARE: {len(combined):,} patients | positive (CPC≥3): {pos_rate:.2%}")
    logger.info("  I-CARE = static admission summaries — 1 row/patient for tabular models")
    return combined


###############################################################################
# ── SECTION 2: TABULAR ENSEMBLE (LightGBM + XGBoost) ──────────────────────
###############################################################################

LGBM_PARAMS = {
    "boosting_type":    "gbdt",
    "objective":        "binary",
    "metric":           ["auc", "average_precision"],
    # NOTE: n_estimators is NOT here — it goes to lgb.train() as num_boost_round.
    # Putting n_estimators in params dict + passing num_boost_round causes
    # "Found n_estimators in params, will use instead of argument" UserWarning.
    "learning_rate":    0.03,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples": 30,
    "subsample":        0.80,
    "colsample_bytree": 0.70,
    "reg_alpha":        0.1,
    "reg_lambda":       0.3,
    "is_unbalance":     True,         # BUG-R6-1 fix: class_weight is sklearn-only; lgb.train() silently ignores it
    "device":           "cpu",        # LightGBM CPU is extremely fast on M4
    "n_jobs":           -1,
    "random_state":     42,
    "verbose":          -1,
}
LGBM_NUM_BOOST_ROUND = 1500  # passed as num_boost_round to lgb.train()

XGB_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      ["auc", "aucpr"],
    "n_estimators":     1000,
    "learning_rate":    0.05,
    "max_depth":        6,
    "subsample":        0.80,
    "colsample_bytree": 0.70,
    "reg_alpha":        0.1,
    "reg_lambda":       0.3,
    # Bug 51 fix: scale_pos_weight MUST NOT be set to None here.
    # None is passed to XGBoost constructor as a kwarg and may be silently
    # ignored or cause version-specific behaviour. Always compute dynamically:
    #   pos_weight = (neg_count / pos_count)   set at training time per fold.
    # The training code at lines ~754, ~1874, ~1915 correctly injects this.
    "tree_method":      "hist",   # fastest on CPU
    "device":           "cpu",
    "random_state":     42,
}


###############################################################################
# ── OPTUNA HYPERPARAMETER TUNING FOR GBT MODELS ─────────────────────────────
###############################################################################

def optuna_tune_gbt(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    target: str,
    n_trials: int = 50,
    cache_dir: Optional[Path] = None,
) -> tuple[dict, dict]:
    """Tunes LightGBM and XGBoost hyperparameters using Optuna with 3-fold CV.

    Uses StratifiedGroupKFold (3-fold for speed, 5-fold in real training).
    Returns tuned (lgbm_params, xgb_params) dicts ready for training.
    Results cached to JSON — skipped on resume if cache exists.

    Expected runtime: ~15-30 min per target (50 trials × 3 folds each).
    Expected gain: +2-4% AUROC vs hardcoded defaults.
    """
    # Check cache first (resume support)
    if cache_dir is not None:
        _lgbm_cache = cache_dir / "optuna_lgbm_params.json"
        _xgb_cache = cache_dir / "optuna_xgb_params.json"
        if _lgbm_cache.exists() and _xgb_cache.exists():
            with open(_lgbm_cache) as f:
                cached_lgbm = json.load(f)
            with open(_xgb_cache) as f:
                cached_xgb = json.load(f)
            # BUG-R6-FIX: Merge cached tuned params OVER defaults so that keys
            # stripped during JSON serialization (metric, n_jobs, device, verbose,
            # is_unbalance) are restored from LGBM_PARAMS/XGB_PARAMS.
            merged_lgbm = LGBM_PARAMS.copy()
            merged_lgbm.update(cached_lgbm)
            merged_xgb = XGB_PARAMS.copy()
            merged_xgb.update(cached_xgb)
            logger.info(f"  \u26a1 Optuna: loaded cached tuned params from {cache_dir.name}/ (merged with defaults)")
            return merged_lgbm, merged_xgb

    logger.info(f"\n  🔍 Optuna: tuning LightGBM + XGBoost hyperparameters ({n_trials} trials each)...")
    _tune_start = time.time()

    # Use 3-fold for speed (objective is called 50x per model)
    kfold_tune = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)

    # ── LightGBM Tuning ─────────────────────────────────────────────────────
    def _lgbm_objective(trial):
        params = {
            "boosting_type":    "gbdt",
            "objective":        "binary",
            "metric":           "auc",
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "is_unbalance":     True,
            "device":           "cpu",
            "n_jobs":           -1,
            "random_state":     SEED,
            "verbose":          -1,
        }
        aurocs = []
        for tr_idx, vl_idx in kfold_tune.split(X, y, groups):
            ds_tr = lgb.Dataset(X[tr_idx], label=y[tr_idx], feature_name=feature_names)
            ds_vl = lgb.Dataset(X[vl_idx], label=y[vl_idx], reference=ds_tr)
            m = lgb.train(
                params, ds_tr, num_boost_round=500, valid_sets=[ds_vl],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
            )
            preds = m.predict(X[vl_idx])
            if y[vl_idx].sum() > 0:
                aurocs.append(roc_auc_score(y[vl_idx], preds))
        return float(np.mean(aurocs)) if aurocs else 0.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    lgbm_study = optuna.create_study(direction="maximize", study_name=f"lgbm_{target}")
    lgbm_study.optimize(_lgbm_objective, n_trials=n_trials, show_progress_bar=True)
    _lgbm_elapsed = time.time() - _tune_start

    best_lgbm = LGBM_PARAMS.copy()
    best_lgbm.update(lgbm_study.best_params)
    logger.success(
        f"  LightGBM Optuna: best AUROC={lgbm_study.best_value:.4f} "
        f"in {_fmt_duration(_lgbm_elapsed)} ({n_trials} trials)\n"
        f"    Best params: lr={best_lgbm['learning_rate']:.4f}, "
        f"leaves={best_lgbm['num_leaves']}, "
        f"subsample={best_lgbm['subsample']:.2f}, "
        f"colsample={best_lgbm['colsample_bytree']:.2f}"
    )

    # ── XGBoost Tuning ───────────────────────────────────────────────────────
    _xgb_tune_start = time.time()
    pos_weight_global = (y == 0).sum() / max((y == 1).sum(), 1)

    def _xgb_objective(trial):
        params = {
            "objective":        "binary:logistic",
            "eval_metric":      "auc",
            "n_estimators":     500,
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": pos_weight_global,
            "tree_method":      "hist",
            "device":           "cpu",
            "random_state":     SEED,
            "early_stopping_rounds": 30,
        }
        aurocs = []
        for tr_idx, vl_idx in kfold_tune.split(X, y, groups):
            m = xgb.XGBClassifier(**params)
            m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[vl_idx], y[vl_idx])], verbose=False)
            preds = m.predict_proba(X[vl_idx])[:, 1]
            if y[vl_idx].sum() > 0:
                aurocs.append(roc_auc_score(y[vl_idx], preds))
        return float(np.mean(aurocs)) if aurocs else 0.0

    xgb_study = optuna.create_study(direction="maximize", study_name=f"xgb_{target}")
    xgb_study.optimize(_xgb_objective, n_trials=n_trials, show_progress_bar=True)
    _xgb_elapsed = time.time() - _xgb_tune_start

    best_xgb = XGB_PARAMS.copy()
    best_xgb.update(xgb_study.best_params)
    logger.success(
        f"  XGBoost Optuna: best AUROC={xgb_study.best_value:.4f} "
        f"in {_fmt_duration(_xgb_elapsed)} ({n_trials} trials)\n"
        f"    Best params: lr={best_xgb['learning_rate']:.4f}, "
        f"depth={best_xgb['max_depth']}, "
        f"subsample={best_xgb['subsample']:.2f}, "
        f"colsample={best_xgb['colsample_bytree']:.2f}"
    )

    _total_elapsed = time.time() - _tune_start
    logger.info(f"  ⚡ Optuna total tuning time: {_fmt_duration(_total_elapsed)}")

    # Cache results for resume
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Only cache the tunable params (not fixed ones like device, verbose)
        with open(cache_dir / "optuna_lgbm_params.json", "w") as f:
            json.dump({k: v for k, v in best_lgbm.items()
                       if isinstance(v, (int, float, str, bool))}, f, indent=2)
        with open(cache_dir / "optuna_xgb_params.json", "w") as f:
            json.dump({k: v for k, v in best_xgb.items()
                       if isinstance(v, (int, float, str, bool, list))}, f, indent=2)
        logger.info(f"  Tuned params cached to {cache_dir.name}/optuna_*_params.json")

    return best_lgbm, best_xgb

# AUDIT-FIX: train_tabular_ensemble() REMOVED — dead code, zero call sites.
# The K-fold loop at lines ~2458-2480 duplicates this logic inline with proper
# checkpointing, patient-level splitting, and fold-averaged iteration counts.
# Keeping it was confusing for maintainers.


###############################################################################
# ── SECTION 3: GRU-D SEQUENTIAL MODEL ─────────────────────────────────────
###############################################################################

class GRUD(nn.Module):
    """
    GRU-D: Gated Recurrent Unit with Decay (Che et al., 2018).
    
    Specifically designed for clinical time-series with:
      - Irregular time intervals between measurements
      - Missing data (vitals not measured every hour)
    
    Key innovation over standard GRU:
      The hidden state DECAYS exponentially between observations:
        h_t = γ_h ⊙ h_{t-1}
      Missing values are a WEIGHTED MIXTURE of the last observation and the mean:
        x̃_t = m_t ⊙ x_t + (1 - m_t) ⊙ (γ_x ⊙ x_{t-1} + (1 - γ_x) ⊙ x̄)
    
    This allows the model to "forget" stale observations that haven't been
    updated in a long time — exactly what skilled clinicians do intuitively.
    
    Reference: Che Z et al. "Recurrent Neural Networks for Multivariate Time 
    Series with Missing Values." Scientific Reports, 2018.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size  = input_size
        
        # Decay parameters for missing data handling
        self.W_gamma_x = nn.Linear(input_size, input_size)   # input decay
        self.W_gamma_h = nn.Linear(input_size, hidden_size)  # hidden state decay
        
        # Standard GRU gates
        self.gru_cell = nn.GRUCell(input_size * 2, hidden_size)  # x2 for masking
        
        # Output head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)  # Binary classification
    
    def forward(self, x, mask, delta):
        """
        Args:
            x:     (batch, seq_len, input_size) — raw vitals (NaN replaced with 0)
            mask:  (batch, seq_len, input_size) — 1 if observed, 0 if missing
            delta: (batch, seq_len, input_size) — time since last observation (hours)
        Returns:
            (batch,) — probability of target event
        """
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        x_last = torch.zeros_like(x[:, 0, :])  # last observed value per feature (zero = mean proxy)
        
        for t in range(x.size(1)):
            x_t     = x[:, t, :]      # current vitals
            m_t     = mask[:, t, :]   # observation mask
            delta_t = delta[:, t, :]  # time gaps
            
            # ── Input decay: stale unobserved features trend toward last known value ──
            # Che 2018 (GRU-D paper) formula:
            #   x̃_t = m_t ⊙ x_t + (1 - m_t) ⊙ (γ_x ⊙ x_last + (1 - γ_x) ⊙ x_mean)
            # where x_last is the last observed value per feature, x_mean ≈ 0 (post-scaling).
            # We approximate x_mean = 0 (training data is centered by StandardScaler).
            gamma_x = torch.exp(-torch.relu(self.W_gamma_x(delta_t)))
            x_hat = m_t * x_t + (1 - m_t) * (gamma_x * x_last)  # decay toward 0 (= mean)
            # Update last-known value only where observed
            x_last = m_t * x_t + (1 - m_t) * x_last
            
            # ── Hidden state decay: model forgets between time steps ──
            gamma_h = torch.exp(-torch.relu(self.W_gamma_h(delta_t)))
            h = gamma_h * h
            
            # ── GRU update with concatenated [x_hat | mask] ──
            gru_input = torch.cat([x_hat, m_t], dim=-1)
            h = self.gru_cell(gru_input, h)
        
        # Classify using final hidden state
        out = self.dropout(h)
        logit = self.fc(out).squeeze(-1)
        return torch.sigmoid(logit)


def prepare_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int = SEQUENCE_LENGTH
) -> tuple:
    """
    Converts a DataFrame to 3D tensors for GRU-D.
    Returns: (X, mask, delta, y) as torch tensors.

    OOM FIX: Pre-allocates numpy arrays instead of building Python lists.
    Python float objects are 24-32 bytes each vs 4 bytes in float32.
    For 225K sequences, the old list-based approach used ~30 GB;
    pre-allocation uses ~1.27 GB for the same data.
    Uses torch.from_numpy() for zero-copy tensor creation (saves another 1.27 GB).
    """
    n_features = len(feature_cols)

    # ── Pass 1: Count total sequences (fast, no allocation) ───────────────
    total_seqs = 0
    for pid, group in df.groupby("patient_id"):
        n_rows = len(group)
        if n_rows > seq_len:
            total_seqs += n_rows - seq_len

    if total_seqs == 0:
        return None, None, None, None

    # ── Pre-allocate contiguous float32 arrays ────────────────────────────
    # This is the key OOM fix: allocating 1.27 GB upfront instead of
    # accumulating 30 GB of Python float objects in lists.
    alloc_gb = total_seqs * seq_len * n_features * 3 * 4 / (1024**3)
    logger.info(
        f"  prepare_sequences: {total_seqs:,} sequences × "
        f"{seq_len}×{n_features}×3 → pre-allocating {alloc_gb:.2f} GB (float32)"
    )
    X_all = np.zeros((total_seqs, seq_len, n_features), dtype=np.float32)
    M_all = np.zeros((total_seqs, seq_len, n_features), dtype=np.float32)
    D_all = np.zeros((total_seqs, seq_len, n_features), dtype=np.float32)
    y_all = np.zeros(total_seqs, dtype=np.float32)

    # ── Pass 2: Fill arrays directly (no Python list intermediates) ───────
    idx = 0
    for pid, group in df.groupby("patient_id"):
        group = group.sort_values("timestamp").reset_index(drop=True) if "timestamp" in group.columns else group.reset_index(drop=True)
        if len(group) <= seq_len:
            continue

        feat_vals = group[feature_cols].values.astype(np.float32)
        mask_vals = (~group[feature_cols].isna()).values.astype(np.float32)
        labels = group[target_col].values

        for t in range(len(group) - seq_len):
            x_win = feat_vals[t:t + seq_len]
            m_win = mask_vals[t:t + seq_len]

            # Delta: time since last valid observation per feature
            d_win = np.zeros_like(x_win)
            last_obs = np.zeros(n_features)
            for row_i in range(seq_len):
                for col_j in range(n_features):
                    if m_win[row_i, col_j] == 1:
                        last_obs[col_j] = row_i
                    d_win[row_i, col_j] = row_i - last_obs[col_j]
            # BUG-AUDIT-4 FIX: Clamp deltas to [0, seq_len] to match
            # PatientSequenceDataset.__iter__() (line 1156). Without this,
            # validation/test data sees unbounded deltas while training data
            # sees clamped deltas — a distribution mismatch that causes
            # activation explosions in TCN.
            d_win = np.clip(d_win, 0.0, float(seq_len))

            X_all[idx] = np.nan_to_num(x_win, nan=0.0)
            M_all[idx] = m_win
            D_all[idx] = d_win
            y_all[idx] = float(labels[t + seq_len]) if (t + seq_len) < len(group) else 0.0
            idx += 1

    # Trim if fewer sequences were produced than counted (shouldn't happen,
    # but guard against edge cases in sort order changes)
    if idx < total_seqs:
        X_all = X_all[:idx]
        M_all = M_all[:idx]
        D_all = D_all[:idx]
        y_all = y_all[:idx]

    # OOM FIX: torch.from_numpy() shares memory (zero-copy) instead of
    # torch.tensor() which creates a full copy. Saves ~1.27 GB.
    return (
        torch.from_numpy(X_all),
        torch.from_numpy(M_all),
        torch.from_numpy(D_all),
        torch.from_numpy(y_all),
    )

# ─────────────────────────────────────────────────────────────────────────────
# OOM FIX: Streaming IterableDataset for memory-efficient sequence building
# ─────────────────────────────────────────────────────────────────────────────
# The old prepare_sequences() built ALL sequences into Python lists, then
# np.stack'd into a giant array, then torch.tensor'd — three copies of ~8 GB
# existed simultaneously = ~24 GB peak → OOM on 24 GB Mac.
#
# PatientSequenceDataset streams sequences on-the-fly per patient per batch.
# Peak memory: ~batch_size × 9504 bytes = ~600 KB. Zero OOM risk.
# ─────────────────────────────────────────────────────────────────────────────

class PatientSequenceDataset(torch.utils.data.IterableDataset):
    """
    Streams (X, mask, delta, y) sequences on-the-fly from a DataFrame.
    
    Instead of materializing all ~850K sequences into RAM at once (8+ GB),
    this iterates over patients one at a time and yields individual sequences.
    The DataLoader handles batching, so peak memory = batch_size × seq_size.
    
    Args:
        df: DataFrame with patient data (must have 'patient_id' column)
        feature_cols: List of feature column names
        target_col: Target column name
        seq_len: Sequence window length
        patient_ids: List of patient IDs to iterate over (pre-shuffled)
    """
    
    def __init__(self, df, feature_cols, target_col, seq_len, patient_ids):
        super().__init__()
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.patient_ids = list(patient_ids)
        # Pre-index patient groups for O(1) lookup
        self._groups = {pid: grp for pid, grp in df.groupby("patient_id")}
    
    def __iter__(self):
        # AUDIT-FIX-2: Shuffle patient order each epoch for better SGD convergence.
        # Previously iterated in the same fixed order every epoch — the model
        # could memorize patient-specific batch patterns.
        import random as _rand
        pids = list(self.patient_ids)
        _rand.shuffle(pids)
        for pid in pids:
            group = self._groups.get(pid)
            if group is None or len(group) <= self.seq_len:
                continue
            
            group = group.sort_values("timestamp").reset_index(drop=True) if "timestamp" in group.columns else group.reset_index(drop=True)
            feat_vals = group[self.feature_cols].values.astype(np.float32)
            labels = group[self.target_col].values
            
            for t in range(len(group) - self.seq_len):
                x = feat_vals[t:t + self.seq_len]
                m = (~np.isnan(x)).astype(np.float32)
                
                # Delta: time since last valid observation per feature
                d = np.zeros_like(x)
                last_obs = np.zeros(len(self.feature_cols))
                for r in range(x.shape[0]):
                    for c in range(x.shape[1]):
                        if m[r, c] == 1:
                            last_obs[c] = r
                        d[r, c] = r - last_obs[c]
                
                # R13-FIX-2: Clamp deltas to [0, seq_len]. Values > seq_len mean
                # "not observed in this window" — unbounded deltas cause activation
                # explosions in TCN (no learned decay gate like GRU-D).
                d = np.clip(d, 0.0, float(self.seq_len))
                
                x = np.nan_to_num(x, nan=0.0)
                y = float(labels[t + self.seq_len]) if (t + self.seq_len) < len(group) else 0.0
                
                yield (
                    torch.tensor(x, dtype=torch.float32),
                    torch.tensor(m, dtype=torch.float32),
                    torch.tensor(d, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32),
                )
    
    def approx_len(self):
        """Approximate sequence count for logging (not exact — avoids full scan)."""
        sample_size = min(100, len(self.patient_ids))
        sample_pids = self.patient_ids[:sample_size]
        total = sum(max(0, len(self._groups.get(p, [])) - self.seq_len) for p in sample_pids)
        return int(total * len(self.patient_ids) / max(sample_size, 1))


def _collate_seq_batch(batch):
    """Custom collate for PatientSequenceDataset — stacks tuple of (x, m, d, y)."""
    xs, ms, ds, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ms), torch.stack(ds), torch.stack(ys)


def train_grud(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_epochs: int = 30,
    batch_size: int = 64,
    seq_len: int = SEQUENCE_LENGTH,
    ckpt_dir: Optional[Path] = None,
) -> GRUD:
    """Trains the GRU-D sequential model using streaming IterableDataset. Returns trained model."""

    min_rows_needed = seq_len + 1
    if len(df) < min_rows_needed:
        logger.warning(f"  GRU-D skipped — only {len(df)} rows (need ≥ {min_rows_needed}).")
        return None
    logger.info(f"  GRU-D: {len(df):,} rows available, seq_len={seq_len}")

    if "patient_id" not in df.columns:
        logger.warning(
            "  GRU-D skipped — dataset has no patient_id column. "
            "Patient-level split is required to prevent sequence leakage."
        )
        return None

    # ── Patient-level train/val split ─────────────────────────────────────
    unique_pids = df["patient_id"].unique()
    local_rng = np.random.default_rng(SEED)
    local_rng.shuffle(unique_pids)
    n_val = max(1, int(len(unique_pids) * 0.2))
    val_pids = set(unique_pids[-n_val:])
    train_pids = [p for p in unique_pids if p not in val_pids]
    logger.info(f"  GRU-D patient split: {len(train_pids):,} train / {n_val:,} val patients")

    # ── Z-normalization (Bug 36 fix) ──────────────────────────────────────
    feat_numerics = [f for f in feature_cols if f in df.columns]
    df_tr_raw = df[~df["patient_id"].isin(val_pids)]
    tr_medians = df_tr_raw[feat_numerics].median()
    grud_scaler = StandardScaler()
    grud_scaler.fit(df_tr_raw[feat_numerics].fillna(tr_medians))
    del df_tr_raw  # Free immediately

    if ckpt_dir is not None:
        joblib.dump(grud_scaler, ckpt_dir.parent / "grud_scaler.pkl")
        # BUG-R7-1 fix: Save training-set medians for consistent NaN-fill at inference.
        # api.py loads these to match the exact fill values used before z-scaling.
        joblib.dump(tr_medians.to_dict(), ckpt_dir.parent / "grud_medians.pkl")
        logger.info("  GRU-D: z-scaler + training medians fitted and saved")

    # ── Scale the full DataFrame (preserving NaN structure for mask) ──────
    # BUG-R9-1 FIX: Must copy df — do NOT modify in-place.
    # train_target() passes the same df_for_seq object to both train_grud()
    # and train_tcn(). In-place z-scaling here would corrupt TCN's input
    # (double z-scaling) and test-set sequential eval (also double z-scaled).
    #
    # OOM FIX preserved: cast to float32 BEFORE copying, so the copy is
    # ~0.97 GB (float32) instead of ~1.94 GB (float64). This still saves
    # ~1 GB compared to the original float64 df.copy().
    df[feat_numerics] = df[feat_numerics].astype(np.float32)  # halve in-place (safe — only precision)
    df_scaled = df.copy()  # copy the float32 version (~0.97 GB, not 1.94 GB)
    was_missing = df[feat_numerics].isna()
    df_scaled[feat_numerics] = grud_scaler.transform(df[feat_numerics].fillna(tr_medians)).astype(np.float32)
    df_scaled[feat_numerics] = df_scaled[feat_numerics].where(~was_missing, other=np.nan)
    del was_missing
    import gc; gc.collect()
    log_ram("pre-IterableDataset GRU-D")
    # Free df_scaled after building dataset + val sequences (~0.97 GB)
    # PatientSequenceDataset._groups holds grouped subsets, not df_scaled itself.
    _df_scaled_ref = df_scaled  # prevent early GC before dataset is built

    # ── Streaming IterableDataset (OOM FIX) ───────────────────────────────
    train_dataset = PatientSequenceDataset(df_scaled, feature_cols, target_col, seq_len, train_pids)
    approx_seqs = train_dataset.approx_len()
    logger.info(f"  GRU-D IterableDataset: ~{approx_seqs:,} sequences (streaming, ~600KB/batch)")

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=_collate_seq_batch,
        num_workers=0,  # Must be 0 for IterableDataset with in-memory DataFrame
    )

    # ── Validation set (small — safe to build in memory) ──────────────────
    val_df = df_scaled[df_scaled["patient_id"].isin(val_pids)]
    logger.info(f"  GRU-D val: building sequences for {val_df['patient_id'].nunique():,} patients ({len(val_df):,} rows)...")
    log_ram("pre-val-sequences GRU-D")
    X_v_raw, M_v_raw, D_v_raw, y_v_raw = prepare_sequences(
        val_df, feature_cols, target_col, seq_len=seq_len
    )
    del val_df, _df_scaled_ref
    # BUG-R9-2 FIX: Free df_scaled (~0.97 GB) now that dataset + val are built
    del df_scaled
    gc.collect()
    log_ram("post-val-sequences GRU-D")
    if X_v_raw is None:
        logger.warning("  GRU-D skipped — insufficient val sequences.")
        return None
    X_v = X_v_raw.to(DEVICE); del X_v_raw
    M_v = M_v_raw.to(DEVICE); del M_v_raw
    D_v = D_v_raw.to(DEVICE); del D_v_raw
    y_v = y_v_raw.to(DEVICE); del y_v_raw
    gc.collect()
    logger.info(f"  GRU-D val sequences: {len(X_v):,} | val positive rate: {float(y_v.mean()):.2%}")

    # ── Model + optimizer ─────────────────────────────────────────────────
    model = GRUD(input_size=len(feature_cols)).to(DEVICE)
    # Determine loss based on positive rate (approximate from val set)
    pos_rate = float(y_v.mean())
    if pos_rate < 0.03:
        criterion = FocalBCE(gamma=2.0, alpha=1.0 - pos_rate).to(DEVICE)
        logger.info(f"  GRU-D using FocalBCE (pos_rate={pos_rate:.2%} — severe imbalance)")
    else:
        # BUG-R6-3 fix: GRUD.forward() returns sigmoid output (probabilities).
        # BCEWithLogitsLoss applies sigmoid internally → double-sigmoid.
        # Use nn.BCELoss which expects probabilities directly.
        pos_weight = torch.tensor([(1.0 - pos_rate) / max(pos_rate, 1e-6)]).to(DEVICE)
        criterion = nn.BCELoss(weight=None)  # class weighting handled by FocalBCE path or sample weights
        # For weighted BCE on probabilities, we apply pos_weight manually:
        _bce_pos_weight = pos_weight
        _orig_criterion = criterion
        def _weighted_bce(pred, target):
            bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
            weight = target * _bce_pos_weight + (1 - target)
            return (bce * weight).mean()
        criterion = _weighted_bce
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_auroc = 0.0
    best_state = None
    _patience = 8      # AUDIT-FIX: early-stop patience (saves 5-20 hours of wasted GPU time)
    _patience_ctr = 0  # counts epochs since last improvement
    _prev_loss = None  # LOG-FIX: track loss trend between epochs
    _best_epoch = 0    # LOG-FIX: track which epoch produced the best model

    training_start = time.time()
    for epoch in range(n_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, mb, db, yb in loader:
            xb, mb, db, yb = xb.to(DEVICE), mb.to(DEVICE), db.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb, mb, db)
            # BUG-R6-3 fix: both FocalBCE and _weighted_bce now accept
            # probability outputs directly — no logit conversion needed.
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            # Heartbeat every 2000 batches so you can see progress within an epoch
            if n_batches % 2000 == 0:
                logger.info(f"    GRU-D epoch {epoch+1} batch {n_batches:,}... (loss: {total_loss/n_batches:.4f})")

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Batched validation forward pass (avoids 213K sequences at once)
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for i in range(0, len(X_v), batch_size):
                bp = model(X_v[i:i+batch_size], M_v[i:i+batch_size], D_v[i:i+batch_size])
                val_preds_list.append(bp.cpu().numpy())
        val_preds = np.concatenate(val_preds_list)

        val_y_np = y_v.cpu().numpy()
        if val_y_np.sum() > 0:
            val_auroc = roc_auc_score(val_y_np, val_preds)
            if val_auroc > best_auroc:
                best_auroc = val_auroc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                _patience_ctr = 0  # reset patience on improvement
            else:
                _patience_ctr += 1

            # LOG-FIX: Track best epoch and compute loss trend
            if val_auroc >= best_auroc:
                _best_epoch = epoch + 1
            avg_loss = total_loss / max(n_batches, 1)
            _loss_trend = ""
            if _prev_loss is not None and _prev_loss > 0:
                pct = ((avg_loss - _prev_loss) / _prev_loss) * 100
                _loss_trend = f" (↓{abs(pct):.1f}%)" if pct < 0 else f" (↑{pct:.1f}%)"
            _prev_loss = avg_loss

            elapsed = time.time() - training_start
            eta_min = (elapsed / (epoch + 1)) * (n_epochs - epoch - 1) / 60
            logger.info(
                f"  GRU-D Epoch {epoch+1}/{n_epochs} | "
                f"Loss: {avg_loss:.4f}{_loss_trend} | "
                f"Val AUROC: {val_auroc:.4f} (best: {best_auroc:.4f} @ ep{_best_epoch}) | "
                f"{epoch_time:.0f}s | ETA: {_fmt_duration(eta_min * 60)} | "
                f"patience: {_patience_ctr}/{_patience}"
            )
            if (epoch + 1) % 10 == 0:
                log_ram(f"GRU-D epoch {epoch+1}")

            # AUDIT-FIX: Early stopping based on patience
            if _patience_ctr >= _patience:
                logger.info(f"  GRU-D early stopping at epoch {epoch+1} (no improvement for {_patience} epochs)")
                break

    _grud_duration = time.time() - training_start
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.success(f"  ✅ GRU-D complete in {_fmt_duration(_grud_duration)} | Best AUROC: {best_auroc:.4f} (epoch {_best_epoch})")
        # R13-FIX-6: Save best AUROC alongside checkpoint for resume logging
        if ckpt_dir is not None:
            _auroc_path = ckpt_dir / "grud_best_auroc.json"
            with open(_auroc_path, "w") as f:
                json.dump({"best_auroc": round(best_auroc, 6)}, f)

    return model


###############################################################################
# ── SECTION 3b: TEMPORAL CONVOLUTIONAL NETWORK (TCN) ───────────────────────
###############################################################################

class _CausalConv1d(nn.Module):
    """Causal conv: output at t only sees inputs ≤ t (no future leakage)."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        # R13-FIX-1: Replace deprecated weight_norm with parametrizations.weight_norm.
        # The old weight_norm computes w = g*v/||v|| — when ||v|| → 0, division by zero → NaN.
        # parametrizations.weight_norm adds internal epsilon to prevent this.
        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=self.pad)
        )

    def forward(self, x):
        return self.conv(x)[:, :, :-self.pad] if self.pad > 0 else self.conv(x)


class _TCNResBlock(nn.Module):
    def __init__(self, n_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        # FIX-1: Added GroupNorm(1, C) after each CausalConv1d.
        # GroupNorm(1, C) = LayerNorm over channels for conv layers (B, C, T).
        # Prevents activation magnitudes from exploding across residual additions
        # — the root cause of NaN loss at batch 2,227 (sepsis) and 849 (hypotension).
        self.net = nn.Sequential(
            _CausalConv1d(n_ch, n_ch, kernel, dilation),
            nn.GroupNorm(1, n_ch),
            nn.ReLU(), nn.Dropout(dropout),
            _CausalConv1d(n_ch, n_ch, kernel, dilation),
            nn.GroupNorm(1, n_ch),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + x)   # residual skip


class TCN(nn.Module):
    """Temporal Convolutional Network with exponential dilation.

    Receptive field = 2 * (kernel-1) * sum(2^i for i in 0..n_levels-1)
    Default (kernel=3, n_levels=6): RF = 4*(1+2+4+8+16+32) = 252 time-steps.

    Advantages over GRU-D:
      - Fully parallel (no sequential dependency) → 5-10× faster on MPS
      - No vanishing gradients, deterministic across seeds
      - WeightNorm on all convolutions for stable training

    Missing-data handling: time-delta appended as extra input channel
    (same strategy as GRU-D, simpler implementation).
    """
    def __init__(self, input_size: int, n_channels: int = 64,  # TEMP: 64 for old checkpoint compat, change to 48 after retrain
                 kernel_size: int = 3, n_levels: int = 6, dropout: float = 0.3):
        super().__init__()
        # BUG-CLAUDE-10-1 fix: TCN now uses all three inputs: x, mask, delta.
        # mask=0 means the value was imputed (not actually observed) — a strong
        # clinical signal. Previously mask was accepted by forward() but silently
        # discarded. Now concatenated: input has 3×input_size channels.
        self.input_proj = nn.Conv1d(input_size * 3, n_channels, 1)
        # R13-FIX-5: BatchNorm after input_proj normalizes the 468→64 channel
        # projection output, preventing activation explosions from large deltas.
        self.input_bn = nn.BatchNorm1d(n_channels)
        dilations = [2 ** i for i in range(n_levels)]
        self.blocks = nn.ModuleList([
            _TCNResBlock(n_channels, kernel_size, d, dropout) for d in dilations
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_channels, 1)

        # TCN-FIX-B: Xavier initialization for all Conv1d layers.
        # Default Kaiming init scales weights by 1/√fan_in. With 468 input channels,
        # initial outputs can be huge (±50), causing sigmoid saturation → log(0) → NaN.
        # Xavier scales by 1/√(fan_in + fan_out), reducing initial magnitude by ~3×.
        #
        # IMPORTANT: _CausalConv1d wraps Conv1d in parametrizations.weight_norm,
        # which decomposes weight into weight_v (direction) and weight_g (magnitude).
        # nn.init.xavier_uniform_(module.weight) is a NO-OP for parametrized modules
        # because module.weight is a computed property — the init writes to a temp
        # tensor that's immediately discarded. We must target weight_v directly.
        nn.init.xavier_uniform_(self.input_proj.weight)  # not weight-normed, works directly
        nn.init.zeros_(self.input_proj.bias)
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, nn.Conv1d):
                    # Check if this Conv1d is wrapped in weight_norm (has weight_v param)
                    if hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'):
                        # Weight-normed: init the direction vector (weight_v)
                        nn.init.xavier_uniform_(module.parametrizations.weight.original0)
                    else:
                        nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, mask, delta):
        # BUG-CLAUDE-10-1 fix: concatenate [x, mask, delta] — mask=0 means imputed.
        # Gives TCN the same missingness information GRU-D uses in its gating.
        # (B, T, F) → concat → (B, T, 3F) → permute → (B, 3F, T) for Conv1d
        inp = torch.cat([x, mask, delta], dim=-1).permute(0, 2, 1)
        out = self.input_bn(self.input_proj(inp))
        for block in self.blocks:
            out = block(out)
        # Global average pool over time
        out = out.mean(dim=-1)            # (B, n_channels)
        out = self.dropout(out)
        return torch.sigmoid(self.fc(out).squeeze(-1))


def train_tcn(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_epochs: int = 50,
    batch_size: int = 64,
    seq_len: int = SEQUENCE_LENGTH,   # Added: per-target sequence length
    ckpt_dir: Optional[Path] = None,
) -> TCN:
    """Trains the TCN sequential model. Returns trained model."""
    # OOM-FIX (mirrors GRU-D fix): Do NOT call prepare_sequences() on the full
    # DataFrame here as an upfront data check. For CinC 2019 (1.55M rows) this
    # allocates ~9.45 GB of float32 tensors (X + mask + delta) before any split.
    # When TCN runs after GRU-D, macOS may not have fully reclaimed GRU-D's
    # ~9.3 GB of freed tensors yet — worst-case peak hits ~22 GB on a 24 GB system,
    # causing the same memory-pressure CPU freeze observed in the first training run.
    #
    # Fix: use a lightweight row-count guard. Actual sequences are built later,
    # AFTER the patient split, on only 80% of patients (~7.5 GB instead of 9.45 GB).
    min_rows_needed = seq_len + 1
    if len(df) < min_rows_needed:
        logger.warning(f"  TCN skipped — only {len(df)} rows (need ≥ {min_rows_needed}).")
        return None
    logger.info(f"  TCN: {len(df):,} rows available, seq_len={seq_len} — building sequences after patient split.")

    # NEW-BUG-FIX (C10): TCN must use the same grud_scaler as GRU-D.
    # At inference, api.py applies grud_scaler.transform() to ALL sequential
    # inputs (both GRU-D and TCN). If TCN is trained on raw-unit data but
    # fed z-scored data at inference, predictions will be wrong.
    # Fix: apply the saved grud_scaler before building sequences for TCN.
    # If the scaler file doesn't exist yet (GRU-D not trained first or skipped),
    # we fit a fresh scaler on TCN training data and save it.
    feat_numerics = [f for f in feature_cols if f in df.columns]
    scaler_path = ckpt_dir.parent / "grud_scaler.pkl" if ckpt_dir is not None else None
    if scaler_path is not None and scaler_path.exists():
        tcn_scaler = joblib.load(scaler_path)
        logger.info("  TCN: loaded existing grud_scaler for z-normalisation")
    else:
        # Fit a new scaler if GRU-D was skipped; save it for inference
        tcn_scaler = StandardScaler()
        tcn_scaler.fit(df[feat_numerics].fillna(df[feat_numerics].median()))
        if scaler_path is not None:
            joblib.dump(tcn_scaler, scaler_path)
            logger.info("  TCN: fitted and saved new grud_scaler (GRU-D was skipped)")

    # Bug 23 fix: patient-level split (mirrors fix in train_grud)
    if "patient_id" in df.columns:
        unique_pids = df["patient_id"].unique()
        # BUG-AUDIT-1 FIX: Use SEED + 1 (not SEED) so TCN gets a DIFFERENT
        # patient split from GRU-D. Previously both used SEED=42, producing
        # identical train/val splits — the adaptive blend weight was evaluated
        # on in-sample data for both models simultaneously.
        local_rng = np.random.default_rng(SEED + 1)
        local_rng.shuffle(unique_pids)
        n_val = max(1, int(len(unique_pids) * 0.2))
        val_pids = set(unique_pids[-n_val:])
        df_tr_raw = df[~df["patient_id"].isin(val_pids)].copy()
        df_vl_raw = df[df["patient_id"].isin(val_pids)].copy()
    else:
        split = int(0.8 * len(df))
        df_tr_raw = df.iloc[:split].copy()
        df_vl_raw = df.iloc[split:].copy()

    # Apply scaler to train/val splits, preserving NaN structure for mask
    tr_medians = df_tr_raw[feat_numerics].median()
    tr_was_missing = df_tr_raw[feat_numerics].isna()
    vl_was_missing = df_vl_raw[feat_numerics].isna()
    df_tr = df_tr_raw.copy()
    df_vl = df_vl_raw.copy()
    df_tr[feat_numerics] = tcn_scaler.transform(df_tr_raw[feat_numerics].fillna(tr_medians))
    df_vl[feat_numerics] = tcn_scaler.transform(df_vl_raw[feat_numerics].fillna(tr_medians))
    # Restore NaNs so prepare_sequences() computes the correct mask
    df_tr[feat_numerics] = df_tr[feat_numerics].where(~tr_was_missing, other=np.nan)
    df_vl[feat_numerics] = df_vl[feat_numerics].where(~vl_was_missing, other=np.nan)

    # OOM-FIX: Free large intermediates BEFORE building sequences/datasets.
    del tr_was_missing, vl_was_missing  # free boolean masks
    del df_tr_raw, df_vl_raw            # free originals — only need scaled versions
    import gc; gc.collect()
    log_ram("pre-sequence-build TCN")

    # ── STREAMING FIX: Use PatientSequenceDataset for training (mirrors GRU-D) ──
    # The old code called prepare_sequences() on 80% of data, materializing
    # 857K sequences × 12 × 156 × 3 × 4 bytes = 17.94 GB → OOM on 24 GB Mac.
    # PatientSequenceDataset streams sequences on-the-fly: ~600 KB per batch.
    train_pids = list(df_tr["patient_id"].unique()) if "patient_id" in df_tr.columns else None
    if train_pids is None:
        logger.warning("  TCN skipped — no patient_id column for streaming.")
        return None

    train_dataset = PatientSequenceDataset(df_tr, feature_cols, target_col, seq_len, train_pids)
    approx_seqs = train_dataset.approx_len()
    logger.info(f"  TCN IterableDataset: ~{approx_seqs:,} train sequences (streaming, ~600KB/batch)")
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=_collate_seq_batch,
        num_workers=0,  # Must be 0 for IterableDataset with in-memory DataFrame
    )

    # ── Validation set (20% of patients — small enough to pre-allocate) ──
    logger.info(f"  TCN val: building sequences for {df_vl['patient_id'].nunique():,} patients ({len(df_vl):,} rows)...")
    log_ram("pre-val-sequences TCN")
    X_vl_raw, M_vl_raw, D_vl_raw, y_vl_raw = prepare_sequences(df_vl, feature_cols, target_col, seq_len=seq_len)
    del df_vl; gc.collect()
    log_ram("post-val-sequences TCN")

    if X_vl_raw is None:
        logger.warning("  TCN skipped — insufficient val sequences after patient-level split.")
        del df_tr; gc.collect()
        return None
    if "patient_id" in df.columns:
        logger.info(f"  TCN patient-level split: {len(unique_pids)-n_val} train / {n_val} val patients")

    # Move val tensors to device one at a time (frees CPU copies immediately)
    X_v = X_vl_raw.to(DEVICE); del X_vl_raw
    M_v = M_vl_raw.to(DEVICE); del M_vl_raw
    D_v = D_vl_raw.to(DEVICE); del D_vl_raw
    y_v = y_vl_raw.to(DEVICE); del y_vl_raw
    gc.collect()
    log_ram("post-val-tensor-load TCN")

    model = TCN(input_size=len(feature_cols)).to(DEVICE)

    # TCN-FIX-C: Always use FocalBCE for TCN (not just when pos_rate < 3%).
    # The old label-smoothing BCE path used nn.functional.binary_cross_entropy()
    # which does NOT clamp predictions — if TCN outputs exactly 0.0 or 1.0 (common
    # at initialization before BatchNorm warms up), log(0) = -inf → NaN loss.
    # FocalBCE already clamps to [1e-7, 1-1e-7] (line 200) AND focuses on hard
    # examples, which is better for all ICU targets (all have class imbalance).
    pos_rate_tcn = float(y_v.mean())
    tcn_criterion = FocalBCE(gamma=2.0, alpha=max(1.0 - pos_rate_tcn, 0.5)).to(DEVICE)
    logger.info(f"  TCN using FocalBCE (pos_rate={pos_rate_tcn:.2%}, alpha={max(1.0 - pos_rate_tcn, 0.5):.2f})")
    def smooth_bce(pred, target):
        # TCN-FIX-A: Clamp predictions to [1e-6, 1-1e-6] as defense-in-depth.
        # FocalBCE already clamps internally, but this guards against any code
        # path that might bypass the criterion (e.g., future refactors).
        pred = pred.clamp(1e-6, 1 - 1e-6)
        return tcn_criterion(pred, target)

    # FIX-1: Lower LR from 3e-4 → 1e-4 with linear warmup.
    # Combined with GroupNorm in _TCNResBlock, this prevents NaN loss that
    # crashed sepsis (batch 2,227) and hypotension (batch 849) in the last run.
    _tcn_lr = 1e-4
    _warmup_batches = 500
    optimizer = torch.optim.Adam(model.parameters(), lr=_tcn_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    best_auroc, best_state = 0.0, None
    _global_batch = 0
    _nan_consecutive = 0
    _MAX_NAN_CONSECUTIVE = 100  # abort if 100 consecutive NaN batches
    _patience = 10     # AUDIT-FIX: early-stop patience (TCN gets more patience than GRU-D
    _patience_ctr = 0  # because warmup takes ~500 batches and early epochs may be noisy)
    _prev_loss_tcn = None   # LOG-FIX: track loss trend between epochs
    _best_epoch_tcn = 0     # LOG-FIX: track which epoch produced the best model

    tcn_training_start = time.time()
    for epoch in range(n_epochs):
        epoch_start_tcn = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0
        n_valid_batches = 0
        _epoch_nan_batches = 0  # LOG-FIX: count NaN batches per epoch
        for xb, mb, db, yb in loader:
            xb, mb, db, yb = xb.to(DEVICE), mb.to(DEVICE), db.to(DEVICE), yb.to(DEVICE)

            # R13-FIX-3: Linear warmup — gradually increase LR over first 500 batches
            _global_batch += 1
            if _global_batch <= _warmup_batches:
                warmup_lr = _tcn_lr * (_global_batch / _warmup_batches)
                for pg in optimizer.param_groups:
                    pg['lr'] = warmup_lr

            optimizer.zero_grad()
            pred = model(xb, mb, db)
            loss = smooth_bce(pred, yb)

            # R13-FIX-4: NaN-aware batch skip. If loss is NaN, don't backward/step.
            # This prevents NaN from poisoning ALL model weights.
            if torch.isnan(loss) or torch.isinf(loss):
                _nan_consecutive += 1
                _epoch_nan_batches += 1  # LOG-FIX: per-epoch NaN counter
                if _nan_consecutive >= _MAX_NAN_CONSECUTIVE:
                    logger.error(f"  TCN: {_MAX_NAN_CONSECUTIVE} consecutive NaN batches — aborting.")
                    break
                if _nan_consecutive == 1 or _nan_consecutive % 50 == 0:
                    logger.warning(f"    TCN batch {n_batches}: NaN/Inf loss — skipping update ({_nan_consecutive} consecutive)")
                n_batches += 1
                continue
            _nan_consecutive = 0

            loss.backward()
            # FIX-1: Tighter norm clip (1.0→0.5) + value clip (cap individual
            # gradient elements at ±1.0). Norm clip alone doesn't prevent a
            # single large gradient from dominating the update direction.
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            n_valid_batches += 1
            if n_batches % 2000 == 0:
                logger.info(f"    TCN epoch {epoch+1} batch {n_batches:,}... (loss: {total_loss/max(n_valid_batches,1):.4f})")

        if _nan_consecutive >= _MAX_NAN_CONSECUTIVE:
            logger.error("  TCN training aborted due to persistent NaN loss.")
            break
        scheduler.step()

        # Batched validation forward pass (avoids 213K sequences at once)
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for i in range(0, len(X_v), batch_size):
                bp = model(X_v[i:i+batch_size], M_v[i:i+batch_size], D_v[i:i+batch_size])
                val_preds_list.append(bp.cpu().numpy())
        val_preds = np.concatenate(val_preds_list)
        val_y = y_v.cpu().numpy()
        epoch_time_tcn = time.time() - epoch_start_tcn
        if val_y.sum() > 0:
            auroc = roc_auc_score(val_y, val_preds)
            if auroc > best_auroc:
                best_auroc = auroc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                _patience_ctr = 0  # reset patience on improvement
                _best_epoch_tcn = epoch + 1
            else:
                _patience_ctr += 1
            if ckpt_dir and (epoch + 1) % 10 == 0 and best_state is not None:
                torch.save(best_state, ckpt_dir / f"tcn_epoch_{epoch+1}.pt")

            # LOG-FIX: Loss trend + NaN batch summary per epoch
            avg_loss_tcn = total_loss / max(n_valid_batches, 1)
            _loss_trend_tcn = ""
            if _prev_loss_tcn is not None and _prev_loss_tcn > 0:
                pct = ((avg_loss_tcn - _prev_loss_tcn) / _prev_loss_tcn) * 100
                _loss_trend_tcn = f" (↓{abs(pct):.1f}%)" if pct < 0 else f" (↑{pct:.1f}%)"
            _prev_loss_tcn = avg_loss_tcn
            _nan_str = f" | NaN: {_epoch_nan_batches}/{n_batches}" if _epoch_nan_batches > 0 else ""

            elapsed = time.time() - tcn_training_start
            eta_min = (elapsed / (epoch + 1)) * (n_epochs - epoch - 1) / 60
            logger.info(
                f"  TCN Epoch {epoch+1}/{n_epochs} | "
                f"Loss: {avg_loss_tcn:.4f}{_loss_trend_tcn} | "
                f"Val AUROC: {auroc:.4f} (best: {best_auroc:.4f} @ ep{_best_epoch_tcn}) | "
                f"{epoch_time_tcn:.0f}s | ETA: {_fmt_duration(eta_min * 60)} | "
                f"patience: {_patience_ctr}/{_patience}{_nan_str}"
            )
            if (epoch + 1) % 10 == 0:
                log_ram(f"TCN epoch {epoch+1}")

            # AUDIT-FIX: Early stopping based on patience
            if _patience_ctr >= _patience:
                logger.info(f"  TCN early stopping at epoch {epoch+1} (no improvement for {_patience} epochs)")
                break

    _tcn_duration = time.time() - tcn_training_start
    # Free streaming dataset (holds reference to df_tr via _groups dict)
    del train_dataset, loader, df_tr
    gc.collect()

    if best_state:
        model.load_state_dict(best_state)
        logger.success(f"  ✅ TCN complete in {_fmt_duration(_tcn_duration)} | Best AUROC: {best_auroc:.4f} (epoch {_best_epoch_tcn})")
        return model
    else:
        # R14-FIX: Return None if no valid epoch completed (e.g., NaN abort).
        # A model object is always truthy in Python — returning it would cause
        # train_target() to save an undertrained model as tcn_model.pt and use
        # its random-quality predictions for 30% of the final ensemble blend.
        logger.warning("  TCN: no valid epoch completed — returning None (model will be excluded from ensemble)")
        return None


###############################################################################
# ── SECTION 3c: 1D-RESNET CNN (ECG WAVEFORM — CARDIAC ARREST ONLY) ─────────
###############################################################################

class _Res1DBlock(nn.Module):
    def __init__(self, ch: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(ch),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + x)


class ResNet1D(nn.Module):
    """1-D Residual CNN for short (30-second) ECG segments at 250 Hz.

    Architecture: Input → 3 progressive blocks (channels: 32→64→128) →
    GlobalAvgPool → Dropout(0.5) → FC → sigmoid.

    Kept deliberately shallow (3 blocks × 2 conv each) because CUDB+SDDB
    have very few records. Deep ResNets overfit on <100 samples;
    heavy dropout + BatchNorm + weight decay compensate.

    Reference: Hannun et al., 2019; Chen et al., PhysioNet 2017.
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            _Res1DBlock(32), _Res1DBlock(32),
            nn.Conv1d(32, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            _Res1DBlock(64), _Res1DBlock(64),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            _Res1DBlock(128),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, 1, T) — single ECG channel
        out = self.stem(x)
        out = self.blocks(out)
        out = out.mean(dim=-1)       # global average pool over time
        out = self.dropout(out)
        return torch.sigmoid(self.fc(out).squeeze(-1))


###############################################################################
# ── SECTION 3d: CHECKPOINT HELPERS ─────────────────────────────────────────
###############################################################################

def save_checkpoint(state: dict, path: Path) -> None:
    """Atomically saves a checkpoint dict (overwrites safely via temp file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.rename(path)


def load_checkpoint(path: Path) -> Optional[dict]:
    """Loads a checkpoint if it exists, otherwise returns None."""
    if path.exists():
        logger.info(f"  Resuming from checkpoint: {path}")
        return torch.load(path, map_location="cpu", weights_only=True)
    return None


def fold_done(target: str, fold: int) -> bool:
    """Returns True if fold {fold} artifacts already exist on disk (resume mode)."""
    fold_dir = MODELS_DIR / target / f"fold_{fold}"
    return (fold_dir / "lgbm_model.pkl").exists()


###############################################################################
# ── SECTION 3e: SAMPLING (SMOTE-ENN / ADASYN) ──────────────────────────────
###############################################################################

def apply_sampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target: str,
) -> tuple:
    """Class-imbalance handling — SMOTE DISABLED, native class weighting only.

    SMOTE-DISABLED rationale:
      LightGBM uses class_weight='balanced' and XGBoost uses scale_pos_weight,
      both of which handle imbalance natively and correctly without generating
      synthetic data. SMOTE on top of native weighting caused:
        1. ~39 min per fold overhead (confirmed in training logs)
        2. Probability miscalibration: model trained on 55% synthetic positives
           outputs inflated probabilities at inference on 2.42%-positive real data.
           AUROC is rank-based so it's unaffected, but AUPRC (precision-based)
           is suppressed — observed AUPRC 0.075 vs expected 0.12-0.18 without SMOTE.
        3. Physiologically unrealistic interpolations between patient feature vectors
           (especially harmful for features derived from temporal trajectories).

    The SMOTE code is preserved below (but never executed) for reference.
    To re-enable: remove the early return on the next line.
    """
    # SMOTE disabled — return original data, let LGB/XGB native weighting handle imbalance
    return X_train, y_train

    # ── PRESERVED SMOTE CODE (not executed) ─────────────────────────────────
    if not HAS_IMBLEARN:  # noqa: unreachable
        logger.warning("  Sampling skipped (imbalanced-learn not installed)")
        return X_train, y_train

    n_positive = int(y_train.sum())
    if n_positive < 6:
        logger.warning(f"  Sampling skipped — only {n_positive} positive samples in fold")
        return X_train, y_train

    try:
        if target == "sepsis":
            sampler = SMOTEENN(random_state=SEED, smote=SMOTE(random_state=SEED, k_neighbors=min(5, n_positive - 1)))
        elif target == "hypotension":
            sampler = SMOTE(random_state=SEED, k_neighbors=min(5, n_positive - 1))
        else:  # hemodynamic_collapse — ADASYN for tabular only
            sampler = ADASYN(random_state=SEED, n_neighbors=min(5, n_positive - 1))

        X_res, y_res = sampler.fit_resample(X_train, y_train)
        logger.info(f"  After sampling: {X_res.shape[0]:,} rows ({y_res.mean():.2%} positive)")
        return X_res, y_res
    except Exception as e:
        logger.warning(f"  Sampling failed ({e}) — using original imbalanced data")
        return X_train, y_train


###############################################################################
# ── SECTION 3f: WFDB ECG LOADERS (CUDB + SDDB) ─────────────────────────────
###############################################################################

def _extract_ecg_features_window(signal: np.ndarray, fs: int = 250) -> dict:
    """Extracts tabular features from a short ECG window (used for CUDB/SDDB)."""
    if len(signal) < fs * 5:   # need at least 5 seconds
        return {}
    # Basic statistical features (cheap, no peak detection needed)
    features = {
        "ecg_mean":     float(np.nanmean(signal)),
        "ecg_std":      float(np.nanstd(signal)),
        "ecg_rms":      float(np.sqrt(np.nanmean(signal ** 2))),
        "ecg_range":    float(np.nanmax(signal) - np.nanmin(signal)),
        "ecg_kurtosis": float(pd.Series(signal).kurt()),
        "ecg_skewness": float(pd.Series(signal).skew()),
    }
    # Spectral power in LF (0.04-0.15 Hz) and HF (0.15-0.4 Hz) bands
    try:
        from scipy.signal import welch
        freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal) // 2))
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.40)
        lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
        hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0
        features["ecg_lf_power"] = lf_power
        features["ecg_hf_power"] = hf_power
        features["ecg_lf_hf_ratio"] = lf_power / max(hf_power, 1e-9)
    except Exception:
        pass
    return features


def load_cudb_wfdb(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Loads Creighton University VT Database ECG recordings via WFDB.

    35 records × ~8.5 min, 250 Hz, single ECG channel.
    Each record ends in VT → VF; annotation marks exact VF onset.

    For each record we create:
      - Positive windows: last 5 min before VF onset → label = 1
      - Negative windows: first 2 min of record → label = 0
    Features extracted: ECG statistical + spectral features.
    """
    if not HAS_WFDB:
        logger.warning("CUDB skipped — wfdb not installed")
        return None

    cudb_dir = data_dir / "cudb_ventricular_tachyarrhythmia"
    # Find the actual versioned subdirectory
    subdirs = sorted([d for d in cudb_dir.rglob("cu01.hea")])
    if not subdirs:
        logger.warning(f"CUDB .hea files not found in {cudb_dir}")
        return None

    record_dir = subdirs[0].parent
    record_stems = sorted(set(p.stem for p in record_dir.glob("cu*.dat")))
    logger.info(f"Loading CUDB: {len(record_stems)} records from {record_dir}")

    rows = []
    for stem in record_stems:
        try:
            rec_path = str(record_dir / stem)
            record = wfdb.rdrecord(rec_path)
            # Try .atr (reference annotations) first, fall back to .ari (unaudited)
            ann = None
            for ann_ext in ("atr", "ari"):
                try:
                    ann = wfdb.rdann(rec_path, ann_ext)
                    break
                except Exception:
                    continue
            if ann is None:
                logger.warning(f"  CUDB {stem}: no .atr or .ari annotation file")
                continue
            fs     = record.fs
            signal = record.p_signal[:, 0]   # first ECG channel

            # Find VF onset sample index.
            # CUDB WFDB convention: VF onset is marked with symbol '[' (start of
            # ventricular flutter/fibrillation). aux_note is always empty in this DB.
            # WFDB standard VF codes:
            #   '['  = start of ventricular flutter/fibrillation
            #   ']'  = end of ventricular flutter/fibrillation
            #   '!'  = ventricular flutter wave
            #   'f'  = ventricular fibrillation wave (less common)
            VF_SYMBOLS = frozenset(['[', '!', 'f'])
            vf_onset_idx = None
            for i, sym in enumerate(ann.symbol):
                if sym in VF_SYMBOLS:
                    vf_onset_idx = ann.sample[i]
                    break
            # Also try aux_note as legacy fallback.
            # WFDB standard for rhythm annotations uses parenthesis-prefix format
            # like '(VF', '(VT', '(VFIB' per PhysioNet WFDB specification.
            # Bare 'vf' would also match '(vf' after .lower() but would also
            # false-match on notes like 'pvf' (premature ventricular focus).
            # Use parenthesis-prefix patterns to be specific.
            if vf_onset_idx is None:
                VF_AUX_PATTERNS = ('(vf', '(vfib', '(vfl', '[vf')  # WFDB aux string patterns
                for i, note in enumerate(ann.aux_note):
                    if note:
                        note_l = note.strip().lower()
                        if any(pat in note_l for pat in VF_AUX_PATTERNS):
                            vf_onset_idx = ann.sample[i]
                            break
            if vf_onset_idx is None or vf_onset_idx < fs * 60:
                continue   # need at least 1 min before event

            # Negative: first 2 min
            neg_window = signal[: fs * 120]
            neg_feats  = _extract_ecg_features_window(neg_window, fs)
            if neg_feats:
                neg_feats.update({"patient_id": f"cudb_{stem}", "target_hemodynamic_collapse": 0})
                rows.append(neg_feats)

            # Positive: last 5 min before VF onset
            pos_start  = max(0, vf_onset_idx - fs * 300)
            pos_window = signal[pos_start:vf_onset_idx]
            pos_feats  = _extract_ecg_features_window(pos_window, fs)
            if pos_feats:
                pos_feats.update({"patient_id": f"cudb_{stem}", "target_hemodynamic_collapse": 1})
                rows.append(pos_feats)

        except Exception as e:
            logger.warning(f"  CUDB {stem}: {e}")

    if not rows:
        return None
    df = pd.DataFrame(rows)
    logger.success(f"CUDB: {len(df)} rows | Positive rate: {df['target_hemodynamic_collapse'].mean():.2%}")
    return df


def load_sddb_wfdb(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Loads Sudden Cardiac Death Holter Database ECG recordings via WFDB.

    23 complete Holter recordings (up to 24h), 250 Hz.
    Extracts pre-arrest windows: [6h-before, 4h-before, 2h-before] → label = 1
    Also extracts control windows from same patients 12h+ before event → label = 0.
    """
    if not HAS_WFDB:
        logger.warning("SDDB skipped — wfdb not installed")
        return None

    sddb_dir = data_dir / "sddb_sudden_cardiac"
    hea_files = sorted(sddb_dir.rglob("*.hea"))
    if not hea_files:
        logger.warning(f"SDDB .hea files not found in {sddb_dir} — still downloading?")
        return None

    logger.info(f"Loading SDDB: {len(hea_files)} records")
    rows = []
    WINDOWS_H = [6, 4, 2]   # hours before arrest to create positive windows

    for hea_path in hea_files:
        stem = hea_path.stem
        rec_path = str(hea_path.parent / stem)
        try:
            record = wfdb.rdrecord(rec_path)
            # SDDB: 12/23 records have .atr (reference), ALL 23 have .ari (unaudited)
            # Try .atr first (more reliable VF labels), fall back to .ari
            ann = None
            for ann_ext in ("atr", "ari"):
                try:
                    ann = wfdb.rdann(rec_path, ann_ext)
                    break
                except Exception:
                    continue
            if ann is None:
                logger.warning(f"  SDDB {stem}: no annotation file found")
                continue
            fs     = record.fs
            signal = record.p_signal[:, 0]
            n_samp = len(signal)

            # Find arrest index: last VT or VF annotation.
            # SDDB WFDB convention: VF/VT onset encoded in ann.symbol not aux_note.
            # Symbol codes:
            #   '['  = start of ventricular flutter/fibrillation
            #   '!'  = ventricular flutter wave
            #   'f'  = ventricular fibrillation wave  
            #   'T'  = ventricular tachycardia (some SDDB records)
            # We also check aux_note as fallback for re-annotated versions.
            VF_SYMBOLS_SDDB = frozenset(['[', '!', 'f', 'T'])
            arrest_idx = None
            # Search backwards for last VF/VT marker
            for i in range(len(ann.symbol) - 1, -1, -1):
                if ann.symbol[i] in VF_SYMBOLS_SDDB:
                    arrest_idx = ann.sample[i]
                    break
            # Legacy fallback: aux_note text search
            # WFDB standard rhythm annotations use parenthesis-prefix like '(VF', '(VT'
            if arrest_idx is None:
                VF_AUX_SDDB = ('(vf', '(vfib', '(vfl', '(vt', '[vf')  # WFDB rhythm patterns
                for i in range(len(ann.aux_note) - 1, -1, -1):
                    note = ann.aux_note[i]
                    if note:
                        note_l = note.strip().lower()
                        if any(pat in note_l for pat in VF_AUX_SDDB):
                            arrest_idx = ann.sample[i]
                            break
            if arrest_idx is None or arrest_idx < fs * 3600 * 2:
                continue   # need 2h minimum pre-arrest

            for hours in WINDOWS_H:
                win_end   = arrest_idx - int(fs * 3600 * (hours - 1))
                win_start = arrest_idx - int(fs * 3600 * hours)
                if win_start < 0:
                    continue
                window = signal[win_start:win_end]
                feats  = _extract_ecg_features_window(window[:fs * 60], fs)  # 1-min sample
                if feats:
                    feats.update({"patient_id": f"sddb_{stem}", "target_hemodynamic_collapse": 1})
                    rows.append(feats)

            # Control: window 12h before arrest
            ctrl_end   = max(0, arrest_idx - int(fs * 3600 * 12))
            ctrl_start = max(0, ctrl_end - int(fs * 3600))
            ctrl_window = signal[ctrl_start:ctrl_end]
            ctrl_feats  = _extract_ecg_features_window(ctrl_window[:fs * 60], fs)
            if ctrl_feats:
                ctrl_feats.update({"patient_id": f"sddb_{stem}", "target_hemodynamic_collapse": 0})
                rows.append(ctrl_feats)

        except Exception as e:
            logger.warning(f"  SDDB {stem}: {e}")

    if not rows:
        return None
    df = pd.DataFrame(rows)
    logger.success(f"SDDB: {len(df)} rows | Positive rate: {df['target_hemodynamic_collapse'].mean():.2%}")
    return df


###############################################################################
# ── SECTION 3g: THRESHOLD + META-STACKING ──────────────────────────────────
###############################################################################

def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 2.0) -> tuple:
    """Finds optimal classification threshold maximizing F-beta (default β=2).

    In ICU: missing a true positive (patient crashes) >> false alarm cost.
    β=2 weights recall 2× more than precision — standard for critical-care alerting.

    FIX-3: For rare events (<5% positive), also checks percentile-based thresholds.
    Grid search picks absolute values (e.g., 0.053) that may not transfer from
    dev→test. Percentiles are relative to the score distribution — more robust.

    Returns: (optimal_threshold, f_beta_score, sensitivity, specificity)
    """
    best_t, best_fb = 0.5, 0.0
    # AUDIT-FIX: Finer search (200 points) + lower bound 0.01 (not 0.05).
    # For hemodynamic_collapse (0.5% positive rate), optimal thresholds may
    # be in the 0.01-0.05 range that the old 0.05 start missed.
    for t in np.linspace(0.01, 0.95, 200):
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        fb = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if fb > best_fb:
            best_t, best_fb = t, fb

    # FIX-3: Percentile-based threshold fallback for rare events.
    # At <5% positive rate, absolute thresholds sit at fragile points in the
    # score distribution. Percentile thresholds adapt to distribution shifts.
    pos_rate = float(y_true.mean())
    if pos_rate < 0.05:
        for pct in [90, 92, 95, 97]:
            t_pct = float(np.percentile(y_prob, pct))
            y_pred = (y_prob >= t_pct).astype(int)
            if y_pred.sum() == 0:
                continue
            fb = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            if fb > best_fb:
                best_t, best_fb = t, fb
                logger.info(f"    FIX-3: Percentile-based threshold p{pct}={t_pct:.4f} beat grid search (F{beta}={fb:.4f})")

    y_pred_opt = (y_prob >= best_t).astype(int)
    tp = ((y_pred_opt == 1) & (y_true == 1)).sum()
    fn = ((y_pred_opt == 0) & (y_true == 1)).sum()
    fp = ((y_pred_opt == 1) & (y_true == 0)).sum()
    tn = ((y_pred_opt == 0) & (y_true == 0)).sum()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    logger.success(f"  Optimal threshold: {best_t:.3f} | F{beta}: {best_fb:.4f} | Sens: {sens:.3f} | Spec: {spec:.3f}")
    return best_t, best_fb, sens, spec


def train_meta_stacker(
    oof_preds: dict,     # {"lgbm": array, "xgb": array, "cat": array, "grud": array, "tcn": array}
    y_true:   np.ndarray,
) -> lgb.Booster:
    """Trains a LightGBM meta-learner on out-of-fold (OOF) model predictions.

    Stacked generalization: instead of hand-tuning blend weights, a meta-learner
    discovers when each base model is more accurate — e.g., LightGBM dominates on
    lab-heavy samples, GRU-D/TCN dominate on rapid vital-sign trajectories.

    Input:  OOF predictions from all base models (already on validation rows)
    Output: Trained LightGBM meta-model + calibrated wrapper
    """
    model_names = list(oof_preds.keys())
    X_meta = np.column_stack([oof_preds[name] for name in model_names])

    meta_params = {
        "objective":       "binary",
        "metric":          ["auc", "average_precision"],
        # BUG-R6-4 fix: removed n_estimators (sklearn alias, not used by lgb.train).
        # Training rounds controlled by early_stopping(50) callback.
        "learning_rate":   0.05,
        "num_leaves":      15,    # small tree: meta-learner should not overfit
        "min_child_samples": 10,
        "reg_alpha":       0.1,
        "reg_lambda":      0.5,
        "is_unbalance":    True,  # BUG-R6-1 fix: class_weight is sklearn-only; use is_unbalance for lgb.train()
        "random_state":    SEED,
        "verbose":         -1,
    }

    # 5-fold inner CV on meta-features (prevents meta-overfitting)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y_true))

    for fold, (tr, vl) in enumerate(skf.split(X_meta, y_true)):
        ds_tr = lgb.Dataset(X_meta[tr], label=y_true[tr])
        ds_vl = lgb.Dataset(X_meta[vl], label=y_true[vl], reference=ds_tr)
        m = lgb.train(
            meta_params, ds_tr, num_boost_round=200, valid_sets=[ds_vl],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        meta_oof[vl] = m.predict(X_meta[vl])

    meta_auroc = roc_auc_score(y_true, meta_oof) if y_true.sum() > 0 else 0.0
    logger.success(f"  Meta-stacker OOF AUROC: {meta_auroc:.4f}")

    # Final fit on all OOF data
    ds_full = lgb.Dataset(X_meta, label=y_true)
    final_meta = lgb.train(
        meta_params, ds_full, num_boost_round=200,
        callbacks=[lgb.log_evaluation(0)],
    )
    # AUDIT-FIX: Return meta_oof (true OOF predictions from inner CV) so
    # the isotonic calibrator can be fitted on genuine held-out predictions
    # instead of the retrained model's in-sample predictions.
    return final_meta, model_names, meta_oof


###############################################################################
# ── SECTION 4: OPTUNA ENSEMBLE WEIGHT TUNING (kept as fallback) ─────────────
###############################################################################

def tune_ensemble_weights(
    lgbm_preds: np.ndarray,
    xgb_preds:  np.ndarray,
    cat_preds:  np.ndarray,     # Bug 53 fix: added CatBoost (was missing)
    grud_preds: np.ndarray,
    tcn_preds:  np.ndarray,     # Bug 53 fix: added TCN (was missing)
    y_true:     np.ndarray,
    n_trials:   int = 100,
) -> dict:
    """
    Uses Optuna to find optimal blending weights for the full 5-engine ensemble.
    Maximizes AUROC on the validation set.

    NOTE: This is a DEBUG/FALLBACK function only. The main train_target() path uses
    LightGBM meta-stacker (train_meta_stacker) which is more powerful since it
    learns conditional weights rather than fixed global weights.

    w1..w5 >= 0, w1+w2+w3+w4+w5 = 1
    final = w1*lgbm + w2*xgb + w3*cat + w4*grud + w5*tcn
    """
    def objective(trial):
        w1 = trial.suggest_float("w_lgbm", 0.0, 1.0)
        w2 = trial.suggest_float("w_xgb",  0.0, 1.0 - w1)
        w3 = trial.suggest_float("w_cat",  0.0, 1.0 - w1 - w2)
        w4 = trial.suggest_float("w_grud", 0.0, 1.0 - w1 - w2 - w3)
        w5 = max(0.0, 1.0 - w1 - w2 - w3 - w4)
        if any(w < 0 for w in [w1, w2, w3, w4, w5]):
            return 0.0
        ensemble = w1*lgbm_preds + w2*xgb_preds + w3*cat_preds + w4*grud_preds + w5*tcn_preds
        if y_true.sum() == 0:
            return 0.0
        return roc_auc_score(y_true, ensemble)

    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    w1 = best["w_lgbm"]; w2 = best["w_xgb"]; w3 = best["w_cat"]; w4 = best["w_grud"]
    w5 = max(0.0, 1.0 - w1 - w2 - w3 - w4)
    weights = {"lgbm": round(w1,4), "xgb": round(w2,4), "cat": round(w3,4),
               "grud": round(w4,4), "tcn":  round(w5,4)}
    logger.success(f"  Optimal 5-engine ensemble weights: {weights}")
    return weights


###############################################################################
# ── SECTION 5: MASTER TRAINING FUNCTION (GOD-MODE) ──────────────────────────
###############################################################################

def validate_datasets() -> dict:
    """Pre-flight check: validates all data sources before training starts.

    Prints a summary table of row counts, positive rates, and missing columns.
    Run with: python backend/train_models.py --validate
    """
    results = {}

    checks = [
        ("CinC 2019 (sepsis)",              lambda: load_cinc2019(DATA_DIR),               "target_sepsis"),
        ("eICU (hypotension)",              lambda: load_eicu_hypotension(DATA_DIR),        "target_hypotension"),
        ("VitalDB (hemo. collapse)",        lambda: load_vitaldb(DATA_DIR),                 "target_hemodynamic_collapse"),
        ("Zenodo cardiac",                  lambda: load_zenodo_cardiac_arrest(DATA_DIR),   "target_hemodynamic_collapse"),
        ("CUDB ECG",                        lambda: load_cudb_wfdb(DATA_DIR),               "target_hemodynamic_collapse"),
        ("SDDB ECG",                        lambda: load_sddb_wfdb(DATA_DIR),               "target_hemodynamic_collapse"),
        ("I-CARE (real arrest, PhysioNet)", lambda: load_icare(DATA_DIR),                   "target_hemodynamic_collapse"),
    ]

    print("\n" + "="*70)
    print(f"{'DATASET':<30} {'ROWS':>10} {'POS RATE':>10} {'STATUS':>10}")
    print("="*70)

    for name, loader_fn, target_col in checks:
        try:
            df = loader_fn()
            if df is None:
                print(f"{name:<30} {'N/A':>10} {'N/A':>10} {'⚠ None':>10}")
                results[name] = {"status": "none"}
            else:
                n = len(df)
                pos = df[target_col].mean() if target_col in df.columns else float("nan")
                print(f"{name:<30} {n:>10,} {pos:>10.2%} {'✅ OK':>10}")
                results[name] = {"rows": n, "pos_rate": pos, "status": "ok"}
        except Exception as e:
            print(f"{name:<30} {'ERR':>10} {'ERR':>10} {str(e)[:15]:>10}")
            results[name] = {"status": "error", "error": str(e)}

    print("="*70)
    return results


def train_target(target: str, resume: bool = False, tune: bool = True) -> Optional[dict]:
    """God-Mode: trains the 5-engine ensemble with checkpointing and meta-stacking.

    Engines:
      1. LightGBM (tabular, class_weight balanced)
      2. XGBoost  (tabular, scale_pos_weight)
      3. CatBoost (tabular, native categoricals) — if installed
      4. GRU-D    (sequential, irregular time-series)
      5. TCN      (sequential, dilated causal convolutions)
    + WFDB ECG supplement for hemodynamic_collapse (CUDB + SDDB features)
    + SMOTE-ENN / ADASYN oversampling per target (tabular only)
    + FocalBCE loss for GRU-D/TCN on severe imbalance targets
    + LightGBM meta-stacker on OOF predictions
    + Isotonic calibration
    + F-beta=2 threshold optimization

    Checkpoints: saved per model to models/{target}/checkpoints/
    Resume: if resume=True, already-saved model artifacts are reloaded and skipped.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING: {target.upper()}  [resume={resume}]")
    logger.info(f"{'='*60}\n")

    out_dir  = MODELS_DIR / target
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # BUG-AUDIT-2 FIX: Initialize cat_final to None. CatBoost training code
    # is absent from this function, but line ~2687 references cat_final.
    # Without this init, a NameError is raised (silently caught by try/except
    # in main()), meaning CatBoost is never included in test-set evaluation
    # and the meta-stacker only sees LightGBM + XGBoost.
    cat_final = None

    # ── 1. Load + merge data ──────────────────────────────────────────────────
    if target == "sepsis":
        df = load_cinc2019(DATA_DIR)
        target_col = "target_sepsis"

    elif target == "hypotension":
        df = load_eicu_hypotension(DATA_DIR)
        target_col = "target_hypotension"

    elif target == "hemodynamic_collapse":
        target_col = "target_hemodynamic_collapse"
        df = load_vitaldb(DATA_DIR)

        # FIX-2: Tag VitalDB rows with dataset source before merging.
        # Dataset indicator features let the model learn source-specific patterns
        # (VitalDB=surgical, Zenodo=cardiac, CUDB/SDDB=ECG, I-CARE=post-arrest).
        if df is not None:
            df["_dataset_source"] = "vitaldb"

        # Supplement with real arrest labels: Zenodo, CUDB, SDDB, and I-CARE
        # I-CARE is the most valuable — 1020+ patients with true post-arrest physiology
        extra_sources = {
            "Zenodo": load_zenodo_cardiac_arrest(DATA_DIR),
            "CUDB":   load_cudb_wfdb(DATA_DIR),
            "SDDB":   load_sddb_wfdb(DATA_DIR),
            "I-CARE": load_icare(DATA_DIR),  # PhysioNet 2023 — real arrest events
        }
        _source_key_map = {"Zenodo": "zenodo", "CUDB": "cudb", "SDDB": "sddb", "I-CARE": "icare"}
        for src_name, extra_df in extra_sources.items():
            if extra_df is None:
                continue
            # FIX-2: Tag each extra source before merging
            extra_df["_dataset_source"] = _source_key_map[src_name]
            if df is not None:
                # COLUMN-CHAIN-BUG-FIX: Use UNION concat, NOT intersection.
                # The old code:
                #   shared = list(set(df.columns) & set(extra_df.columns))
                #   df = pd.concat([df[shared], extra_df[shared]])
                # was destructive. When CUDB (only ECG features: ecg_mean, ecg_std...)
                # was merged into df (which had VitalDB vitals: MAP, HR, SpO2...):
                #   intersection = {patient_id, target_hemodynamic_collapse} only.
                # ALL vital sign columns were stripped from the DataFrame. After CUDB
                # merge, df had only 2 columns — feature_engineering() returned all
                # NaNs, and fillna(0) made X an all-zero matrix. The hemodynamic_collapse
                # model would then train on [zeros → random labels] = learns nothing.
                #
                # Fix: concat with union of columns. Rows from sources without a column
                # (e.g. CUDB has no heart_rate) get NaN there, which fillna(0) handles
                # correctly — identical to how missing vitals are treated in training.
                df = pd.concat([df, extra_df], ignore_index=True, sort=False)
                logger.info(f"  +{src_name}: now {len(df):,} total rows")
            else:
                df = extra_df
                logger.warning(f"  VitalDB missing; using {src_name} as primary source")

        # FIX-2: One-hot encode _dataset_source into indicator columns.
        # These features let the model learn domain-specific patterns instead of
        # forcing heterogeneous populations into a single distribution.
        if df is not None and "_dataset_source" in df.columns:
            for src in ["vitaldb", "zenodo", "cudb", "sddb", "icare"]:
                df[f"dataset_source_{src}"] = (df["_dataset_source"] == src).astype(np.float32)
            df.drop(columns=["_dataset_source"], inplace=True)
            logger.info(f"  FIX-2: Added 5 dataset_source_* indicator columns")
    else:
        raise ValueError(f"Unknown target: {target}")

    if df is None:
        logger.error(f"No data available for [{target}]. Skipping.")
        return None

    # ── 2. Feature prep ───────────────────────────────────────────────────────
    # R13-FIX-7: Feature engineering disk caching with automatic invalidation.
    # AUDIT-FIX: Previously cache was never invalidated when features.py changed.
    # Now computes a SHA256 hash of features.py and stores it in the cache filename.
    # Any change to feature engineering code automatically produces a cache miss.
    import hashlib
    _features_py = Path(__file__).parent / "features.py"
    # AUDIT-FIX-2: Include BOTH features.py AND train_models.py in cache key.
    # Data loaders in train_models.py perform feature-relevant operations
    # (resampling, NaT filtering, column intersection) that affect the
    # engineered features. Changes to train_models.py should also invalidate.
    _train_py = Path(__file__)
    _combined_hash = hashlib.sha256(
        _features_py.read_bytes() + _train_py.read_bytes()
    ).hexdigest()[:12]
    feature_cols = get_feature_columns()
    _feat_cache_path = out_dir / f"_features_cache_{target}_{_combined_hash}.parquet"
    _feat_cache_valid = False
    if resume and _feat_cache_path.exists():
        try:
            df = pd.read_parquet(_feat_cache_path)
            logger.info(f"  Feature engineering: loaded from cache ({len(df):,} rows, hash={_combined_hash})")
            _feat_cache_valid = True
        except Exception as e:
            logger.warning(f"  Feature cache load failed ({e}) — recomputing...")
    elif resume:
        logger.info(f"  Feature cache miss (features.py changed or first run, hash={_combined_hash}) — recomputing...")

    if not _feat_cache_valid:
        # Per-patient feature engineering (prevents cross-patient leakage)
        if "patient_id" in df.columns:
            logger.info("  Engineering features per patient (prevents cross-patient leakage)...")
            patient_frames = []
            n_patients = df["patient_id"].nunique()
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
                for i, (pid, pat_df) in enumerate(df.groupby("patient_id", sort=False)):
                    pat_df = pat_df.sort_values("timestamp").reset_index(drop=True) if "timestamp" in pat_df.columns else pat_df.reset_index(drop=True)
                    patient_frames.append(engineer_features(pat_df))
                    if (i + 1) % 5000 == 0 or (i + 1) == n_patients:
                        logger.info(f"    Feature engineering: {i+1:,}/{n_patients:,} patients ({(i+1)/n_patients*100:.0f}%)")
            df = pd.concat(patient_frames, ignore_index=True)
        else:
            # No patient_id (static datasets like Zenodo cardiac): safe to call once
            df = engineer_features(df)
        # Save cache for future resume runs
        try:
            df.to_parquet(_feat_cache_path, index=False)
            logger.info(f"  Feature cache saved: {_feat_cache_path.name} ({_feat_cache_path.stat().st_size / 1e6:.0f} MB)")
        except Exception as e:
            logger.warning(f"  Could not save feature cache ({e}) — will recompute next time")

    avail = [f for f in feature_cols if f in df.columns]

    if target_col not in df.columns:
        logger.error(f"Target col '{target_col}' missing. Check loader.")
        return None

    df_clean = df.dropna(subset=[target_col])
    X = df_clean[avail].fillna(0).values.astype(np.float32)
    y = df_clean[target_col].values.astype(int)

    # Target-specific sequence length for GRU-D/TCN
    seq_len = TARGET_SEQ_LEN.get(target, SEQUENCE_LENGTH)
    logger.info(f"Dataset: {len(X):,} rows | {y.mean():.2%} positive | {len(avail)} features | seq_len={seq_len}h")

    # ── 3. Patient-level hold-out test set (20%) ──────────────────────────────
    groups = df_clean["patient_id"].values if "patient_id" in df_clean.columns else np.arange(len(X))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    dev_idx, test_idx = next(splitter.split(X, y, groups))
    X_dev, X_test = X[dev_idx], X[test_idx]
    y_dev, y_test = y[dev_idx], y[test_idx]
    groups_dev = groups[dev_idx]

    # ── 3b. Optuna hyperparameter tuning (before K-fold) ───────────────────────
    if tune:
        _lgbm_params, _xgb_params = optuna_tune_gbt(
            X_dev, y_dev, groups_dev, avail, target,
            n_trials=50, cache_dir=ckpt_dir,
        )
    else:
        _lgbm_params, _xgb_params = LGBM_PARAMS.copy(), XGB_PARAMS.copy()
        logger.info("  Optuna tuning skipped (--no-tune flag). Using default hyperparameters.")

    # ── 4. 5-Fold StratifiedGroupKFold on development set ───────────────────────
    # NEW-BUG-FIX (C1): Use StratifiedGroupKFold instead of GroupKFold.
    # GroupKFold does NOT guarantee each fold has both classes. With rare events
    # (0.5–5% positive rate) some folds may have 0 positive validation samples,
    # causing undefined AUC. StratifiedGroupKFold maintains class balance per
    # fold while still respecting patient-level group boundaries.
    kfold     = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds: dict = {"lgbm": np.zeros(len(y_dev)), "xgb": np.zeros(len(y_dev))}
    fold_aurocs      = []
    fold_best_iters  = []  # BUG-D: track best_iteration for final model
    xgb_fold_best_iters = []  # AUDIT-FIX: track XGBoost best_iteration per fold
    saved_vl_indices = []  # AUDIT-FIX: save fold indices to avoid re-randomized AUPRC
    _kfold_start = time.time()  # LOG-FIX: K-fold timing
    _trained_fold_times = []     # BUG-R8-FIX: track only actually-trained fold durations for accurate ETA


    for fold, (tr_idx, vl_idx) in enumerate(kfold.split(X_dev, y_dev, groups_dev)):
        saved_vl_indices.append(vl_idx)  # Save for AUPRC summary below
        fold_ckpt = ckpt_dir / f"fold_{fold}_lgbm.pkl"
        if resume and fold_ckpt.exists():
            logger.info(f"  [Fold {fold+1}] Resuming from checkpoint — loading saved models")
            lgb_m = joblib.load(fold_ckpt)
            xgb_m = joblib.load(ckpt_dir / f"fold_{fold}_xgb.pkl")
            oof_preds["lgbm"][vl_idx] = lgb_m.predict(X_dev[vl_idx])
            oof_preds["xgb"][vl_idx]  = xgb_m.predict_proba(X_dev[vl_idx])[:, 1]
            # R12-BUG-2 FIX: Recover fold_best_iters from resumed models.
            fold_best_iters.append(getattr(lgb_m, 'best_iteration', None) or LGBM_NUM_BOOST_ROUND)
            # BUG-R8-FIX: Also recover XGBoost best_iteration on resume.
            # Previously xgb_fold_best_iters was empty on full resume, causing
            # final XGBoost to train 1000 rounds instead of fold-averaged optimal.
            xgb_fold_best_iters.append(
                xgb_m.best_iteration if hasattr(xgb_m, 'best_iteration') and xgb_m.best_iteration else 1000
            )
            resumed_fold_auroc = roc_auc_score(y_dev[vl_idx], oof_preds["lgbm"][vl_idx]) if y_dev[vl_idx].sum() > 0 else 0.0
            fold_aurocs.append(resumed_fold_auroc)
            logger.info(f"  [Fold {fold+1}] LightGBM AUROC (resumed): {resumed_fold_auroc:.4f}")
            continue

        logger.info(f"\n  ── Fold {fold+1}/5 ──────────────────────────────────")
        X_tr_f, X_vl_f = X_dev[tr_idx], X_dev[vl_idx]
        _fold_start = time.time()  # LOG-FIX: per-fold timing
        y_tr_f, y_vl_f = y_dev[tr_idx], y_dev[vl_idx]

        # SMOTE-ENN / ADASYN oversampling on training fold only
        log_ram(f"pre-fold {fold+1}")
        X_tr_s, y_tr_s = apply_sampling(X_tr_f, y_tr_f, target)

        pos_weight = (y_tr_s == 0).sum() / max((y_tr_s == 1).sum(), 1)

        # LightGBM
        lgb_tr = lgb.Dataset(X_tr_s, label=y_tr_s, feature_name=avail)
        lgb_vl = lgb.Dataset(X_vl_f, label=y_vl_f, reference=lgb_tr)
        lgb_m = lgb.train(
            _lgbm_params, lgb_tr, num_boost_round=LGBM_NUM_BOOST_ROUND, valid_sets=[lgb_vl],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof_preds["lgbm"][vl_idx] = lgb_m.predict(X_vl_f)
        lgb_fold_auroc = roc_auc_score(y_vl_f, oof_preds["lgbm"][vl_idx]) if y_vl_f.sum() > 0 else 0
        lgb_fold_auprc = average_precision_score(y_vl_f, oof_preds["lgbm"][vl_idx]) if y_vl_f.sum() > 0 else 0
        logger.success(f"  Fold {fold+1} LightGBM — AUROC: {lgb_fold_auroc:.4f} | AUPRC: {lgb_fold_auprc:.4f}")
        joblib.dump(lgb_m, ckpt_dir / f"fold_{fold}_lgbm.pkl")
        fold_best_iters.append(lgb_m.best_iteration or LGBM_NUM_BOOST_ROUND)

        # XGBoost
        # AUDIT-FIX: Added early_stopping_rounds=50 so XGBoost doesn't blindly
        # train all 1000 rounds. XGBoost 2.1.3 defaults to None (no early stop).
        xgb_p = _xgb_params.copy(); xgb_p["scale_pos_weight"] = pos_weight
        xgb_p["early_stopping_rounds"] = 50
        xgb_m = xgb.XGBClassifier(**xgb_p)
        xgb_m.fit(X_tr_s, y_tr_s, eval_set=[(X_vl_f, y_vl_f)], verbose=False)
        oof_preds["xgb"][vl_idx] = xgb_m.predict_proba(X_vl_f)[:, 1]
        xgb_fold_auroc = roc_auc_score(y_vl_f, oof_preds["xgb"][vl_idx]) if y_vl_f.sum() > 0 else 0
        xgb_fold_auprc = average_precision_score(y_vl_f, oof_preds["xgb"][vl_idx]) if y_vl_f.sum() > 0 else 0
        logger.success(f"  Fold {fold+1} XGBoost  — AUROC: {xgb_fold_auroc:.4f} | AUPRC: {xgb_fold_auprc:.4f}")
        joblib.dump(xgb_m, ckpt_dir / f"fold_{fold}_xgb.pkl")
        xgb_fold_best_iters.append(xgb_m.best_iteration if hasattr(xgb_m, 'best_iteration') and xgb_m.best_iteration else 1000)

        # LOG-FIX: per-fold timing + ETA (only counts actually-trained folds)
        _fold_elapsed = time.time() - _fold_start
        # BUG-R8-FIX: Track only actually-trained fold durations, not resumed ones.
        # Resumed folds take ~50ms; counting them in the average makes ETA
        # wildly inaccurate (underestimates by 3-4x on partial resume).
        _trained_fold_times.append(_fold_elapsed)
        _avg_fold_time = np.mean(_trained_fold_times)
        _remaining_folds = 5 - (fold + 1)
        _kfold_eta = _avg_fold_time * _remaining_folds
        logger.info(f"  Fold {fold+1} completed in {_fmt_duration(_fold_elapsed)} | ETA remaining: {_fmt_duration(_kfold_eta)}")
        log_ram(f"after fold {fold+1}")

        fold_auroc = lgb_fold_auroc
        fold_aurocs.append(fold_auroc)

    # AUDIT-FIX: Use saved_vl_indices (from the training loop above) instead of
    # re-calling kfold.split() which re-randomizes fold assignments and computes
    # AUPRC on WRONG validation indices (the OOF predictions are from different folds).
    fold_auprc_mean = np.mean([average_precision_score(y_dev[vi], oof_preds["lgbm"][vi])
                               for vi in saved_vl_indices
                               if y_dev[vi].sum() > 0])
    logger.success(f"  5-Fold CV LightGBM — AUROC: {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f} | AUPRC: {fold_auprc_mean:.4f}")

    # XGB 5-fold CV summary
    xgb_fold_auroc_mean = np.mean([roc_auc_score(y_dev[vi], oof_preds["xgb"][vi])
                                   for vi in saved_vl_indices if y_dev[vi].sum() > 0])
    xgb_fold_auprc_mean = np.mean([average_precision_score(y_dev[vi], oof_preds["xgb"][vi])
                                   for vi in saved_vl_indices if y_dev[vi].sum() > 0])
    logger.success(f"  5-Fold CV XGBoost   — AUROC: {xgb_fold_auroc_mean:.4f} | AUPRC: {xgb_fold_auprc_mean:.4f}")

    # ── 5. Train final tabular models on all dev data ──────────────────────────
    logger.info("\n  Training final tabular models on full development set...")

    lgbm_ckpt = out_dir / "lgbm_model.pkl"
    xgb_ckpt  = out_dir / "xgb_model.pkl"
    _all_final_cached = resume and lgbm_ckpt.exists() and xgb_ckpt.exists()

    # R12-WASTE-1 FIX: Skip SMOTE when all final models are cached.
    # apply_sampling() on 1.2M rows wastes ~30s + ~2 GB on resume.
    if _all_final_cached:
        X_dev_s, y_dev_s, pos_w_final = None, None, None
    else:
        X_dev_s, y_dev_s = apply_sampling(X_dev, y_dev, target)
        pos_w_final = (y_dev_s == 0).sum() / max((y_dev_s == 1).sum(), 1)

    if resume and lgbm_ckpt.exists():
        logger.info("  Resuming: loading saved final LightGBM")
        lgb_final = joblib.load(lgbm_ckpt)
    else:
        lgb_tr_final = lgb.Dataset(X_dev_s, label=y_dev_s, feature_name=avail)
        _final_rounds = int(np.mean(fold_best_iters) * 1.1) if fold_best_iters else LGBM_NUM_BOOST_ROUND
        logger.info(f"  Final LightGBM: {_final_rounds} rounds (1.1× avg fold best_iteration)")
        lgb_final = lgb.train(_lgbm_params, lgb_tr_final, num_boost_round=_final_rounds, callbacks=[lgb.log_evaluation(0)])
        joblib.dump(lgb_final, lgbm_ckpt)

    if resume and xgb_ckpt.exists():
        xgb_final = joblib.load(xgb_ckpt)
    else:
        # AUDIT-FIX: Use fold-averaged best_iteration for final XGBoost (same as LightGBM).
        # Previously always trained all 1000 rounds without early stopping.
        _xgb_final_iters = int(np.mean(xgb_fold_best_iters) * 1.1) if xgb_fold_best_iters else 1000
        logger.info(f"  Final XGBoost: {_xgb_final_iters} estimators (1.1× avg fold best_iteration)")
        xgb_p2 = _xgb_params.copy()
        xgb_p2["scale_pos_weight"] = pos_w_final
        xgb_p2["n_estimators"] = _xgb_final_iters
        xgb_final = xgb.XGBClassifier(**xgb_p2)
        xgb_final.fit(X_dev_s, y_dev_s, verbose=False)
        joblib.dump(xgb_final, xgb_ckpt)

    # R12-WASTE-2 FIX: Load cached SHAP explainer on resume instead of recomputing.
    _shap_path = out_dir / "shap_explainer.pkl"
    if resume and _shap_path.exists():
        explainer = joblib.load(_shap_path)
    else:
        explainer = shap.TreeExplainer(lgb_final)
        joblib.dump(explainer, _shap_path)

    # ── MEMORY FIX: Spill tabular data to disk during sequential training ────
    # X_dev (~1.55 GB), y_dev, oof_preds, models, and explainer are NOT needed
    # during the 13+ hours of GRU-D + TCN training. Spill to disk, reload later.
    _cache_dir = out_dir
    joblib.dump(X_dev, _cache_dir / "_x_dev_cache.pkl")
    joblib.dump(y_dev, _cache_dir / "_y_dev_cache.pkl")
    joblib.dump(oof_preds, _cache_dir / "_oof_cache.pkl")
    joblib.dump(X_test, _cache_dir / "_x_test_cache.pkl")
    joblib.dump(y_test, _cache_dir / "_y_test_cache.pkl")
    joblib.dump(groups_dev, _cache_dir / "_groups_dev_cache.pkl")
    del X_dev, y_dev, oof_preds, X_test, y_test, groups_dev
    del explainer  # already saved to shap_explainer.pkl
    # Keep lgb_final, xgb_final — small (<15 MB total), needed for test eval

    # ── MEMORY FIX R10-3: Extract NEWS2 column BEFORE freeing df_clean ────────
    # df_clean (~2 GB) was previously retained for the entire 20-hour run just for
    # 2 lines of NEWS2 benchmarking at the end. Extract what we need now.
    _news2_test = None
    if "news2_score" in df_clean.columns:
        _news2_test = df_clean.iloc[test_idx]["news2_score"].fillna(0).values

    import gc; gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    log_ram("post-tabular-spill")

    # ── 6. GRU-D and TCN sequential models ───────────────────────────────────
    grud_ckpt_path = ckpt_dir / "grud_best.pt"
    tcn_ckpt_path  = ckpt_dir / "tcn_best.pt"

    # EDGE-CASE FIX: df_for_seq MUST be defined BEFORE the resume check.
    # Previously it was inside the else-branch, so --resume with grud_best.pt
    # but no tcn_best.pt would crash with NameError when TCN tried to use it.
    # Also needed by test-set sequential eval for hemodynamic_collapse.
    df_for_seq = df_clean
    if target == "hemodynamic_collapse" and "timestamp" in df_clean.columns:
        # NAT-FIX: After the union merge, CUDB/SDDB rows have NaT timestamps.
        # Filter to only rows with valid timestamps before resampling.
        df_seq_base = df_clean[df_clean["timestamp"].notna()].copy()
        n_dropped = len(df_clean) - len(df_seq_base)
        if n_dropped > 0:
            logger.info(
                f"  GRU-D/TCN resample: dropped {n_dropped:,} rows with NaT timestamps "
                f"(ECG-only sources: CUDB/SDDB). {len(df_seq_base):,} rows remain."
            )
        num_cols = df_seq_base.select_dtypes(include=[np.number]).columns.tolist()
        _resampled = (
            df_seq_base.set_index("timestamp")
            .groupby("patient_id")[num_cols]
            .resample("1h")
            .mean()
            .reset_index()
        )
        # BUG-A FIX: .mean() averaged the binary target column (0/1 → fractional
        # like 0.33). Clinically: if ANY minute in the hour had a collapse event,
        # the hourly label should be 1. Threshold at >0 implements max() semantics.
        if target_col in _resampled.columns:
            _resampled[target_col] = (_resampled[target_col] > 0).astype(int)
        if "level_1" in _resampled.columns:
            _resampled = _resampled.rename(columns={"level_1": "timestamp"})
        elif "timestamp" not in _resampled.columns:
            _dt_cols = [
                c for c in _resampled.columns
                if c != "patient_id" and pd.api.types.is_datetime64_any_dtype(_resampled[c])
            ]
            if _dt_cols:
                _resampled = _resampled.rename(columns={_dt_cols[0]: "timestamp"})
            else:
                logger.warning(
                    "  GRU-D/TCN resample: could not locate timestamp column "
                    "after hourly resample — sequential models will be skipped."
                )
                _resampled = None
        df_for_seq = _resampled
        if df_for_seq is not None:
            logger.info(f"  GRU-D/TCN: resampled hemodynamic_collapse to hourly ({len(df_for_seq):,} rows)")

    if resume and grud_ckpt_path.exists():
        logger.info("  Resuming: loading saved GRU-D")
        grud_model = GRUD(input_size=len(avail)).to(DEVICE)
        grud_model.load_state_dict(torch.load(grud_ckpt_path, map_location=DEVICE, weights_only=True))
        # R13-FIX-6: Log GRU-D best AUROC from saved metadata on resume
        _grud_auroc_path = ckpt_dir / "grud_best_auroc.json"
        if _grud_auroc_path.exists():
            with open(_grud_auroc_path) as f:
                _grud_meta = json.load(f)
            logger.success(f"  GRU-D best Val AUROC (from training): {_grud_meta['best_auroc']:.4f}")
        else:
            logger.info("  GRU-D AUROC: (not recorded — trained before R13 fix)")
    else:
        log_ram("pre-GRU-D")
        logger.info("  Training GRU-D...")
        if df_for_seq is None:
            logger.warning("  GRU-D skipped — df_for_seq is None (resample timestamp issue).")
            grud_model = None
        else:
            grud_model = train_grud(df_for_seq, avail, target_col, seq_len=seq_len, ckpt_dir=ckpt_dir)
        if grud_model:
            torch.save(grud_model.state_dict(), grud_ckpt_path)
            torch.save(grud_model.state_dict(), out_dir / "grud_model.pt")

    # ── MEMORY FIX R10-1: Move grud_model to CPU during TCN training ──────────
    # grud_model was staying on GPU for 6+ hours of TCN training. It's only needed
    # again at test eval — move to CPU now, move back to GPU for test eval later.
    if grud_model is not None:
        grud_model = grud_model.cpu()

    # ── MEMORY FIX R10-2: Spill df_for_seq to disk during TCN training ────────
    # df_for_seq (~1 GB) coexists with TCN's internal copies during training.
    # Spill to disk, reload for TCN and test eval.
    if df_for_seq is not None:
        joblib.dump(df_for_seq, _cache_dir / "_df_for_seq_cache.pkl")
        _df_for_seq_cols = list(df_for_seq.columns)  # save column list for reference

    # ── MEMORY FIX R10-3 continued: Free df_clean now ─────────────────────────
    # NEWS2 column already extracted above. df_for_seq already spilled.
    # For non-hemodynamic targets, df_for_seq IS df_clean (same object), so
    # we can't del df_clean without also losing df_for_seq — but it's spilled.
    del df_clean
    del df_for_seq
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    log_ram("post-GRU-D-cleanup")

    if resume and tcn_ckpt_path.exists():
        logger.info("  Resuming: loading saved TCN")
        tcn_model = TCN(input_size=len(avail)).to(DEVICE)
        tcn_model.load_state_dict(torch.load(tcn_ckpt_path, map_location=DEVICE, weights_only=True))
    else:
        logger.info("  Training TCN...")
        # Reload df_for_seq from spill cache for TCN training
        _df_for_seq_tcn = joblib.load(_cache_dir / "_df_for_seq_cache.pkl") if (_cache_dir / "_df_for_seq_cache.pkl").exists() else None
        if _df_for_seq_tcn is None:
            logger.warning("  TCN skipped — df_for_seq is None (resample timestamp issue).")
            tcn_model = None
        else:
            tcn_model = train_tcn(_df_for_seq_tcn, avail, target_col, seq_len=seq_len, ckpt_dir=ckpt_dir)
        del _df_for_seq_tcn
        gc.collect()
        if tcn_model:
            torch.save(tcn_model.state_dict(), tcn_ckpt_path)
            torch.save(tcn_model.state_dict(), out_dir / "tcn_model.pt")

    # ── MEMORY FIX: Cleanup after TCN, reload spilled data ────────────────────
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    X_dev = joblib.load(_cache_dir / "_x_dev_cache.pkl")
    y_dev = joblib.load(_cache_dir / "_y_dev_cache.pkl")
    oof_preds = joblib.load(_cache_dir / "_oof_cache.pkl")
    X_test = joblib.load(_cache_dir / "_x_test_cache.pkl")
    y_test = joblib.load(_cache_dir / "_y_test_cache.pkl")
    groups_dev = joblib.load(_cache_dir / "_groups_dev_cache.pkl")
    log_ram("post-reload-tabular")

    # ── 7. Get test-set predictions from all engines ──────────────────────────
    test_preds: dict = {}
    test_preds["lgbm"] = lgb_final.predict(X_test)
    test_preds["xgb"]  = xgb_final.predict_proba(X_test)[:, 1]
    if cat_final is not None:
        test_preds["cat"] = cat_final.predict_proba(X_test)[:, 1]

    # Sequential models need sequences built from test patients.
    # R10 FIX: df_clean and df_for_seq were freed to save ~3 GB during sequential
    # training. Reload df_for_seq from spill cache for test eval.
    _df_for_seq_eval = None
    _df_for_seq_cache_path = _cache_dir / "_df_for_seq_cache.pkl"
    if _df_for_seq_cache_path.exists():
        _df_for_seq_eval = joblib.load(_df_for_seq_cache_path)
        log_ram("post-reload-df_for_seq for test eval")

    # R10-1 FIX: Move grud_model back to GPU for test eval
    if grud_model is not None:
        grud_model = grud_model.to(DEVICE)

    _seq_scaler_path = out_dir / "grud_scaler.pkl"
    seq_y_test = {}   # per model: ground-truth labels aligned to sequence indices
    for seq_name, seq_model in [("grud", grud_model), ("tcn", tcn_model)]:
        if seq_model is None:
            continue

        # Select test patients from the correct DataFrame (resolution must match training)
        _test_pids = set(groups[test_idx]) if _df_for_seq_eval is not None and "patient_id" in _df_for_seq_eval.columns else None
        if (target == "hemodynamic_collapse"
                and _df_for_seq_eval is not None
                and "patient_id" in _df_for_seq_eval.columns
                and _test_pids is not None):
            # Use hourly-resampled data — same resolution GRU-D/TCN were trained on
            test_df = _df_for_seq_eval[_df_for_seq_eval["patient_id"].isin(_test_pids)].reset_index(drop=True)
        elif _df_for_seq_eval is not None:
            # sepsis / hypotension: df_for_seq is at the correct resolution
            test_df = _df_for_seq_eval.iloc[test_idx].reset_index(drop=True) if len(_df_for_seq_eval) > max(test_idx) else _df_for_seq_eval[_df_for_seq_eval["patient_id"].isin(set(groups[test_idx]))].reset_index(drop=True)
        else:
            logger.warning(f"  {seq_name.upper()} test eval: df_for_seq not available — skipping")
            continue

        # Apply z-scaler (models trained on scaled data; eval must use same scale)
        if _seq_scaler_path.exists():
            try:
                _seq_scaler   = joblib.load(_seq_scaler_path)
                _feat_num     = [f for f in avail if f in test_df.columns]
                test_df       = test_df.copy()
                _was_missing  = test_df[_feat_num].isna()
                # BUG-R11-1 fix: use TRAINING medians (grud_medians.pkl), not test medians.
                # Training saves tr_medians to grud_medians.pkl at line 1238.
                # Using test medians introduces test→train data leakage.
                _medians_path = out_dir / "grud_medians.pkl"
                if _medians_path.exists():
                    _tr_medians_dict = joblib.load(_medians_path)
                    _medians = pd.Series(_tr_medians_dict).reindex(_feat_num)
                    logger.info(f"  {seq_name.upper()} test eval: using training medians from grud_medians.pkl")
                else:
                    _medians = test_df[_feat_num].median()  # fallback if file missing
                    logger.warning(f"  {seq_name.upper()} test eval: grud_medians.pkl missing, using test medians (fallback)")
                test_df[_feat_num] = _seq_scaler.transform(test_df[_feat_num].fillna(_medians))
                # Re-introduce NaNs at originally-missing positions so mask is correct
                test_df[_feat_num] = test_df[_feat_num].where(~_was_missing, other=np.nan)
                logger.info(f"  {seq_name.upper()} test eval: z-scaler applied, {len(test_df):,} rows")
            except Exception as e:
                logger.warning(f"  {seq_name.upper()} test eval: scaler load failed ({e}) — using unscaled data")
        else:
            logger.warning(f"  {seq_name.upper()} test eval: grud_scaler.pkl not found — unscaled data (AUROC estimate only)")

        logger.info(f"  {seq_name.upper()} test eval: building sequences for {test_df['patient_id'].nunique():,} patients ({len(test_df):,} rows)...")
        log_ram(f"pre-test-sequences {seq_name}")
        X_seq, M_seq, D_seq, y_seq = prepare_sequences(test_df, avail, target_col, seq_len=seq_len)
        if X_seq is not None:
            # Batched forward pass (consistent with training validation)
            seq_model.eval()
            _batch_preds = []
            with torch.no_grad():
                _bs = 64
                for i in range(0, len(X_seq), _bs):
                    bp = seq_model(
                        X_seq[i:i+_bs].to(DEVICE),
                        M_seq[i:i+_bs].to(DEVICE),
                        D_seq[i:i+_bs].to(DEVICE)
                    )
                    _batch_preds.append(bp.cpu().numpy())
            preds_seq = np.concatenate(_batch_preds)
            del X_seq, M_seq, D_seq  # free test tensors
            test_preds[seq_name] = preds_seq
            seq_y_test[seq_name] = y_seq.numpy()  # ground-truth aligned to sequence positions
        else:
            test_preds[seq_name] = test_preds["lgbm"].copy()  # fallback
            seq_y_test[seq_name] = y_test

    # R15-FIX-1: Align predictions for final blending, but DO NOT truncate tabular
    # predictions to sequential length when the sequential set is tiny.
    # Previously, min_n = min(len(y_test), len(grud_preds), len(tcn_preds)) would
    # truncate 240K tabular predictions to ~766 sequential sequences — with a
    # 1.24% positive rate, 766 rows often had ZERO positive labels, making AUROC
    # undefined and crashing the NEWS2 baseline comparison.
    #
    # Fix: Use tabular-length predictions for the final blend. Sequential preds
    # are blended only for the rows they cover (the tail of the array, since
    # prepare_sequences produces rows from index [seq_len, N]).
    tabular_keys = [k for k in ["lgbm", "xgb", "cat"] if k in test_preds]
    seq_keys_for_blend = [k for k in ["grud", "tcn"] if k in test_preds]

    # Tabular predictions are all the same length as y_test
    tab_n = len(y_test)
    y_test_aligned = y_test[:tab_n]

    # ── 8. Meta-stacker ───────────────────────────────────────────────────────
    logger.info("  Training LightGBM meta-stacker...")
    # Build OOF meta-features (only tabular OOF available for meta-training)
    oof_keys = [k for k in ["lgbm", "xgb", "cat"] if k in oof_preds]
    oof_meta_X = np.column_stack([oof_preds[k] for k in oof_keys])

    # AUDIT-FIX: train_meta_stacker now returns meta_oof (true OOF predictions
    # from inner CV) for isotonic calibration.
    meta_model, meta_model_names, meta_oof = train_meta_stacker(
        {k: oof_preds[k] for k in oof_keys}, y_dev
    )
    joblib.dump(meta_model, out_dir / "meta_stacker.pkl")
    with open(out_dir / "meta_model_names.json", "w") as f:
        json.dump(oof_keys, f)

    # Meta predictions on test set (tabular length)
    test_meta_X = np.column_stack([test_preds[k][:tab_n] for k in oof_keys])
    meta_preds  = meta_model.predict(test_meta_X)

    # ── 9a. Isotonic calibration ──────────────────────────────────────────────
    # AUDIT-FIX: Previously used `meta_model.predict(oof_meta_X)` — in-sample
    # predictions from the retrained-on-all-dev model. Now uses `meta_oof`
    # (true OOF from 5-fold inner CV inside train_meta_stacker).
    try:
        from sklearn.isotonic import IsotonicRegression
        iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso_reg.fit(meta_oof, y_dev)
        meta_preds = np.clip(iso_reg.predict(meta_preds), 0.0, 1.0)
        joblib.dump(iso_reg, out_dir / "isotonic_calibrator.pkl")
        logger.info("  Isotonic calibrator fitted on TRUE OOF meta-preds (from inner CV) and applied to test meta_preds.")
    except Exception as e:
        logger.warning(f"  Isotonic calibration failed ({e}) — using uncalibrated predictions")
        iso_reg = None

    # AUDIT-FIX: Replace hardcoded 70/30 blend with adaptive agreement-based blend
    # matching api.py _blend_sequential() exactly. The isotonic calibrator is fitted
    # on these predictions — if training uses a different blend than inference, the
    # calibrator maps the wrong distribution and probabilities become miscalibrated.
    final_preds = meta_preds.copy()  # start with tabular-length meta predictions
    if seq_keys_for_blend:
        seq_lens = [len(test_preds[k]) for k in seq_keys_for_blend]
        min_seq_n = min(seq_lens)
        if min_seq_n > 0:
            seq_avg = np.mean(
                [test_preds[k][:min_seq_n] for k in seq_keys_for_blend], axis=0
            )
            blend_start = max(0, tab_n - min_seq_n)
            tab_slice   = meta_preds[blend_start:blend_start + min_seq_n]
            # Adaptive blend: agreement-based weight matching api.py inference path
            abs_diff    = np.abs(tab_slice - seq_avg)
            seq_w       = np.where(abs_diff <= 0.05, 0.50,
                          np.where(abs_diff <= 0.15, 0.35,
                          np.where(abs_diff <= 0.25, 0.25, 0.15)))
            final_preds[blend_start:blend_start + min_seq_n] = (
                (1 - seq_w) * tab_slice + seq_w * seq_avg
            )
            logger.info(
                f"  Sequential blend: {min_seq_n:,} rows blended (adaptive agreement-based weight) "
                f"out of {tab_n:,} total test rows"
            )

    # ── 9b. Calibration + threshold optimization ──────────────────────────────
    final_auroc = roc_auc_score(y_test_aligned, final_preds) if y_test_aligned.sum() > 0 else 0
    final_auprc = average_precision_score(y_test_aligned, final_preds) if y_test_aligned.sum() > 0 else 0

    # AUDIT-FIX: Optimize threshold on DEV OOF predictions (not test set).
    # Previously optimized on y_test_aligned — textbook data leakage because the
    # threshold was selected to maximize F2 on the data it's measured on.
    # Now: find threshold on dev OOF, then REPORT test-set metrics at that threshold.
    # BUG-R6-FIX: Use meta_oof (true inner-CV OOF predictions from
    # train_meta_stacker) instead of meta_model.predict(oof_meta_X) which is
    # in-sample — the retrained meta_model saw oof_meta_X during its final fit.
    # Using in-sample predictions makes the threshold slightly optimistic.
    oof_meta_preds_for_threshold = meta_oof.copy()
    if iso_reg is not None:
        oof_meta_preds_for_threshold = np.clip(iso_reg.predict(oof_meta_preds_for_threshold), 0.0, 1.0)
    best_threshold, best_fbeta, _, _ = optimize_threshold(
        y_dev, oof_meta_preds_for_threshold, beta=2.0
    )
    logger.info(f"  Threshold optimized on DEV OOF predictions (not test set).")

    # Report test-set metrics at the dev-optimized threshold (no re-tuning)
    y_pred_test = (final_preds >= best_threshold).astype(int)
    _tp = ((y_pred_test == 1) & (y_test_aligned == 1)).sum()
    _fn = ((y_pred_test == 0) & (y_test_aligned == 1)).sum()
    _fp = ((y_pred_test == 1) & (y_test_aligned == 0)).sum()
    _tn = ((y_pred_test == 0) & (y_test_aligned == 0)).sum()
    sensitivity = _tp / max(_tp + _fn, 1)
    specificity = _tn / max(_tn + _fp, 1)
    test_fbeta = fbeta_score(y_test_aligned, y_pred_test, beta=2.0, zero_division=0) if y_test_aligned.sum() > 0 else 0.0

    logger.success(f"\n📊 FINAL GOD-MODE RESULTS — {target.upper()}")
    logger.success(f"  AUROC: {final_auroc:.4f} | AUPRC: {final_auprc:.4f}")
    logger.success(f"  5-Fold CV: {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")
    logger.success(f"  Threshold (dev-optimized): {best_threshold:.3f} | F2 (dev): {best_fbeta:.4f} | F2 (test): {test_fbeta:.4f}")
    logger.success(f"  Test @ threshold: Sens: {sensitivity:.3f} | Spec: {specificity:.3f}")

    # Benchmark vs NEWS2 if available (using pre-extracted column from R10-3 fix)
    # R15-FIX-2: Guard against all-zero y_test_aligned (crashes roc_auc_score).
    if _news2_test is not None:
        n2 = _news2_test[:tab_n]
        if n2.max() > 0 and y_test_aligned.sum() > 0:
            n2_auroc = roc_auc_score(y_test_aligned, n2 / n2.max())
            logger.info(f"  NEWS2 Baseline AUROC: {n2_auroc:.4f} | Chronos improvement: {final_auroc - n2_auroc:+.4f}")
        elif y_test_aligned.sum() == 0:
            logger.warning("  NEWS2 baseline skipped — no positive labels in test set")
    del _news2_test

    # ── 10. Save all artifacts ────────────────────────────────────────────────
    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(avail, f)

    # ── Sequential enabled flag ──────────────────────────────────────────────
    # Evaluate GRU-D and TCN on test set to decide if their blend is helpful.
    # If either sequential model AUROC < 0.74 for hypotension, disable blending
    # at inference (a near-random model at 15% weight degrades the ensemble).
    # NEW-BUG-FIX (C11 continued): Use correctly aligned y labels for sequential AUROC.
    # Previously used y_test_aligned (tabular rows) for sequential predictions
    # (which are indexed differently); now uses seq_y_test[name] populated above.
    seq_aurocs = {}
    for seq_name in ["grud", "tcn"]:
        if seq_name not in test_preds:
            continue
        y_for_seq = seq_y_test.get(seq_name, y_test_aligned)
        n_seq = min(len(y_for_seq), len(test_preds[seq_name]))
        if n_seq > 0 and y_for_seq[:n_seq].sum() > 0:
            try:
                seq_aurocs[seq_name] = float(roc_auc_score(y_for_seq[:n_seq], test_preds[seq_name][:n_seq]))
            except Exception:
                seq_aurocs[seq_name] = 0.0
    sequential_enabled = True
    # AUDIT-FIX: Apply sequential gate to ALL targets, not just hypotension.
    # A near-random sequential model at even 15% weight degrades the ensemble.
    # Threshold 0.74 = minimum acceptable AUROC for sequential contribution.
    if seq_aurocs:
        min_seq_auc = min(seq_aurocs.values())
        if min_seq_auc < 0.74:
            sequential_enabled = False
            logger.warning(
                f"  ⚠️  Sequential models underperformed for {target} "
                f"(AUROC={min_seq_auc:.4f} < 0.74 threshold). "
                f"sequential_enabled=False written to metadata — "
                f"api.py will use tabular-only blend for this target."
            )

    metadata = {
        "target":                   target,
        "trained_at":               datetime.now().isoformat(),
        "seed":                     SEED,
        "n_features":               len(avail),
        "n_engines":                len(test_preds),
        "engines":                  list(test_preds.keys()),
        "val_auroc":                round(final_auroc, 4),
        "val_auprc":                round(final_auprc, 4),
        "kfold_auroc_mean":         round(float(np.mean(fold_aurocs)), 4) if fold_aurocs else None,
        "kfold_auroc_std":          round(float(np.std(fold_aurocs)), 4) if fold_aurocs else None,
        "optimal_threshold":        round(best_threshold, 4),
        "f2_at_threshold":          round(best_fbeta, 4),
        "sensitivity":              round(sensitivity, 4),
        "specificity":              round(specificity, 4),
        "prediction_horizon_hours": PREDICTION_HORIZON_HOURS,
        "sequence_length_hours":    seq_len,
        # Sequential model individual AUROCs + enabled flag (used by api.py at inference)
        "sequential_aurocs":        seq_aurocs,
        "sequential_enabled":       sequential_enabled,
        "calibrated":               iso_reg is not None,
        "label_note":               (
            "Proxy labels: MAP<50 (5+min) OR SpO2<85 (3+min) in VitalDB surgical cases, "
            "supplemented with real arrest labels from I-CARE (PhysioNet 2023), "
            "Zenodo ETU dataset, CUDB, and SDDB ECG databases."
        ) if target == "hemodynamic_collapse" else None,
    }
    with open(out_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.success(f"✅ All [{target}] artifacts saved to {out_dir}")

    # LOG-FIX: Comparison vs previous training run
    _prev_meta_path = out_dir / "model_metadata_prev.json"
    _curr_meta_path = out_dir / "model_metadata.json"
    try:
        # We saved the new metadata above; check if a prev backup exists
        if _prev_meta_path.exists():
            with open(_prev_meta_path) as f:
                prev = json.load(f)
            logger.info(f"\n  📊 COMPARISON vs PREVIOUS RUN ({prev.get('trained_at', 'unknown')[:19]})")
            logger.info(f"  {'Metric':<22} {'Previous':>10} {'Current':>10} {'Change':>10}")
            logger.info(f"  {'='*54}")
            for key in ["val_auroc", "val_auprc", "kfold_auroc_mean", "optimal_threshold",
                        "f2_at_threshold", "sensitivity", "specificity"]:
                p_val = prev.get(key)
                c_val = metadata.get(key)
                if p_val is not None and c_val is not None:
                    diff = c_val - p_val
                    arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
                    logger.info(f"  {key:<22} {p_val:>10.4f} {c_val:>10.4f} {arrow}{abs(diff):>8.4f}")
            logger.info(f"  {'='*54}")
    except Exception:
        pass  # No previous run to compare — skip silently

    # Back up current metadata for next run's comparison
    try:
        import shutil
        shutil.copy2(_curr_meta_path, _prev_meta_path)
    except Exception:
        pass

    # Fix 4: Cleanup temporary spill-to-disk cache files
    for tmp_name in ["_x_dev_cache.pkl", "_y_dev_cache.pkl", "_oof_cache.pkl",
                     "_x_test_cache.pkl", "_y_test_cache.pkl", "_groups_dev_cache.pkl",
                     "_df_for_seq_cache.pkl"]:
        (_cache_dir / tmp_name).unlink(missing_ok=True)

    return metadata


###############################################################################
# ── MAIN ─────────────────────────────────────────────────────────────────────
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Project Chronos — God-Mode Ensemble Trainer")
    parser.add_argument("--target", choices=["sepsis", "hypotension", "hemodynamic_collapse"],
                        help="Train a specific target")
    parser.add_argument("--all",      action="store_true", help="Train all three targets")
    parser.add_argument("--resume",   action="store_true",
                        help="Resume from last checkpoint (skips completed folds/models)")
    parser.add_argument("--no-tune",  action="store_true",
                        help="Skip Optuna hyperparameter tuning (use default params)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate all data sources and exit (no training)")
    args = parser.parse_args()

    # --validate: check datasets before wasting 20+ hours
    if args.validate:
        validate_datasets()
        return

    if not (args.all or args.target):
        parser.print_help()
        sys.exit(1)

    targets = ["sepsis", "hypotension", "hemodynamic_collapse"] if args.all else [args.target]

    all_metadata = {}
    for t in targets:
        try:
            meta = train_target(t, resume=args.resume, tune=not args.no_tune)
            if meta:
                all_metadata[t] = meta
        except Exception as e:
            logger.error(f"\n{'='*60}")
            logger.error(f"FATAL ERROR training [{t}]: {type(e).__name__}: {e}")
            logger.error(f"{'='*60}")
            import traceback
            logger.error(traceback.format_exc())
            logger.error(f"Continuing to next target (if any)...\n")
        # ── MEMORY FIX R10-4: Force GC + MPS cache flush between targets ──────
        import gc; gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    # Summary table
    print("\n" + "="*75)
    print(f"{'TARGET':<20} {'AUROC':>8} {'AUPRC':>8} {'5-FOLD AUROC':>14} {'THRESH':>8} {'SENS':>6}")
    print("="*75)
    for t, m in all_metadata.items():
        kf_str = f"{m['kfold_auroc_mean']:.4f}±{m['kfold_auroc_std']:.4f}" if m.get("kfold_auroc_mean") else "N/A"
        print(f"{t:<20} {m['val_auroc']:>8.4f} {m['val_auprc']:>8.4f} {kf_str:>14} {m['optimal_threshold']:>8.3f} {m['sensitivity']:>6.3f}")
    print("="*75)


if __name__ == "__main__":
    main()

