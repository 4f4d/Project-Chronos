#!/bin/bash
# Project Chronos — Complete Command Reference
# Last updated: 2026-02-26
# All commands run from: /Users/falcon/.gemini/antigravity/scratch/health-tech/backend/
# Run: source .venv/bin/activate  FIRST for every new terminal session

# ══════════════════════════════════════════════════════════════════════════════
# 0. SETUP (run once)
# ══════════════════════════════════════════════════════════════════════════════

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install all Python dependencies
pip install -r requirements.txt

# Install PyTorch (Apple Silicon — standard pip install INCLUDES MPS support on macOS)
# BUG-CLAUDE-7-2 fix: The old command used --index-url https://download.pytorch.org/whl/cpu
# which installs a CPU-ONLY build — MPS backend is NOT included in the cpu wheel!
# This means the MPS accelerator is silently unavailable, and training falls back to slow CPU.
# Standard pip install from PyPI auto-selects the right macOS wheel WITH MPS support.
pip install torch

# Install system tools
brew install wget jq


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA DOWNLOADS  (run from backend/ directory)
# ══════════════════════════════════════════════════════════════════════════════

# ── VitalDB — 6388 surgical cases (already running in bg, ~4h total):
python scripts/download_datasets.py --vitaldb --vitaldb-cases 6388
# Check how many have downloaded so far:
ls data/vitaldb/ | wc -l

# ── CinC 2019 sepsis dataset (40k patients, ~2 GB):
python scripts/download_datasets.py --cinc

# ── eICU demo (1.6M vital rows — requires manual DUA acceptance):
# → Go to: https://physionet.org/content/eicu-crd-demo/2.0.1/
# → Accept DUA, download all CSVs → extract to data/eicu_demo/

# ── Zenodo cardiac arrest CSV (112 real arrest patients):
mkdir -p data/zenodo_cardiac
curl -L 'https://zenodo.org/records/7603772/files/CardiacPatientData.csv' \
  -o data/zenodo_cardiac/CardiacPatientData.csv

# ── SDDB (Sudden Cardiac Death Holter) — 23 records, 1.2 GB:
wget -r -np -nH --cut-dirs=3 \
  https://physionet.org/files/sddb/1.0.0/ \
  -P data/sddb_sudden_cardiac/

# ── CUDB (Creighton University VT/VF database) — 35 records:
wget -r -np -nH --cut-dirs=3 \
  https://physionet.org/files/cudb/1.0.0/ \
  -P data/cudb_ventricular_tachyarrhythmia/

# ── I-CARE (PhysioNet 2023, 1020+ post-arrest patients) — OPEN ACCESS:
# Create PhysioNet account at https://physionet.org/ (free, no institutional access needed)
# After account setup — I-CARE is open-access, no DUA required:
mkdir -p data/icare
wget -r -N -c -np \
  https://physionet.org/files/i-care/2.1/ \
  -P data/icare/
# → This downloads ~17.5 GB. Place all .tsv patient files in data/icare/
# → Loader is READY: load_icare() in train_models.py — will auto-detect files


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATASET PREPARATION & VERIFICATION  (always run BEFORE training)
# ══════════════════════════════════════════════════════════════════════════════

# Full verification: checks all datasets, unpacks zips, validates columns
python scripts/prepare_datasets.py

# Resource check only (RAM, swap — run to decide if safe to train)
python scripts/prepare_datasets.py --resource-check

# Save a JSON report
python scripts/prepare_datasets.py --save-report prep_report.json

# Verify a single target only (NEW NAMES — use hemodynamic_collapse not cardiac_arrest)
python scripts/prepare_datasets.py --target sepsis
python scripts/prepare_datasets.py --target hypotension
python scripts/prepare_datasets.py --target hemodynamic_collapse  # ← not cardiac_arrest!

# Quick validation inside train_models.py (fast, checks loaders+columns)
python train_models.py --validate


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

# IMPORTANT: target name is now "hemodynamic_collapse" not "cardiac_arrest"

# ── Training directly ──
python train_models.py --target sepsis
python train_models.py --target hypotension
python train_models.py --target hemodynamic_collapse   # ← new name
python train_models.py --all   # trains all 3 targets sequentially

# ── OVERNIGHT RUN (background, keeps running if you close terminal) ──
mkdir -p logs
nohup python train_models.py --all > logs/training_$(date +%Y%m%d_%H%M).log 2>&1 &
echo "Training PID: $!"   # note this PID to monitor/kill later

# ── CRASH RECOVERY: resume from last checkpoint ──
python train_models.py --target hemodynamic_collapse --resume
python train_models.py --all --resume

# ── Check if training is still running ──
ps aux | grep train_models.py

# ── Watch live training log ──
tail -f logs/training_YYYYMMDD_HHMM.log

# ── Kill training if needed ──
kill <PID>


# ══════════════════════════════════════════════════════════════════════════════
# 4. CHECKING TRAINED MODELS
# ══════════════════════════════════════════════════════════════════════════════

# See what's been saved after training
ls models/sepsis/
ls models/hypotension/
ls models/hemodynamic_collapse/   # ← new folder name

# View model performance metadata (AUROC, AUPRC, threshold, sensitivity)
cat models/sepsis/model_metadata.json
cat models/hypotension/model_metadata.json
cat models/hemodynamic_collapse/model_metadata.json

# Quick Python check that all model artifacts loaded correctly
python3 -c "
import joblib, json
for t in ['sepsis', 'hypotension', 'hemodynamic_collapse']:
    try:
        m = joblib.load(f'models/{t}/lgbm_model.pkl')
        meta = json.load(open(f'models/{t}/model_metadata.json'))
        print(f'✅ {t}: AUROC={meta[\"val_auroc\"]} AUPRC={meta[\"val_auprc\"]} engines={meta[\"engines\"]}')
    except Exception as e:
        print(f'❌ {t}: {e}')
"

# Verify feature count matches across models (should all be 75; was 70 before NEE feature added on 2026-02-27)
python3 -c "
import json
for t in ['sepsis', 'hypotension', 'hemodynamic_collapse']:
    try:
        cols = json.load(open(f'models/{t}/feature_columns.json'))
        print(f'  {t}: {len(cols)} features')
    except: print(f'  {t}: not trained yet')
"


# ══════════════════════════════════════════════════════════════════════════════
# 5. SHADOW EVALUATION (runs models on MIMIC demo — inference only, no training)
# ══════════════════════════════════════════════════════════════════════════════

# Run on both MIMIC-III and MIMIC-IV demo (100 patients each)
python scripts/shadow_evaluate.py --all

# Run on just MIMIC-III demo
python scripts/shadow_evaluate.py --mimic3

# Run on just MIMIC-IV demo
python scripts/shadow_evaluate.py --mimic4

# Print summary from existing reports (no rerun)
python scripts/shadow_evaluate.py --report-only

# View the full report
cat reports/shadow_eval_mimic3_demo.json | python3 -m json.tool | head -50


# ══════════════════════════════════════════════════════════════════════════════
# 6. RUNNING THE API SERVER
# ══════════════════════════════════════════════════════════════════════════════

# Start the FastAPI backend (from backend/ directory)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# In background (no auto-reload):
nohup uvicorn api:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &

# Test the API is alive
curl http://localhost:8000/health

# Test a prediction endpoint (example payload)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "test_001",
    "heart_rate": 112,
    "systolic_bp": 88,
    "mean_arterial_pressure": 62,
    "spo2": 94,
    "temperature": 38.5,
    "lactate": 3.2,
    "vasopressor_dose": 0.05
  }'

# The API response "cardiac_arrest" key = hemodynamic_collapse model output
# (key intentionally kept as cardiac_arrest for frontend compatibility)


# ══════════════════════════════════════════════════════════════════════════════
# 7. MEMORY / PERFORMANCE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

# Check current RAM usage before training
python scripts/prepare_datasets.py --resource-check

# Check available RAM (quick)
python3 -c "
import subprocess, re
vm = subprocess.check_output(['vm_stat'], text=True)
free = re.search(r'Pages free:\s+(\d+)', vm)
inact = re.search(r'Pages inactive:\s+(\d+)', vm)
free_gb = (int(free.group(1)) + int(inact.group(1))) * 16384 / 1e9
print(f'Available RAM: ~{free_gb:.1f} GB')
"

# Check swap usage (if > 4 GB, RAM is under heavy pressure)
sysctl vm.swapusage

# Expected training RAM usage:
#   Sepsis (CinC 2019, 40k patients):     ~8-12 GB peak
#   Hypotension (eICU 1.6M rows):         ~10-14 GB peak
#   Hemodynamic Collapse (VitalDB 6388):  ~6-10 GB peak
# Total sequential run: ~14 GB max at any one time


# ══════════════════════════════════════════════════════════════════════════════
# 8. SYNTAX / CODE CORRECTNESS CHECK
# ══════════════════════════════════════════════════════════════════════════════

# Quick syntax check for all critical files (run after any code change):
source .venv/bin/activate
python -m py_compile train_models.py      && echo "train_models.py ✅"
python -m py_compile features.py          && echo "features.py ✅"
python -m py_compile api.py               && echo "api.py ✅"
python -m py_compile scripts/prepare_datasets.py && echo "prepare_datasets.py ✅"
python -m py_compile scripts/shadow_evaluate.py  && echo "shadow_evaluate.py ✅"
python -m py_compile scripts/mimic_mapper.py     && echo "mimic_mapper.py ✅"

# Full feature engineering smoke test
python3 -c "
from features import get_feature_columns, LAB_VITALS
cols = get_feature_columns()
mf = [c for c in cols if c.endswith('_measured')]
nee_cols = [c for c in cols if 'nee' in c]
print(f'✅ {len(cols)} features | {len(mf)} missingness flags: {mf}')
print(f'   NEE features ({len(nee_cols)}): {nee_cols}')
# Expected: 75 features as of 2026-02-27 (was 70 before NEE addition)
"


# ══════════════════════════════════════════════════════════════════════════════
# 9. FILE STRUCTURE REFERENCE
# ══════════════════════════════════════════════════════════════════════════════

# Critical files:
# backend/
#   train_models.py              ← Main training pipeline (~2230+ lines)
#   features.py                  ← Feature engineering (543+ lines, 75 features)
#   api.py                       ← FastAPI server
#   physics_engine.py            ← Deterministic clinical rules (safety net)
#   data_streamer.py             ← ICU data simulator
#   requirements.txt             ← All Python dependencies
#   PROJECT_HANDOFF.md           ← Context for incoming AI agent (READ THIS FIRST)
#   scripts/
#     download_datasets.py       ← Dataset downloader
#     prepare_datasets.py        ← Pre-training verification & unpack
#     shadow_evaluate.py         ← Shadow evaluation on MIMIC demo
#     mimic_mapper.py            ← MIMIC itemid → Chronos feature mapper
#   data/
#     cinc2019/                  ← 40,336 sepsis PSV files
#     eicu_demo/                 ← 1.6M eICU vital rows
#     vitaldb/                   ← 6388 cardiac arrest cases (downloading)
#     cudb_ventricular_*/        ← 35 VT/VF ECG records (WFDB)
#     sddb_sudden_cardiac/       ← 23 Holter ECG records (WFDB, 1.2 GB)
#     zenodo_cardiac/            ← 112 real cardiac arrest patients
#     icare/                     ← I-CARE post-arrest (pending user download)
#     mimic3_demo/               ← 100 MIMIC-III patients (shadow eval only)
#     mimic4_demo/               ← 100 MIMIC-IV patients (shadow eval only)
#   models/
#     sepsis/                    ← Saved LGBM, XGBoost, CatBoost, GRU-D, TCN
#     hypotension/               ← Same structure
#     hemodynamic_collapse/      ← Same + meta_stacker.pkl (was cardiac_arrest/)
#       checkpoints/             ← Per-fold .pkl and per-epoch .pt files


# ══════════════════════════════════════════════════════════════════════════════
# 10. TROUBLESHOOTING
# ══════════════════════════════════════════════════════════════════════════════

# "ModuleNotFoundError: No module named 'wfdb'"
pip install wfdb

# "ModuleNotFoundError: No module named 'catboost'"
pip install catboost

# "ModuleNotFoundError: No module named 'imblearn'"
pip install imbalanced-learn

# "MPS backend not available"
# → You're on CPU. Training will be 3-5× slower but still works.
# → Update PyTorch: pip install --upgrade torch

# "Killed" during training (OOM / out of memory)
# → Close all other apps, then:
python train_models.py --target sepsis --resume     # picks up where it died

# "No data for hemodynamic_collapse. Skipping."
# → VitalDB download hasn't reached enough files. Check:
ls data/vitaldb/ | wc -l
# → If < 100, wait for download or re-run:
python scripts/download_datasets.py --vitaldb --vitaldb-cases 2000

# "I-CARE .tsv files not found"
# → User is downloading this separately (~17.5 GB). Put files in data/icare/
# → Code does NOT crash without I-CARE, it skips with a warning.

# Training completes but AUROC is 0.5 (random)
# → Positive rate is 0 in test fold (data too sparse). Check:
python train_models.py --validate

# API error: "models/hemodynamic_collapse not found"
# → If old code saved to models/cardiac_arrest/:
#   mv models/cardiac_arrest models/hemodynamic_collapse
# → Or retrain: python train_models.py --target hemodynamic_collapse

# WFDB ECG loader returns None (CUDB/SDDB empty)
# → wfdb.rdann() uses "atr" extension; SDDB has 12 .atr + 23 .ari
# → Loader already handles fallback to .ari automatically

# Feature count mismatch error (model expects 70 but got 75)
# → Old models trained before 2026-02-27 had 70 features (before NEE vasopressor features)
# → Models trained before 2026-02-26 had 62 features (before 8 lab missingness flags were added)
# → Solution: retrain. Current code produces 75 features.
# → Or: mv models/cardiac_arrest models/hemodynamic_collapse (if old folder name)

# "KeyError: cardiac_arrest" in models/ folder
# → Target was renamed to hemodynamic_collapse on Feb 26 2026
# → Either retrain or mv models/cardiac_arrest models/hemodynamic_collapse
