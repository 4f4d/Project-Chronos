# Project Chronos — Agent Handoff Document
# Last updated: 2026-02-26 | Conversation: 6be5c943-b67f-430d-a320-b1e6473de881
# Project root: /Users/falcon/.gemini/antigravity/scratch/health-tech/

================================================================================
## HOW TO READ THIS FILE (for the incoming agent)
================================================================================

You are a new AI agent taking over Project Chronos from a previous session.
This file contains everything you need to understand the full state of the project
without reading any conversation history.

Key memory/brain files for this conversation are at:
  /Users/falcon/.gemini/antigravity/brain/6be5c943-b67f-430d-a320-b1e6473de881/
    ├── implementation_plan.md   ← Full God-Mode architecture plan
    ├── task.md                  ← Task checklist with completion status
    ├── dataset_analysis.md      ← Per-dataset deep analysis
    ├── ai_peer_review_analysis.md ← Synthesis of 4 AI agent reviews (ChatGPT, Gemini, Grok x2)
    └── walkthrough.md           ← Latest session change log

Read those files FIRST, then this document. Together they give you full context.

================================================================================
## WHAT THIS PROJECT IS
================================================================================

Project Chronos is an AI-powered ICU "early warning system" that predicts three
critical clinical events before they happen:

  1. SEPSIS                →  Predicts septic shock onset 2-6 hours in advance
  2. HYPOTENSION           →  Predicts acute hypotensive episodes (MAP collapse)
  3. HEMODYNAMIC COLLAPSE  →  Predicts circulatory failure / cardiac arrest risk
     (formerly called "CARDIAC ARREST" internally — renamed Feb 26 2026)

⚠️  IMPORTANT NAMING NOTE:
  Internal target name: "hemodynamic_collapse"
  API JSON key:         "cardiac_arrest"  (intentionally kept for frontend compat)
  Models folder:        models/hemodynamic_collapse/  (not cardiac_arrest/)
  All argparse choices: --target hemodynamic_collapse

The outputs are:
  a) A probability [0.0-1.0] for each target
  b) SHAP feature importance explaining WHY the model thinks this
  c) A physics-engine "safety net" using deterministic clinical rules
  d) A UI that shows these predictions as a "Triage Radar"

================================================================================
## ARCHITECTURE (as of 2026-02-26)
================================================================================

### ML Pipeline (train_models.py — 1,960 lines, God-Mode v2)

5 model engines → LightGBM meta-stacker → isotonic calibration → F-beta threshold:

  Engine 1: LightGBM     (tabular, class_weight balanced)
  Engine 2: XGBoost      (tabular, scale_pos_weight)
  Engine 3: CatBoost     (tabular, native categoricals — optional if installed)
  Engine 4: GRU-D        (sequential, irregular time-series with missingness)
  Engine 5: TCN          (sequential, dilated causal convolutions, faster than GRU-D)
  ECG layer: ResNet1D    (for hemodynamic_collapse only, uses CUDB/SDDB waveform data)

  Meta-stacker: LightGBM trained on out-of-fold predictions from all engines
  Calibration: Isotonic regression (probabilities become clinically meaningful)
  Threshold: F-beta=2 optimization (sensitivity-weighted for ICU context)

### Key design parameters:
  PREDICTION_HORIZON_HOURS = 4     # predict event in next 4 hours (mid of 2-6h window)
  SEED = 42

  Per-target sequence lengths (CRITICAL — different from old code):
    TARGET_SEQ_LEN = {
        "sepsis":                12,   # 12h lookback for sepsis trajectory
        "hypotension":           12,   # 12h lookback for MAP trends
        "hemodynamic_collapse":   8,   # 8h lookback — matches VitalDB surgical duration
    }

### Training pipeline features:
  - SEED=42 globally (random, numpy, torch, PYTHONHASHSEED, optuna)
  - 5-fold GroupKFold cross-validation (grouped by patient_id)
  - Tabular imbalance handling:
      sepsis               → SMOTE-ENN  (moderate imbalance ~6.5%)
      hypotension          → plain SMOTE (clean MAP labels)
      hemodynamic_collapse → ADASYN     (severe imbalance ~0.5-2%)
  - Sequential imbalance handling (NEW — replaces SMOTE for temporal data):
      if positive_rate < 3% → FocalBCE loss (Lin et al. 2017, γ=2.0)
      else                  → pos_weight BCEWithLogitsLoss (GRU-D) /
                              label-smoothed BCE α=0.05 (TCN)
  - Per-fold checkpoint saves (recover from crash with --resume)
  - Per-epoch neural model checkpoints every 10 epochs
  - Label smoothing (α=0.05) for TCN on noisy ICU proxy labels
  - WeightNorm on TCN convolutions, BatchNorm on ResNet1D
  - Gradient clipping (norm=1.0) on all neural models
  - CosineAnnealingLR scheduler on neural models

### Feature Engineering (features.py — 514 lines):

  Two-stage imputation (NEW — critical distinction):
    Monitor vitals (HR, BP, SpO2, RR, Temp):  forward-fill ONLY
    Lab vitals (Lactate, WBC, Creatinine, Bilirubin, Platelets, PaO2, FiO2, GCS):
      → Add {lab}_measured binary flag FIRST (before NaN is overwritten)
      → Then forward-fill + median fallback

  Total feature set: 70 features
    - 15 base vitals (all forward-filled)
    - 8 lab missingness flags (lactate_measured, wbc_measured, etc.)
    - 12 clinical scores (SOFA, NEWS2, Shock Index, MAP/Lactate ratio,
                         PF ratio, SOFA delta flags, RPP, DO2, A-a gradient, CRI)
    - 35 temporal deltas (7 vitals × 3 windows: 1h, 2h, 4h)

================================================================================
## DATASETS (all in backend/data/)
================================================================================

### SEPSIS target datasets:
  ✅ cinc2019/          → 40,336 patient PSV files, CinC 2019 challenge
                          SepsisLabel column, gold standard
                          Loader: load_cinc2019()

  ✅ eicu_demo/         → Full eICU demo (all 30 CSVs), 2,520 ICU stays, 1.6M vital rows
                          Used for: hypotension primarily, enrichment for sepsis
                          Loader: load_eicu_hypotension()

### HEMODYNAMIC COLLAPSE target datasets:
  ✅ vitaldb/           → 6388 surgical case CSVs (downloading — already 2400+)
                          250+ signals at 1-sec resolution, MAP/SpO2 proxy labels
                          Loader: load_vitaldb()
                          LABEL: target_hemodynamic_collapse (NOT target_cardiac_arrest)

  ✅ zenodo_cardiac/    → 112 real cardiac arrest patients (Sri Lanka)
                          Columns: SBP, DBP, HR, SpO2, GCS, Outcome
                          NOTE: "Ceratinine" column (typo, handled in loader)
                          Loader: load_zenodo_cardiac_arrest()
                          LABEL: target_hemodynamic_collapse

  ✅ cudb_ventricular_tachyarrhythmia/  → 35 WFDB records, 250 Hz, ~8.5 min each
                                          VF onset annotated, uses .atr ext
                                          Loader: load_cudb_wfdb()
                                          LABEL: target_hemodynamic_collapse

  ✅ sddb_sudden_cardiac/  → 23 WFDB records, 250 Hz, 2-channel Holter, ~25h each
                             IMPORTANT: 12/23 have .atr, ALL 23 have .ari
                             Loader tries .atr first, falls back to .ari (fixed)
                             Loader: load_sddb_wfdb()
                             LABEL: target_hemodynamic_collapse

  🔲 icare/             → I-CARE PhysioNet 2023 (1020+ post-arrest patients)
                          Open access, no credentialing needed
                          Download: wget -r -N -c -np --user USERNAME \
                            https://physionet.org/files/i-care/2.1/ -P data/icare/
                          Loader: load_icare() — READY, waiting for download
                          LABEL: target_hemodynamic_collapse (all positive — real arrests)
                          NOTE: User is downloading this separately, will provide when done.
                          The code WILL NOT CRASH without this — it gracefully skips with warning.

### HYPOTENSION target datasets:
  ✅ eicu_demo/vitalPeriodic.csv → Primary source, MAP-based AHE labels
  ⚠️  cinc2009_hypotension/      → Downloaded but WFDB records not yet verified
                                   Loader exists: load_cinc2009_ahe() (commented out)

### Shadow evaluation (inference-only on real MIMIC data):
  ✅ mimic3_demo/       → 100 MIMIC-III patient admissions (demo dataset)
  ✅ mimic4_demo/       → 100 MIMIC-IV patient admissions (demo dataset)
  Both used for shadow evaluation ONLY (not training)
  Mapper: scripts/mimic_mapper.py  ← MIMIC itemid → Chronos feature name
  Evaluator: scripts/shadow_evaluate.py  ← Runs inference, generates alert log

================================================================================
## FILE STRUCTURE (as of 2026-02-26)
================================================================================

/Users/falcon/.gemini/antigravity/scratch/health-tech/
├── backend/
│   ├── train_models.py            ← MAIN TRAINING (God-Mode v2, ~1960 lines)
│   ├── features.py                ← Feature engineering + imputation (514 lines, 70 features)
│   ├── api.py                     ← FastAPI server (complete)
│   ├── physics_engine.py          ← Deterministic clinical safety net
│   ├── data_streamer.py           ← ICU data simulator
│   ├── requirements.txt           ← All Python deps (wfdb, catboost, etc.)
│   ├── COMMANDS.sh                ← Complete command reference sheet (this file's companion)
│   ├── PROJECT_HANDOFF.md         ← This file (incoming agent context)
│   ├── scripts/
│   │   ├── download_datasets.py   ← Dataset downloader (--vitaldb, --cinc, etc.)
│   │   ├── prepare_datasets.py    ← Pre-training verification + unpack script
│   │   ├── shadow_evaluate.py     ← Shadow evaluation on MIMIC demo
│   │   └── mimic_mapper.py        ← MIMIC itemid → Chronos feature mapper
│   ├── data/                      ← All datasets (see above)
│   └── models/                    ← Trained model artifacts (after training)
│       ├── sepsis/
│       ├── hypotension/
│       └── hemodynamic_collapse/  ← NOTE: was 'cardiac_arrest' before Feb 26 2026
│           ├── lgbm_model.pkl
│           ├── xgb_model.pkl
│           ├── cat_model.pkl      (if catboost installed)
│           ├── grud_model.pt
│           ├── tcn_model.pt
│           ├── meta_stacker.pkl
│           ├── shap_explainer.pkl
│           ├── feature_columns.json
│           ├── model_metadata.json  ← AUROC, AUPRC, threshold, sensitivity, seq_len
│           └── checkpoints/         ← Resume points (fold_N_lgbm.pkl, etc.)
└── frontend/                      ← React Triage Radar UI (not started)

================================================================================
## WHAT'S DONE vs WHAT'S LEFT
================================================================================

### ✅ COMPLETED (as of 2026-02-26):
  - [x] Full dataset research and acquisition strategy
  - [x] requirements.txt (wfdb, catboost, imbalanced-learn, antropy added)
  - [x] train_models.py — God-Mode v2 (5 engines + meta-stacker + checkpointing)
  - [x] All data loaders (cinc2019, eicu, vitaldb, zenodo, cudb, sddb, icare)
  - [x] FocalBCE loss class (for GRU-D + TCN — replaces SMOTE on time-series)
  - [x] Per-target sequence lengths (TARGET_SEQ_LEN dict)
  - [x] Two-stage imputation: lab missingness flags + forward-fill
  - [x] 70-feature engineering pipeline (8 new missingness indicator flags)
  - [x] SMOTE-ENN / ADASYN / plain SMOTE per target (tabular only)
  - [x] 5-fold GroupKFold cross-validation
  - [x] LightGBM meta-stacker replacing linear weight blending
  - [x] F-beta=2 threshold optimization
  - [x] AUPRC metric (alongside AUROC) in all training output
  - [x] Checkpoint / resume system (--resume flag)
  - [x] Global seed management (SEED=42)
  - [x] Label smoothing (α=0.05) for TCN
  - [x] scripts/prepare_datasets.py (verify + unpack + column checks)
  - [x] scripts/shadow_evaluate.py (MIMIC-III/IV inference evaluation)
  - [x] scripts/mimic_mapper.py (MIMIC itemid → Chronos feature name)
  - [x] COMMANDS.sh (complete command reference)
  - [x] hemodynamic_collapse rename (from cardiac_arrest)
  - [x] SDDB .ari fallback bug fixed
  - [x] vasopressor_dose field in API payload

### 🔲 NEXT STEPS (in priority order):
  1. Wait for VitalDB download (~4h remaining for 6388 cases)
  2. Add I-CARE dataset when user downloads it → place in data/icare/
  3. Run dataset verification: python scripts/prepare_datasets.py
  4. Run training: python train_models.py --all --resume (overnight, ~15-20 hours)
  5. Run shadow evaluation: python scripts/shadow_evaluate.py --all
  6. Build frontend Triage Radar (Phase 5) — React dashboard

### 🔲 KNOWN GAPS (secondary priority):
  - Per-model Optuna HP tuning not yet implemented
    (current code uses LGBM_PARAMS / XGB_PARAMS — hand-tuned defaults)
  - ResNet1D ECG classifier class exists but not wired into train_target()
  - cinc2009 hypotension loader written but not connected (commented out)
  - uq_vital_signs loader not written

================================================================================
## EXPECTED AUROC TARGETS (realistic, based on literature + AI peer review)
================================================================================

  Sepsis              → Internal: 0.86-0.91 AUROC, External: -0.05 to -0.10
  Hypotension         → Internal: 0.88-0.93 AUROC, External: -0.05 to -0.08
  Hemodynamic Collapse→ Internal: 0.78-0.85 AUROC (harder — proxy labels)
                        With I-CARE real labels: +0.03-0.05 improvement expected

These are realistic — NOT the 0.96 AUROC the AI agents mentioned (that's on eICU full dataset).

================================================================================
## EXACT PROMPT FOR INCOMING AGENT
================================================================================

Copy-paste this to any AI agent to give them full project context:

---PROMPT START---

You are taking over development of **Project Chronos**, an ICU early warning AI.
The project is at: /Users/falcon/.gemini/antigravity/scratch/health-tech/

Your FIRST step MUST be to read ALL of these files before doing anything else:
  1. /Users/falcon/.gemini/antigravity/brain/6be5c943-b67f-430d-a320-b1e6473de881/implementation_plan.md
  2. /Users/falcon/.gemini/antigravity/brain/6be5c943-b67f-430d-a320-b1e6473de881/task.md
  3. /Users/falcon/.gemini/antigravity/brain/6be5c943-b67f-430d-a320-b1e6473de881/walkthrough.md
  4. /Users/falcon/.gemini/antigravity/scratch/health-tech/backend/PROJECT_HANDOFF.md
  5. /Users/falcon/.gemini/antigravity/scratch/health-tech/backend/COMMANDS.sh

CRITICAL CONTEXT (memorize this before touching any code):
  - The 3rd prediction target is "hemodynamic_collapse" (NOT "cardiac_arrest")
    Internal model dir: models/hemodynamic_collapse/
    API JSON key: cardiac_arrest (intentionally kept for frontend — DO NOT change this)
  - Features: 70 total. Includes 8 lab missingness flags (lactate_measured, etc.)
  - Sequence lengths: sepsis=12h, hypotension=12h, hemodynamic_collapse=8h
  - GRU-D and TCN use FocalBCE loss (not SMOTE) for sequences < 3% positive rate
  - SMOTE/ADASYN is for tabular models (LightGBM/XGBoost/CatBoost) ONLY
  - I-CARE dataset: user is downloading separately. Code handles missing I-CARE gracefully.
  - MIMIC-III/IV demo datasets: used for shadow evaluation only, NOT training.
  - VitalDB download is in progress (user running: python scripts/download_datasets.py
    --vitaldb --vitaldb-cases 6388, has been running 2+ hours).

Ask the user what they need next, OR proceed with the first unchecked item in task.md.

---PROMPT END---

================================================================================
## QUICK REFERENCE: KEY DECISIONS MADE (rationale)
================================================================================

Q: Why hemodynamic_collapse not cardiac_arrest?
A: VitalDB proxy labels (MAP<50 for 5+min OR SpO2<85 for 3+min) don't predict
   arrest itself — they predict the circulatory failure cascade BEFORE arrest.
   "Hemodynamic collapse" is the correct clinical term for this pre-arrest state.

Q: Why FocalBCE instead of SMOTE for GRU-D/TCN?
A: SMOTE generates synthetic time-series by interpolating between patients — creating
   physiologically impossible vital sign trajectories. FocalBCE handles imbalance
   at the loss level WITHOUT touching the data (Lin et al., 2017, γ=2.0).

Q: Why per-target SEQ_LEN?
A: VitalDB surgical cases average 2-4 hours total. A 12h lookback window would
   be empty. 8h matches the actual case length and avoids padding artifacts.
   Sepsis/hypotension need 12h because SOFA trajectory patterns evolve slowly.

Q: Why lab missingness flags?
A: NIH 2023 study: missingness indicators = >40% of top predictors in ICU mortality.
   A missing lactate at 14:00 means the clinician didn't ORDER it — a real signal.
   We add {lab}_measured flags BEFORE imputing NaN so the model sees both the
   imputed value AND whether it was actually measured.

Q: Why SMOTE-ENN for sepsis, ADASYN for hemodynamic_collapse?
A: SMOTE-ENN removes noisy borderline synthetic samples — better for sepsis where
   class boundary is fuzzy (proxy labels via SepsisLabel, onset ambiguity).
   ADASYN focuses synthesis in hard decision regions — better for hemodynamic_collapse
   where positive rate is 0.5-2% and we need samples in the hardest subspace.

Q: Why LightGBM meta-stacker instead of Optuna weight blending?
A: Linear blending assumes models are equally right in all regions. Meta-stacker
   learns WHEN each model is more accurate — LightGBM dominates on lab-heavy
   rows, GRU-D/TCN dominate on rapid vital trajectory rows.

Q: Why F-beta=2 threshold (not F1 or accuracy)?
A: In ICU: missing true positive (patient crashes) >> 10 false alarms.
   F-beta=2 weights recall 2× over precision. Standard for critical care alerts.

Q: Why MIMIC demo for shadow eval only (not training)?
A: MIMIC-III demo = 100 patients, MIMIC-IV demo = 100 patients. That's 200 total.
   Using 50% for training (100 patients) is too small to improve a model already
   trained on 40k+ CinC 2019 patients. Shadow eval proves generalization across
   hospital systems without data leakage.

Q: Why not MIMIC-IV full / eICU full?
A: Both require PhysioNet credentialing ($0 but institutional DUA). Not available
   to this project. CinC 2019 (40k) + eICU demo (2.5k) is adequate for Phase 1.

Q: Why not MIMIC-III/IV full for training even with 10-15% sample?
A: Credentialing required. The demo datasets (100 each) are freely available but
   too small for meaningful model improvement on top of existing training data.
