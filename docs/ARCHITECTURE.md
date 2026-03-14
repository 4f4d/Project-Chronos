# Architecture

Detailed technical architecture documentation for Project Chronos.

## Table of Contents

- [System Components](#system-components)
- [Data Flow](#data-flow)
- [Backend Architecture](#backend-architecture)
- [Frontend Architecture](#frontend-architecture)
- [ML Training Pipeline](#ml-training-pipeline)
- [Data Contract](#data-contract)
- [Deployment Model](#deployment-model)

---

## System Components

Project Chronos is composed of four independent processes that communicate via HTTP and WebSocket:

| Process | Technology | Port | Role |
| ------- | --------- | ---- | ---- |
| API Server | FastAPI (Python 3.12) | 8000 | Prediction, SHAP, physics, model management |
| Data Streamer | Python script | N/A | Simulates real-time ICU data via REST POST |
| Frontend | React + Vite | 5173 | Triage radar dashboard |
| LLM Server | Ollama | 11434 | Clinical debate inference (optional) |

All processes run on the same machine. The system is designed for a single-node deployment on an Apple Silicon Mac with 16-24 GB of unified memory.

---

## Data Flow

```text
 ┌──────────────────┐
 │  ICU Monitor /   │
 │  data_streamer   │
 └────────┬─────────┘
          │ POST /api/patient_data
          ▼
 ┌──────────────────────────────────────────────┐
 │                  FastAPI                      │
 │                                              │
 │  1. Validate input (Pydantic schema)         │
 │  2. Feature engineering (70 features)        │
 │  3. Run ML ensemble (LGB, XGB, GRU-D, TCN)  │
 │  4. Meta-stacker + isotonic calibration      │
 │  5. Physics engine (DO2, THI, HIS)           │
 │  6. SHAP feature importance                  │
 │  7. Format response (data_contract.py)       │
 │  8. Push via WebSocket to frontend           │
 │                                              │
 └────────┬─────────────────────────────────────┘
          │ WebSocket /ws
          ▼
 ┌──────────────────────────────────────────────┐
 │               React Frontend                  │
 │                                              │
 │  ┌────────────┐  ┌──────────────┐            │
 │  │ Triage     │  │ Detail       │            │
 │  │ Radar      │  │ Panel        │            │
 │  │ (all pts)  │  │ (selected)   │            │
 │  └────────────┘  └──────────────┘            │
 │  ┌────────────┐  ┌──────────────┐            │
 │  │ Analytics  │  │ Settings     │            │
 │  │ Dashboard  │  │ Panel        │            │
 │  └────────────┘  └──────────────┘            │
 │  ┌────────────────────────────────┐          │
 │  │ LLM Clinical Debate           │          │
 │  │ (Hawkeye / Reed / Foreman)    │──────┐   │
 │  └────────────────────────────────┘      │   │
 └──────────────────────────────────────────┼───┘
                                            │ POST /api/chat
                                            ▼
                                   ┌────────────────┐
                                   │  Ollama        │
                                   │  Med42-v2 8B   │
                                   │  localhost:11434│
                                   └────────────────┘
```

---

## Backend Architecture

### API Endpoints

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/api/health` | GET | System status, loaded models, feature count |
| `/api/patient_data` | POST | Accept patient vitals, return predictions |
| `/api/predictions` | GET | Current predictions for all active patients |
| `/api/patient/{id}` | GET | Detailed prediction for a specific patient |
| `/api/chat` | POST | Stream LLM response for clinical debate |
| `/ws` | WebSocket | Real-time prediction push to frontend |

### Model Loading

On startup, the API server loads model artifacts from `backend/models/{target}/`:

```text
models/{target}/
├── lgbm_model.pkl          # Final LightGBM model
├── xgb_model.pkl           # Final XGBoost model
├── grud_model.pt           # GRU-D PyTorch state dict
├── tcn_model.pt            # TCN PyTorch state dict (if trained)
├── meta_stacker.pkl        # LightGBM meta-stacker
├── isotonic_calibrator.pkl # Isotonic regression calibrator
├── shap_explainer.pkl      # SHAP TreeExplainer (LightGBM-based)
├── grud_scaler.pkl         # StandardScaler for z-normalization
├── grud_medians.pkl        # Training-set medians for NaN fill
└── model_metadata.json     # Performance metrics and hyperparameters
```

Models are loaded lazily and cached. The system tolerates missing components: if TCN artifacts are absent, the ensemble runs with available engines only.

### Physics Engine (`physics_engine.py`)

The physics engine is stateless and deterministic. It receives a single patient's vitals at each time step and computes physiological indices. It maintains no internal state and makes no stochastic decisions. All formulas are derived from published physiology literature.

Key design principle: the physics engine cannot be trained, tuned, or influenced by data. It applies the same physiological constraints regardless of what the ML models predict.

---

## Frontend Architecture

### Component Hierarchy

```text
App.jsx
├── SettingsPanel.jsx          # System configuration (view switching)
├── PatientCard.jsx            # Individual patient triage card
│   └── Risk level indicator   # Color-coded risk pills
├── DetailPanel.jsx            # Selected patient detail view
│   ├── SHAP bar chart         # Feature importance visualization
│   └── Radar chart            # Multi-axis vital sign overview
├── AnalyticsDashboard.jsx     # Population-level metrics
│   └── Charts.jsx             # Shared chart components
└── HouseTeam.jsx              # Multi-agent LLM debate
    ├── Agent streaming view   # Real-time token streaming
    └── Clinical disclaimer    # Mandatory non-diagnostic notice
```

### State Management

The `useChronos.js` hook manages all data fetching and state:

- WebSocket connection with automatic reconnection and exponential backoff
- Patient data caching and deduplication
- API health polling (5-second interval)
- Connection status tracking and error counting

No external state management library is used. React's built-in `useState` and `useRef` handle all state.

### Real-Time Updates

Patient data streams via WebSocket at configurable intervals (default: 5 seconds). The triage radar re-sorts patients by risk level on each update. The detail panel refreshes SHAP values and radar chart data when the selected patient receives new predictions.

---

## ML Training Pipeline

### Execution Order

```text
train_models.py --target {target} --tune
│
├── 1. Load datasets
│   └── Dataset-specific loaders (CinC 2019, eICU, VitalDB, Zenodo, CUDB, SDDB)
│
├── 2. Feature engineering (per-patient, prevents cross-patient leakage)
│   └── features.py: engineer_features()
│
├── 3. Patient-level train/test split (GroupShuffleSplit 80/20)
│
├── 4. Optuna hyperparameter tuning (50 trials, LightGBM + XGBoost)
│
├── 5. 5-Fold StratifiedGroupKFold
│   ├── LightGBM per fold
│   ├── XGBoost per fold
│   └── Out-of-fold predictions saved
│
├── 6. Sequential model training
│   ├── GRU-D (z-scaled sequences, patient-level split, early stopping)
│   └── TCN (separate z-scaler, separate patient split)
│
├── 7. Meta-stacker (LightGBM on concatenated OOF predictions)
│
├── 8. Isotonic calibration
│
├── 9. F-beta=2 threshold optimization
│
├── 10. Hold-out test evaluation
│   ├── AUROC, AUPRC, sensitivity, specificity
│   └── vs NEWS2 baseline comparison
│
├── 11. SHAP explainer generation
│
└── 12. Save artifacts to backend/models/{target}/
```

### Important Implementation Details

**No imputation leakage**: The feature engineering step (`engineer_features()`) uses hardcoded population-median fallback values (e.g., HR=75, MAP=93, SpO2=97), not data-derived statistics. These are physiological normal values from clinical literature, applied identically to train and test data. The z-normalization scalers for GRU-D and TCN are fitted exclusively on the training split.

**SMOTE is disabled**: Initial experiments with SMOTE-ENN and ADASYN showed that oversampling on top of native class weighting (LightGBM `is_unbalance=True`, XGBoost `scale_pos_weight`) caused overfit and lower AUPRC. The current pipeline uses only native class weighting. The SMOTE code is preserved but not executed.

**NaN protection**: 13 defensive mechanisms protect sequential model training from numerical instability, including gradient clipping (max_norm=1.0), per-batch NaN detection and skip, Xavier initialization, input BatchNorm, GroupNorm in TCN blocks, and loss scaling guards.

---

## Data Contract

The backend-frontend communication follows a strict JSON schema defined in `shared/data_contract.py` using Pydantic models. See that file for the complete schema specification.

Key response fields per patient:

| Field | Type | Description |
| ----- | ---- | ----------- |
| `patient_id` | string | Patient identifier |
| `predictions` | object | Sepsis, hypotension, cardiac_arrest probabilities with risk levels |
| `vitals` | object | Current vital signs with units |
| `shap_values` | object | Top SHAP feature drivers per prediction |
| `physics` | object | DO2, THI, HIS, overrides |
| `clinical_scores` | object | SOFA, NEWS2, Shock Index, PF Ratio |
| `timestamp` | string | ISO 8601 timestamp |

---

## Deployment Model

The entire system is designed for single-machine, air-gapped deployment:

| Concern | Approach |
| ------- | -------- |
| Data privacy | No network egress; all computation local |
| Model inference | CPU-based (no GPU required; MPS available for training) |
| LLM inference | Ollama runs locally; no API keys or cloud LLM calls |
| Authentication | Not implemented (planned: role-based access control) |
| Logging | Loguru to local filesystem |
| Monitoring | `/api/health` endpoint for system status |

For a two-machine deployment (e.g., one for training, one for inference), the only configuration change is setting `VITE_API_URL` in `frontend/.env` to the backend machine's IP address.
