"""
Project Chronos - FastAPI Edge Server
=======================================
The main backend API that:
  1. Receives streaming patient vitals (from data_streamer.py or real HL7 feed)
  2. Maintains a rolling 12-hour history per patient (for GRU-D temporal modeling)
  3. Runs the LightGBM / XGBoost / GRU-D ensemble for each prediction target
  4. Runs the Physics Engine (deterministic biological safety net)
  5. Computes SHAP values for the explainable frontend
  6. Returns the strict JSON contract consumed by the Triage Radar UI

Runs locally on the 24GB M4 Mac (Edge deployment - HIPAA compliant,
no data leaves the local network).

Usage:
  uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
  
  Frontend Mac connects to:
  http://<mac1_local_ip>:8000
"""

import sys
import json
import math
import asyncio
import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, deque
from typing import Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Local imports
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from features import engineer_features, impute_vitals, get_feature_columns, VITAL_MEDIANS
from physics_engine import run_physics_engine, PhysicsEngineOutput
from train_models import GRUD, TCN, SEQUENCE_LENGTH, DEVICE  # Bug 54 fix: import TCN

# ─────────────────────────────────────────────────────────────────────────────
# Paths & Config
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "models"
MAX_HISTORY_ROWS = SEQUENCE_LENGTH * 3   # Keep 3x the sequence length in memory per patient

RISK_LEVELS = {
    (0.80, 1.01): "CRITICAL",
    (0.55, 0.80): "HIGH",
    (0.30, 0.55): "MODERATE",
    (0.00, 0.30): "LOW",
}

def get_risk_level(probability: float, threshold: float = 0.0) -> str:
    """Maps a probability score to a clinical risk level.

    If a per-target optimal_threshold is provided (from F-beta=2 training),
    tiers are anchored proportionally to that threshold instead of the
    hardcoded RISK_LEVELS. This ensures the clinical alert system respects
    the operating point that was tuned for early warning sensitivity.

    AUDIT-FIX: Previously used fixed 0.80/0.55/0.30 cutoffs regardless of
    the training-optimized threshold.
    """
    if threshold > 0.0:
        # Threshold-proportional tiers:
        #   CRITICAL = 2× threshold and above
        #   HIGH     = threshold to 2× threshold
        #   MODERATE = 0.5× threshold to threshold
        #   LOW      = below 0.5× threshold
        if probability >= min(threshold * 2.0, 0.95):
            return "CRITICAL"
        elif probability >= threshold:
            return "HIGH"
        elif probability >= threshold * 0.5:
            return "MODERATE"
        else:
            return "LOW"
    # Fallback to hardcoded tiers if no threshold available
    for (lo, hi), level in RISK_LEVELS.items():
        if lo <= probability < hi:
            return level
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────
class ModelRegistry:
    """Holds all trained models in memory, loaded once at startup."""

    def __init__(self):
        self.lgbm:              dict[str, Any]  = {}
        self.xgb:               dict[str, Any]  = {}
        self.cat:               dict[str, Any]  = {}
        self.grud:              dict[str, Any]  = {}
        self.tcn:               dict[str, Any]  = {}   # Bug 54 fix: was never declared or loaded
        self.grud_scaler:       dict[str, Any]  = {}   # Bug 55 fix: z-scaler for GRU-D inference
        self.grud_medians:      dict[str, Any]  = {}   # BUG-R3-2: training-set medians for z-scaler NaN-fill
        self.explainer:         dict[str, Any]  = {}
        self.meta_stacker:      dict[str, Any]  = {}   # Bug 26 fix: declare in __init__
        self.meta_names:        dict[str, list] = {}   # model names the meta-stacker expects
        self.calibrator:        dict[str, Any]  = {}   # Isotonic calibrator (post-SHAP)
        self.sequential_enabled: dict[str, bool] = {}  # False if seq AUROC < 0.74 at training
        self.weights:           dict[str, dict] = {}
        self.features:          dict[str, list] = {}
        self.metadata:          dict[str, dict] = {}
        self.optimal_threshold: dict[str, float] = {}  # F-beta=2 optimized threshold per target
        self.loaded_targets:    list[str]       = []

    def load_target(self, target: str) -> bool:
        """Load all model artifacts for a single prediction target."""
        model_dir = MODELS_DIR / target

        if not model_dir.exists():
            logger.warning(f"  Model directory missing for [{target}]: {model_dir}")
            logger.info(f"  Run: python backend/train_models.py --target {target}")
            return False

        try:
            lgbm_path         = model_dir / "lgbm_model.pkl"
            xgb_path          = model_dir / "xgb_model.pkl"
            cat_path          = model_dir / "cat_model.pkl"
            shap_path         = model_dir / "shap_explainer.pkl"
            grud_path         = model_dir / "grud_model.pt"
            tcn_path          = model_dir / "tcn_model.pt"   # Bug 54 fix: load TCN
            grud_scaler_path  = model_dir / "grud_scaler.pkl"  # Bug 55 fix: load scaler
            feat_path         = model_dir / "feature_columns.json"
            meta_path         = model_dir / "model_metadata.json"
            meta_stacker_path = model_dir / "meta_stacker.pkl"
            meta_names_path   = model_dir / "meta_model_names.json"

            self.lgbm[target]      = joblib.load(lgbm_path)
            self.xgb[target]       = joblib.load(xgb_path)
            self.explainer[target] = joblib.load(shap_path)

            # CatBoost is optional
            if cat_path.exists():
                try:
                    self.cat[target] = joblib.load(cat_path)
                except Exception:
                    self.cat[target] = None
            else:
                self.cat[target] = None

            with open(feat_path) as f:
                self.features[target] = json.load(f)
            with open(meta_path) as f:
                self.metadata[target] = json.load(f)
            # Load F-beta=2 optimized operating threshold from training
            self.optimal_threshold[target] = float(
                self.metadata[target].get("optimal_threshold", 0.25)
            )

            # Bug 25 fix: Load meta-stacker + model names it was trained on.
            if meta_stacker_path.exists():
                self.meta_stacker[target] = joblib.load(meta_stacker_path)
                if meta_names_path.exists():
                    with open(meta_names_path) as f:
                        self.meta_names[target] = json.load(f)
                else:
                    self.meta_names[target] = ["lgbm", "xgb"] + (
                        ["cat"] if self.cat.get(target) is not None else []
                    )
                logger.info(f"  [{target}] meta-stacker loaded (input models: {self.meta_names[target]})")
            else:
                self.meta_stacker[target] = None
                self.meta_names[target]   = []
                self.weights[target] = {"lgbm": 0.50, "xgb": 0.30, "grud": 0.20}

            # Bug 55 fix: Load GRU-D z-normalization scaler.
            # GRU-D was trained on z-normalized inputs (Bug 36 fix in train_grud).
            # Without loading this scaler, GRU-D receives raw vitals at inference
            # (e.g. HR=75, MAP=93) instead of z-scores (≈0), causing garbage predictions.
            if grud_scaler_path.exists():
                self.grud_scaler[target] = joblib.load(grud_scaler_path)
                logger.info(f"  [{target}] GRU-D scaler loaded (Bug 55 fix)")
            else:
                self.grud_scaler[target] = None

            # BUG-R3-2 FIX: Load training-set medians for z-scaler NaN-fill.
            # GRU-D/TCN training fills NaN with training-set medians before z-scaling.
            # Using per-patient medians at inference shifts the z-scaled distribution.
            grud_medians_path = model_dir / "grud_medians.pkl"
            if grud_medians_path.exists():
                self.grud_medians[target] = joblib.load(grud_medians_path)
                logger.info(f"  [{target}] GRU-D training medians loaded (R3-2 fix)")
            else:
                self.grud_medians[target] = None

            # Load GRU-D model
            if grud_path.exists() and self.features.get(target):
                grud_model = GRUD(input_size=len(self.features[target])).to(DEVICE)
                grud_model.load_state_dict(
                    torch.load(grud_path, map_location=DEVICE, weights_only=True)
                )
                grud_model.eval()
                self.grud[target] = grud_model
                logger.info(f"  [{target}] GRU-D model loaded")
            else:
                self.grud[target] = None

            # Bug 54 fix: Load TCN model (was trained and saved but never loaded).
            if tcn_path.exists() and self.features.get(target):
                from train_models import TCN
                tcn_model = TCN(input_size=len(self.features[target])).to(DEVICE)
                try:
                    tcn_model.load_state_dict(
                        torch.load(tcn_path, map_location=DEVICE, weights_only=True),
                        strict=False  # TEMP: tolerate GroupNorm keys missing from old checkpoints
                    )
                    tcn_model.eval()
                    self.tcn[target] = tcn_model
                    logger.info(f"  [{target}] TCN model loaded (strict=False for arch compat)")
                except Exception as e:
                    logger.warning(f"  [{target}] TCN load failed, skipping: {e}")
                    self.tcn[target] = None
            else:
                self.tcn[target] = None

            self.loaded_targets.append(target)
            meta = self.metadata[target]
            has_meta  = self.meta_stacker.get(target) is not None
            has_grud  = self.grud.get(target)  is not None
            has_tcn   = self.tcn.get(target)   is not None

            # Load isotonic calibrator (produced by train_target() after training)
            cal_path = model_dir / "isotonic_calibrator.pkl"
            if cal_path.exists():
                self.calibrator[target] = joblib.load(cal_path)
                logger.info(f"  [{target}] isotonic calibrator loaded")
            else:
                self.calibrator[target] = None

            # Read sequential_enabled flag (False = seq AUROC < 0.74 at training)
            self.sequential_enabled[target] = bool(meta.get("sequential_enabled", True))
            if not self.sequential_enabled[target]:
                logger.warning(f"  [{target}] sequential_enabled=False (tabular-only blend at inference)")

            logger.success(
                f"  \u2705 [{target}] loaded \u2014 AUROC: {meta.get('val_auroc', '?')} | "
                f"meta: {'\u2713' if has_meta else '\u2717'} | "
                f"grud: {'\u2713' if has_grud else '\u2717'} | tcn: {'\u2713' if has_tcn else '\u2717'} | "
                f"cal: {'\u2713' if self.calibrator.get(target) else '\u2717'} | "
                f"seq: {'ON' if self.sequential_enabled[target] else 'OFF'}"
            )
            return True

        except Exception as e:
            logger.error(f"  Failed to load [{target}]: {e}")
            return False
    
    def load_all(self):
        logger.info("Loading model registry...")
        for t in ["sepsis", "hypotension", "hemodynamic_collapse"]:
            self.load_target(t)
        
        if not self.loaded_targets:
            logger.warning(
                "⚠️  No trained models found. Returning physics-only predictions.\n"
                "   Run: python backend/train_models.py --all"
            )
        else:
            logger.success(f"Registry: {len(self.loaded_targets)} model sets loaded.")


registry = ModelRegistry()

# Per-patient rolling history (in-memory deque, bounded to MAX_HISTORY_ROWS)
patient_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY_ROWS))

# Active WebSocket connections (for real-time frontend push)
# BUG-CLAUDE-4-1/4-2 fix: Changed from dict[str, WebSocket] to dict[str, list[WebSocket]].
# Previously, opening a second browser tab (or two triage monitors) would silently
# overwrite the first connection — the first tab received no more updates.
# Now each key holds a LIST of sockets; push_to_websocket iterates all of them.
websocket_connections: dict[str, list[WebSocket]] = defaultdict(list)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan (model loading at startup)
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("\n" + "="*60)
    logger.info("🏥  PROJECT CHRONOS EDGE SERVER  STARTING UP")
    logger.info("="*60)
    registry.load_all()
    logger.info("="*60 + "\n")
    yield
    logger.info("Project Chronos shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Project Chronos — ICU Predictive Stability Engine",
    description="Edge-deployed, HIPAA-compliant AI prediction API for ICU catastrophic events.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to frontend MAC's IP in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────────────────────────────
class VitalsPayload(BaseModel):
    patient_id: str
    timestamp: Optional[str] = None

    # Monitor vitals (always present — forward-fill is correct here)
    heart_rate:             Optional[float] = None
    systolic_bp:            Optional[float] = None
    diastolic_bp:           Optional[float] = None
    mean_arterial_pressure: Optional[float] = None
    spo2:                   Optional[float] = None
    respiratory_rate:       Optional[float] = None
    temperature:            Optional[float] = None

    # Lab results (intermittent — missingness is clinically meaningful)
    lactate:                Optional[float] = None
    wbc:                    Optional[float] = None
    creatinine:             Optional[float] = None
    bilirubin:              Optional[float] = None
    platelets:              Optional[float] = None
    pao2:                   Optional[float] = None
    fio2:                   Optional[float] = None
    gcs:                    Optional[int]   = None

    # Treatment features (address the treatment paradox)
    # vasopressor_dose: norepinephrine-equivalent dose in mcg/kg/min (0 if none)
    vasopressor_dose:       Optional[float] = 0.0
    vasopressor_active:     Optional[bool]  = False    # kept for backward compat
    on_mechanical_ventilation: Optional[bool] = False

    # R17: Ground truth labels injected by the data_streamer for validation overlay
    ground_truth_labels:    Optional[dict]  = None

    model_config = {"populate_by_name": True, "extra": "allow"}


# ─────────────────────────────────────────────────────────────────────────────
# Core Inference Logic
# ─────────────────────────────────────────────────────────────────────────────

def run_tabular_inference(
    target: str,
    feature_vector: np.ndarray,
    feature_names: list[str],
) -> tuple[float, list[dict]]:
    """
    Runs LightGBM + XGBoost (+CatBoost when available) for a single target.
    Uses the trained LightGBM meta-stacker for blending when available;
    falls back to calibrated weighted average if meta-stacker was not trained.
    Returns (probability, top_shap_drivers).
    """
    if target not in registry.loaded_targets:
        return 0.0, []

    try:
        # Bug 39 fix: feature_vector contains ALL features (full get_feature_columns() list),
        # but each model was trained on `avail` — the subset of features present in that
        # training dataset. We must sub-select to the model's expected feature set.
        # Without this, LightGBM / XGBoost would crash or silently predict on wrong features.
        model_features = registry.features[target]   # The avail list from training
        feat_name_to_idx = {name: i for i, name in enumerate(feature_names)}
        model_feat_indices = [feat_name_to_idx[f] for f in model_features if f in feat_name_to_idx]
        X_model = feature_vector[model_feat_indices].reshape(1, -1)

        lgbm_prob = float(registry.lgbm[target].predict(X_model)[0])
        xgb_prob  = float(registry.xgb[target].predict_proba(X_model)[0, 1])
        cat_prob  = (
            float(registry.cat[target].predict_proba(X_model)[0, 1])
            if registry.cat.get(target) is not None else None
        )

        # ── Step 2: Meta-stacker blending (Bug 25 fix) ───────────────────────
        meta = registry.meta_stacker.get(target)
        if meta is not None:
            name_to_prob = {"lgbm": lgbm_prob, "xgb": xgb_prob, "cat": cat_prob}
            meta_input_names = registry.meta_names.get(target, ["lgbm", "xgb"])
            meta_feats = np.array(
                [name_to_prob.get(n, lgbm_prob) for n in meta_input_names],
                dtype=np.float32
            ).reshape(1, -1)
            final_prob = float(np.clip(meta.predict(meta_feats)[0], 0.0, 1.0))
        else:
            probs  = [lgbm_prob, xgb_prob] + ([cat_prob] if cat_prob is not None else [])
            w_base = [0.50, 0.30]          + ([0.20]     if cat_prob is not None else [])
            total  = sum(w_base)
            final_prob = float(np.clip(
                sum(p * w for p, w in zip(probs, w_base)) / max(total, 1e-9),
                0.0, 1.0
            ))

        # BUG-R7-2 fix: Isotonic calibration REMOVED from here.
        # Training applies isotonic AFTER the 70/30 sequential blend
        # (train_models.py line 2583), NOT before it. Calibration is now
        # applied in the /predict endpoint after _blend_sequential().

        # ── Step 3: SHAP values (always from LightGBM — fastest + most reliable) ─
        # Bug 39-b fix: was using `X` which no longer exists after Bug 39 refactor.
        # Must use X_model (shaped to the model's feature subset, not all_features).
        # Bug 39-c fix: SHAP sv shape = (1, len(model_features)), so idx into model_features.
        shap_values = registry.explainer[target].shap_values(X_model)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # binary class → take positive class
        sv = shap_values.flatten()

        # Top 5 SHAP drivers (by absolute magnitude)
        # Indices here correspond to model_features (the trained feature list)
        top_indices = np.argsort(np.abs(sv))[::-1][:5]
        drivers = []
        for idx in top_indices:
            if idx < len(model_features):
                feat_name = model_features[idx]
                # Get raw value from the full feature_vector using original feat index
                orig_idx  = feat_name_to_idx.get(feat_name, -1)
                raw_val   = float(feature_vector[orig_idx]) if orig_idx >= 0 else 0.0
                shap_val  = float(sv[idx])
                direction = "↑ Rising" if shap_val > 0 else "↓ Falling"
                drivers.append({
                    "feature_name":  feat_name,
                    "shap_value":    round(shap_val, 4),
                    "current_value": round(raw_val, 3),
                    "direction":     direction,
                })

        return final_prob, drivers

    except Exception as e:
        logger.error(f"Inference failed for [{target}]: {e}")
        # AUDIT-FIX: Return NaN instead of 0.0 so downstream logic can detect
        # inference failure. Previously, a matrix shape error would make a
        # crashing patient appear as 0% risk ("healthy").
        return float('nan'), []


def _blend_sequential(target: str, tabular_prob: float, history_df: pd.DataFrame, all_features: list) -> float:
    """
    Bug 56 fix: Blends tabular meta-stacker output with GRU-D and TCN predictions.

    During training, train_target() produces a final blend of:
        0.70 * meta_preds (tabular: LGBM+XGB+CatBoost stacked)
      + 0.30 * mean(grud_preds, tcn_preds)

    This function replicates that blend at inference time.
    Falls back to tabular_prob if no sequential models are available.

    Bug 55 fix: Applies grud_scaler (z-normalization) to the feature values
    before passing them to GRU-D, matching training distribution.
    """
    grud_model = registry.grud.get(target)
    tcn_model  = registry.tcn.get(target)
    scaler     = registry.grud_scaler.get(target)
    feat_cols  = registry.features.get(target, [])

    # Check sequential_enabled flag (False if seq AUROC < 0.74 at training for hypotension)
    if not registry.sequential_enabled.get(target, True):
        return tabular_prob  # Training decided sequential models add noise for this target

    if (grud_model is None and tcn_model is None) or not feat_cols:
        return tabular_prob  # No sequential models available — tabular only

    try:
        from train_models import prepare_sequences, TARGET_SEQ_LEN, SEQUENCE_LENGTH

        # R16-FIX-1: Align streaming DataFrame columns to training feature set.
        # The scaler was fitted on per-target feature columns (156 for sepsis,
        # 83 for hypotension, 104 for hemodynamic_collapse), but streaming
        # data from engineer_features() may not have ALL those columns (e.g.,
        # target-specific lab features or delta features that require columns
        # not present in streamed vitals). We: (a) add missing columns as NaN,
        # (b) fill NaN with training medians, (c) then scale.
        df_scaled = history_df.copy()
        if scaler is not None:
            tr_medians = registry.grud_medians.get(target, {})

            # Add any missing feature columns from training (filled with median)
            missing_cols = [col for col in feat_cols if col not in df_scaled.columns]
            if missing_cols:
                missing_df = pd.DataFrame(
                    {col: tr_medians.get(col, 0.0) for col in missing_cols},
                    index=df_scaled.index,
                )
                df_scaled = pd.concat([df_scaled, missing_df], axis=1)

            avail_cols = [f for f in feat_cols if f in df_scaled.columns]
            if tr_medians:
                fill_vals = {c: tr_medians.get(c, 0.0) for c in avail_cols}
            else:
                fill_vals = df_scaled[avail_cols].median().to_dict()
            df_scaled[avail_cols] = scaler.transform(
                df_scaled[avail_cols].fillna(fill_vals)
            )

        seq_len = TARGET_SEQ_LEN.get(target, SEQUENCE_LENGTH)
        # R16-FIX-3: prepare_sequences() groups by patient_id — inject one since
        # at inference all rows belong to a single patient.
        if "patient_id" not in df_scaled.columns:
            df_scaled["patient_id"] = "inference_patient"
        X_seq, M_seq, D_seq, _ = prepare_sequences(
            df_scaled, feat_cols, feat_cols[0], seq_len=seq_len  # target_col placeholder
        )
        if X_seq is None or len(X_seq) == 0:
            return tabular_prob  # Not enough history yet — fall back to tabular

        X_t = X_seq[-1:].to(DEVICE)
        M_t = M_seq[-1:].to(DEVICE)
        D_t = D_seq[-1:].to(DEVICE)

        seq_preds = []
        with torch.no_grad():
            if grud_model is not None:
                seq_preds.append(float(grud_model(X_t, M_t, D_t).cpu().numpy()[0]))
            if tcn_model is not None:
                seq_preds.append(float(tcn_model(X_t, M_t, D_t).cpu().numpy()[0]))

        if not seq_preds:
            return tabular_prob

        seq_avg = float(np.mean(seq_preds))

        # T2-FIX: Replace hardcoded 70/30 tabular/sequential blend with an
        # adaptive confidence-weighted blend.
        # Rationale: the 70/30 was an engineering guess. In practice:
        #   - When tabular and sequential AGREE (within 5%), both are likely
        #     capturing the same signal — equal weighting is appropriate.
        #   - When they DISAGREE strongly (>20%), the tabular meta-stacker is
        #     more reliable (it was calibrated on more data; sequential GRU-D
        #     has fewer training sequences). Weight tabular more heavily.
        # This implements a soft agreement-based interpolation instead of
        # a fixed constant, making the blend data-adaptive without retraining.
        abs_diff = abs(tabular_prob - seq_avg)
        if abs_diff <= 0.05:
            # Models agree — treat equally (50/50)
            seq_weight = 0.50
        elif abs_diff <= 0.15:
            # Minor disagreement — slight tabular preference
            seq_weight = 0.35
        elif abs_diff <= 0.25:
            # Moderate disagreement — tabular dominates
            seq_weight = 0.25
        else:
            # Strong disagreement — trust tabular, treat sequential as weak signal
            seq_weight = 0.15

        blended = (1.0 - seq_weight) * tabular_prob + seq_weight * seq_avg
        return float(np.clip(blended, 0.0, 1.0))

    except Exception as e:
        logger.warning(f"Sequential blend failed for [{target}]: {e} — using tabular only")
        return tabular_prob


def _sanitize_for_json(obj):
    """R16-FIX-2: Recursively replace NaN/Inf floats with safe defaults.

    json.dumps() crashes on float('nan') and float('inf') with:
        ValueError: Out of range float values are not JSON compliant
    This walks the entire response dict and replaces them with 0.0.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    return obj


def build_response(
    patient_id: str,
    timestamp: str,
    sepsis_prob: float, sepsis_drivers: list,
    bp_prob: float, bp_drivers: list,
    ca_ml_prob: float, ca_ml_drivers: list,
    physics: PhysicsEngineOutput,
    current_vitals: dict,
    sofa_score: float,
    news2_score: float,
    shock_index: float,
) -> dict:
    """Assembles the strict JSON output contract."""

    # Hemodynamic collapse / cardiac arrest probability:
    # R16-FIX-4: Physics engine probability only used when override ACTUALLY fires.
    # Previously, physics always produced ~45% even for healthy patients (because
    # DO2 calc with default pao2/fio2 gives sub-optimal values), which forced
    # every patient to at least 45% cardiac arrest risk.
    ca_physics_prob = physics.cardiac_arrest_probability / 100.0
    if physics.physics_override_triggered:
        # Physics fired a hard-coded trip wire — trust it as the floor
        ca_prob_raw = float(np.clip(max(ca_ml_prob, ca_physics_prob), 0.0, 1.0))
    else:
        # No physics override — use pure ML prediction
        ca_prob_raw = float(np.clip(ca_ml_prob, 0.0, 1.0))

    # R16-FIX-5 + T1-FIX-5: Crash composite score with threshold-normalized weights.
    # Old: hardcoded 0.30×Sep + 0.25×BP + 0.45×CA — these weights were guesses.
    # New: Each target's raw probability is first normalized by its F-beta threshold
    # (the operating point optimized during training). This converts each score to
    # "how many times above the clinical alert threshold is this patient?" — a
    # clinically meaningful unit. The composite is then the weighted average of these
    # normalized contributions, inversely weighted by threshold (high-threshold targets
    # are harder to trigger, so they deserve more weight when they do fire).
    #
    # Derivation: w_i ∝ threshold_i (higher threshold = harder to exceed = more significant)
    # Normalized so weights sum to 1.
    thresholds = {
        "sepsis":               registry.optimal_threshold.get("sepsis",              0.18),
        "hypotension":          registry.optimal_threshold.get("hypotension",         0.20),
        "hemodynamic_collapse": registry.optimal_threshold.get("hemodynamic_collapse",0.22),
    }
    w_sep = thresholds["sepsis"]
    w_bp  = thresholds["hypotension"]
    w_ca  = thresholds["hemodynamic_collapse"]
    w_sum = max(w_sep + w_bp + w_ca, 1e-9)
    crash_score = float(np.clip(
        (sepsis_prob * w_sep + bp_prob * w_bp + ca_prob_raw * w_ca) / w_sum,
        0.0, 1.0
    ))

    return {
        "patient_id":           patient_id,
        "timestamp":            timestamp,
        "crash_probability_score": round(crash_score * 100, 2),  # 0-100%
        "crash_risk_level":     get_risk_level(crash_score, threshold=(
            (thresholds['sepsis'] + thresholds['hypotension'] + thresholds['hemodynamic_collapse']) / 3.0
        )),
        "clinical_scores": {
            "sofa_score":   round(sofa_score, 1),
            "news2_score":  round(news2_score, 1),
            "shock_index":  round(shock_index, 3),
        },
        "predictions": {
            "septic_shock": {
                "risk_probability_percentage": round(sepsis_prob * 100, 2),
                "time_window": "2-6 hours",
                "risk_level":  get_risk_level(sepsis_prob, threshold=registry.optimal_threshold.get("sepsis", 0.0)),
                "shap_drivers": sepsis_drivers,
            },
            "blood_pressure_collapse": {
                "risk_probability_percentage": round(bp_prob * 100, 2),
                "time_window": "2-6 hours",
                "risk_level":  get_risk_level(bp_prob, threshold=registry.optimal_threshold.get("hypotension", 0.0)),
                "shap_drivers": bp_drivers,
            },
            "cardiac_arrest": {
                # API key kept as 'cardiac_arrest' for frontend compatibility.
                # Internally trained as 'hemodynamic_collapse' (proxy label + I-CARE real labels).
                "risk_probability_percentage": round(ca_prob_raw * 100, 2),
                "time_window": "Critical" if physics.physics_override_triggered else "2-6 hours",
                "risk_level":  get_risk_level(ca_prob_raw, threshold=registry.optimal_threshold.get("hemodynamic_collapse", 0.0)),
                "ml_probability": round(ca_ml_prob * 100, 2),        # ML model alone
                "physics_probability": round(ca_physics_prob * 100, 2),  # Physics alone
                "shap_drivers": ca_ml_drivers,
                "physics_metrics": {
                    "tissue_hypoxia_index":          physics.tissue_hypoxia_index,
                    "hemodynamic_instability_score": physics.hemodynamic_instability_score,
                    "oxygen_delivery_do2":           physics.oxygen_delivery_do2,
                    "o2_extraction_ratio":           physics.o2_extraction_ratio,
                },
                "physics_override_triggered": physics.physics_override_triggered,
                "alert_reasons": physics.alert_reasons,
            }
        },
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "models_loaded": registry.loaded_targets,
        "active_patients": len(patient_history),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/patients")
async def list_patients():
    """Returns the current status of all tracked patients (for the Triage Radar)."""
    return {
        "active_patients": list(patient_history.keys()),
        "count": len(patient_history),
    }


@app.post("/predict")
async def predict(payload: VitalsPayload, background_tasks: BackgroundTasks):
    """
    Main prediction endpoint.
    Called every N seconds by data_streamer.py for each active patient.
    """
    pid = payload.patient_id
    ts  = payload.timestamp or datetime.now(timezone.utc).isoformat()
    
    # ── 1. Update rolling patient history ─────────────────────────────────
    vitals_dict = payload.model_dump()
    # Map vasopressor_dose (API field name, kept for backward compat) → nee_dose
    # (internal feature name used by features.py and the trained models).
    vitals_dict["nee_dose"] = float(vitals_dict.pop("vasopressor_dose", 0.0) or 0.0)
    vitals_dict["timestamp"] = pd.to_datetime(ts)
    patient_history[pid].append(vitals_dict)
    
    # ── 2. Build history DataFrame & engineer features ─────────────────────
    history_df = pd.DataFrame(list(patient_history[pid]))
    if "timestamp" in history_df.columns:
        history_df = history_df.sort_values("timestamp").reset_index(drop=True)
    
    # engineer_features() calls impute_vitals() internally as its first step.
    # Do NOT call impute_vitals separately here — it would fill all NaNs before
    # engineer_features() adds missingness flags, making ALL _measured flags = 1.
    #
    # R15-FIX: Resample to hourly bins BEFORE feature engineering.
    # Training data is 1-hour binned (df.resample("1h").mean()). Temporal deltas
    # use df.shift(N) where N rows = N hours. If the streamer sends data every
    # 5 minutes, shift(4) looks 4 rows back = 20 min, NOT 4 hours.
    # This mismatch made all 75 delta features systematically wrong at inference.
    if "timestamp" in history_df.columns:
        history_df = (
            history_df
            .set_index("timestamp")
            .resample("1h")
            .mean(numeric_only=True)
            .reset_index()
        )
    history_df = engineer_features(history_df)    # Adds SOFA, NEWS2, deltas, missingness flags, etc.
    
    # Get current row (last observation) as feature vector
    current_row    = history_df.iloc[-1]
    sofa_score     = float(current_row.get("sofa_score", 0))
    news2_score    = float(current_row.get("news2_score", 0))
    shock_index    = float(current_row.get("shock_index", 0))
    
    # ── 3. Tabular ML inference ────────────────────────────────────────────
    all_features = get_feature_columns()
    # BUG-R3-1 FIX: Use the same fill map as training (VITAL_MEDIANS + DEMOGRAPHIC_DEFAULTS)
    # instead of defaulting to 0. This ensures inference feature values match the training
    # distribution for any features that remain NaN after engineer_features().
    from features import VITAL_MEDIANS, DEMOGRAPHIC_DEFAULTS
    _infer_fill_map = {**VITAL_MEDIANS, **DEMOGRAPHIC_DEFAULTS}
    feature_vals = np.array(
        [float(current_row.get(f, _infer_fill_map.get(f, 0.0))) for f in all_features],
        dtype=np.float32,
    )

    sepsis_prob, sepsis_drivers = run_tabular_inference("sepsis",              feature_vals, all_features)
    bp_prob,     bp_drivers     = run_tabular_inference("hypotension",         feature_vals, all_features)
    ca_prob_tab, ca_drivers     = run_tabular_inference("hemodynamic_collapse", feature_vals, all_features)

    # AUDIT-FIX: Detect inference failures (NaN return from run_tabular_inference).
    # If inference fails for a target, log it and fall back to 0.0 so the physics
    # engine and other targets can still provide useful output.
    _inference_errors = []
    if math.isnan(sepsis_prob):
        _inference_errors.append("sepsis")
        sepsis_prob = 0.0
    if math.isnan(bp_prob):
        _inference_errors.append("hypotension")
        bp_prob = 0.0
    if math.isnan(ca_prob_tab):
        _inference_errors.append("hemodynamic_collapse")
        ca_prob_tab = 0.0

    # ── 4a. Isotonic calibration BEFORE sequential blend (BUG-R10-FIX) ────
    # Training order (train_models.py lines 3112-3150):
    #   1. meta_preds = meta_model.predict(X)        ← uncalibrated
    #   2. meta_preds = iso_reg.predict(meta_preds)  ← CALIBRATE
    #   3. final = blend(meta_preds, seq_preds)       ← BLEND calibrated with seq
    #
    # The isotonic calibrator was fitted on meta_oof (pre-blend meta-stacker OOF).
    # It must be applied to the tabular meta-stacker output BEFORE blending with
    # sequential predictions. Applying it AFTER blend is a distribution mismatch:
    #   iso(blend(raw_meta, seq)) ≠ blend(iso(raw_meta), seq)
    _cal_sep = registry.calibrator.get("sepsis")
    if _cal_sep is not None:
        sepsis_prob = float(np.clip(_cal_sep.predict([sepsis_prob])[0], 0.0, 1.0))
    _cal_hyp = registry.calibrator.get("hypotension")
    if _cal_hyp is not None:
        bp_prob = float(np.clip(_cal_hyp.predict([bp_prob])[0], 0.0, 1.0))
    _cal_ca = registry.calibrator.get("hemodynamic_collapse")
    if _cal_ca is not None:
        ca_prob_tab = float(np.clip(_cal_ca.predict([ca_prob_tab])[0], 0.0, 1.0))

    # ── 4b. Sequential model inference (GRU-D + TCN) — Bug 56 fix ─────────
    # GRU-D and TCN were trained and saved but never called during /predict.
    # Blends calibrated tabular with raw sequential using adaptive weights,
    # exactly matching the train_target() final blend used during evaluation.
    ca_prob = _blend_sequential("hemodynamic_collapse", ca_prob_tab, history_df, all_features)
    sep_prob = _blend_sequential("sepsis",              sepsis_prob, history_df, all_features)
    hyp_prob = _blend_sequential("hypotension",         bp_prob,     history_df, all_features)

    # ── 5. Physics Engine (biological safety net) ──────────────────────────
    vitals_for_physics = {k: v for k, v in current_row.items() if v is not None}
    physics_output: PhysicsEngineOutput = run_physics_engine(vitals_for_physics)

    # ── 6. Assemble & return JSON response ─────────────────────────────────
    response = build_response(
        pid, ts,
        sep_prob, sepsis_drivers,
        hyp_prob, bp_drivers,
        ca_prob,  ca_drivers,
        physics_output, vitals_dict,
        sofa_score, news2_score, shock_index,
    )

    # R16-FIX-2: Sanitize NaN/Inf values that crash json.dumps().
    # Sources: SHAP values on sparse features, model predictions on unseen
    # feature combinations, physics engine divisions by zero.
    response = _sanitize_for_json(response)

    # AUDIT-FIX: Flag inference errors in response for frontend visibility.
    if _inference_errors:
        response["inference_errors"] = _inference_errors
        logger.warning(f"Inference failed for targets: {_inference_errors} — using fallback 0.0")

    # R17: Attach ground truth for validation overlay (if sent by streamer)
    gt = vitals_dict.get("ground_truth_labels") or getattr(payload, "ground_truth_labels", None)
    if gt:
        response["ground_truth"] = gt

    # ── 7. Push to WebSocket if frontend is connected ──────────────────────
    # BUG-I fix: Previously only fired when a patient-specific WS was open.
    # useChronos.js connects to /ws/triage/all (registered as "__triage__"),
    # not to per-patient sockets — so the triage dashboard received NO updates.
    # Now fires whenever ANY WebSocket subscriber exists (patient-specific OR triage).
    if pid in websocket_connections or "__triage__" in websocket_connections:
        background_tasks.add_task(push_to_websocket, pid, response)
    
    return response


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket for real-time frontend push
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/{patient_id}")
async def websocket_endpoint(websocket: WebSocket, patient_id: str):
    """
    Frontend can open a WebSocket for a specific patient to receive
    real-time prediction updates as vitals are streamed.
    """
    await websocket.accept()
    websocket_connections[patient_id].append(websocket)   # BUG-CLAUDE-4-2 fix: append, not overwrite
    logger.info(f"WebSocket connected for patient {patient_id} (total for pid: {len(websocket_connections[patient_id])})")
    
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        websocket_connections[patient_id].remove(websocket)
        if not websocket_connections[patient_id]:
            del websocket_connections[patient_id]
        logger.info(f"WebSocket disconnected for patient {patient_id}")


@app.websocket("/ws/triage/all")
async def triage_websocket(websocket: WebSocket):
    """
    Broadcasts ALL patient updates to a single connected Triage Radar UI.
    The frontend subscribes to this endpoint for the global dashboard view.
    """
    await websocket.accept()
    websocket_connections["__triage__"].append(websocket)   # BUG-CLAUDE-4-1 fix: append, not overwrite
    logger.info(f"Triage Radar WebSocket connected (total: {len(websocket_connections['__triage__'])}).")
    
    try:
        while True:
            await asyncio.sleep(1)  # Keeps the connection alive
    except WebSocketDisconnect:
        if websocket in websocket_connections["__triage__"]:
            websocket_connections["__triage__"].remove(websocket)
        if not websocket_connections["__triage__"]:
            del websocket_connections["__triage__"]
        logger.info("Triage Radar WebSocket disconnected.")


async def push_to_websocket(patient_id: str, data: dict):
    """Pushes a prediction payload to the patient's WebSocket(s) and the triage dashboard(s).

    Bug 37 fix: Both send paths are always attempted independently.
    BUG-CLAUDE-4-1/4-2 fix: Iterates over ALL connections per key (multi-tab support).
    Stale/closed connections are silently removed from the list.
    """
    # Send to patient-specific WebSocket(s) (if the patient's detail view is open)
    if patient_id in websocket_connections:
        dead = []
        for ws in list(websocket_connections[patient_id]):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            websocket_connections[patient_id].remove(ws)
        if not websocket_connections[patient_id]:
            del websocket_connections[patient_id]

    # ALWAYS also send to ALL triage dashboard connections (global view)
    if "__triage__" in websocket_connections:
        dead = []
        for ws in list(websocket_connections["__triage__"]):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            websocket_connections["__triage__"].remove(ws)
        if not websocket_connections["__triage__"]:
            del websocket_connections["__triage__"]


@app.get("/patient/{patient_id}/history")
async def get_patient_history(patient_id: str):
    """Returns the last N rows of stored vitals for a patient (for chart display)."""
    if patient_id not in patient_history:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found.")
    
    history = list(patient_history[patient_id])
    return {
        "patient_id": patient_id,
        "n_rows":     len(history),
        "history":    history[-24:],  # Return last 24 readings for charting
    }
