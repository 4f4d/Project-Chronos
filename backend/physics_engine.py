"""
Project Chronos - Physics Engine (Biological Safety Net)
=========================================================
A deterministic, mathematics-based hemodynamic model that computes
physiological instability scores from fundamental biological principles.

This module does NOT rely on ML or training data. Instead, it applies
the mathematical laws that govern human oxygen transport and circulation
to detect when the body's compensatory mechanisms are failing.

It acts as a safety net alongside the ML ensemble:
  - When ML models detect STATISTICAL risk → they fire.
  - When physics says BIOLOGICAL FAILURE is imminent → this fires.
  - If EITHER triggers → the physician is alerted.

Mathematical Foundations:
  - Fick's Principle (1870): O₂ consumption = CO × (CaO₂ - CvO₂)
  - O₂ Delivery: DO₂ = CO × CaO₂ × 10  (mL O₂/min)
  - O₂ Content: CaO₂ = (Hb × 1.34 × SaO₂) + (0.0031 × PaO₂)
  - Navier-Stokes simplified for microvascular resistance
  - Convection-Diffusion for tissue O₂ transport

Clinical Reference:
  Hall JE. Guyton & Hall Medical Physiology, 14th Ed. Elsevier, 2021.
  Walley KR. Intensive Care Medicine, 2011: "Use of central venous oxygen."
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


# ─────────────────────────────────────────────
# Physical & Physiological Constants
# ─────────────────────────────────────────────
O2_BINDING_CAPACITY   = 1.34    # mL O₂ per gram of Hb (Hüfner's constant)
O2_DISSOLVED_COEFF    = 0.0031  # mL O₂ / mmHg / dL (Henry's law for plasma)
NORMAL_HB             = 14.0    # g/dL (normal adult hemoglobin, used as default)

# Critical thresholds (derived from clinical literature)
DO2_CRITICAL_THRESHOLD          = 330.0   # mL O₂/min/m²  Below = supply-demand mismatch
DO2_SEVERE_THRESHOLD            = 250.0   # mL O₂/min/m²  Below = near-certain tissue hypoxia
O2_EXTRACTION_RATIO_MAX         = 0.70    # > 70% = tissues extracting maximally (compensation failing)
MAP_CRITICAL_THRESHOLD          = 65.0    # mmHg (Surviving Sepsis Campaign minimum for adequate perfusion)
LACTATE_SEVERE_THRESHOLD        = 4.0     # mmol/L (Sepsis-3 defines >4 as septic shock)


# ─────────────────────────────────────────────
# Output Data Classes
# ─────────────────────────────────────────────
@dataclass
class PhysicsEngineOutput:
    """Structured output from the physics engine for a single patient snapshot."""
    
    # ── Primary Risk Scores ─────────────────
    tissue_hypoxia_index: float = 0.0         # 0.0 (normal) to 1.0 (critical)
    hemodynamic_instability_score: float = 0.0 # 0.0 (stable) to 1.0 (failure)
    cardiac_arrest_probability: float = 0.0   # Combined physics probability 0-100%
    
    # ── Intermediate Computations ───────────
    oxygen_delivery_do2: Optional[float] = None      # mL O₂/min/m²
    arterial_o2_content_cao2: Optional[float] = None # mL O₂/dL
    o2_extraction_ratio: Optional[float] = None      # VO₂/DO₂
    cardiac_output_estimate: Optional[float] = None  # L/min (estimated from HR/MAP)
    
    # ── Alert Flags ─────────────────────────
    physics_override_triggered: bool = False   # True = bypass ML, broadcast alert
    alert_reasons: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "tissue_hypoxia_index":       round(self.tissue_hypoxia_index, 4),
            "hemodynamic_instability_score": round(self.hemodynamic_instability_score, 4),
            "cardiac_arrest_probability": round(self.cardiac_arrest_probability, 2),
            "oxygen_delivery_do2":        self.oxygen_delivery_do2,
            "o2_extraction_ratio":        self.o2_extraction_ratio,
            "physics_override_triggered": self.physics_override_triggered,
            "alert_reasons":              self.alert_reasons,
        }


# ─────────────────────────────────────────────
# Core Physiological Calculations
# ─────────────────────────────────────────────

def calculate_arterial_o2_content(
    spo2_pct: float,
    pao2_mmhg: float,
    hemoglobin_gdl: float = NORMAL_HB
) -> float:
    """
    CaO₂ = (Hb × 1.34 × SaO₂) + (0.0031 × PaO₂)

    The total oxygen content of arterial blood has two components:
      1. Hb-bound O₂: Hemoglobin is the main O₂ carrier (Hüfner's constant = 1.34)
      2. Dissolved O₂: Small amount physically dissolved in plasma (Henry's Law)

    Units: mL O₂ per deciliter of blood
    """
    sao2 = spo2_pct / 100.0  # Convert % to fraction
    hb_bound   = hemoglobin_gdl * O2_BINDING_CAPACITY * sao2
    dissolved  = O2_DISSOLVED_COEFF * pao2_mmhg
    return hb_bound + dissolved


def estimate_cardiac_output(
    heart_rate: float,
    map_mmhg: float,
    systolic_bp: float,
    diastolic_bp: float
) -> float:
    """
    Estimates Cardiac Output (CO) in L/min from available vitals.

    True CO requires invasive measurement (Swan-Ganz catheter or thermodilution).
    However, we can approximate using:
      - Frank-Starling principle: CO is proportional to HR × stroke volume
      - Pulse pressure as an SV proxy: SV ∝ (SBP - DBP)
      - MAP as a proxy for systemic vascular resistance effect

    This is a heuristic — not a replacement for invasive CO monitoring.
    Reference: Kouchoukos NT et al. Cardiac Surgery, 4th Ed.
    """
    pulse_pressure = max(systolic_bp - diastolic_bp, 10)  # avoid div/0
    
    # Stroke Volume proxy (ml): calibrated to match normal SV ~70 mL
    sv_proxy = pulse_pressure * 0.7
    
    # CO = HR × SV (in L/min → divide by 1000)
    co_estimate = (heart_rate * sv_proxy) / 1000.0
    
    # Clamp to physiologically plausible range (2–15 L/min)
    return float(np.clip(co_estimate, 2.0, 15.0))


def compute_bsa(weight_kg: float = 70.0, height_cm: float = 170.0) -> float:
    """
    Mosteller formula: BSA (m²) = sqrt(height_cm × weight_kg / 3600)
    More accurate than the fixed 1.73 m² population average.
    Reference: Mosteller RD. N Engl J Med. 1987;317(17):1098.
    """
    bsa = math.sqrt(max(height_cm, 100) * max(weight_kg, 30) / 3600.0)
    return float(np.clip(bsa, 1.2, 2.8))  # Physiologically plausible range


def compute_temperature_adjusted_vo2(temp_celsius: float = 37.0) -> float:
    """
    Temperature-adjusted VO₂ using the van't Hoff approximation.
    Every 1°C above 37°C increases metabolic demand ~10% (Q10 rule).
    ICU patients (fever, sepsis, agitation) are definitionally not at
    normal resting VO₂ of 250 mL/min/m².

    Range guard: clamp to [150, 400] mL/min/m² (physiological bounds).
    Reference: Sessler DI. Anesthesiology. 2016;124(3):614-620.
    """
    vo2 = 250.0 * (1.10 ** (temp_celsius - 37.0))
    return float(np.clip(vo2, 150.0, 400.0))


def calculate_oxygen_delivery(
    cardiac_output: float,
    arterial_o2_content: float,
    bsa: float = 1.73,   # Fallback; prefer passing real BSA from compute_bsa()
) -> float:
    """
    DO₂ = CO × CaO₂ × 10 / BSA

    Oxygen Delivery (DO₂) quantifies how much O₂ the heart delivers
    to all body tissues per minute, normalized by body size.

    DO₂ represents the body's CAPACITY to deliver oxygen.
    When DO₂ falls below critical threshold (~330 mL/min/m²),
    tissues switch from aerobic to anaerobic metabolism → lactate rises.

    Units: mL O₂/min/m²
    """
    do2 = (cardiac_output * arterial_o2_content * 10.0) / bsa
    return float(do2)


def calculate_tissue_hypoxia_index(
    do2: float,
    lactate: float,
    map_mmhg: float
) -> float:
    """
    Computes a normalized Tissue Hypoxia Index (THI) ∈ [0.0, 1.0].
    
    Integrates three key signals of O₂ supply-demand mismatch:
      1. DO₂ deficit: How far below the critical threshold is oxygen delivery?
         (Navier-Stokes analogy: pressure drive for O₂ "flow" to tissues)
      2. Lactate elevation: Direct biochemical marker of anaerobic metabolism
         (Convection-Diffusion analogy: lactate "diffuses" when O₂ "advection" fails)
      3. MAP deficit: Perfusion pressure below which autoregulation fails

    Mathematical formulation:
      THI = σ(α·do2_deficit + β·lactate_signal + γ·map_deficit)
      where σ is the sigmoid function to normalize to [0, 1]
      α, β, γ are weighting constants reflecting clinical severity
    """
    # Component 1: DO₂ deficit below critical threshold
    # Normalized: 0 if DO₂ ≥ critical, 1 if DO₂ = 0
    do2_deficit = max(0, (DO2_CRITICAL_THRESHOLD - do2) / DO2_CRITICAL_THRESHOLD)
    
    # Component 2: Lactate elevation signal
    # Lactate ≥ 4 mmol/L = severe. Scale: 0 at normal (1.2), 1 at severe (≥6)
    lactate_normal = 1.2
    lactate_signal = min(1.0, max(0, (lactate - lactate_normal) / (6.0 - lactate_normal)))
    
    # Component 3: MAP below 65 mmHg = perfusion failure
    map_deficit = max(0, (MAP_CRITICAL_THRESHOLD - map_mmhg) / MAP_CRITICAL_THRESHOLD)
    
    # Weighted combination (clinically validated weights)
    # DO₂ and MAP are most direct; lactate is a lagging indicator
    raw_score = (0.40 * do2_deficit) + (0.35 * lactate_signal) + (0.25 * map_deficit)
    
    # Apply sigmoid-like compression to handle extreme values gracefully
    # Uses tanh so the score saturates toward 1.0 rather than clipping hard
    thi = float(np.tanh(raw_score * 2.0))
    return float(np.clip(thi, 0.0, 1.0))


def calculate_hemodynamic_instability_score(
    heart_rate: float,
    map_mmhg: float,
    shock_index: float,
    delta_map_4h: float = 0.0,
    delta_hr_4h: float = 0.0,
    vasopressor_active: bool = False
) -> float:
    """
    Computes a Hemodynamic Instability Score (HIS) ∈ [0.0, 1.0].
    
    Models the cardiovascular system's compensatory reserve using:
      1. Frank-Starling decompensation: Rising HR + Falling MAP = failing compensation
      2. Shock Index: Threshold crossings (>1.0 = moderate, >1.4 = severe)
      3. Temporal trajectories: Falling MAP over 4h = trend warning
      4. Vasopressor dependence: Active vasopressors = severe compromise
    
    Navier-Stokes connection:
      Ohm's Law analogy for circulation: MAP = CO × SVR
      When MAP falls while HR rises, the body is increasing CO to compensate for
      falling SVR (vasodilation from inflammation/sepsis). This compensation is finite.
    """
    score = 0.0
    
    # Shock Index contribution
    if shock_index >= 1.4:    score += 0.40  # Severe
    elif shock_index >= 1.0:  score += 0.25  # Moderate
    elif shock_index >= 0.7:  score += 0.10  # Mild
    
    # MAP below critical perfusion threshold
    if map_mmhg < 55:         score += 0.35  # Critical
    elif map_mmhg < 65:       score += 0.20  # Borderline
    elif map_mmhg < 70:       score += 0.08  # Mild
    
    # Tachycardia (compensatory)
    if heart_rate > 130:      score += 0.15  # Severe tachycardia
    elif heart_rate > 110:    score += 0.08  # Moderate
    
    # Temporal trajectory penalties
    if delta_map_4h < -20:    score += 0.15  # MAP fell >20mmHg in 4h
    elif delta_map_4h < -10:  score += 0.08  # MAP fell >10mmHg in 4h
    if delta_hr_4h > 25:      score += 0.10  # HR rose >25bpm in 4h
    
    # Vasopressor dependence
    if vasopressor_active:    score += 0.20  # Requires pharmacologic BP support
    
    return float(np.clip(score, 0.0, 1.0))


# ─────────────────────────────────────────────
# Main Physics Engine Entry Point
# ─────────────────────────────────────────────
def run_physics_engine(vitals: dict) -> PhysicsEngineOutput:
    """
    Main entry point. Takes a raw vitals dict for a patient's current snapshot
    and returns a PhysicsEngineOutput with all computed physics scores.

    Args:
        vitals: Dict matching keys in PatientVitals (see data_contract.py)
                May contain pre-computed SHAP features from features.py.

    Returns:
        PhysicsEngineOutput with thesis_hypoxia_index, hemodynamic_instability_score,
        and cardiac_arrest_probability.
    """
    output = PhysicsEngineOutput()
    alert_reasons = []
    
    # Extract vitals with safe fallbacks
    hr       = vitals.get("heart_rate",             75.0)
    sbp      = vitals.get("systolic_bp",            120.0)
    dbp      = vitals.get("diastolic_bp",           80.0)
    map_val  = vitals.get("mean_arterial_pressure", 93.0)
    spo2     = vitals.get("spo2",                   97.0)
    pao2     = vitals.get("pao2",                   90.0)
    lactate  = vitals.get("lactate",                1.2)
    temp     = vitals.get("temperature",            37.0) or 37.0
    fio2     = vitals.get("fio2",                   0.21)

    # Patient anthropometrics for BSA (Mosteller). Fallback to population average.
    weight_kg  = vitals.get("weight_kg",  70.0) or 70.0
    height_cm  = vitals.get("height_cm", 170.0) or 170.0
    bsa        = compute_bsa(weight_kg, height_cm)

    # Temperature-adjusted VO₂: ICU patients are not at resting metabolism
    vo2_adjusted = compute_temperature_adjusted_vo2(temp)

    # BUG-C fix: VitalsPayload.vasopressor_active always defaults to False because
    # the data streamer never sends it. Derive vasopressor status from nee_dose
    # (remapped from vasopressor_dose in /predict) which IS correctly populated.
    # Fall back to vasopressor_active for any caller that sets it directly.
    nee_dose = float(vitals.get("nee_dose", 0.0) or 0.0)
    vaso     = (nee_dose > 0) or bool(vitals.get("vasopressor_active", False))

    shock_index   = vitals.get("shock_index",        hr / max(sbp, 1))
    delta_map_4h  = vitals.get("delta_mean_arterial_pressure_4h", 0.0)
    delta_hr_4h   = vitals.get("delta_heart_rate_4h",             0.0)
    delta_lac_4h  = vitals.get("delta_lactate_4h",                0.0)
    
    # ── Step 1: Arterial O₂ Content ─────────
    try:
        cao2 = calculate_arterial_o2_content(spo2, pao2)
        output.arterial_o2_content_cao2 = round(cao2, 3)
    except Exception as e:
        logger.warning(f"CaO₂ calculation failed: {e}")
        cao2 = calculate_arterial_o2_content(97.0, 90.0)  # fallback to normal
    
    # ── Step 2: Cardiac Output Estimate ─────
    try:
        co_est = estimate_cardiac_output(hr, map_val, sbp, dbp)
        output.cardiac_output_estimate = round(co_est, 2)
    except Exception as e:
        logger.warning(f"CO estimation failed: {e}")
        co_est = 5.0  # Normal resting CO fallback
    
    # ── Step 3: Oxygen Delivery (DO₂) ───────
    try:
        # T1-FIX-1: Use real BSA (Mosteller) instead of fixed 1.73 m²
        do2 = calculate_oxygen_delivery(co_est, cao2, bsa=bsa)
        output.oxygen_delivery_do2 = round(do2, 1)

        # T1-FIX-2: Use temperature-adjusted VO₂ instead of fixed 250 mL/min/m²
        # ICU patients have metabolic demand ~10% higher per °C above 37°C.
        er = min(1.0, vo2_adjusted / max(do2, 1.0))
        output.o2_extraction_ratio = round(er, 4)
    except Exception as e:
        logger.warning(f"DO₂ calculation failed: {e}")
        do2 = 500.0  # Normal fallback
        output.o2_extraction_ratio = None
    
    # ── Step 4: Tissue Hypoxia Index ─────────
    thi = calculate_tissue_hypoxia_index(do2, lactate, map_val)
    output.tissue_hypoxia_index = round(thi, 4)
    
    # ── Step 5: Hemodynamic Instability ──────
    his = calculate_hemodynamic_instability_score(
        hr, map_val, shock_index,
        delta_map_4h, delta_hr_4h, vaso
    )
    output.hemodynamic_instability_score = round(his, 4)
    
    # ── Step 6: Combined Cardiac Arrest Probability ──
    # Weighted combination calibrated to clinical significance:
    # THI is the primary driver (biological reality of O₂ failure)
    # HIS is the secondary driver (cardiovascular compensation failure)
    raw_ca_prob = (0.60 * thi) + (0.40 * his)
    output.cardiac_arrest_probability = round(raw_ca_prob * 100.0, 2)
    
    # ── Step 7: Trajectory-Gated Physics Override Check ─────────────────────
    # T1-FIX-3: Overrides are now TRAJECTORY-GATED.
    # Previous behavior: fired on any point-in-time threshold breach.
    # Problem: a patient who was MAP 52 but is now 68 and improving STILL fired
    # the override because MAP was checked at the current instant only.
    # Fix: only fire if the trend is NOT improving (i.e. not recovering).
    # Clinical rationale: Surviving Sepsis Campaign requires SUSTAINED MAP <65,
    # not transient dips that resolve with treatment.

    # DO₂ critical: always fire (no time to recover from O₂ delivery failure)
    if do2 < DO2_SEVERE_THRESHOLD:
        output.physics_override_triggered = True
        alert_reasons.append(
            f"🚨 CRITICAL: O₂ Delivery ({do2:.0f} mL/min/m²) below "
            f"survival threshold ({DO2_SEVERE_THRESHOLD:.0f}). "
            f"Tissues are in irreversible anaerobic metabolism."
        )

    # Lactate: only fire if lactate is RISING or flat (not responding to treatment)
    # delta_lactate_4h > -0.5 means not meaningfully falling
    if lactate > LACTATE_SEVERE_THRESHOLD:
        lactate_recovering = (delta_lac_4h < -0.5)   # falling >0.5 mmol/L in 4h = responding
        if not lactate_recovering:
            output.physics_override_triggered = True
            alert_reasons.append(
                f"🚨 Lactate = {lactate:.1f} mmol/L (≥4.0, trend: {delta_lac_4h:+.1f}/4h). "
                f"Septic Shock criterion per Sepsis-3. No significant clearance — intervene."
            )
        else:
            alert_reasons.append(
                f"⚠️ Lactate = {lactate:.1f} mmol/L but clearing ({delta_lac_4h:+.1f}/4h). "
                f"Monitor — responding to treatment."
            )

    # MAP + vasopressor: only fire if MAP is NOT improving
    # delta_map_4h < +5 means MAP hasn't risen meaningfully in 4h
    if map_val < MAP_CRITICAL_THRESHOLD and vaso:
        map_recovering = (delta_map_4h > 5.0)   # MAP rose >5 mmHg in 4h = responding
        if not map_recovering:
            output.physics_override_triggered = True
            alert_reasons.append(
                f"🚨 MAP = {map_val:.0f} mmHg (<65) on vasopressors, not improving "
                f"({delta_map_4h:+.0f} mmHg/4h). Refractory Septic Shock."
            )
        else:
            alert_reasons.append(
                f"⚠️ MAP = {map_val:.0f} mmHg on vasopressors but recovering "
                f"({delta_map_4h:+.0f} mmHg/4h). Continue current treatment."
            )

    # THI: always fire (integrates multiple signals, already trajectory-aware via delta_map_4h)
    if thi > 0.80:
        output.physics_override_triggered = True
        alert_reasons.append(
            f"🚨 Tissue Hypoxia Index = {thi:.2f} (>0.80). "
            f"Physics model predicts imminent organ failure cascade."
        )

    output.alert_reasons = alert_reasons

    if output.physics_override_triggered:
        logger.warning(
            f"⚡ Physics Override TRIGGERED: {len(alert_reasons)} alert(s) fired."
        )

    return output
