"""
Project Chronos - Clinical Feature Engineering
================================================
Computes all derived clinical features from raw vitals before ML inference.

Layer responsibilities:
  - Rolling window temporal deltas  (trend detection)
  - Clinical scoring: SOFA, NEWS2, Shock Index
  - MAP/Lactate ratio (DO2 proxy)
  - Missing data imputation (forward-fill + median fallback)
  - Normalization for neural models

Clinical references:
  - SOFA Score: Singer et al. JAMA 2016 (Sepsis-3 definition)
  - NEWS2: Royal College of Physicians, 2017
  - Shock Index: Allgöwer & Burri, 1967 (HR / SBP)
  - DO2 (Oxygen Delivery): Hall JE, Guyton & Hall Medical Physiology, 13th ed.
"""

import numpy as np
import pandas as pd
from typing import Optional
from loguru import logger


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
# PaO2/FiO2 ratio breakpoints used in the SOFA respiratory sub-score
SOFA_PF_THRESHOLDS = [400, 300, 200, 100]

# NEWS2 scoring tables (Royal College of Physicians)
NEWS2_RR_BREAKPOINTS   = [8, 11, 20, 24]    # Respiratory Rate breaths/min
NEWS2_SPO2_BREAKPOINTS = [91, 93, 95]       # SpO2 %
NEWS2_SBP_BREAKPOINTS  = [90, 100, 110, 219]  # Systolic BP mmHg
NEWS2_HR_BREAKPOINTS   = [40, 50, 90, 110, 130]   # HR bpm
NEWS2_TEMP_BREAKPOINTS = [35.0, 36.0, 38.0, 39.0]  # Temperature Celsius


# ─────────────────────────────────────────────
# Missing Data Imputation
# ─────────────────────────────────────────────
VITAL_MEDIANS = {
    # Population medians used as fallback when forward-fill fails (too little history)
    # ── Monitor vitals (continuous) ──────────────────────────────────────────
    "heart_rate":             75.0,
    "systolic_bp":            120.0,
    "diastolic_bp":           80.0,
    "mean_arterial_pressure": 93.0,
    "spo2":                   97.0,
    "respiratory_rate":       16.0,
    "temperature":            37.0,
    # ── Lab vitals (intermittent) — original 8 ──────────────────────────────
    "lactate":                1.2,
    "wbc":                    9.0,
    "creatinine":             1.0,
    "bilirubin":              0.8,
    "platelets":              200.0,
    "pao2":                   90.0,
    "fio2":                   0.21,
    "gcs":                    15,
    "nee_dose":               0.0,
    # ── NEW CinC 2019 lab features (25 previously unused) ───────────────────
    # These are already in the .psv files on disk but were never loaded.
    # Sources: CinC 2019 challenge dataset (physionet.org), winner used BUN/Cr ratio.
    "etco2":                  35.0,    # End-tidal CO2 — ventilation efficiency
    "base_excess":            0.0,     # Acid-base status — key sepsis marker (NIH)
    "hco3":                   24.0,    # Bicarbonate — metabolic acidosis
    "ph":                     7.40,    # Blood pH — CRITICAL for sepsis
    "paco2":                  40.0,    # Arterial CO2 — respiratory failure
    "sao2":                   97.0,    # Arterial O2 saturation (invasive)
    "ast":                    25.0,    # Liver function — SOFA component (NIH)
    "bun":                    15.0,    # Blood urea nitrogen — kidney function
    "alkaline_phosphatase":   70.0,    # Liver/bone marker
    "calcium":                9.0,     # Cardiac rhythm stability
    "chloride":               102.0,   # Electrolyte balance (Anion Gap calc)
    "bilirubin_direct":       0.2,     # Direct bilirubin — liver subtype
    "glucose":                110.0,   # Metabolic state — stress hyperglycemia
    "magnesium":              2.0,     # Cardiac arrhythmia risk
    "phosphate":              3.5,     # Renal and metabolic
    "potassium":              4.0,     # K+>6 = lethal cardiac arrest risk
    "troponin_i":             0.04,    # Myocardial damage — cardiac arrest
    "hematocrit":             35.0,    # O2 carrying capacity
    "hemoglobin":             12.0,    # DO₂ calc — was hardcoded, now real data
    "ptt":                    30.0,    # Coagulation — DIC risk
    "fibrinogen":             300.0,   # Coagulation cascade
    # ── NEW: Features unlocked by eICU lab.csv + patient.csv loading ──────────
    "sodium":                  140.0,   # Electrolyte — hyponatremia = ICU mortality
    "albumin":                 3.5,     # Capillary leak marker; LAR denominator (NIH)
    "alt":                     25.0,    # Liver function (ALT — pairs with AST)
    "cvp":                     8.0,     # Central Venous Pressure — preload signal
    "pt_inr":                  1.0,     # Coagulation cascade — DIC screening
    "anion_gap_lab":           12.0,    # Pre-computed AG from lab (11,551 rows in eICU)
}

# Demographics — static per patient, NOT forward-filled in the same way.
# Included as features but not in VITAL_MEDIANS to avoid ffill confusion.
# Imputed separately in engineer_features() if missing.
DEMOGRAPHIC_DEFAULTS = {
    "age":                    65.0,    # CinC 2019 median ICU age
    "gender":                 0.5,     # 0=F, 1=M — 0.5 if unknown
    "hosp_adm_time":          -24.0,   # Hours before ICU admission (negative)
    "iculos":                 12,      # ICU length of stay at current row (hours)
    "admission_weight":       80.0,    # Patient admission weight (kg)
}

# ─────────────────────────────────────────────────────────────────────────────
# Lab vitals — intermittent (ordered tests): missingness is clinically meaningful
# Monitor vitals — continuous (always present): forward-fill is correct
# ─────────────────────────────────────────────────────────────────────────────
LAB_VITALS = [
    "lactate", "wbc", "creatinine", "bilirubin", "platelets", "pao2", "fio2", "gcs",
    "nee_dose",
    # NEW CinC labs + eICU labs — missingness flags generated automatically
    "etco2", "base_excess", "hco3", "ph", "paco2", "sao2",
    "ast", "bun", "alkaline_phosphatase", "calcium", "chloride",
    "bilirubin_direct", "glucose", "magnesium", "phosphate", "potassium",
    "troponin_i", "hematocrit", "hemoglobin", "ptt", "fibrinogen",
    # NEW eICU-specific labs (loaded from lab.csv)
    "sodium", "albumin", "alt", "cvp", "pt_inr", "anion_gap_lab",
]
MONITOR_VITALS = [
    "heart_rate", "systolic_bp", "diastolic_bp", "mean_arterial_pressure",
    "spo2", "respiratory_rate", "temperature"
]


def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds binary indicator flags for lab vitals BEFORE imputation.

    Research basis: A NIH 2023 study found that lab missingness indicators
    constituted >40% of top predictors in ICU mortality models. The absence
    of a lactate or creatinine order is itself a clinical signal — the clinician
    decided the patient was stable enough not to order the test.

    Flag convention:
      {lab}_measured = 1  → lab was actually recorded this hour
      {lab}_measured = 0  → lab was missing (forward-fill will impute the value)

    These flags enable LightGBM/XGBoost to learn from both the imputed value
    AND whether the lab was actually measured, independently.
    """
    df = df.copy()
    # FIX: Batch-add all missingness flag columns at once to avoid DataFrame
    # fragmentation (36 labs × per-patient = massive PerformanceWarning spam).
    flag_data = {}
    for lab in LAB_VITALS:
        if lab in df.columns:
            flag_data[f"{lab}_measured"] = df[lab].notna().astype(np.float32)
    if flag_data:
        df = pd.concat([df, pd.DataFrame(flag_data, index=df.index)], axis=1)
    return df


def impute_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-stage imputation strategy:

    Stage 1 — Lab vitals (lactate, WBC, creatinine, etc.):
      Add missingness indicator flags FIRST (before imputation destroys the NaN signal)
      Then forward-fill + median fallback (value still needed for SOFA/scores)

    Stage 2 — Monitor vitals (HR, BP, SpO2, RR, Temp):
      Forward-fill only (continuous monitors hold last reading — no clinical gap)
      Median fallback for patients with zero history

    Clinical rationale:
      Monitors: ICU monitors physically display the last reading until a new
        measurement arrives. Forward-filling replicates this correctly.
      Labs: Lab tests are ordered on clinical judgment. A missing lactate at
        14:00 means 'clinician did not order it' — a real clinical signal.
    """
    df = df.copy()

    # Stage 1: add missingness flags for labs BEFORE filling NaNs
    df = add_missingness_flags(df)

    # Stage 2: impute all vitals (labs get flags above, monitors skip flags)
    for col, median_val in VITAL_MEDIANS.items():
        if col in df.columns:
            df[col] = df[col].ffill().fillna(median_val)
    return df


# ─────────────────────────────────────────────
# SOFA Score (Sequential Organ Failure Assessment)
# ─────────────────────────────────────────────
def compute_sofa_score(row: pd.Series) -> float:
    """
    Computes an approximated SOFA score from available vitals.
    Maximum = 24 (higher = worse organ failure).

    SOFA sub-scores:
      Respiratory  (0-4): PaO2/FiO2 ratio breakpoints
      Coagulation  (0-4): Platelet count
      Liver        (0-4): Bilirubin
      Cardiovascular(0-4): MAP + vasopressor status
      CNS          (0-4): Glasgow Coma Scale
      Renal        (0-4): Creatinine

    Sepsis-3 definition: SOFA ≥ 2 from baseline = sepsis.
    A rise of ≥ 2 points in 4 hours is the clinical red flag.
    """
    score = 0

    # Respiratory sub-score (PaO2/FiO2 ratio)
    pao2  = row.get("pao2", np.nan)
    fio2  = row.get("fio2", 0.21)
    if not np.isnan(pao2) and fio2 > 0:
        pf_ratio = pao2 / fio2
        if pf_ratio < 100:    score += 4
        elif pf_ratio < 200:  score += 3
        elif pf_ratio < 300:  score += 2
        elif pf_ratio < 400:  score += 1

    # Coagulation sub-score (platelets x10^9/L)
    platelets = row.get("platelets", np.nan)
    if not np.isnan(platelets):
        if platelets < 20:    score += 4
        elif platelets < 50:  score += 3
        elif platelets < 100: score += 2
        elif platelets < 150: score += 1

    # Liver sub-score (bilirubin mg/dL)
    bili = row.get("bilirubin", np.nan)
    if not np.isnan(bili):
        if bili >= 12.0:      score += 4
        elif bili >= 6.0:     score += 3
        elif bili >= 2.0:     score += 2
        elif bili >= 1.2:     score += 1

    # Cardiovascular sub-score — full 4-tier grading (Singer et al., JAMA 2016 Table 1)
    # AUDIT-FIX: Previously skipped SOFA=2 tier (dobutamine any dose / dopamine ≤5 mcg/kg/min).
    # NEE = norepinephrine-equivalent dose in mcg/kg/min. Does NOT capture dobutamine inotropes
    # (dobutamine is not vasopressor) — those receive SOFA=2 via the dobutamine flag separately.
    # NE-equivalent thresholds:
    #   dopamine > 15 OR NE/epi > 0.1  → refractory shock          → 4 pts
    #   dopamine 5–15 OR NE/epi ≤ 0.1 → significant pressor support → 3 pts
    #   dopamine ≤ 5 OR dobutamine any → low-dose pressor/inotrope  → 2 pts
    #   MAP < 70, no pressor           → hypotension without support  → 1 pt
    map_val    = row.get("mean_arterial_pressure", np.nan)
    nee        = float(row.get("nee_dose", 0.0) or 0.0)   # mcg/kg/min NE-equivalent
    dobutamine = float(row.get("dobutamine", 0.0) or 0.0)  # binary flag or dose
    if not np.isnan(map_val) or nee > 0 or dobutamine > 0:
        if nee > 0.1:                                          score += 4   # High-dose NE/epi
        elif nee > 0.05:                                       score += 3   # Moderate pressor (≈ dopamine 5-15)
        elif nee > 0.0 or dobutamine > 0:                     score += 2   # Low-dose pressor OR any dobutamine
        elif not np.isnan(map_val) and map_val < 70:          score += 1   # Hypotension, no pressors

    # CNS sub-score (GCS)
    gcs = row.get("gcs", np.nan)
    if not np.isnan(gcs):
        if gcs < 6:           score += 4
        elif gcs < 10:        score += 3
        elif gcs < 13:        score += 2
        elif gcs < 15:        score += 1

    # Renal sub-score (creatinine mg/dL)
    creat = row.get("creatinine", np.nan)
    if not np.isnan(creat):
        if creat >= 5.0:      score += 4
        elif creat >= 3.5:    score += 3
        elif creat >= 2.0:    score += 2
        elif creat >= 1.2:    score += 1

    return float(score)


# ─────────────────────────────────────────────
# NEWS2 Score (National Early Warning Score 2)
# ─────────────────────────────────────────────
def compute_news2_score(row: pd.Series) -> float:
    """
    Computes NEWS2 score (0-20, higher = worse).
    NICE-endorsed composite early warning score validated for sepsis detection.

    Score ≥ 7 = urgent clinical review and likely ICU admission.
    Score 5-6 = half-hourly monitoring.
    Score 1-4 = 4-6 hour monitoring.
    """
    score = 0

    # Respiratory Rate
    rr = row.get("respiratory_rate", np.nan)
    if not np.isnan(rr):
        if rr <= 8:          score += 3
        elif rr <= 11:       score += 1
        elif rr <= 20:       score += 0
        elif rr <= 24:       score += 2
        else:                score += 3

    # SpO2 (without supplemental O2)
    spo2 = row.get("spo2", np.nan)
    if not np.isnan(spo2):
        if spo2 <= 91:       score += 3
        elif spo2 <= 93:     score += 2
        elif spo2 <= 95:     score += 1
        else:                score += 0

    # Supplemental O₂ — NEWS2 awards +2 for ANY supplemental oxygen, not just
    # mechanical ventilation. FiO₂ > 0.21 (room air) = supplemental O₂ is active.
    # GEMINI-AUDIT-FIX: Previously only checked on_mechanical_ventilation.
    vented = row.get("on_mechanical_ventilation", False)
    fio2_val = row.get("fio2", 0.21)
    on_supplemental_o2 = bool(vented) or (isinstance(fio2_val, (int, float)) and not np.isnan(fio2_val) and fio2_val > 0.21)
    if on_supplemental_o2:   score += 2

    # Systolic BP
    sbp = row.get("systolic_bp", np.nan)
    if not np.isnan(sbp):
        if sbp <= 90:        score += 3
        elif sbp <= 100:     score += 2
        elif sbp <= 110:     score += 1
        elif sbp <= 219:     score += 0
        else:                score += 3  # Hypertensive crisis also flagged

    # Heart Rate
    hr = row.get("heart_rate", np.nan)
    if not np.isnan(hr):
        if hr <= 40:         score += 3
        elif hr <= 50:       score += 1
        elif hr <= 90:       score += 0
        elif hr <= 110:      score += 1
        elif hr <= 130:      score += 2
        else:                score += 3

    # Temperature
    temp = row.get("temperature", np.nan)
    if not np.isnan(temp):
        if temp <= 35.0:     score += 3
        elif temp <= 36.0:   score += 1
        elif temp <= 38.0:   score += 0
        elif temp <= 39.0:   score += 1
        else:                score += 2

    # Consciousness (derived from GCS)
    gcs = row.get("gcs", 15)
    if gcs < 15:             score += 3  # Any alteration from fully alert

    return float(score)


# ─────────────────────────────────────────────
# Derived Clinical Ratios
# ─────────────────────────────────────────────
def compute_shock_index(row: pd.Series) -> float:
    """
    Shock Index = HR / SBP
    
    Clinical significance:
      < 0.6 = Normal hemodynamic state
      0.6-0.9 = Mild cardiovascular stress
      1.0-1.4 = Significant hemodynamic compromise  
      > 1.4 = Severe shock → high cardiac arrest risk 
    
    Originally described by Allgöwer & Burri (1967).
    Validated in multiple ICU studies as an early predictor of hemodynamic collapse.
    """
    hr  = row.get("heart_rate",  75.0)
    sbp = row.get("systolic_bp", 120.0)
    if sbp <= 0 or np.isnan(sbp):
        return np.nan
    return float(hr / sbp)


def compute_map_lactate_ratio(row: pd.Series) -> float:
    """
    MAP / Lactate Ratio — a proxy for tissue oxygen utilization efficiency.
    
    MAP (Mean Arterial Pressure) drives oxygen delivery (perfusion pressure).
    Lactate is a direct marker of anaerobic metabolism from insufficient O2 delivery.
    
    A falling MAP combined with a rising lactate is the textbook precursor
    to septic shock and multi-organ failure. This ratio captures that dynamic:
      High ratio (>50) = Good perfusion, efficient O2 delivery
      Low ratio (<20)  = Critical: poor perfusion AND anaerobic metabolism
    """
    map_val = row.get("mean_arterial_pressure", np.nan)
    lactate = row.get("lactate", np.nan)
    if np.isnan(map_val) or np.isnan(lactate) or lactate <= 0:
        return np.nan
    return float(map_val / lactate)


def compute_pf_ratio(row: pd.Series) -> float:
    """PaO₂/FiO₂ ratio — the gold standard for quantifying lung oxygenation."""
    pao2 = row.get("pao2", np.nan)
    fio2 = row.get("fio2", 0.21)
    if np.isnan(pao2) or fio2 <= 0:
        return np.nan
    return float(pao2 / fio2)


# ─────────────────────────────────────────────────────────────────
# Physics-Derived Training Features
# Computed once → fed as ML input features (zero inference overhead)
# ─────────────────────────────────────────────────────────────────

def compute_rate_pressure_product(row: pd.Series) -> float:
    """
    Rate Pressure Product (RPP) = HR × Systolic BP
    Validated index of myocardial oxygen demand.
    > 20,000 = High risk of myocardial failure.
    Reference: Gobel FL et al. Circulation 1978.
    """
    hr  = row.get("heart_rate",  np.nan)
    sbp = row.get("systolic_bp", np.nan)
    if np.isnan(hr) or np.isnan(sbp):
        return np.nan
    return float(hr * sbp)


def compute_do2_estimate(row: pd.Series) -> float:
    """
    Estimated Oxygen Delivery (DO₂) without invasive catheterization.
    DO₂ ≥ 450 mL/min/m² = Adequate. < 300 = Critical hypoxia.
    Reference: Rivers et al. NEJM 2001 (EGDT trial).

    CO estimate uses the pulse-pressure proxy (Frank-Starling principle):
      SV ∝ (SBP - DBP)  →  CO = HR × SV / 1000
    Matches physics_engine.estimate_cardiac_output() for consistency.

    BUG-FIX: Hemoglobin was previously hardcoded at 12.0 g/dL. Now uses real
    hemoglobin from CinC data (column 'hemoglobin') with 12.0 as fallback.
    """
    hr      = row.get("heart_rate",             75.0)
    sbp     = row.get("systolic_bp",            120.0)
    dbp     = row.get("diastolic_bp",            80.0)
    spo2    = row.get("spo2",                    97.0)
    hb      = row.get("hemoglobin",              12.0)  # Now uses real Hgb from CinC data
    if any(not isinstance(v, (int, float)) or np.isnan(v) for v in [hr, sbp, dbp, spo2]):
        return np.nan
    if np.isnan(hb):
        hb = 12.0  # Fallback for datasets without hemoglobin
    # Pulse-pressure stroke volume proxy (calibrated to match ~70 mL at normal vitals)
    pulse_pressure = max(sbp - dbp, 10.0)
    sv_proxy       = pulse_pressure * 0.7          # ~70 mL at PP=40 mmHg (normal)
    co_estimate    = float(np.clip((hr * sv_proxy) / 1000.0, 2.0, 15.0))  # L/min
    cao2 = hb * 1.34 * (spo2 / 100.0)
    return float(co_estimate * cao2 * 10)



def compute_aa_gradient(row: pd.Series) -> float:
    """
    Alveolar-Arterial (A-a) O₂ Gradient. Captures early lung dysfunction
    before SpO₂ visibly drops. > 30 mmHg = lung injury.
    Reference: West JB. Respiratory Physiology, 10th Ed, Ch 5.
    """
    pao2  = row.get("pao2",  np.nan)
    fio2  = row.get("fio2",  0.21)
    paco2 = row.get("paco2", 40.0)
    if np.isnan(pao2) or fio2 <= 0:
        return np.nan
    pao2_alveolar = fio2 * (760.0 - 47.0) - (paco2 / 0.8)
    return float(max(0.0, pao2_alveolar - pao2))


def compute_compensatory_reserve_index(row: pd.Series) -> float:
    """
    Compensatory Reserve Index (CRI) — approximation of Frank-Starling reserve.
    CRI → 1.0 = Full reserve. CRI → 0.0 = Reserve exhausted, decompensation imminent.
    Reference: Moulton SL et al. J Trauma 2011.
    """
    sbp = row.get("systolic_bp",  120.0)
    dbp = row.get("diastolic_bp",  80.0)
    hr  = row.get("heart_rate",    75.0)
    if any(np.isnan(v) for v in [sbp, dbp, hr]):
        return np.nan
    pp_norm  = min(max(sbp - dbp, 1.0) / 40.0, 1.0)
    hr_norm  = max(0.0, 1.0 - (max(hr - 70.0, 0) / 80.0))
    return float(np.clip(pp_norm * 0.6 + hr_norm * 0.4, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────
# NEW: Scientifically-Backed Derived Features
# Sources: CinC 2019 winning solution + NIH clinical literature
# ─────────────────────────────────────────────────────────────────

def compute_anion_gap(row: pd.Series) -> float:
    """
    Anion Gap = Na+ - Cl- - HCO3-
    
    NIH study: AG > 18 mmol/L predicts serum lactate > 4 mmol/L in sepsis.
    Longitudinal AG trajectories linked to all-cause ICU mortality.
    AG does NOT require a direct sodium measurement — CinC 2019 data lacks Na+,
    so we approximate Na+ ≈ Cl + HCO3 + 12 (the normal AG). When Cl and HCO3
    are measured, deviations from this identity reveal unmeasured anions (lactate,
    ketones, etc.).
    
    Reference: NIH PMC, "Longitudinal AG trajectories and mortality in sepsis"
    """
    cl   = row.get("chloride", np.nan)
    hco3 = row.get("hco3",    np.nan)
    na_val = row.get("sodium", np.nan)
    if np.isnan(cl) or np.isnan(hco3):
        return np.nan
    # Use real sodium if available (eICU lab.csv / Zenodo), else approximate
    na_use = na_val if not np.isnan(na_val) else 140.0
    return float(na_use - cl - hco3)


def compute_bun_creatinine_ratio(row: pd.Series) -> float:
    """
    BUN / Creatinine Ratio (BCR).
    
    CinC 2019 winning solution ("Can I get your signature?") used BCR as a key
    input feature to their gradient boosting model.
    
    NIH: BCR ≥ 27.3 mg/dL = highest sepsis mortality risk.
    Clinically: elevated BCR distinguishes pre-renal azotemia (dehydration/shock)
    from intrinsic renal failure — crucial for sepsis management.
    
    Reference: NIH PMC, "BCR and mortality in septic shock"
    """
    bun  = row.get("bun",        np.nan)
    creat = row.get("creatinine", np.nan)
    if np.isnan(bun) or np.isnan(creat) or creat <= 0:
        return np.nan
    return float(bun / creat)


def compute_qsofa_score(row: pd.Series) -> float:
    """
    Quick SOFA (qSOFA) — the Sepsis-3 bedside screening tool.
    
    Score 0-3:
      +1 if SBP ≤ 100 mmHg
      +1 if RR ≥ 22 breaths/min
      +1 if GCS < 15 (altered mentation)
    
    qSOFA ≥ 2 = high risk of poor outcome (sepsis guideline trigger).
    Simpler than full SOFA but validated for sepsis screening.
    
    Reference: Singer et al. JAMA 2016 (Sepsis-3 definition)
    """
    score = 0
    sbp = row.get("systolic_bp", np.nan)
    rr  = row.get("respiratory_rate", np.nan)
    gcs = row.get("gcs", 15)
    if not np.isnan(sbp) and sbp <= 100:
        score += 1
    if not np.isnan(rr) and rr >= 22:
        score += 1
    if not np.isnan(gcs) and gcs < 15:
        score += 1
    return float(score)


def compute_modified_shock_index(row: pd.Series) -> float:
    """
    Modified Shock Index = HR / MAP (instead of HR / SBP).
    
    More sensitive than classical Shock Index because MAP accounts for
    diastolic contribution (vascular resistance). MSI > 1.3 = severe shock.
    
    Reference: Singh et al. "Modified Shock Index as predictor of mortality
    in emergency department" J Emerg Med 2014.
    """
    hr  = row.get("heart_rate",             np.nan)
    map_val = row.get("mean_arterial_pressure", np.nan)
    if np.isnan(hr) or np.isnan(map_val) or map_val <= 0:
        return np.nan
    return float(hr / map_val)


def compute_sao2_fio2_ratio(row: pd.Series) -> float:
    """
    SaO2 / FiO2 Ratio — non-invasive alternative to PaO2/FiO2.
    
    When arterial blood gas (ABG) is unavailable, SaO2/FiO2 correlates
    well with PaO2/FiO2 for ARDS grading. Low ratio = respiratory failure.
    
    Reference: NIH PMC, "SaO2/FiO2 as non-invasive PF ratio surrogate"
    """
    sao2 = row.get("sao2", np.nan)
    fio2 = row.get("fio2", 0.21)
    if np.isnan(sao2) or fio2 <= 0:
        return np.nan
    return float(sao2 / fio2)


def compute_pulse_pressure(row: pd.Series) -> float:
    """
    Pulse Pressure = SBP - DBP.
    
    Reflects stroke volume and arterial compliance (Frank-Starling mechanism).
    Narrowing PP (< 25 mmHg) = failing cardiac output → decompensation.
    Wide PP (> 60 mmHg) = septic vasodilation or aortic regurgitation.
    
    Reference: Guyton & Hall Medical Physiology, 13th Ed.
    """
    sbp = row.get("systolic_bp",  np.nan)
    dbp = row.get("diastolic_bp", np.nan)
    if np.isnan(sbp) or np.isnan(dbp):
        return np.nan
    return float(sbp - dbp)


def compute_potassium_risk_flag(row: pd.Series) -> float:
    """
    Potassium Risk Flag: K+ > 5.5 (hyperkalemia) OR K+ < 3.0 (hypokalemia).
    
    Both extremes cause lethal cardiac arrhythmias:
      K+ > 6.0 = peaked T waves → ventricular fibrillation
      K+ < 3.0 = U waves → torsades de pointes
    
    Reference: AHA ACLS Guidelines for hyperkalemia-induced cardiac arrest.
    """
    k = row.get("potassium", np.nan)
    if np.isnan(k):
        return 0.0  # Unknown = assume not at risk
    return 1.0 if (k > 5.5 or k < 3.0) else 0.0


def compute_troponin_elevation_flag(row: pd.Series) -> float:
    """
    Troponin I Elevation Flag: TroponinI > 0.04 ng/mL = myocardial injury.
    
    Elevated troponin in sepsis indicates stress cardiomyopathy and is an
    independent predictor of mortality (even without acute coronary syndrome).
    
    Reference: Mehta et al. "Cardiac troponin I predicts myocardial dysfunction
    and adverse outcome in septic shock" Int J Cardiol 2014.
    """
    trop = row.get("troponin_i", np.nan)
    if np.isnan(trop):
        return 0.0  # Unknown = assume not elevated
    return 1.0 if trop > 0.04 else 0.0


def compute_lactate_albumin_ratio(row: pd.Series) -> float:
    """
    Lactate / Albumin Ratio (LAR).
    
    Multiple NIH studies show LAR outperforms lactate alone AND APACHE II
    for sepsis mortality prediction. LAR integrates a measure of acute
    metabolic distress (lactate) with a marker of capillary leak/inflammation
    (albumin), providing a more comprehensive risk assessment.
    
    Cut-offs:
      LAR > 0.58 = reliable indicator of death (AnesthesiologyPaper 2023)
      LAR > 1.2  = 28-day mortality predictor (ResearchGate meta-analysis)
    
    Reference: NIH PMC "LAR and mortality in AKI"; Frontiers 2023
    """
    lac = row.get("lactate", np.nan)
    alb = row.get("albumin", np.nan)
    if np.isnan(lac) or np.isnan(alb) or alb <= 0:
        return np.nan
    return float(lac / alb)


def compute_de_ritis_ratio(row: pd.Series) -> float:
    """
    AST / ALT Ratio (De Ritis Ratio).
    
    Ratio > 2.0 = alcoholic liver disease or severe hepatocellular injury.
    Ratio < 1.0 = normal or mild hepatitis.
    Prognostic in multi-organ failure and ICU mortality prediction.
    
    Reference: De Ritis et al., Biochimica 1957; validated in ICU setting.
    """
    ast_val = row.get("ast", np.nan)
    alt_val = row.get("alt", np.nan)
    if np.isnan(ast_val) or np.isnan(alt_val) or alt_val <= 0:
        return np.nan
    return float(ast_val / alt_val)


def compute_cvp_map_gradient(row: pd.Series) -> float:
    """
    CVP-MAP Gradient = MAP - CVP.
    
    Represents the perfusion pressure gradient driving organ blood flow.
    Low gradient (< 60 mmHg) = poor organ perfusion.
    
    Reference: Guyton physiology — tissue perfusion = MAP - venous pressure.
    """
    map_val = row.get("mean_arterial_pressure", np.nan)
    cvp_val = row.get("cvp", np.nan)
    if np.isnan(map_val) or np.isnan(cvp_val):
        return np.nan
    return float(map_val - cvp_val)


# ─────────────────────────────────────────────────────────────────
# NEW: Best-of-Both-Audits Feature Engineering
# Sources: Gemini audit (d²Lac, BCEI, dDSI, RCCR)
#          Claude audit (VDI, Lactate Kinetics, Interactions)
# ─────────────────────────────────────────────────────────────────

def compute_vasopressor_dependency_index(row: pd.Series) -> float:
    """
    Vasopressor Dependency Index (VDI) — composite vasopressor dose
    normalized to norepinephrine equivalents.

    VDI > 0.5 correlates with 72h mortality (AUROC 0.82).
    Rising VDI trajectory is the strongest predictor of refractory shock.

    Reference: Russell JA et al. "Vasopressin versus Norepinephrine
    Infusion in Patients with Septic Shock." NEJM 2008;358:877-887.
    """
    nee = float(row.get("nee_dose", 0.0) or 0.0)
    vasopressin = float(row.get("vasopressin", 0.0) or 0.0)
    dopamine = float(row.get("dopamine", 0.0) or 0.0)
    # NE-equivalent normalization (standard ICU conversion factors)
    return float(nee + vasopressin * 2.5 + dopamine * 0.01)


def compute_resp_cardiac_coupling(row: pd.Series) -> float:
    """
    Respiratory-Cardiac Coupling Ratio (RCCR).

    Healthy respiratory sinus arrhythmia binds RR:HR at ~1:4 (ratio ~0.25).
    Sympathetic crisis uncouples parasympathetic dampening, shattering this
    intrinsic ratio before overt tachypnea manifests.

    Source: Gemini audit — autonomic decoupling precedes tachypnea.
    """
    rr = row.get("respiratory_rate", np.nan)
    hr = row.get("heart_rate", np.nan)
    if np.isnan(rr) or np.isnan(hr) or hr <= 0:
        return np.nan
    return float((rr / hr) / 0.25)  # Normalized to healthy baseline


def compute_shock_index_x_lactate(row: pd.Series) -> float:
    """
    Shock Index × Lactate — synergistic interaction term.

    SI alone misses occult shock when compensated by tachycardia.
    Multiplying by lactate captures the metabolic dimension.

    Source: Claude audit — feature interaction recommendation.
    """
    si = row.get("shock_index", np.nan)
    lac = row.get("lactate", np.nan)
    if np.isnan(si) or np.isnan(lac):
        return np.nan
    return float(si * lac)


def compute_pf_ratio_peep_adjusted(row: pd.Series) -> float:
    """
    PEEP-adjusted PaO₂/FiO₂ ratio.

    Raw PF ratio is misleading at high PEEP — a PF of 200 at PEEP 15
    is much worse than PF 200 at PEEP 5. Dividing by (PEEP + 1)
    normalizes oxygenation for ventilatory support level.

    Source: Claude audit — feature interaction recommendation.
    """
    pf = row.get("pf_ratio", np.nan)
    peep = row.get("peep", 0.0)
    if np.isnan(pf):
        return np.nan
    peep_val = float(peep) if not np.isnan(peep) else 0.0
    return float(pf / (peep_val + 1.0))


# ─────────────────────────────────────────────
# Temporal Delta Features (Trend Detection)
# ─────────────────────────────────────────────
DELTA_VITALS = [
    "heart_rate", "mean_arterial_pressure", "spo2",
    "respiratory_rate", "temperature", "lactate",
    "sofa_score", "news2_score", "shock_index",
    "rate_pressure_product", "do2_estimate", "compensatory_reserve_index",
    "nee_dose",
    # NEW — high-value dynamic features for temporal trend tracking:
    # Source: CinC 2019 winners + NIH clinical literature.
    "ph",                 # pH trend — falling pH = metabolic acidosis progression
    "base_excess",        # Acid-base trajectory — worsening = sepsis progression
    "glucose",            # Stress hyperglycemia trajectory
    "potassium",          # Rising K+ = renal failure / cardiac arrest imminent
    "bun",                # BUN trajectory — acute kidney injury
    "hematocrit",         # Falling Hct = bleeding / hemodilution
    # NOTE: anion_gap, bun_creatinine_ratio, lactate_albumin_ratio are DERIVED features
    # computed in engineer_features() — they do not exist in raw input and cannot use
    # the raw-value delta path. Their deltas are computed from imputed values, which is
    # acceptable since they are algebraic combinations of already-imputed lab values.
    "anion_gap",             # AG trajectory — widening = unmeasured anion accumulation
    "bun_creatinine_ratio",  # CinC winner feature — tracks renal pathophysiology
    # NEW eICU-enriched dynamic features
    "creatinine",            # CRoC = creatinine rate of change (JMIR 2023 AKI predictor)
    "albumin",               # Falling albumin = capillary leak / sepsis progression
    "cvp",                   # CVP trend — preload assessment
    "lactate_albumin_ratio",  # LAR trajectory — stronger than lactate alone (NIH)
]

DELTA_WINDOWS = [1, 2, 4]  # Hours of look-back for trend calculation

# Additional DELTA_VITALS for new audit features
# vasopressor_dependency_index: track escalation trajectory
# resp_cardiac_coupling: autonomic ratio drift
DELTA_VITALS_AUDIT = [
    "vasopressor_dependency_index",
    "resp_cardiac_coupling",
]


def compute_temporal_deltas(
    patient_history: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes temporal delta features for EVERY row (not just the last row).

    For each vital V and window W (1h, 2h, 4h):
      delta_V_Wh = V_current - V_(W hours ago)

    These features capture TRENDS — the rate of physiological change —
    which are far more clinically informative than any single snapshot.

    Example clinical significance:
      delta_map_4h = -20 mmHg  → MAP fell 20mmHg over 4 hours → high sepsis risk
      delta_lactate_2h = +1.5  → Lactate rising → tissue hypoxia
      delta_hr_1h = +25 bpm    → Tachycardic escalation → cardiac strain

    BUG-FIX (NEW-BUG-2): The previous implementation only wrote delta values to
    df.index.max() (the last row). During training where engineer_features() is
    called on a per-patient group, this is correct. During the old usage on the
    entire multi-patient DataFrame it meant 99.99% of rows had zero deltas,
    causing a silent train/inference mismatch on all 39 delta features.

    Fix: use vectorised pandas .shift(N) which is O(n) and writes a real delta
    value for every row. All training datasets are already resampled to 1-hour
    bins before reaching this function, so shift(N) = N hours back is exact.
    Rows with fewer than N hours of history get 0.0 (no history = no trend).

    Args:
        patient_history: DataFrame for a SINGLE patient, sorted chronologically,
                         with 'timestamp' as column or index, 1-hour bins.
    Returns:
        DataFrame with additional delta_* columns set for every row.
    """
    df = patient_history.copy()

    had_timestamp_col = "timestamp" in df.columns
    if had_timestamp_col:
        df = df.reset_index(drop=True)

    # FIX: Batch-add ALL delta columns at once to avoid DataFrame fragmentation.
    # This double-nested loop adds ~75 NEW columns (25 vitals × 3 windows).
    # Adding them one-at-a-time fragments the block manager and triggers
    # PerformanceWarning for every patient × every column = millions of warnings.
    delta_data = {}
    for vital in DELTA_VITALS:
        if vital not in df.columns:
            continue
        for window_hours in DELTA_WINDOWS:
            col_name = f"delta_{vital}_{window_hours}h"
            # shift(N) looks N rows back; since data is 1h-bin, N rows = N hours.
            # fillna(0.0): rows with < N hours of history have no valid past value
            # → treat as "no change" rather than NaN (avoids downstream fillna(0)
            # masking real trailing NaN issues).
            delta_data[col_name] = (df[vital] - df[vital].shift(window_hours)).fillna(0.0)
    if delta_data:
        df = pd.concat([df, pd.DataFrame(delta_data, index=df.index)], axis=1)

    if had_timestamp_col:
        return df
    return df


# ─────────────────────────────────────────────
# Master Feature Engineering Pipeline
# ─────────────────────────────────────────────
def engineer_features(patient_history: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point for feature engineering.
    
    Runs the entire pipeline on a patient's historical vitals DataFrame:
      1. Impute missing values (with missingness flags for labs)
      2. Compute classical clinical scores (SOFA, NEWS2, SI)
      3. Compute physics-derived features (RPP, DO₂, A-a gradient, CRI)
      4. Compute NEW scientifically-backed derived features:
         - Anion Gap (NIH: AG>18 = lactate>4)
         - BUN/Creatinine Ratio (CinC 2019 winner; BCR≥27.3 = high mortality)
         - qSOFA (Sepsis-3 quick bedside screen)
         - Modified Shock Index (HR/MAP — more sensitive than HR/SBP)
         - SaO2/FiO2 Ratio (non-invasive PF ratio alternative)
         - Pulse Pressure (Frank-Starling reserve)
         - Clinical risk flags (potassium, troponin, glucose, lactate clearance)
      5. Compute temporal deltas across 1h, 2h, 4h windows
      6. Impute demographics with population defaults
    
    Returns the DataFrame with all new feature columns appended.
    The last row represents the current state used for inference.
    """
    # BUG-R2-2 FIX: Compute temporal deltas on RAW (pre-imputation) values.
    # Previously deltas were computed after impute_vitals(), which forward-fills
    # and applies median fallbacks. This created false trends:
    #   e.g., Hour 0: lactate=NaN → imputed to 1.2 (median)
    #         Hour 4: lactate=4.0 (real measurement)
    #         delta_lactate_4h = 4.0 - 1.2 = 2.8 ← FALSE TREND (1.2 was fabricated)
    #
    # Fix: Save raw values for DELTA_VITALS before imputation. Compute deltas on
    # raw values where NaN means "not measured" → delta = NaN → fillna(0) in
    # compute_temporal_deltas() correctly yields "no trend info available."
    # Clinical scores (SOFA, NEWS2, etc.) still use imputed values (correct —
    # they need a complete row for their calculations).
    #
    # Save raw values for delta computation before imputation
    _raw_delta_cols = [v for v in DELTA_VITALS if v in patient_history.columns]
    _raw_for_deltas = patient_history[_raw_delta_cols].copy() if _raw_delta_cols else None

    df = impute_vitals(patient_history)

    # ── Classical clinical scores ────────────────────────────────────────────
    df["sofa_score"]       = df.apply(compute_sofa_score, axis=1)
    df["news2_score"]      = df.apply(compute_news2_score, axis=1)
    df["shock_index"]      = df.apply(compute_shock_index, axis=1)
    df["map_lactate_ratio"]= df.apply(compute_map_lactate_ratio, axis=1)
    df["pf_ratio"]         = df.apply(compute_pf_ratio, axis=1)

    # ── Physics-derived features ─────────────────────────────────────────────
    df["rate_pressure_product"]      = df.apply(compute_rate_pressure_product, axis=1)
    df["do2_estimate"]               = df.apply(compute_do2_estimate, axis=1)
    df["aa_gradient"]                = df.apply(compute_aa_gradient, axis=1)
    df["compensatory_reserve_index"] = df.apply(compute_compensatory_reserve_index, axis=1)

    # ── NEW: Scientifically-backed derived features ──────────────────────────
    # Source: CinC 2019 winning solution + NIH clinical literature
    df["anion_gap"]             = df.apply(compute_anion_gap, axis=1)
    df["bun_creatinine_ratio"]  = df.apply(compute_bun_creatinine_ratio, axis=1)
    df["qsofa_score"]           = df.apply(compute_qsofa_score, axis=1)
    df["modified_shock_index"]  = df.apply(compute_modified_shock_index, axis=1)
    df["sao2_fio2_ratio"]       = df.apply(compute_sao2_fio2_ratio, axis=1)
    df["pulse_pressure"]        = df.apply(compute_pulse_pressure, axis=1)
    # Clinical risk flags (binary)
    df["potassium_risk_flag"]   = df.apply(compute_potassium_risk_flag, axis=1)
    df["troponin_elevated"]     = df.apply(compute_troponin_elevation_flag, axis=1)
    # NEW eICU-enriched derived features
    df["lactate_albumin_ratio"] = df.apply(compute_lactate_albumin_ratio, axis=1)
    df["de_ritis_ratio"]        = df.apply(compute_de_ritis_ratio, axis=1)
    df["cvp_map_gradient"]      = df.apply(compute_cvp_map_gradient, axis=1)
    df["glucose_variability"]   = 0.0  # Computed after deltas are available
    df["lactate_clearance_failed"] = 0.0  # Computed after deltas

    # ── NEW: Best-of-Both-Audits derived features ────────────────────────────
    # Gemini audit features
    df["resp_cardiac_coupling"]         = df.apply(compute_resp_cardiac_coupling, axis=1)
    # Claude audit features
    df["vasopressor_dependency_index"]  = df.apply(compute_vasopressor_dependency_index, axis=1)
    df["shock_index_x_lactate"]         = df.apply(compute_shock_index_x_lactate, axis=1)
    df["pf_ratio_peep_adjusted"]        = df.apply(compute_pf_ratio_peep_adjusted, axis=1)

    # ── Temporal deltas (trend detection) ────────────────────────────────────
    # BUG-R2-2 FIX continued: Temporarily swap in raw (pre-imputation) values
    # for the DELTA_VITALS columns, compute deltas, then restore imputed values.
    # This ensures deltas reflect real measurement-to-measurement changes only.
    if _raw_for_deltas is not None and len(_raw_for_deltas) == len(df):
        _imputed_backup = df[_raw_delta_cols].copy()
        df[_raw_delta_cols] = _raw_for_deltas.values  # raw NaNs for delta calc
        df = compute_temporal_deltas(df)
        df[_raw_delta_cols] = _imputed_backup.values   # restore imputed values
    else:
        df = compute_temporal_deltas(df)

    # ── Post-delta features (depend on delta values) ─────────────────────────
    # SOFA delta (rise of ≥2 in 4h is the Sepsis-3 diagnostic criterion)
    if "delta_sofa_score_4h" in df.columns:
        df["sofa_rise_2h_flag"] = (df.get("delta_sofa_score_2h", 0) >= 2.0).astype(int)
        df["sofa_rise_4h_flag"] = (df["delta_sofa_score_4h"] >= 2.0).astype(int)
    # Glucose variability: |Δglucose_2h| > 50 mg/dL = stress hyperglycemia signal
    if "delta_glucose_2h" in df.columns:
        df["glucose_variability"] = (df["delta_glucose_2h"].abs() > 50.0).astype(int)
    # Lactate clearance failure: rising lactate over 2h = tissue hypoxia worsening
    # Source: Rivers NEJM 2001 (EGDT trial)
    if "delta_lactate_2h" in df.columns:
        df["lactate_clearance_failed"] = (df["delta_lactate_2h"] > 0).astype(int)

    # ── NEW: Post-delta audit features (Gemini + Claude) ─────────────────────
    # Gemini: d²Lactate (Metabolic Decoupling Velocity)
    # Second derivative of lactate = acceleration of lactic acidosis.
    # Captures liver clearance collapse inflection point.
    if "delta_lactate_2h" in df.columns:
        df["lactate_acceleration"] = (
            df["delta_lactate_2h"] - df["delta_lactate_2h"].shift(2)
        ).fillna(0.0)
    else:
        df["lactate_acceleration"] = 0.0

    # Gemini: Buffer Capacity Exhaustion Index (BCEI)
    # Ratio of HCO₃ depletion rate to pH depletion rate.
    # When buffer ceiling breaks, pH collapses faster than HCO₃.
    if "delta_hco3_4h" in df.columns and "delta_ph_4h" in df.columns:
        _dph = df["delta_ph_4h"].replace(0, np.nan)
        df["buffer_exhaustion_index"] = (
            df.get("delta_hco3_4h", pd.Series(0.0, index=df.index)) / _dph
        ).fillna(0.0)
    else:
        df["buffer_exhaustion_index"] = 0.0

    # Gemini: Dynamic Diastolic Instability (dDSI)
    # Rolling StdDev of HR/DBP ratio over 4 hours.
    # Captures sympathetic oscillation preceding MAP collapse.
    if "heart_rate" in df.columns and "diastolic_bp" in df.columns:
        _dbp_safe = df["diastolic_bp"].replace(0, np.nan)
        _hr_dbp_ratio = df["heart_rate"] / _dbp_safe
        df["diastolic_instability"] = _hr_dbp_ratio.rolling(
            window=4, min_periods=2
        ).std().fillna(0.0)
    else:
        df["diastolic_instability"] = 0.0

    # Claude: Lactate clearance rate (% clearance over 6h)
    # <10% at 6h = 60% mortality (Rivers EGDT trial NEJM 2001)
    if "lactate" in df.columns:
        _lac_6h_ago = df["lactate"].shift(6)
        _lac_6h_safe = _lac_6h_ago.replace(0, np.nan)
        df["lactate_clearance_rate"] = (
            (_lac_6h_ago - df["lactate"]) / _lac_6h_safe * 100.0
        ).fillna(0.0)
    else:
        df["lactate_clearance_rate"] = 0.0

    # Claude: Lactate AUC 12h (trapezoidal integral)
    # AUC_lactate outperforms single-point measurement (+0.04 AUROC)
    if "lactate" in df.columns:
        df["lactate_auc_12h"] = df["lactate"].rolling(
            window=12, min_periods=1
        ).apply(lambda x: np.trapz(x, dx=1.0), raw=True).fillna(0.0)
    else:
        df["lactate_auc_12h"] = 0.0

    # Claude: MAP_trend × Lactate_trend (convergent decompensation signal)
    if "delta_mean_arterial_pressure_2h" in df.columns and "delta_lactate_2h" in df.columns:
        df["map_trend_x_lactate_trend"] = (
            df["delta_mean_arterial_pressure_2h"] * df["delta_lactate_2h"]
        ).fillna(0.0)
    else:
        df["map_trend_x_lactate_trend"] = 0.0

    # ── Demographics — impute with population defaults if missing ────────────
    # FIX: Collect new columns in a dict first, then assign all at once via
    # pd.concat(axis=1). Adding columns one-at-a-time in a loop fragments
    # the DataFrame's internal block manager, triggering PerformanceWarning
    # on every call (17K+ warnings flooding the training log).
    _existing_demo = {}
    _new_demo = {}
    for demo_col, default_val in DEMOGRAPHIC_DEFAULTS.items():
        if demo_col in df.columns:
            _existing_demo[demo_col] = default_val
        else:
            _new_demo[demo_col] = default_val
    # Fillna for existing columns (in-place, no fragmentation)
    for col, val in _existing_demo.items():
        df[col] = df[col].fillna(val)
    # Add ALL new columns at once (single concat, zero fragmentation)
    if _new_demo:
        new_cols = pd.DataFrame(
            {col: [val] * len(df) for col, val in _new_demo.items()},
            index=df.index,
        )
        df = pd.concat([df, new_cols], axis=1)

    return df


def get_feature_columns() -> list[str]:
    """
    Returns the ordered list of feature columns expected by the ML models.
    Must be kept in sync with train_models.py.

    Includes:
      - Base vitals (monitors + labs, forward-filled) — 43 features
      - Demographics (age, gender, hosp_adm_time, iculos, admission_weight) — 5 features
      - Lab missingness indicator flags ({lab}_measured = 0/1) — 36 flags
        Research basis: NIH 2023 showed missingness flags = >40% of top ICU predictors
      - Clinical scores (SOFA, NEWS2, SI, qSOFA, etc.) — 11 scores
      - Derived features (Anion Gap, BUN/Cr, MSI, LAR, CVP-MAP, etc.) — 15 features
      - Physics-derived features (RPP, DO2, A-a gradient, CRI) — 4 features
      - Zenodo/eICU passthrough features — 5 features
      - Temporal deltas (1h, 2h, 4h windows) — 75 deltas
      Total: ~194 features
    """
    base_vitals = list(VITAL_MEDIANS.keys())
    demographics = list(DEMOGRAPHIC_DEFAULTS.keys())
    # Missingness indicator flags for lab vitals (not monitors — monitors are always present)
    missingness_flags = [f"{lab}_measured" for lab in LAB_VITALS]
    scores      = [
        "sofa_score", "news2_score", "shock_index",
        "map_lactate_ratio", "pf_ratio",
        "sofa_rise_2h_flag", "sofa_rise_4h_flag",
        # Physics-derived (fed to ML ensemble as input features)
        "rate_pressure_product",
        "do2_estimate",
        "aa_gradient",
        "compensatory_reserve_index",
    ]
    # Scientifically-backed derived features
    derived = [
        "anion_gap",              # NIH: AG>18 = lactate>4
        "bun_creatinine_ratio",   # CinC 2019 winner feature; BCR≥27.3 = high mortality
        "qsofa_score",            # Sepsis-3 quick bedside screen
        "modified_shock_index",   # HR/MAP — more sensitive than classic SI
        "sao2_fio2_ratio",        # Non-invasive PF ratio alternative (NIH)
        "pulse_pressure",         # Frank-Starling reserve indicator
        "potassium_risk_flag",    # K+>5.5 or K+<3.0 → cardiac arrhythmia
        "troponin_elevated",      # TroponinI>0.04 → myocardial injury
        "glucose_variability",    # |ΔGlucose_2h|>50 → stress hyperglycemia
        "lactate_clearance_failed",  # ΔLactate_2h>0 → tissue hypoxia worsening
        # NEW eICU-enriched derived features
        "lactate_albumin_ratio",  # NIH: LAR > 0.58 = death predictor; outperforms APACHE II
        "de_ritis_ratio",         # AST/ALT ratio — liver injury / multi-organ failure
        "cvp_map_gradient",       # MAP - CVP — organ perfusion pressure (Guyton physiology)
        # NEW: Best-of-Both-Audits features (Gemini + Claude)
        "resp_cardiac_coupling",          # Gemini: autonomic decoupling ratio
        "vasopressor_dependency_index",   # Claude: multi-pressor normalization
        "shock_index_x_lactate",          # Claude: synergistic interaction
        "pf_ratio_peep_adjusted",         # Claude: PEEP-normalized oxygenation
        "lactate_acceleration",           # Gemini: d²Lactate — liver clearance collapse
        "buffer_exhaustion_index",        # Gemini: metabolic buffer saturation
        "diastolic_instability",          # Gemini: sympathetic oscillation
        "lactate_clearance_rate",         # Claude: % clearance over 6h
        "lactate_auc_12h",                # Claude: 12h trapezoidal integral
        "map_trend_x_lactate_trend",      # Claude: convergent decompensation signal
    ]
    # Zenodo/eICU static features that pass through (renamed in loader)
    # These may be NaN for datasets that don't have them — handled by fillna(0) at training
    passthrough = [
        "alcohol_use",              # Zenodo: alcoholic status
        "smoking",                  # Zenodo: smoking status
        "family_hx_cardiac_disease", # Zenodo: FHCD
        "triage_score",             # Zenodo: clinical acuity score
        # NOTE: gcs is already in VITAL_MEDIANS (base_vitals), not repeated here
        # FIX-2: Dataset source indicators (hemodynamic_collapse only).
        # Let the model learn domain-specific patterns across heterogeneous sources.
        # For sepsis/hypotension (single source), these columns won't exist in df,
        # so avail filtering removes them — zero impact on other targets.
        "dataset_source_vitaldb",
        "dataset_source_zenodo",
        "dataset_source_cudb",
        "dataset_source_sddb",
        "dataset_source_icare",
    ]
    deltas      = [f"delta_{v}_{w}h" for v in DELTA_VITALS for w in DELTA_WINDOWS]
    return base_vitals + demographics + missingness_flags + scores + derived + passthrough + deltas
