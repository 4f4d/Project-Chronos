"""
Project Chronos - Shared Data Contract
=====================================
Defines the strict JSON schema / Pydantic models used by
BOTH the backend API and frontend consumer.

Any change here must be reflected on both sides.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ─────────────────────────────────────────────
# INBOUND: Vitals streamed from the simulator
# ─────────────────────────────────────────────

class PatientVitals(BaseModel):
    """One snapshot of a patient's vitals at a given timestamp."""
    patient_id: str
    timestamp: datetime
    
    # Core Vital Signs
    heart_rate: Optional[float] = Field(None, description="HR in bpm")
    systolic_bp: Optional[float] = Field(None, description="SBP in mmHg")
    diastolic_bp: Optional[float] = Field(None, description="DBP in mmHg")
    mean_arterial_pressure: Optional[float] = Field(None, description="MAP in mmHg")
    spo2: Optional[float] = Field(None, description="SpO2 in %")
    respiratory_rate: Optional[float] = Field(None, description="RR in breaths/min")
    temperature: Optional[float] = Field(None, description="Temp in Celsius")
    
    # Lab Values
    lactate: Optional[float] = Field(None, description="Serum lactate in mmol/L")
    wbc: Optional[float] = Field(None, description="White blood cells x10^9/L")
    creatinine: Optional[float] = Field(None, description="Creatinine in mg/dL")
    bilirubin: Optional[float] = Field(None, description="Total bilirubin in mg/dL")
    platelets: Optional[float] = Field(None, description="Platelets x10^9/L")
    pao2: Optional[float] = Field(None, description="PaO2 in mmHg")
    fio2: Optional[float] = Field(None, description="FiO2 as fraction (0.21–1.0)")
    gcs: Optional[int] = Field(None, description="Glasgow Coma Scale 3-15")

    # Context
    vasopressor_active: Optional[bool] = Field(False, description="Is vasopressor running?")
    # BUG-D fix: vasopressor_dose was in api.py VitalsPayload but missing here.
    # The /predict endpoint remaps vasopressor_dose → nee_dose internally.
    vasopressor_dose: Optional[float] = Field(0.0, description="NE-equivalent vasopressor dose in mcg/kg/min")
    on_mechanical_ventilation: Optional[bool] = Field(False, description="Intubated?")


# ─────────────────────────────────────────────
# OUTBOUND: Prediction response JSON contract
# ─────────────────────────────────────────────

class SHAPDrivers(BaseModel):
    """Top feature contributions to a prediction (SHAP values)."""
    feature_name: str
    shap_value: float          # Positive = increases risk, negative = decreases
    current_value: float       # Actual measured value
    direction: str             # "↑ Rising" | "↓ Falling" | "→ Stable"


class SepticShockPrediction(BaseModel):
    risk_probability_percentage: float
    time_window: str = "2-6 hours"
    risk_level: str             # "CRITICAL" | "HIGH" | "MODERATE" | "LOW"
    shap_drivers: list[SHAPDrivers]
    # BUG-CLAUDE-3-2 fix: sofa_score/news2_score are NOT in the per-prediction block.
    # build_response() places them in the top-level clinical_scores dict only.
    # Keeping these dead fields here caused confusion about the real API contract.


class HypotensionPrediction(BaseModel):
    risk_probability_percentage: float
    time_window: str = "2-6 hours"
    risk_level: str
    shap_drivers: list[SHAPDrivers]
    # BUG-CLAUDE-3-2 fix: shock_index lives in clinical_scores, not here.


class CardiacArrestPrediction(BaseModel):
    """BUG-Q fix: Updated to match actual build_response() output."""
    risk_probability_percentage: float
    time_window: str             # "2-6 hours" | "Critical" (when physics override fires)
    risk_level: str              # "CRITICAL" | "HIGH" | "MODERATE" | "LOW"
    shap_drivers: list           # SHAP feature contributions from hemodynamic_collapse model
    ml_probability: float        # ML model probability alone (pre-physics blend)
    physics_probability: float   # Physics engine cardiac_arrest_probability
    physics_metrics: dict = Field(
        ...,
        description="Keys: tissue_hypoxia_index, hemodynamic_instability_score, oxygen_delivery_do2, o2_extraction_ratio"
    )
    physics_override_triggered: bool = False  # True if physics hard-override fired
    alert_reasons: list[str] = Field(default_factory=list)


class PredictionPayload(BaseModel):
    """
    The strict output contract returned by /predict.
    Consumed by the Frontend for the Triage Radar UI.
    """
    patient_id: str
    timestamp: datetime
    predictions: dict = Field(
        ...,
        description="Keys: 'septic_shock', 'blood_pressure_collapse', 'cardiac_arrest'"
    )
    # BUG-CLAUDE-3-1 fix: clinical_scores was always in the API response but missing here.
    # PatientCard.jsx and DetailPanel.jsx both read patient.clinical_scores.sofa_score,
    # patient.clinical_scores.news2_score, patient.clinical_scores.shock_index.
    clinical_scores: dict = Field(
        default_factory=dict,
        description="Keys: sofa_score, news2_score, shock_index"
    )
    # The single aggregated "Crash Probability Score" shown in the Triage Radar
    # Computed as: max(sepsis_prob, hypotension_prob, cardiac_arrest_prob)
    # weighted by clinical acuity
    crash_probability_score: float
    crash_risk_level: str   # "CRITICAL" | "HIGH" | "MODERATE" | "LOW"
    last_updated: datetime
