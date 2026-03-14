"""
Project Chronos - MIMIC Demo Column Mapper
==========================================
Maps MIMIC-III and MIMIC-IV column names / itemids to the internal
feature schema used by Project Chronos models.

MIMIC uses integer 'itemid' codes for almost everything in chartevents.
This module translates those codes into our feature naming convention
(heart_rate, mean_arterial_pressure, etc.) so the shadow evaluator
can run inference on MIMIC patients without any retraining.

Supports:
  - MIMIC-III Demo (100 patients, data in mimic3_demo/)
  - MIMIC-IV Demo  (100 patients, data in mimic4_demo/)

Usage (internal only — called by shadow_evaluate.py):
    mapper = MIMICMapper(data_dir / "mimic3_demo", "mimic3_demo")
    patients = mapper.list_patients()
    vitals_df = mapper.load_patient_vitals(patient_id)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# MIMIC-III itemid → our feature name
# Source: d_items.csv, MIMIC-III ClinicalDB (MIT-LCP)
# ─────────────────────────────────────────────────────────────────────────────
MIMIC3_ITEMID_MAP = {
    # Heart Rate
    211:   "heart_rate",
    220045:"heart_rate",
    # Arterial BP
    51:    "systolic_bp",
    220179:"systolic_bp",
    8368:  "diastolic_bp",
    220180:"diastolic_bp",
    52:    "mean_arterial_pressure",
    220052:"mean_arterial_pressure",
    456:   "mean_arterial_pressure",
    # SpO2
    646:   "spo2",
    220277:"spo2",
    # Respiratory Rate
    615:   "respiratory_rate",    # CareVue: 'Resp Rate (Total)'
    618:   "respiratory_rate",    # MetaVision: 'Respiratory Rate' (Bug 38 fix: was MISSING)
    220210:"respiratory_rate",    # MetaVision: 'Respiratory Rate'
    224690:"respiratory_rate",    # MetaVision: 'Respiratory Rate (Total)' (Bug 38 fix: was MISSING)
    # Temperature (Celsius)
    223762:"temperature",
    676:   "temperature",
    # Labs
    50813: "lactate",    # metavision
    818:   "lactate",    # carevue
    51300: "wbc",
    51301: "wbc",
    50912: "creatinine",
    50885: "bilirubin",
    51265: "platelets",
    50821: "pao2",
    50816: "fio2",
    # GCS components — sum for approximate total
    198:   "gcs_motor",
    220739:"gcs_eye",
    223900:"gcs_verbal",
    223901:"gcs_motor",
    184:   "gcs_eye",
    723:   "gcs_verbal",
    # Vasopressor (binary flag)
    30047: "vasopressor_norepinephrine_dose",
    221906:"vasopressor_norepinephrine_dose",
    30120: "vasopressor_norepinephrine_dose",
}

# MIMIC-IV items are in chartevents with similar IDs but some differ
MIMIC4_ITEMID_MAP = {
    **MIMIC3_ITEMID_MAP,  # Most IDs overlap, only update differences
    220045: "heart_rate",
    220179: "systolic_bp",
    220180: "diastolic_bp",
    220052: "mean_arterial_pressure",
    220277: "spo2",
    220210: "respiratory_rate",
    223762: "temperature",
    # MIMIC-IV lab itemids (from labevents, not chartevents)
    50813:  "lactate",
    51301:  "wbc",
    50912:  "creatinine",
    50885:  "bilirubin",
    51265:  "platelets",
    50821:  "pao2",
    50816:  "fio2",
}


class MIMICMapper:
    """
    Loads and maps MIMIC-III/IV Demo patient data to the Chronos feature schema.

    Key tables used:
      MIMIC-III: CHARTEVENTS.csv, LABEVENTS.csv, PATIENTS.csv, ADMISSIONS.csv
      MIMIC-IV:  chartevents.csv, labevents.csv, patients.csv, admissions.csv

    Output: hourly-resampled DataFrame with columns matching VITAL_MEDIANS keys
    """

    def __init__(self, data_dir: Path, version: str = "mimic3_demo"):
        self.data_dir = Path(data_dir)
        self.version  = version
        self.itemid_map = MIMIC4_ITEMID_MAP if "mimic4" in version else MIMIC3_ITEMID_MAP

        # MIMIC-III uses uppercase filenames but lowercase column names inside the CSV
        if "mimic3" in version:
            self._chart_file   = self._find_file("CHARTEVENTS.csv")
            self._lab_file     = self._find_file("LABEVENTS.csv")
            self._patient_file = self._find_file("PATIENTS.csv")
            self._admit_file   = self._find_file("ADMISSIONS.csv")
            # Column names inside the CSV are lowercase in MIMIC-III demo
            self._id_col       = "hadm_id"
            self._pid_col      = "subject_id"
            self._itemid_col   = "itemid"
            self._value_col    = "valuenum"
            self._time_col     = "charttime"
        else:
            self._chart_file   = self._find_file("chartevents.csv")
            self._lab_file     = self._find_file("labevents.csv")
            self._patient_file = self._find_file("patients.csv")
            self._admit_file   = self._find_file("admissions.csv")
            self._id_col       = "hadm_id"
            self._pid_col      = "subject_id"
            self._itemid_col   = "itemid"
            self._value_col    = "valuenum"
            self._time_col     = "charttime"

    def _find_file(self, name: str) -> Optional[Path]:
        """Finds a CSV file recursively within data_dir."""
        matches = list(self.data_dir.rglob(name))
        if matches:
            return matches[0]
        return None

    def list_patients(self) -> list[str]:
        """Returns all unique patient (hadm_id or stay_id) identifiers."""
        if self._admit_file and self._admit_file.exists():
            adm = pd.read_csv(self._admit_file, usecols=[self._id_col, self._pid_col])
            return [str(x) for x in adm[self._id_col].unique().tolist()]

        if self._chart_file and self._chart_file.exists():
            chunk = pd.read_csv(self._chart_file, usecols=[self._id_col], nrows=10000)
            return [str(x) for x in chunk[self._id_col].unique().tolist()]

        logger.warning(f"Cannot list patients from {self.data_dir} — no admissions/chartevents CSV found")
        return []

    def load_patient_vitals(self, hadm_id: str) -> Optional[pd.DataFrame]:
        """
        Loads and maps all chart + lab events for one hospital admission.

        Steps:
          1. Filter chartevents + labevents by hadm_id
          2. Map itemid → internal feature name
          3. Resample to 1-hour intervals (mean)
          4. Forward-fill + median fallback

        Returns: DataFrame with columns like heart_rate, mean_arterial_pressure, etc.
        """
        hadm_id = str(hadm_id)
        rows = []

        # Load chart events for this admission
        for source_file, label in [
            (self._chart_file, "chart"),
            (self._lab_file,   "lab"),
        ]:
            if source_file is None or not source_file.exists():
                continue
            try:
                df = pd.read_csv(
                    source_file,
                    usecols=[self._id_col, self._itemid_col, self._value_col, self._time_col],
                    dtype={self._id_col: str},
                )
                df = df[df[self._id_col] == hadm_id].copy()
                if df.empty:
                    continue

                df[self._time_col] = pd.to_datetime(df[self._time_col], errors="coerce")
                df = df.dropna(subset=[self._time_col, self._value_col])
                df[self._itemid_col] = df[self._itemid_col].astype(int)

                # Map itemid → feature name
                df["feature_name"] = df[self._itemid_col].map(self.itemid_map)
                df = df.dropna(subset=["feature_name"])

                # Aggregate GCS components to total (motor + eye + verbal)
                gcs_components = ["gcs_motor", "gcs_eye", "gcs_verbal"]
                gcs_df = df[df["feature_name"].isin(gcs_components)].copy()
                non_gcs = df[~df["feature_name"].isin(gcs_components)].copy()

                if not gcs_df.empty:
                    # Bug 21 fix: take mean per (charttime, component) FIRST to collapse
                    # duplicate chartevents rows (same itemid, same time, edit history),
                    # then sum across components to get total GCS.
                    gcs_totals = (
                        gcs_df.groupby([self._time_col, "feature_name"])[self._value_col]
                        .mean()                                 # collapse duplicates per component
                        .groupby(level=self._time_col).sum()   # sum motor + eye + verbal
                        .reset_index()
                    )
                    gcs_totals.columns = [self._time_col, self._value_col]
                    gcs_totals["feature_name"] = "gcs"
                    non_gcs = pd.concat([non_gcs, gcs_totals], ignore_index=True)

                rows.append(non_gcs[[self._time_col, "feature_name", self._value_col]])

            except Exception as e:
                logger.warning(f"  [{label}] {hadm_id}: {e}")

        if not rows:
            return None

        all_events = pd.concat(rows, ignore_index=True)
        all_events.columns = ["timestamp", "feature_name", "value"]

        # Pivot to wide format (one column per feature)
        wide = (
            all_events
            .groupby(["timestamp", "feature_name"])["value"]
            .mean()
            .unstack("feature_name")
            .reset_index()
        )

        # Resample to 1-hour bins
        wide["hour_bin"] = wide["timestamp"].dt.floor("1h")
        hourly = wide.groupby("hour_bin").mean(numeric_only=True).reset_index()
        hourly.rename(columns={"hour_bin": "timestamp"}, inplace=True)
        hourly["patient_id"] = hadm_id

        return hourly
