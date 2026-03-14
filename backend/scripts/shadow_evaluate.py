"""
Project Chronos - MIMIC Demo Shadow Evaluation Script
======================================================
Runs the trained model silently on MIMIC-III Demo (100 patients) and
MIMIC-IV Demo (100 patients) without any training — inference only.

This is called 'shadow mode deployment': the model predicts without
intervening, and we compare what it WOULD have alerted to what actually
happened. Proves generalization to real hospital data formats.

No patient data leaves this machine. All inference is local.

Usage:
    source .venv/bin/activate
    python scripts/shadow_evaluate.py --mimic3 --mimic4
    python scripts/shadow_evaluate.py --report-only   # just the summary table

Output:
    reports/shadow_eval_mimic3.json   — per-patient alert log
    reports/shadow_eval_mimic4.json
    reports/shadow_summary.html        — human-readable report for hackathon panel
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
DATA_DIR    = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

from scripts.mimic_mapper import MIMICMapper
from features import engineer_features, impute_vitals, get_feature_columns

import joblib


def load_models() -> dict:
    """Load the trained tabular models for each target."""
    models_dir = BASE_DIR / "models"
    registry = {}
    for target in ["sepsis", "hypotension", "hemodynamic_collapse"]:
        target_dir = models_dir / target
        lgbm_path = target_dir / "lgbm_model.pkl"
        feat_path = target_dir / "feature_columns.json"
        meta_path = target_dir / "model_metadata.json"

        if not lgbm_path.exists():
            logger.warning(f"  [{target}] model not found — skipping. Train first: python train_models.py --all")
            continue

        registry[target] = {
            "lgbm": joblib.load(lgbm_path),
            "features": json.load(open(feat_path)),
            "metadata": json.load(open(meta_path)),
            "threshold": json.load(open(meta_path)).get("optimal_threshold", 0.5),
        }
        meta = registry[target]["metadata"]
        logger.success(
            f"  [{target}] loaded — AUROC: {meta.get('val_auroc', '?')} | "
            f"AUPRC: {meta.get('val_auprc', '?')} | Threshold: {registry[target]['threshold']:.3f}"
        )

    return registry


def predict_patient(patient_vitals_df: pd.DataFrame, models: dict) -> list[dict]:
    """
    Runs hourly inference on a single patient's vitals timeline.

    Returns a list of hourly prediction dicts containing:
      - timestamp
      - sepsis_prob, hypotension_prob, hemodynamic_collapse_prob
      - any_alert_fired (bool): whether any threshold was exceeded
      - alert_targets: which targets crossed the threshold
    """
    # engineer_features() calls impute_vitals() internally — do NOT call separately
    # (double-imputation would zero out all missingness flags before they are set)
    df = engineer_features(patient_vitals_df)

    feature_cols = get_feature_columns()
    hourly_preds = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        feat_vec = np.array([float(row.get(f, 0.0)) for f in feature_cols], dtype=np.float32)

        preds = {}
        alerts = []
        for target, m in models.items():
            avail = m["features"]
            avail_vec = feat_vec[[feature_cols.index(f) for f in avail if f in feature_cols]]

            try:
                prob = float(m["lgbm"].predict(avail_vec.reshape(1, -1))[0])
            except Exception:
                prob = 0.0

            preds[target] = round(prob, 4)
            if prob >= m["threshold"]:
                alerts.append(target)

        hourly_preds.append({
            "hour":                  int(idx),
            "sepsis_prob":           preds.get("sepsis", 0.0),
            "hypotension_prob":      preds.get("hypotension", 0.0),
            "cardiac_risk_prob":     preds.get("hemodynamic_collapse", 0.0),
            "any_alert_fired":       len(alerts) > 0,
            "alert_targets":         alerts,
            "timestamp":             str(row.get("timestamp", "")),
        })

    return hourly_preds


def shadow_evaluate(mimic_version: str, models: dict) -> dict:
    """
    Main shadow evaluation for a MIMIC demo dataset.

    Args:
        mimic_version: "mimic3_demo" or "mimic4_demo"
        models: dict of loaded model artifacts

    Returns:
        dict with per-patient results and aggregate statistics
    """
    mapper = MIMICMapper(DATA_DIR / mimic_version, mimic_version)
    patients = mapper.list_patients()

    if not patients:
        logger.error(f"No patients found in {DATA_DIR / mimic_version}")
        return {}

    logger.info(f"\nShadow evaluation on {mimic_version}: {len(patients)} patients")
    results = {}
    all_alert_hours = []

    for pid in patients:
        try:
            vitals_df = mapper.load_patient_vitals(pid)
            if vitals_df is None or len(vitals_df) < 2:
                continue

            hourly_preds = predict_patient(vitals_df, models)
            n_alert_hours = sum(1 for h in hourly_preds if h["any_alert_fired"])
            total_hours   = len(hourly_preds)

            results[pid] = {
                "patient_id":      pid,
                "total_hours":     total_hours,
                "alert_hours":     n_alert_hours,
                "alert_rate":      round(n_alert_hours / max(total_hours, 1), 3),
                "max_sepsis":      max((h["sepsis_prob"] for h in hourly_preds), default=0.0),
                "max_hypotension": max((h["hypotension_prob"] for h in hourly_preds), default=0.0),
                "max_cardiac_risk":max((h["cardiac_risk_prob"] for h in hourly_preds), default=0.0),
                "hourly_timeline": hourly_preds,
            }
            all_alert_hours.append(n_alert_hours > 0)
            logger.info(
                f"  {pid}: {total_hours}h | {n_alert_hours} alert hours "
                f"| max sepsis={results[pid]['max_sepsis']:.2f} "
                f"| max cardiac={results[pid]['max_cardiac_risk']:.2f}"
            )

        except Exception as e:
            logger.warning(f"  {pid}: failed ({e})")

    # Aggregate summary
    summary = {
        "dataset":             mimic_version,
        "evaluated_at":        datetime.now().isoformat(),
        "n_patients":          len(results),
        "n_patients_any_alert": sum(all_alert_hours),
        "alert_rate_pct":      round(100 * sum(all_alert_hours) / max(len(all_alert_hours), 1), 1),
        "patient_results":     results,
    }

    # Save JSON report
    report_path = REPORTS_DIR / f"shadow_eval_{mimic_version}.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.success(f"  Report saved: {report_path}")

    return summary


def print_summary_table(all_results: dict[str, dict]) -> None:
    """Prints a clean summary table to terminal."""
    print("\n" + "=" * 70)
    print(f"{'SHADOW EVALUATION SUMMARY':^70}")
    print("=" * 70)
    print(f"{'Dataset':<20} {'Patients':<12} {'Alerting':<12} {'Alert Rate':<14}")
    print("-" * 70)
    for name, results in all_results.items():
        print(
            f"{name:<20} {results.get('n_patients', 0):<12} "
            f"{results.get('n_patients_any_alert', 0):<12} "
            f"{results.get('alert_rate_pct', 0):.1f}%"
        )
    print("=" * 70)
    print("\n📋 This is SHADOW mode: model predictions only, no clinical action taken.")
    print("   Compare alerts to actual clinical events in the patient records.")
    print(f"   Full report: {REPORTS_DIR}/shadow_eval_*.json\n")


def main():
    parser = argparse.ArgumentParser(description="Project Chronos Shadow Evaluation on MIMIC Demo")
    parser.add_argument("--mimic3", action="store_true", help="Run on MIMIC-III Demo (100 patients)")
    parser.add_argument("--mimic4", action="store_true", help="Run on MIMIC-IV Demo (100 patients)")
    parser.add_argument("--all",    action="store_true", help="Run on both MIMIC-III and MIMIC-IV")
    parser.add_argument("--report-only", action="store_true",
                        help="Print summary from existing JSON reports without re-running inference")
    args = parser.parse_args()

    if args.report_only:
        reports = {}
        for f in REPORTS_DIR.glob("shadow_eval_*.json"):
            with open(f) as fp:
                r = json.load(fp)
                reports[r.get("dataset", f.stem)] = r
        print_summary_table(reports)
        return

    if not (args.mimic3 or args.mimic4 or args.all):
        parser.print_help()
        return

    logger.info("Loading Project Chronos models...")
    models = load_models()
    if not models:
        logger.error("No models loaded. Run: python train_models.py --all")
        return

    all_results = {}

    if args.mimic3 or args.all:
        r = shadow_evaluate("mimic3_demo", models)
        if r:
            all_results["mimic3_demo"] = r

    if args.mimic4 or args.all:
        r = shadow_evaluate("mimic4_demo", models)
        if r:
            all_results["mimic4_demo"] = r

    print_summary_table(all_results)


if __name__ == "__main__":
    main()
