"""
Project Chronos - Dataset Preparation & Verification
======================================================
Runs BEFORE training. Does everything needed to make data ready:
  1. Auto-detects and unpacks any zip/tar/gz files in all data dirs
  2. Validates expected files exist and are non-empty
  3. Checks column names match what the loaders expect
  4. Runs row-count + class-balance sanity checks
  5. Prints a go/no-go report with specific fix instructions
  6. Optionally launches training only if all critical checks pass

Usage:
  python scripts/prepare_datasets.py              # check + fix only
  python scripts/prepare_datasets.py --train      # check, fix, then train
  python scripts/prepare_datasets.py --low-memory # optimise for ≤8GB free RAM
  python scripts/prepare_datasets.py --target sepsis --train
"""

import os
import sys
import zipfile
import tarfile
import gzip
import shutil
import subprocess
import argparse
import json
from pathlib import Path
from typing import Optional

from loguru import logger

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent.parent
DATA_DIR    = BACKEND_DIR / "data"

# ─────────────────────────────────────────────────────────────
# Expected dataset specifications (what loaders need)
# ─────────────────────────────────────────────────────────────
DATASET_SPECS = {

    "cinc2019": {
        "critical": True,
        "target":   "sepsis",
        "dir":      DATA_DIR / "cinc2019",
        "check_fn": lambda d: list(d.rglob("*.psv")),
        "min_files": 40000,
        "file_desc": ".psv patient records",
        "fix_hint":  (
            "Download training_setA.zip and training_setB.zip from:\n"
            "  https://physionet.org/files/challenge-2019/1.0.0/training/\n"
            "Then: python scripts/download_datasets.py --cinc"
        ),
    },

    "eicu_demo": {
        "critical": True,
        "target":   "hypotension",
        "dir":      DATA_DIR / "eicu_demo",
        "check_fn": lambda d: list(d.rglob("vitalPeriodic.csv")),
        "min_files": 1,
        "file_desc": "vitalPeriodic.csv",
        "required_columns": {
            "vitalPeriodic.csv": [
                "patientunitstayid", "observationoffset",
                "heartrate", "systemicmean", "systemicsystolic",
                "systemicdiastolic", "sao2", "respiration", "temperature"
            ],
            "patient.csv": [
                "patientunitstayid", "hospitaldischargestatus"
            ],
        },
        "fix_hint": (
            "Download from: https://physionet.org/content/eicu-crd-demo/2.0.1/\n"
            "Accept DUA → Download all CSVs → extract to data/eicu_demo/"
        ),
    },

    "vitaldb": {
        "critical": True,
        "target":   "hemodynamic_collapse",
        "dir":      DATA_DIR / "vitaldb",
        "check_fn": lambda d: list(d.glob("case_*.csv")),
        "min_files": 100,
        "file_desc": "case_*.csv files",
        "fix_hint": (
            "Run: python scripts/download_datasets.py --vitaldb --vitaldb-cases 6388\n"
            "(takes 3-5 hours, run overnight)"
        ),
    },

    "zenodo_cardiac": {
        "critical": False,
        "target":   "hemodynamic_collapse",
        "dir":      DATA_DIR / "zenodo_cardiac",
        "check_fn": lambda d: list(d.glob("CardiacPatientData.csv")),
        "min_files": 1,
        "file_desc": "CardiacPatientData.csv",
        "required_columns": {
            "CardiacPatientData.csv": [
                "ID", "SBP", "DBP", "HR", "SpO2", "GCS", "Outcome"
            ]
        },
        "fix_hint": (
            "curl -L 'https://zenodo.org/records/7603772/files/CardiacPatientData.csv'"
            " -o data/zenodo_cardiac/CardiacPatientData.csv"
        ),
    },

    "cudb_ventricular_tachyarrhythmia": {
        "critical": False,
        "target":   "hemodynamic_collapse",
        "dir":      DATA_DIR / "cudb_ventricular_tachyarrhythmia",
        "check_fn": lambda d: list(d.rglob("cu*.dat")),
        "min_files": 30,
        "file_desc": "WFDB .dat records",
        "fix_hint": (
            "wget -r -np -nH --cut-dirs=3 "
            "https://physionet.org/files/cudb/1.0.0/ "
            "-P data/cudb_ventricular_tachyarrhythmia/"
        ),
    },

    "sddb_sudden_cardiac": {
        "critical": False,
        "target":   "hemodynamic_collapse",
        "dir":      DATA_DIR / "sddb_sudden_cardiac",
        "check_fn": lambda d: list(d.rglob("*.dat")),
        "min_files": 20,
        "file_desc": "WFDB .dat records",
        "fix_hint": (
            "wget -r -np -nH --cut-dirs=3 "
            "https://physionet.org/files/sddb/1.0.0/ "
            "-P data/sddb_sudden_cardiac/ "
            "(still downloading — wait for it to complete)"
        ),
    },

    "cinc2009_hypotension": {
        "critical": False,
        "target":   "hypotension",
        "dir":      DATA_DIR / "cinc2009_hypotension",
        "check_fn": lambda d: list(d.rglob("*.dat")),
        "min_files": 50,
        "file_desc": "WFDB .dat records",
        "fix_hint":  "Data downloaded, may still be missing WFDB records inside the versioned folder.",
    },

    "uq_vital_signs": {
        "critical": False,
        "target":   "hemodynamic_collapse",
        "dir":      DATA_DIR / "uq_vital_signs",
        "check_fn": lambda d: list(d.rglob("*.csv")) + list(d.rglob("*.mat")),
        "min_files": 10,
        "file_desc": "case CSV/MAT files",
        "fix_hint":  "Still downloading? Check data/uq_vital_signs/uqvitalsignsdata/",
    },

    "sepsis_survival": {
        "critical": False,
        "target":   "sepsis",
        "dir":      DATA_DIR / "kaggle_supplements" / "sepsis_survival_clinical",
        "check_fn": lambda d: list(d.rglob("*.csv")),
        "min_files": 1,
        "file_desc": "CSV files",
        "required_columns": {
            "s41598-020-73558-3_sepsis_survival_primary_cohort.csv": [
                "age", "sex_0male_1female", "episode_number",
                "hospital_outcome_1alive_2dead"
            ]
        },
        "fix_hint": "Files should be in the s41598-020-73558-3_sepsis_survival_dataset/ subfolder.",
    },
}

# ─────────────────────────────────────────────────────────────
# Step 1: Auto-unpack archives
# ─────────────────────────────────────────────────────────────
def unpack_archives(root: Path) -> int:
    """Recursively finds and unpacks zip/tar.gz/gz files in root and all subdirs."""
    unpacked = 0
    archives = (
        list(root.rglob("*.zip")) +
        list(root.rglob("*.tar.gz")) +
        list(root.rglob("*.tar.bz2")) +
        list(root.rglob("*.tgz")) +
        list(root.rglob("*.tar"))
    )

    if not archives:
        logger.info("  No archives found — nothing to unpack.")
        return 0

    for archive in archives:
        extract_dir = archive.parent
        logger.info(f"  Unpacking: {archive.relative_to(root)}")
        try:
            if archive.suffix == ".zip":
                with zipfile.ZipFile(archive, "r") as z:
                    z.extractall(extract_dir)
            elif archive.suffix in (".gz", ".tgz") or archive.name.endswith(".tar.gz"):
                with tarfile.open(archive, "r:gz") as t:
                    t.extractall(extract_dir)
            elif archive.suffix in (".bz2",) or archive.name.endswith(".tar.bz2"):
                with tarfile.open(archive, "r:bz2") as t:
                    t.extractall(extract_dir)
            elif archive.suffix == ".tar":
                with tarfile.open(archive, "r") as t:
                    t.extractall(extract_dir)
            else:
                continue

            logger.success(f"    → Extracted to {extract_dir}")
            unpacked += 1

        except Exception as e:
            logger.error(f"    Failed to unpack {archive.name}: {e}")

    return unpacked

# ─────────────────────────────────────────────────────────────
# Step 2: Column validation helpers
# ─────────────────────────────────────────────────────────────
def check_columns(data_dir: Path, required_columns: dict) -> list[str]:
    """Returns list of missing columns (empty if all OK)."""
    problems = []
    for filename, expected_cols in required_columns.items():
        # Search recursively for the file
        matches = list(data_dir.rglob(filename))
        if not matches:
            problems.append(f"File not found: {filename}")
            continue
        found_path = matches[0]
        try:
            import pandas as pd
            df = pd.read_csv(found_path, nrows=1)
            actual_cols = set(df.columns.str.lower())
            for col in expected_cols:
                if col.lower() not in actual_cols:
                    problems.append(f"Missing column '{col}' in {filename}")
        except Exception as e:
            problems.append(f"Cannot read {filename}: {e}")
    return problems

# ─────────────────────────────────────────────────────────────
# Step 3: Class balance + row count spot-check
# ─────────────────────────────────────────────────────────────
def spot_check_data(name: str, data_dir: Path) -> Optional[dict]:
    """Quick spot-check for key datasets — returns stats dict or None."""
    import pandas as pd

    if name == "cinc2019":
        psv_files = list(data_dir.rglob("*.psv"))
        if not psv_files:
            return None
        # Sample 100 files for label distribution
        sample = psv_files[:100]
        sepsis_count = 0
        for f in sample:
            try:
                df = pd.read_csv(f, sep="|")
                if "SepsisLabel" in df.columns and df["SepsisLabel"].max() == 1:
                    sepsis_count += 1
            except Exception:
                pass
        return {"n_files": len(psv_files), "sepsis_rate_sampled": sepsis_count / len(sample)}

    elif name == "eicu_demo":
        vital_path = next(data_dir.rglob("vitalPeriodic.csv"), None)
        if not vital_path:
            return None
        df = pd.read_csv(vital_path, usecols=["systemicmean"], nrows=5000)
        return {"n_vital_rows_sampled": len(df), "map_missing_pct": df["systemicmean"].isna().mean()}

    elif name == "vitaldb":
        case_files = list(data_dir.glob("case_*.csv"))
        if not case_files:
            return None
        sample = case_files[:10]
        non_empty = sum(1 for f in sample if f.stat().st_size > 512)
        return {"n_case_files": len(case_files), "non_empty_sample": f"{non_empty}/10"}

    elif name == "zenodo_cardiac":
        csv_path = next(data_dir.glob("CardiacPatientData.csv"), None)
        if not csv_path:
            return None
        df = pd.read_csv(csv_path)
        outcome_col = "Outcome" if "Outcome" in df.columns else None
        if outcome_col:
            arrest_rate = df[outcome_col].eq(0).mean()  # 0 = died/arrest in this dataset
        else:
            arrest_rate = float("nan")
        return {"n_rows": len(df), "n_cols": len(df.columns), "outcome_col": outcome_col}

    return None

# ─────────────────────────────────────────────────────────────
# Step 4: Full verification run
# ─────────────────────────────────────────────────────────────
def run_verification(target_filter: Optional[str] = None) -> tuple[bool, dict]:
    """Runs all checks. Returns (all_critical_passed, full_report_dict)."""
    import pandas as pd

    logger.info("\n" + "="*70)
    logger.info("PROJECT CHRONOS — DATASET PREPARATION & VERIFICATION")
    logger.info("="*70)

    report = {}
    any_critical_failed = False

    # Step 1: Unpack archives in the data directory
    logger.info("\n[1/4] Scanning for archives to unpack...")
    n_unpacked = unpack_archives(DATA_DIR)
    logger.info(f"  Unpacked {n_unpacked} archive(s).")

    # Step 2: Dataset-by-dataset validation
    logger.info("\n[2/4] Validating each dataset...")

    for ds_name, spec in DATASET_SPECS.items():
        if target_filter and spec.get("target") != target_filter:
            continue

        ds_dir    = spec["dir"]
        critical  = spec["critical"]
        tag       = "🔴 CRITICAL" if critical else "🟡 Optional"
        found     = ds_dir.exists()

        entry = {
            "dir":      str(ds_dir),
            "critical": critical,
            "target":   spec.get("target", "?"),
            "status":   "unknown",
        }

        if not found:
            entry["status"]  = "missing_dir"
            entry["problem"] = f"Directory not found: {ds_dir}"
            entry["fix"]     = spec.get("fix_hint", "Create dir and download data.")
            if critical:
                any_critical_failed = True
            logger.warning(f"  {tag} [{ds_name}] — Directory missing")
            report[ds_name] = entry
            continue

        # Count expected files
        files = spec["check_fn"](ds_dir)
        n = len(files)
        min_f = spec.get("min_files", 1)

        if n < min_f:
            entry["status"]  = "insufficient_files"
            entry["n_found"] = n
            entry["n_min"]   = min_f
            entry["problem"] = f"Only {n} {spec['file_desc']} (need ≥ {min_f})"
            entry["fix"]     = spec.get("fix_hint", "")
            if critical:
                any_critical_failed = True
            logger.warning(f"  {tag} [{ds_name}] — {n}/{min_f} {spec['file_desc']}")
            report[ds_name] = entry
            continue

        # Column validation (if spec defines expected columns)
        col_problems = []
        if "required_columns" in spec:
            col_problems = check_columns(ds_dir, spec["required_columns"])

        if col_problems:
            entry["status"]   = "column_mismatch"
            entry["problems"] = col_problems
            entry["fix"]      = "Check download integrity — possible partial/corrupt file"
            if critical:
                any_critical_failed = True
            logger.error(f"  {tag} [{ds_name}] — Column mismatch: {col_problems}")
        else:
            entry["status"]  = "ok"
            entry["n_files"] = n
            logger.success(f"  ✅ [{ds_name}] — {n} {spec['file_desc']}")

        # Spot-check
        stats = spot_check_data(ds_name, ds_dir)
        if stats:
            entry["stats"] = stats
            for k, v in stats.items():
                logger.info(f"     · {k}: {v}")

        report[ds_name] = entry

    # Step 3: Additional eICU column sanity check (renamed/different versions)
    logger.info("\n[3/4] eICU column deep-check...")
    eicu_dir = DATA_DIR / "eicu_demo"
    vital_csv = next(eicu_dir.rglob("vitalPeriodic.csv"), None)
    if vital_csv:
        import pandas as pd
        df_h = pd.read_csv(vital_csv, nrows=0)
        wanted = ["systemicmean", "systemicsystolic", "systemicdiastolic",
                  "heartrate", "sao2", "respiration", "temperature",
                  "observationoffset", "patientunitstayid"]
        actual_lower = [c.lower() for c in df_h.columns]
        missing = [c for c in wanted if c not in actual_lower]
        if missing:
            logger.error(f"  eICU vitalPeriodic missing columns: {missing}")
            report["eicu_demo"]["eicu_column_check"] = "FAILED"
            report["eicu_demo"]["missing_cols"] = missing
            any_critical_failed = True
        else:
            logger.success(f"  ✅ eICU vitalPeriodic — all {len(wanted)} required columns present")
    else:
        logger.warning("  eICU vitalPeriodic.csv not found — cannot column-check")

    # Step 4: Zenodo column alias check (Ceratinine vs Creatinine typo)
    logger.info("\n[4/4] Zenodo cardiac special checks...")
    zenodo_csv = DATA_DIR / "zenodo_cardiac" / "CardiacPatientData.csv"
    if zenodo_csv.exists():
        import pandas as pd
        df_z = pd.read_csv(zenodo_csv, nrows=0)
        cols = list(df_z.columns)
        has_typo = "Ceratinine" in cols   # original dataset has this typo
        has_correct = "Creatinine" in cols
        if has_typo and not has_correct:
            logger.info("  Zenodo CSV has 'Ceratinine' typo — loader already handles this ✅")
        elif not has_typo and not has_correct:
            logger.warning("  Zenodo: neither Ceratinine nor Creatinine column found")
        logger.info(f"  Zenodo columns ({len(cols)}): {', '.join(cols[:10])}...")

    return not any_critical_failed, report


# ─────────────────────────────────────────────────────────────
# RAM / Resource Report
# ─────────────────────────────────────────────────────────────
def print_resource_check() -> None:
    """Prints current system RAM usage and training recommendations."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        total_gb = vm.total / 1e9
        avail_gb = vm.available / 1e9
        used_pct = vm.percent
        swap = psutil.swap_memory()
        swap_used_gb = swap.used / 1e9
    except ImportError:
        logger.warning("psutil not installed — cannot check RAM. pip install psutil")
        return

    print("\n" + "="*60)
    print("SYSTEM RESOURCE CHECK")
    print("="*60)
    print(f"  Total RAM:      {total_gb:.1f} GB")
    print(f"  Available RAM:  {avail_gb:.1f} GB ({100-used_pct:.0f}% free)")
    print(f"  Swap used:      {swap_used_gb:.1f} GB")

    # RAM requirements for training
    requirements = {
        "sepsis (LightGBM + XGBoost, 1.6M rows)": 8.0,
        "sepsis SMOTE-ENN peak":                   5.0,
        "hemodynamic_collapse (VitalDB, full dataset)":  9.0,
        "GRU-D / TCN neural model (MPS shared)":  2.0,
        "CatBoost":                                3.0,
        "SHAP TreeExplainer":                      2.0,
        "SAFETY BUFFER recommended":               4.0,
    }

    print(f"\n  TRAINING RAM REQUIREMENTS:")
    total_needed = sum(requirements.values())
    for name, gb in requirements.items():
        print(f"  {'↪ ' + name:<50} ~{gb:.1f} GB")
    print(f"  {'TOTAL PEAK (worst case)':.<52} ~{total_needed:.1f} GB")
    print()

    if avail_gb < 6.0:
        print("  ❌ WARNING: Less than 6 GB free — HIGH RISK of OOM crash during training.")
        print("  BEFORE TRAINING:")
        print("    1. Quit Chrome, Discord, Spotify          → frees ~3-4 GB")
        print("    2. Quit Antigravity                        → frees ~5.5 GB")
        print("    3. Run training in Terminal.app (not VS Code)")
        print("    4. Use --low-memory flag to reduce peak usage")
        print(f"\n  After quitting the above, estimated free RAM: ~{avail_gb + 8:.1f} GB")
    elif avail_gb < 10.0:
        print("  ⚠️  WARNING: Borderline RAM — training will cause swap pressure.")
        print("  RECOMMENDED: Quit Chrome/Discord/Spotify before training.")
        print("  USE: --low-memory flag for safer peak usage.")
    else:
        print("  ✅ RAM looks OK for training.")

    print("="*60)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Project Chronos — Dataset Preparation & Pre-Training Verification"
    )
    parser.add_argument("--train", action="store_true",
                        help="Launch training after successful verification")
    parser.add_argument("--target", choices=["sepsis", "hypotension", "hemodynamic_collapse"],
                        help="Only verify datasets for one target")
    parser.add_argument("--low-memory", action="store_true",
                        help="Print low-memory training command for <8GB free RAM")
    parser.add_argument("--save-report", type=str, default="",
                        help="Save JSON report to file (e.g. --save-report prep_report.json)")
    parser.add_argument("--resource-check", action="store_true",
                        help="Print RAM usage and resource recommendations, then exit")
    args = parser.parse_args()

    if args.resource_check:
        print_resource_check()
        return

    # Run resource check first
    print_resource_check()

    # Run full verification
    all_ok, report = run_verification(target_filter=args.target)

    # Save report
    if args.save_report:
        with open(args.save_report, "w") as f:
            json.dump(report, f, indent=2)
        logger.success(f"Report saved to {args.save_report}")

    # Final verdict
    n_ok      = sum(1 for v in report.values() if v.get("status") == "ok")
    n_warn    = sum(1 for v in report.values() if v.get("status") != "ok" and not v.get("critical"))
    n_crit    = sum(1 for v in report.values() if v.get("status") != "ok" and v.get("critical"))

    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"  ✅ Passed:           {n_ok}")
    print(f"  ⚠️  Optional warnings: {n_warn}")
    print(f"  ❌ Critical failures: {n_crit}")

    for ds_name, entry in report.items():
        status = entry.get("status", "?")
        if status != "ok":
            crit_tag = "[CRITICAL]" if entry.get("critical") else "[optional]"
            print(f"\n  {crit_tag} {ds_name}:")
            print(f"    Problem: {entry.get('problem', entry.get('problems', 'unknown'))}")
            if entry.get("fix"):
                print(f"    Fix:     {entry['fix']}")

    print("="*70)

    if not all_ok:
        print("\n❌ Critical checks failed. Fix the issues above before training.")
        print("   Run again after fixing to re-verify.")
        sys.exit(1)

    print("\n✅ All critical checks passed. Data is ready for training.\n")

    if args.train:
        target_arg = f"--target {args.target}" if args.target else "--all"
        if args.low_memory:
            # Low-memory: train one target at a time, let OS breathe between runs
            targets = [args.target] if args.target else ["hypotension", "sepsis", "hemodynamic_collapse"]
            for t in targets:
                cmd = [sys.executable, str(BACKEND_DIR / "train_models.py"),
                       "--target", t, "--resume"]
                print(f"  🚀 Launching: {' '.join(cmd)}")
                subprocess.run(cmd, check=False)
        else:
            # Bug 40 fix: target_arg may be '--target sepsis' (two tokens) as a single string.
            # subprocess list args must be split individually — passing as one string causes
            # argparse inside train_models.py to see '--target sepsis' as one unknown arg.
            base_train_cmd = [sys.executable, str(BACKEND_DIR / "train_models.py")]
            target_tokens = target_arg.split()   # ['--target', 'sepsis'] or ['--all']
            cmd = base_train_cmd + target_tokens
            print(f"  🚀 Launching: {' '.join(cmd)}")
            subprocess.run(cmd, check=False)
    else:
        base_cmd = "python backend/train_models.py"
        target_str = f"--target {args.target}" if args.target else "--all"
        print("  To start training, run:")
        if args.low_memory:
            print(f"    # Low-memory mode: train sequentially with resume support")
            for t in (["hypotension", "sepsis", "hemodynamic_collapse"] if not args.target else [args.target]):
                print(f"    {base_cmd} --target {t} --resume")
        else:
            print(f"    {base_cmd} {target_str} --resume")
        print(f"\n  Or with auto-launch:")
        print(f"    python scripts/prepare_datasets.py {target_str} --train")


if __name__ == "__main__":
    main()
