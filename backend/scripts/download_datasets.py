"""
Project Chronos - Dataset Downloader
=====================================
Downloads and validates all required open-access datasets for training.

Datasets downloaded:
  1. PhysioNet CinC 2019  → ./data/cinc2019/
  2. Health Gym (HuggingFace) → ./data/healthgym/
  3. VitalDB               → ./data/vitaldb/
  4. MIMIC-IV Demo         → ./data/mimic4_demo/  (streamer only)
  5. MIMIC-III Demo        → ./data/mimic3_demo/  (streamer only)

Usage:
  python scripts/download_datasets.py [--all | --cinc | --healthgym | --vitaldb | --mimic]
"""

import os
import sys
import zipfile
import tarfile
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"

DATASET_DIRS = {
    "cinc2019":   DATA_DIR / "cinc2019",
    "healthgym":  DATA_DIR / "healthgym",
    "vitaldb":    DATA_DIR / "vitaldb",
    "mimic4":     DATA_DIR / "mimic4_demo",
    "mimic3":     DATA_DIR / "mimic3_demo",
}

# ─────────────────────────────────────────────
# Dataset Sources (all open access, no auth)
# ─────────────────────────────────────────────
# CinC 2019: Use the content-based URL (files/ path works, challenge/ path does not)
CINC2019_URLS = [
    "https://physionet.org/content/challenge-2019/1.0.0/training/training_setA.zip",
    "https://physionet.org/content/challenge-2019/1.0.0/training/training_setB.zip",
]
# Fallback manual download page if above still 404s:
CINC2019_PAGE = "https://physionet.org/content/challenge-2019/1.0.0/"

# MIMIC IV Demo - requires clicking a public DUA on PhysioNet
# After clicking agree, the download link is public
MIMIC4_DEMO_URL = "https://physionet.org/files/mimic-iv-demo/2.2/"

# MIMIC III Demo
MIMIC3_DEMO_URL = "https://physionet.org/files/mimiciii-demo/1.4/"

# VitalDB - directly accessible via their public API and CSV exports
VITALDB_API = "https://api.vitaldb.net"


def sizeof_fmt(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """Stream-download a file with a progress bar. Returns True on success."""
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest, "wb") as f, tqdm(
            desc=dest.name, total=total, unit="B",
            unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))
        logger.success(f"Downloaded {dest.name} ({sizeof_fmt(dest.stat().st_size)})")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


# ─────────────────────────────────────────────
# 1. CinC 2019 (PhysioNet Open Challenge)
# ─────────────────────────────────────────────
def download_cinc2019():
    """
    Downloads the PhysioNet/CinC 2019 Sepsis Challenge dataset.
    - training_setA: Beth Israel Deaconess Medical Center patients
    - training_setB: Emory University patients
    - 40,000+ patients, 40 clinical variables, labeled with Sepsis-3 criteria
    - Format: pipe-delimited .psv files, one per patient
    """
    out_dir = DATASET_DIRS["cinc2019"]
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading PhysioNet CinC 2019 Sepsis Challenge dataset...")
    
    for url in CINC2019_URLS:
        filename = url.split("/")[-1]
        zip_path = out_dir / filename
        
        if zip_path.exists():
            logger.info(f"  {filename} already exists, skipping download.")
        else:
            success = download_file(url, zip_path)
            if not success:
                logger.warning(
                    f"  Could not auto-download {filename}.\n"
                    f"  Please manually download from: {url}\n"
                    f"  and place in: {out_dir}"
                )
                continue
        
        # Extract
        extract_dir = out_dir / filename.replace(".zip", "")
        if not extract_dir.exists():
            logger.info(f"  Extracting {filename}...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)
            logger.success(f"  Extracted to {extract_dir}")
    
    # Validate
    psv_files = list(out_dir.rglob("*.psv"))
    logger.info(f"  CinC 2019 validation: found {len(psv_files)} patient records.")
    if len(psv_files) < 10000:
        logger.warning("  Expected 40,000+ records. Check download integrity.")
    return len(psv_files) > 0


# ─────────────────────────────────────────────
# 2. Health Gym (HuggingFace Synthetic Data)
# ─────────────────────────────────────────────
def download_healthgym():
    """
    Downloads Health Gym Synthetic Acute Hypotension + Sepsis datasets from HuggingFace.
    - GAN-generated synthetic data derived from MIMIC-III
    - Explicitly bypasses HIPAA constraints (synthetic, 0.045% re-identification risk)
    - Contains 3,910 ICU patient trajectories with acute hypotension events
    - Includes vasopressor data, MAP readings, and hemodynamic trajectories
    """
    out_dir = DATASET_DIRS["healthgym"]
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Health Gym datasets from HuggingFace...")
    
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        
        # Health Gym Hypotension dataset
        logger.info("  Downloading Acute Hypotension dataset...")
        snapshot_download(
            repo_id="ryanwangzf/HealthGym",
            repo_type="dataset",
            local_dir=str(out_dir),
        )
        
        files = list(out_dir.rglob("*.csv")) + list(out_dir.rglob("*.parquet"))
        logger.success(f"  Health Gym downloaded. Found {len(files)} data files.")
        return len(files) > 0
    
    except Exception as e:
        logger.error(f"Health Gym download failed: {e}")
        logger.info(
            "  Alternative: Visit https://huggingface.co/datasets/ryanwangzf/HealthGym\n"
            f"  and manually download CSVs to: {out_dir}"
        )
        return False


# ─────────────────────────────────────────────
# 3. VitalDB — uses the official vitaldb Python library
# ─────────────────────────────────────────────
def download_vitaldb(n_cases: int = 100):
    """
    Downloads VitalDB high-resolution patient cases using the official
    vitaldb Python library (pip install vitaldb).

    - 6,388 surgical patients, 1-7 second granularity vital signs
    - Tracks: HR, MAP, SpO2, RR, EtCO2, BT (body temp)
    - ICU outcomes: mortality flag, ICU stay length
    - Used as the Cardiac Arrest prediction training set

    Library docs: https://vitaldb.net/dataset/#h.gzbn4nvqy89o
    """
    out_dir = DATASET_DIRS["vitaldb"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # The tracks we care about for cardiac arrest prediction
    TARGET_TRACKS = ["HR", "NIBP_MBP", "SpO2", "RR", "BT", "EtCO2", "PLETH"]

    try:
        import vitaldb  # pip install vitaldb
    except ImportError:
        logger.error(
            "vitaldb library not installed. Run: pip install vitaldb\n"
            "Then retry: python scripts/download_datasets.py --vitaldb"
        )
        return False

    logger.info(f"Fetching VitalDB case list via vitaldb library...")

    try:
        # Step 1: Get the public caselist (direct API — confirmed working)
        import pandas as pd
        caselist_path = out_dir / "caselist.csv"

        # Always re-fetch if cached caselist has fewer rows than requested cases
        if caselist_path.exists():
            df_cached = pd.read_csv(caselist_path)
            if len(df_cached) < n_cases:
                logger.warning(
                    f"  Cached caselist has only {len(df_cached)} rows, but {n_cases} cases requested. "
                    f"Re-fetching full caselist from API..."
                )
                caselist_path.unlink()  # delete stale cache
            else:
                logger.info(f"  caselist.csv already exists ({len(df_cached)} cases), using cache.")
                df = df_cached

        if not caselist_path.exists():
            df = pd.read_csv("https://api.vitaldb.net/cases")
            df.to_csv(caselist_path, index=False)
            logger.success(f"  Caselist saved: {len(df)} cases → {caselist_path}")

        # Step 2: Download waveforms using the official vitaldb library
        # vitaldb.load_case(caseid, track_names, interval=1) → numpy ndarray
        new_downloads = 0
        skipped = 0
        failed = 0
        to_download = min(n_cases, len(df))
        logger.info(f"  Requesting {to_download} case waveforms (1s resolution) — {len(list(out_dir.glob('case_*.csv')))} already on disk...")
        for _, row in tqdm(df.head(to_download).iterrows(), total=to_download, desc="VitalDB"):
            case_id = int(row["caseid"])
            out_file = out_dir / f"case_{case_id}.csv"
            if out_file.exists():
                skipped += 1
                continue
            try:
                arr = vitaldb.load_case(case_id, TARGET_TRACKS, interval=1)
                if arr is not None and arr.size > 0:
                    case_df = pd.DataFrame(arr, columns=TARGET_TRACKS)
                    if case_df.notna().any().any():
                        case_df.to_csv(out_file, index=False)
                        new_downloads += 1
                    else:
                        failed += 1
                else:
                    failed += 1
            except Exception as ex:
                logger.debug(f"  case {case_id} failed: {ex}")
                failed += 1
        total_saved = new_downloads + skipped
        logger.success(
            f"  VitalDB: {total_saved}/{to_download} total usable "
            f"({new_downloads} new, {skipped} already existed, {failed} no-data) → {out_dir}"
        )
        return total_saved > 0

    except Exception as e:
        logger.error(f"VitalDB download failed: {e}")
        logger.info(
            "  Manual alternative:\n"
            "  1. Visit https://vitaldb.net/dataset/\n"
            "  2. Use the vitaldb Python library directly:\n"
            "     import vitaldb\n"
            "     df = vitaldb.load_cases(['HR','NIBP_MBP','SpO2','RR','BT'])\n"
            f"  3. Save to: {out_dir}/vitaldb_cases.csv"
        )
        return False


# ─────────────────────────────────────────────
# 4. MIMIC Demos (Simulator source only)
# ─────────────────────────────────────────────
def download_mimic_instructions():
    """
    The MIMIC-III and MIMIC-IV Demo datasets are open access but require
    clicking a Data Use Agreement on PhysioNet (no training required).
    
    Instructions printed here — no automated download attempted
    to ensure users acknowledge the DUA appropriately.
    """
    mimic4_dir = DATASET_DIRS["mimic4"]
    mimic3_dir = DATASET_DIRS["mimic3"]
    mimic4_dir.mkdir(parents=True, exist_ok=True)
    mimic3_dir.mkdir(parents=True, exist_ok=True)
    
    instructions = """
╔══════════════════════════════════════════════════════════════════════════╗
║              MIMIC DEMO DATASETS — MANUAL DOWNLOAD REQUIRED             ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  These datasets require you to click a Data Use Agreement on PhysioNet. ║
║  This is instant (no CITI training, no institutional review needed).    ║
║                                                                          ║
║  MIMIC-IV Clinical Demo (v2.2)                                          ║
║  URL: https://physionet.org/content/mimic-iv-demo/2.2/                  ║
║  Download: Click "Download" → Accept DUA → Download hosp.zip            ║
║  Extract to: ./data/mimic4_demo/                                         ║
║                                                                          ║
║  MIMIC-III Clinical Demo (v1.4)                                         ║
║  URL: https://physionet.org/content/mimiciii-demo/1.4/                  ║
║  Download: Click "Download" → Accept DUA → Download all CSVs            ║
║  Extract to: ./data/mimic3_demo/                                         ║
║                                                                          ║
║  Key files needed:                                                       ║
║    - chartevents.csv  (vital signs over time)                           ║
║    - labevents.csv    (lab values over time)                             ║
║    - patients.csv     (demographics)                                     ║
║    - admissions.csv   (admission info)                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    print(instructions)
    logger.info("MIMIC instructions printed. Please follow the steps above.")
    return True


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Project Chronos — Dataset Downloader"
    )
    parser.add_argument("--all",      action="store_true", help="Download all datasets")
    parser.add_argument("--cinc",     action="store_true", help="Download CinC 2019")
    parser.add_argument("--healthgym",action="store_true", help="Download Health Gym")
    parser.add_argument("--vitaldb",  action="store_true", help="Download VitalDB")
    parser.add_argument("--mimic",    action="store_true", help="Print MIMIC instructions")
    parser.add_argument("--vitaldb-cases", type=int, default=2000,
                        help="Number of VitalDB cases to download (default: 2000)")
    args = parser.parse_args()
    
    if not any([args.all, args.cinc, args.healthgym, args.vitaldb, args.mimic]):
        parser.print_help()
        sys.exit(1)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    
    if args.all or args.cinc:
        results["CinC 2019"] = download_cinc2019()
    
    # Bug 29 fix: HealthGym excluded from --all — no load_healthgym() in train_models.py.
    # Downloading via --all wastes disk/bandwidth for unused data.
    # Only runs when --healthgym is explicitly passed (for future integration).
    if args.healthgym:
        results["Health Gym"] = download_healthgym()
    
    if args.all or args.vitaldb:
        results["VitalDB"] = download_vitaldb(n_cases=args.vitaldb_cases)
    
    if args.all or args.mimic:
        results["MIMIC (instructions)"] = download_mimic_instructions()
    
    # Summary
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    for name, success in results.items():
        status = "✅ OK" if success else "❌ FAILED / Manual Download Required"
        print(f"  {name:25s}: {status}")
    print("="*50)


if __name__ == "__main__":
    main()
