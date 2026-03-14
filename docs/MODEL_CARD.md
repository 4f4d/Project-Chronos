# Model Card — Project Chronos

Following the framework described by Mitchell et al. (FAT* 2019).

---

## Model Details

- **Model name**: Project Chronos Predictive ICU Early Warning System
- **Model version**: 1.0 (trained 2026-03-04/05)
- **Model type**: 4-engine ensemble (LightGBM, XGBoost, GRU-D, TCN) with LightGBM meta-stacker and isotonic calibration
- **Training framework**: scikit-learn 1.6.1, LightGBM 4.6.0, XGBoost 2.1.3, PyTorch 2.6.0
- **Hardware**: Apple M4 (24 GB unified memory, MPS backend for GPU)
- **Training time**: Approximately 19.4 hours for all three targets
- **License**: Academic and research use

---

## Intended Use

**Primary intended use**: Research prototype demonstrating feasibility of multi-engine ensemble ICU prediction with explainable AI (SHAP) and physics-based validation.

**Primary intended users**: Researchers, clinical informaticists, and academic reviewers evaluating approaches to ICU early warning systems.

**Out-of-scope uses**: This system is not approved for clinical deployment. It must not be used for making treatment decisions, triaging patients, or replacing clinical judgment in any healthcare setting. No regulatory review (FDA, CE, or equivalent) has been conducted.

---

## Training Data

| Dataset | Source | Records | Target | License |
| ------- | ------ | ------- | ------ | ------- |
| PhysioNet CinC 2019 | physionet.org | 40,336 | Sepsis | PhysioNet Open Access |
| eICU Collaborative Demo | physionet.org | 2,520 ICU stays | Hypotension | PhysioNet Open Access |
| VitalDB | vitaldb.net | 6,389 surgical cases | Hemodynamic Collapse | Open Access |
| Zenodo Cardiac (Sri Lanka) | zenodo.org | 112 cardiac arrests | Hemodynamic Collapse | Creative Commons |
| CUDB (Creighton University) | physionet.org | 35 records | Hemodynamic Collapse | PhysioNet Open Access |
| SDDB (Sudden Death) | physionet.org | 23 Holter records | Hemodynamic Collapse | PhysioNet Open Access |

**Important**: VitalDB contains elective surgical cases from a single institution (Seoul National University Hospital). The hemodynamic events in this dataset are frequently iatrogenic (e.g., anesthesia-induced hypotension) and do not represent the clinical profile of spontaneous ICU hemodynamic collapse. This domain mismatch is the primary driver of the hemodynamic collapse model's underperformance.

---

## Evaluation Data

Evaluation uses the held-out 20% patient-level test split from GroupShuffleSplit (random_state=42). No additional external validation datasets have been used. MIMIC-III and MIMIC-IV demo datasets are reserved for prospective shadow evaluation.

---

## Performance Metrics

### Primary Metrics

| Target | AUROC | AUPRC | Sensitivity | Specificity | F-beta=2 Threshold |
| ------ | ----- | ----- | ----------- | ----------- | ------------------ |
| Septic Shock | 0.718 | 0.085 | 0.778 | 0.357 | Tuned on dev set |
| Blood Pressure Collapse | 0.942 | 0.458 | 0.888 | 0.880 | Tuned on dev set |
| Hemodynamic Collapse | 0.680 | 0.043 | 0.155 | 0.984 | Tuned on dev set |

### Cross-Validation Metrics (5-fold StratifiedGroupKFold)

| Target | CV AUROC (mean +/- std) | CV-to-Test Gap |
| ------ | ----------------------- | -------------- |
| Septic Shock | 0.817 +/- 0.006 | -0.099 |
| Blood Pressure Collapse | 0.971 +/- 0.008 | -0.029 |
| Hemodynamic Collapse | 0.860 +/- 0.059 | -0.180 |

### Comparison to Clinical Baselines

| System | AUROC (Sepsis) | Source |
| ------ | -------------- | ------ |
| Epic Sepsis Model | 0.63 | Wong et al., JAMA Internal Medicine, 2021 |
| NEWS2 | 0.665 | Royal College of Physicians, 2017 |
| qSOFA | ~0.61 | Seymour et al., JAMA, 2016 |
| **Chronos (Sepsis)** | **0.718** | This work |
| **Chronos (Hypotension)** | **0.942** | This work |

---

## Quantitative Analysis

### Cross-Validation to Test Gap Analysis

The 0.099 AUROC gap for sepsis and 0.180 gap for hemodynamic collapse require explanation:

**Sepsis gap (0.099)**: The sepsis model was trained with 3 of 4 engines (TCN excluded due to NaN training instability). The missing engine likely accounts for a portion of the performance gap. No imputation leakage was found during code audit: feature engineering uses hardcoded population-median fallback values, not data-derived statistics.

**Hemodynamic collapse gap (0.180)**: Two contributing factors: (1) the VitalDB proxy labels are noisy, causing high CV variance (std=0.059), and (2) the small genuine cardiac event datasets (170 total) are insufficient to anchor the decision boundary. The high CV variance (0.059 vs 0.006 for sepsis) indicates the model's performance is unstable across folds.

**Hypotension gap (0.029)**: Within expected range. The eICU dataset provides clean, continuous MAP recordings with well-defined AHE labels, resulting in both high performance and low variance.

### Calibration

Isotonic calibration is applied to raw ensemble scores to produce clinically meaningful probabilities. However, formal calibration metrics (Brier score, calibration curves) have not yet been computed. This is a known gap in the current evaluation.

### Subgroup Analysis

No subgroup analysis by age, sex, race, or comorbidity has been conducted. The CinC 2019 dataset does not include race information. The eICU dataset includes limited demographic fields, but subgroup performance has not been evaluated. This is a significant limitation for any clinical deployment pathway.

---

## Ethical Considerations

1. **Algorithmic bias**: Without subgroup analysis, it is unknown whether the model performs differently across demographic groups. ICU populations are not demographically representative of the general population, and training data may encode biases from historical care patterns.

2. **Automation bias**: Clinical decision support systems can create over-reliance on algorithmic predictions. The system mitigates this through SHAP-based explanations (showing why, not just what) and the multi-agent debate format (presenting conflicting perspectives).

3. **Alarm fatigue**: Poorly calibrated alert thresholds can worsen the alarm fatigue problem the system aims to solve. The F-beta=2 threshold favors sensitivity, which may increase false positive rates. The physics engine's trajectory-gated overrides partially address this.

4. **Privacy**: The system is designed for on-premise deployment with no network egress. However, the LLM component processes patient data locally through Ollama, and LLM inference logs may persist on disk.

---

## Caveats and Recommendations

1. **Do not use for clinical decisions.** This is a research prototype without regulatory approval.

2. **Hemodynamic collapse predictions should be interpreted with extreme caution.** The model performs below the NEWS2 baseline and uses proxy labels from a non-representative dataset.

3. **Sepsis specificity is low (0.357).** The current threshold produces a high false positive rate. In a clinical setting, this would need to be recalibrated based on the institution's tolerance for false alarms versus missed cases.

4. **The system has been evaluated on retrospective data only.** Prospective validation is required before any clinical use.

5. **The cardiac output estimation in the physics engine is a heuristic.** It should not be interpreted as a physiologically accurate measurement.

---

## References

- Mitchell M, et al. Model Cards for Model Reporting. FAT* 2019.
- Wong A, et al. External Validation of a Widely Implemented Proprietary Sepsis Prediction Model. JAMA Internal Medicine. 2021;181(8):1065-1070.
- Kumar V, et al. Duration of hypotension before initiation of effective antimicrobial therapy. Critical Care Medicine. 2006;34(6):1589-1596.
- Singer M, et al. Sepsis-3: Third International Consensus Definitions. JAMA. 2016;315(8):801-810.
- Che Z, et al. Recurrent Neural Networks for Multivariate Time Series with Missing Values. Scientific Reports. 2018;8:6085.
