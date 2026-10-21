# Maintenance Shift Planner — NexAI Hack 2026 (PS 2.1)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Tests: Pytest Passing](https://img.shields.io/badge/tests-10%20passed-brightgreen.svg)](tests/test_core.py)

An end-to-end industrial predictive maintenance and shift scheduling platform developed for **NexAI Hack 2026 (Problem Statement 2.1: Maintenance Shift Planner)**. 

The application couples physics-informed machine learning with an explicit financial cost model to optimize predictive alert thresholds and allocate finite technician maintenance capacity to high-risk milling machines.

---

## 📑 Table of Contents
1. [Problem Statement & Overview](#problem-statement--overview)
2. [Key Architecture & Workflow](#key-architecture--workflow)
3. [Mandatory Requirements Implementation](#mandatory-requirements-implementation)
   - [Requirement A: Machine Failure Prediction](#requirement-a--machine-failure-prediction)
   - [Requirement B: Explicit Cost Model & Alert Threshold](#requirement-b--explicit-cost-model--alert-threshold)
   - [Requirement C: Maintenance Shift Planner](#requirement-c--maintenance-shift-planner)
4. [Dataset Governance, Source & Synthetic Limitation](#dataset-governance-source--synthetic-limitation)
5. [Model Methodology & Actual Test Evaluation Metrics](#model-methodology--actual-test-evaluation-metrics)
6. [Setup and Run Instructions](#setup-and-run-instructions)
7. [Testing & Verification](#testing--verification)
8. [Public Deployment Guide](#public-deployment-guide)
9. [Development Log & AI Tools Used](#development-log--ai-tools-used)

---

## Problem Statement & Overview

In high-throughput manufacturing plants, machine downtime carries asymmetric financial consequences:
- **Unplanned Breakdowns (Missed Failures / FN):** Halt assembly lines, damage cutting tools, ruin workpieces, and incur emergency repair costs ($C_{FN} \approx \$1,000+$).
- **False Alarms (FP):** Dispatch technicians unnecessarily to inspect healthy equipment ($C_{FP} \approx \$100$).
- **Capacity Bottlenecks:** Maintenance teams have fixed technician capacity per shift ($N$ slots). Standard predictive dashboards merely produce probability scores without answering: *Which machines must be serviced during this shift, and which can be safely deferred?*

The **Maintenance Shift Planner** solves this by:
1. Predicting failure probabilities and diagnosing overlapping failure modes for individual machines.
2. Deriving the cost-minimizing alert threshold $\tau^*$ via validation grid optimization rather than using an arbitrary $0.5$ default.
3. Ranking a batch of monitored machines by expected downtime loss and assigning them to available technician slots with transparent telemetry explanations.

---

## Key Architecture & Workflow

```
       Raw Sensor Telemetry
  (Type, Temps, Speed, Torque, Wear)
                 │
                 ▼
     [Physics Feature Engine]
  • ΔT = Process Temp - Air Temp
  • Power = Torque × Speed × 2π/60
  • Overstrain = Tool Wear × Torque
                 │
        ┌────────┴────────┐
        ▼                 ▼
[Binary Classifier]  [Multi-Output Classifier]
 P(Machine Failure)   P(TWF), P(HDF), P(PWF), P(OSF), P(RNF)
        │                 │
        └────────┬────────┘
                 │
                 ▼
    [Operational Cost Model]
  Minimizes Total Cost on Validation Data:
  Cost(τ) = FN(τ)·C_FN + FP(τ)·C_FP + TP(τ)·C_TP
                 │
                 ▼
     [Optimal Threshold τ*]
                 │
                 ▼
     [Technician Shift Planner]
  1. Filter batch where P(Fail) ≥ τ*
  2. Rank by Expected Loss = P(Fail) × C_FN
  3. Dispatch top N machines to technician slots
  4. Explain root-cause telemetry triggers
  5. Quantify deferred breakdown risk
```

---

## Mandatory Requirements Implementation

### Requirement A — Machine Failure Prediction
- **Input Parameters:** Machine Variant (`L`, `M`, `H`), Air Temperature [K], Process Temperature [K], Rotational Speed [rpm], Spindle Torque [Nm], Tool Wear [min].
- **Outputs:**
  - Machine Failure Probability: Calibrated risk score with visual progress indicator.
  - Alert Status: Evaluated dynamically against the cost-optimal threshold $\tau^*$.
  - Multi-Output Failure Modes: Independent probabilities for all 5 physical mechanisms:
    - `TWF`: Tool Wear Failure (cutter fatigue replacement).
    - `HDF`: Heat Dissipation Failure ($\Delta T < 8.6\text{ K}$ and speed $< 1380\text{ rpm}$).
    - `PWF`: Spindle Power Failure ($\text{Power} < 3500\text{ W}$ or $> 9000\text{ W}$).
    - `OSF`: Overstrain Failure (wear $\times$ torque exceeds variant limits).
    - `RNF`: Random / Stochastic Failure (uncorrelated 0.1% anomaly).
  - Overlapping Causes: Correctly models cases where multiple physical failure mechanisms occur simultaneously without forcing an unjustified single-class diagnosis.

### Requirement B — Explicit Cost Model & Alert Threshold
- **Configurable Cost Parameters:**
  - Missed Failure Cost ($C_{FN}$): Default $\$1,000$ (line stoppage, emergency teardown).
  - False Alarm Cost ($C_{FP}$): Default $\$100$ (technician inspection travel & check).
  - Preventive Maintenance Cost ($C_{TP}$): Default $\$150$ (scheduled tool swap).
- **Objective Function:**
  $$\text{Total Cost}(\tau) = \text{FN}(\tau) \times C_{FN} + \text{FP}(\tau) \times C_{FP} + \text{TP}(\tau) \times C_{TP}$$
- **Validation Tuning:** Evaluated strictly on the **Validation split** (1,500 samples) across candidate thresholds $\tau \in [0.01, 0.99]$.
- **Interactive Visualization:** Plotly chart displaying the total operational cost curve, component breakdown, optimal point $\tau^*$, and default $0.50$ baseline comparison.

### Requirement C — Maintenance Shift Planner
- **Batch Processing:** Ingests batches via CSV upload or loads the bundled benchmark 50-machine shift batch.
- **Validation:** Enforces schema integrity, valid machine variants, and physical sensor operating boundaries.
- **Prioritization Logic:**
  $$\text{Expected Breakdown Loss} = P(\text{Failure}) \times C_{FN}$$
  Machines exceeding $\tau^*$ are ranked descending by expected loss.
- **Capacity Constraint:** Allocates exactly $\le N$ machines to available technician slots.
- **Three-Tier Categorization:**
  1. **Selected for Service:** High-risk machines assigned to technician slots ($1 \dots N$).
  2. **Deferred (Action Required):** Machines exceeding $\tau^*$ that cannot be serviced due to capacity constraints, reporting total deferred unaddressed financial risk.
  3. **Nominal (Monitored):** Machines operating safely below $\tau^*$.
- **Explainability:** Each prioritized machine receives a concrete explanation identifying the sensor telemetry trigger (e.g. low temperature delta, excessive mechanical power, overstrain limit exceeded).

---

## Dataset Governance, Source & Synthetic Limitation

- **Dataset:** AI4I 2020 Predictive Maintenance Dataset.
- **Source:** [UCI Machine Learning Repository (Dataset ID 601)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
- **Citation:** Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications with a Digital Twin*. 2020 IEEE International Conference on Artificial Intelligence and Green Energy (ICAIGE).
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Synthetic Limitation Disclosure:** The AI4I 2020 dataset was generated using physical simulation models of milling machines with added noise. **This dataset is synthetic.** High performance on this synthetic benchmark does not guarantee real-world factory deployment without domain transfer, acoustic sensor calibration, and physical on-site telemetry validation.

---

## Model Methodology & Actual Test Evaluation Metrics

All metrics below are computed **strictly on the held-out test split (15% = 1,500 samples)**, which remained completely untouched during model training and validation threshold tuning:

- **Split Protocol:** Stratified split on `Machine failure`:
  - Training: 70% (7,000 samples)
  - Validation: 15% (1,500 samples)
  - Test: 15% (1,500 samples)
- **Feature Preprocessing:** `UDI` and `Product ID` were removed to prevent identity leakage. Preprocessors (`OneHotEncoder`, `StandardScaler`) were fitted exclusively on Training data.
- **Imbalance Handling:** Balanced class weighting (`class_weight='balanced'`) to address the 3.4% failure rate.

### Genuine Held-Out Test Evaluation Results

| Metric | Result | Notes |
|---|---|---|
| **Test PR-AUC** | **0.9487** | Primary metric for severe class imbalance (3.4%) |
| **Test ROC-AUC** | **0.9860** | Strong discrimination across all thresholds |
| **Test Recall (at $\tau=0.50$)** | **94.12%** | Detected 48 of 51 true machine failures |
| **Test Precision (at $\tau=0.50$)** | **65.75%** | 48 TP vs 25 FP |
| **Test F1-Score** | **0.7742** | Harmonic mean of precision and recall |

#### Held-Out Confusion Matrix (N = 1,500 test samples)
- **True Negatives (TN):** 1,424 (Nominal machines correctly operated)
- **False Positives (FP):** 25 (False alarms)
- **False Negatives (FN):** 3 (Missed failures)
- **True Positives (TP):** 48 (Correctly alerted failures)

#### Failure Mode Test PR-AUC
- **Heat Dissipation (HDF):** `1.0000` (Deterministic physical rule $\Delta T < 8.6\text{ K}$)
- **Spindle Power (PWF):** `1.0000` (Deterministic power formula envelope)
- **Overstrain (OSF):** `1.0000` (Deterministic torque $\times$ wear threshold)
- **Tool Wear (TWF):** `0.1952` (Stochastic failure window between 200–240 min)
- **Random Failure (RNF):** `0.0116` (Synthetic uniform 0.1% random Bernoulli trial with zero sensor correlation)

---

## Setup and Run Instructions

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Clone Repository
```bash
git clone https://github.com/Yukesh559/maintenance-shift-planner.git
cd maintenance-shift-planner
```

### 3. Create & Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Download Dataset & Train Models
```bash
# Acquire and verify UCI dataset
python data/download_data.py

# Fit pipelines and serialize artifacts
python -m src.model
```

### 6. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Testing & Verification

Run the comprehensive automated test suite covering all 10 verification areas:
```bash
pytest tests/test_core.py -v
```

### Automated Test Coverage:
1. `test_dataset_loading_and_split`: Verifies 70/15/15 stratified split dimensions.
2. `test_removal_of_identifier_columns`: Confirms no `UDI` or `Product ID` in features.
3. `test_prediction_output_and_probability_ranges`: Asserts probabilities $\in [0, 1]$.
4. `test_threshold_selection_validation_only`: Verifies threshold optimization on validation data only.
5. `test_cost_calculation`: Validates manual mathematical cost breakdown.
6. `test_overlapping_failure_modes`: Checks independent multi-output failure diagnoses.
7. `test_invalid_csv_input_handling`: Verifies input validation and descriptive error reporting.
8. `test_technician_slot_limits`: Enforces slot capacity bounds.
9. `test_selection_and_deferral_logic`: Verifies risk ranking and partitioning.
10. `test_inference_preprocessing_consistency`: Verifies training vs inference transform parity.

**Test Outcome:** `10 passed in 2.95s`.

---

## Public Deployment Guide

The application is structured for direct deployment on Streamlit Community Cloud or Hugging Face Spaces.

### Option A: Streamlit Community Cloud (Recommended)
1. Push this repository to GitHub: `https://github.com/Yukesh559/maintenance-shift-planner`.
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Select the repository `Yukesh559/maintenance-shift-planner`, branch `main`, and main file `app.py`.
4. Click **Deploy**. The application automatically downloads the dataset and runs the trained model.

### Option B: Hugging Face Spaces (Docker / Streamlit)
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces) with SDK set to **Streamlit**.
2. Connect the GitHub repository or push repository files.
3. The Space will build from `requirements.txt` and launch `app.py`.

*Live URL will be updated here upon completing host verification.*

---

## Development Log & AI Tools Used

- **Development Log:** A complete, unedited chronological log of development decisions, experiments, metrics, and resolutions is maintained in [`DEV_LOG.md`](DEV_LOG.md).
- **AI Coding Assistant:** Antigravity (Google DeepMind Advanced Agentic Coding) paired with human engineering review.
- **Libraries Used:** Python, pandas, NumPy, scikit-learn, Streamlit, Plotly, joblib, pytest.
