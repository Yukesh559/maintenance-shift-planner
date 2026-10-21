# Development Log — NexAI Hack 2026: PS 2.1 Maintenance Shift Planner

This development log tracks genuine chronological milestones, technical decisions, experiments, issues encountered, and problem resolutions during development.

---

### Milestone 1: Workspace Inspection & Dependency Audit
- **Timestamp:** 2026-10-22 00:15 - 00:20 IST
- **Status:** Completed
- **Observations:**
  - Workspace `maintenance-shift-planner` contained only `.git` with remote pointing to `https://github.com/Yukesh559/maintenance-shift-planner.git`.
  - Python `3.13.7` available with `pandas`, `numpy`, `scikit-learn`, and `streamlit` installed.
  - Missing dependency detected: `plotly` was not installed (`ModuleNotFoundError: No module named 'plotly'`).
- **Resolution:**
  - Installed `plotly` (version 7.1.0) and confirmed `pytest` availability.

---

### Milestone 2: Dataset Acquisition & Verification
- **Timestamp:** 2026-10-22 00:40 IST
- **Dataset:** UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset (ID 601).
- **Actions:**
  - Implemented `data/download_data.py`.
  - Directly verified HTTP 200 availability from UCI archive.
  - Downloaded and verified dataset: 10,000 instances, 14 columns.
  - Verified class distribution: 339 failures (~3.39% severe class imbalance).
  - Investigated failure modes: TWF (46), HDF (115), PWF (95), OSF (98), RNF (19).
  - Identified 24 cases of overlapping failure modes and 9 cases of non-specific failure, confirming the requirement to avoid single-label multi-class reduction.
  - Created balanced sample batch `data/sample_batch.csv` (50 machines) for testing Requirement C.

---

### Milestone 3: Preprocessing & Physical Feature Engineering
- **Timestamp:** 2026-10-22 00:41 IST
- **File:** `src/preprocess.py`
- **Design Decisions:**
  - Strictly dropped `UDI` and `Product ID` to prevent identifier and temporal leakage.
  - Stratified 3-way split: 70% Train (7,000), 15% Validation (1,500), 15% Test (1,500) stratified on `Machine failure`.
  - Feature engineering derived from domain physics (Matzka, 2020):
    1. Temperature difference $\Delta T = \text{Process temp} - \text{Air temp}$.
    2. Spindle mechanical power $P = \text{Torque} \times (\text{Speed} \times 2\pi/60)$.
    3. Overstrain factor = $\text{Tool wear} \times \text{Torque}$.
  - Preprocessor (`OneHotEncoder` + `StandardScaler`) fitted exclusively on the Training set.

---

### Milestone 4: Model Pipeline & Multi-Output Failure Diagnosis
- **Timestamp:** 2026-10-22 00:43 IST
- **File:** `src/model.py`
- **Design Decisions:**
  - Binary classifier: `RandomForestClassifier(class_weight='balanced', max_depth=10, n_estimators=150)`.
  - Multi-output classifier: `MultiOutputClassifier(RandomForestClassifier(...))` for independent detection of TWF, HDF, PWF, OSF, and RNF.
- **Genuine Held-Out Test Evaluation Results (1,500 samples):**
  - PR-AUC: **0.9487**
  - ROC-AUC: **0.9860**
  - Test Recall at default threshold: **94.12%** (48 TP, 3 FN)
  - Test Precision: **65.75%** (25 FP)
  - Failure Mode Test PR-AUC: HDF = 1.000, PWF = 1.000, OSF = 1.000, TWF = 0.1952, RNF = 0.0116.
  - Note: RNF has near-zero PR-AUC because the synthetic data generator produced it as a 0.1% independent random Bernoulli trial uncorrelated with sensors.

---

### Milestone 5: Explicit Cost Model & Alert Threshold Selection
- **Timestamp:** 2026-10-22 00:44 IST
- **File:** `src/cost_model.py`
- **Design Decisions:**
  - Configurable cost parameters: Missed failure $C_{FN} = \$1,000$, False alarm $C_{FP} = \$100$, Preventive service $C_{TP} = \$150$.
  - Threshold optimization performed strictly on **Validation predictions** across $\tau \in [0.01, 0.99]$.
  - Optimization produced minimum cost threshold $\tau^* = 0.74$ (with $C_{TP}$ included) or $\tau^* = 0.41$ (pure FN+FP).
  - Built interactive Plotly visualization displaying total cost curve, breakdown lines, optimal marker, and default 0.50 baseline.

---

### Milestone 6: Shift Planner & Root-Cause Explainability
- **Timestamp:** 2026-10-22 00:45 IST
- **File:** `src/planner.py`
- **Issue Encountered & Fixed:**
  - Encountered `UnicodeEncodeError: 'charmap'` on Windows cp1252 console output when formatting Greek letter $\tau$ and bullet points.
  - Resolved by standardizing strings (`tau*`, ` | `, `Delta_T`, `wear * torque`) for clean cross-platform safety.
- **Planner Logic:**
  - Validates batch columns and physical sensor ranges.
  - Scores expected risk: $\text{Expected Loss} = P(\text{Failure}) \times C_{FN}$.
  - Ranks alert machines descending by expected loss.
  - Enforces strict technician slot limit ($N$).
  - Partitions into Selected for Service, Deferred (exceeding threshold but capacity constrained), and Nominal.
  - Produces human-readable telemetry explanations based on physical thresholds (e.g. $\Delta T < 8.6\text{ K}$, power $> 9000\text{ W}$, overstrain limits).

---

### Milestone 7: Automated Unit & Integration Testing
- **Timestamp:** 2026-10-22 00:46 IST
- **File:** `tests/test_core.py`
- **Outcomes:**
  - 10 test functions executed via `pytest`.
  - **Result: 10 passed in 2.95s.**
  - Verified: dataset split sizes, absence of `UDI`/`Product ID`, probability bounds $[0, 1]$, validation-only threshold tuning, cost formula accuracy, overlapping failure modes, CSV error handling, slot cap enforcement, selection/deferral logic, and preprocessing transform consistency.

---

### Milestone 8: Full Streamlit Application & Local Verification
- **Timestamp:** 2026-10-22 00:48 IST
- **File:** `app.py`
- **Implementation:**
  - Tab 1: Single-Machine Failure Prediction form, failure probability gauge, alert status badge, overlapping mode metrics, physical sensor analysis.
  - Tab 2: Interactive Cost Model sliders, validation threshold sweep, Plotly interactive curve, and comparative financial matrix.
  - Tab 3: Shift Planner with sample batch / custom CSV upload, technician slot allocation ($N$), categorized tables (Selected, Deferred, Nominal), and policy impact narrative.
  - Tab 4: Model Evaluation & Governance displaying held-out test metrics, confusion matrix, and synthetic digital twin disclosures.
- **Verification:**
  - Streamlit application launched locally on port 8501 and returned HTTP 200 OK.
