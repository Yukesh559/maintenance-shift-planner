"""
NexAI Hack 2026 — PS 2.1: Maintenance Shift Planner
Professional Streamlit Dashboard implementing Requirements A, B, and C.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.preprocess import clean_and_split_data, TARGET_MODES, NUMERICAL_COLS, IDENTIFIER_COLS
from src.model import MaintenanceModelSuite, get_or_train_suite, METRICS_FILE
from src.cost_model import CostConfig, sweep_thresholds, optimize_threshold, generate_cost_curve_figure, compute_cost_components
from src.planner import plan_maintenance_shift

# Page configuration
st.set_page_config(
    page_title="Maintenance Shift Planner | PS 2.1",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for industrial theme styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .alert-badge-danger {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .alert-badge-safe {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_suite_cached():
    return get_or_train_suite()


@st.cache_data
def load_splits_and_validation_predictions():
    suite = get_or_train_suite()
    splits = clean_and_split_data()
    val_probs = suite.binary_model.predict_proba(splits["X_val"])[:, 1]
    test_probs = suite.binary_model.predict_proba(splits["X_test"])[:, 1]
    return splits, val_probs, test_probs


# Initialize Session State
if "active_threshold" not in st.session_state:
    st.session_state.active_threshold = 0.45
if "cost_fn" not in st.session_state:
    st.session_state.cost_fn = 1000.0
if "cost_fp" not in st.session_state:
    st.session_state.cost_fp = 100.0
if "cost_tp" not in st.session_state:
    st.session_state.cost_tp = 150.0

model_suite = load_suite_cached()
splits, val_probs, test_probs = load_splits_and_validation_predictions()

# Sidebar: Governance & Configuration
with st.sidebar:
    st.markdown("### ⚙️ NexAI Hack 2026")
    st.markdown("**Problem Statement 2.1**  \n*Maintenance Shift Planner*")
    st.divider()

    st.markdown("#### 📐 Active Policy Parameters")
    st.metric("Alert Decision Threshold (τ*)", f"{st.session_state.active_threshold:.2f}")
    st.caption("Derived by minimizing operational costs on validation data.")
    
    st.markdown(f"**Cost Model:**  \n"
                f"- Missed Failure ($C_{{FN}}$): **${st.session_state.cost_fn:,.0f}**  \n"
                f"- False Alarm ($C_{{FP}}$): **${st.session_state.cost_fp:,.0f}**  \n"
                f"- Preventive Service ($C_{{TP}}$): **${st.session_state.cost_tp:,.0f}**")
    
    st.divider()
    st.markdown("#### 📚 Dataset & Verification")
    st.markdown("**AI4I 2020 Predictive Maintenance**  \n"
                "- Source: UCI ML Repository (ID 601)  \n"
                "- Author: Stephan Matzka (2020)  \n"
                "- Licence: CC BY 4.0  \n"
                "- Size: 10,000 synthetic milling records  \n"
                "- Total Failures: 339 (3.39% rate)")
    st.warning("⚠️ **Synthetic Dataset Notice:** Results reflect mathematical digital twin physics. Factory deployment requires empirical site validation.", icon="⚠️")

# Main Header
st.markdown('<div class="main-title">Maintenance Shift Planner</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Cost-optimal predictive maintenance, multi-cause failure diagnosis, and technician shift scheduling.</div>', unsafe_allow_html=True)

# Navigation Tabs
tabs = st.tabs([
    "1. ⚙️ Machine Failure Prediction",
    "2. 💰 Cost Analysis & Threshold",
    "3. 📋 Shift Planner",
    "4. 📊 Model Evaluation & Governance"
])

# ==============================================================================
# TAB 1: MACHINE FAILURE PREDICTION (Requirement A)
# ==============================================================================
with tabs[0]:
    st.subheader("Requirement A — Single-Machine Diagnostic Form")
    st.markdown("Input telemetry parameters from an individual milling machine to predict failure probability, identify overlapping physical failure modes, and evaluate alert status.")

    # Preset scenarios for easy demonstration
    st.markdown("**Load Preset Scenario:**")
    preset_cols = st.columns(4)
    if preset_cols[0].button("🟢 Nominal Machine"):
        st.session_state.in_type = "L"
        st.session_state.in_air = 298.5
        st.session_state.in_proc = 308.5
        st.session_state.in_speed = 1550.0
        st.session_state.in_torque = 38.0
        st.session_state.in_wear = 45.0
    if preset_cols[1].button("🔥 Heat Dissipation (HDF)"):
        st.session_state.in_type = "L"
        st.session_state.in_air = 302.5
        st.session_state.in_proc = 308.0 # delta_T = 5.5K < 8.6K
        st.session_state.in_speed = 1250.0 # < 1380 rpm
        st.session_state.in_torque = 48.0
        st.session_state.in_wear = 80.0
    if preset_cols[2].button("⚡ Power Failure (PWF)"):
        st.session_state.in_type = "M"
        st.session_state.in_air = 299.0
        st.session_state.in_proc = 309.5
        st.session_state.in_speed = 1350.0
        st.session_state.in_torque = 68.0 # Power = 9613 W > 9000 W
        st.session_state.in_wear = 60.0
    if preset_cols[3].button("🔨 Overstrain & Tool Wear (OSF+TWF)"):
        st.session_state.in_type = "L"
        st.session_state.in_air = 301.0
        st.session_state.in_proc = 311.0
        st.session_state.in_speed = 1400.0
        st.session_state.in_torque = 58.0
        st.session_state.in_wear = 225.0 # wear * torque = 13050 > 11000

    # Sensor Input Form
    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            val_type = st.selectbox(
                "Machine Variant (Type)",
                options=["L", "M", "H"],
                index=["L", "M", "H"].index(st.session_state.get("in_type", "L")),
                help="L = Low variant (50%), M = Medium (30%), H = High quality variant (20%)."
            )
            val_air = st.number_input(
                "Air Temperature [K]",
                min_value=270.0, max_value=340.0,
                value=float(st.session_state.get("in_air", 298.5)),
                step=0.1
            )
        with col2:
            val_proc = st.number_input(
                "Process Temperature [K]",
                min_value=280.0, max_value=350.0,
                value=float(st.session_state.get("in_proc", 308.5)),
                step=0.1
            )
            val_speed = st.number_input(
                "Rotational Speed [rpm]",
                min_value=800.0, max_value=3200.0,
                value=float(st.session_state.get("in_speed", 1550.0)),
                step=10.0
            )
        with col3:
            val_torque = st.number_input(
                "Spindle Torque [Nm]",
                min_value=0.0, max_value=120.0,
                value=float(st.session_state.get("in_torque", 38.0)),
                step=0.5
            )
            val_wear = st.number_input(
                "Tool Wear [min]",
                min_value=0.0, max_value=350.0,
                value=float(st.session_state.get("in_wear", 45.0)),
                step=1.0
            )

        submit_btn = st.form_submit_button("🔍 Evaluate Machine Health", use_container_width=True)

    # Compute & Display Prediction
    prediction = model_suite.predict_single(
        machine_type=val_type,
        air_temp=val_air,
        process_temp=val_proc,
        rot_speed=val_speed,
        torque=val_torque,
        tool_wear=val_wear,
        alert_threshold=st.session_state.active_threshold
    )

    f_prob = prediction["failure_probability"]
    is_alert = prediction["alert_status"]

    st.markdown("### Diagnostic Outcome")
    res_col1, res_col2, res_col3 = st.columns([1, 1, 1.5])

    with res_col1:
        st.metric(
            label="Machine Failure Probability",
            value=f"{f_prob * 100:.1f}%",
            delta=f"{'+' if is_alert else '-'} Alert Threshold: {st.session_state.active_threshold * 100:.1f}%",
            delta_color="inverse" if is_alert else "normal"
        )
        st.progress(min(max(f_prob, 0.0), 1.0))

    with res_col2:
        st.markdown("**Alert Dispatch Status**")
        if is_alert:
            st.markdown('<div class="alert-badge-danger">🚨 ALERT: Service Required</div>', unsafe_allow_html=True)
            st.caption(f"Failure probability ({f_prob:.3f}) exceeds cost-optimal threshold ({st.session_state.active_threshold:.3f}).")
        else:
            st.markdown('<div class="alert-badge-safe">🟢 NOMINAL: Normal Operation</div>', unsafe_allow_html=True)
            st.caption(f"Failure probability ({f_prob:.3f}) is below alert threshold ({st.session_state.active_threshold:.3f}).")

    with res_col3:
        st.markdown("**Primary Diagnosis / Failure Modes**")
        st.markdown(f"**Identified Mode:** `{prediction['predicted_mode_summary']}`")
        if len(prediction["active_modes"]) > 1:
            st.info(f"⚡ **Overlapping Causes Detected:** {', '.join(prediction['active_modes'])}. Telemetry reflects multiple simultaneous failure strains.", icon="ℹ️")

    # Mode Probability Breakdown
    st.markdown("#### Multi-Output Failure Mode Breakdown")
    mode_cols = st.columns(5)
    mode_descriptions = {
        "TWF": "Tool Wear Failure",
        "HDF": "Heat Dissipation Failure",
        "PWF": "Spindle Power Failure",
        "OSF": "Overstrain Failure",
        "RNF": "Random / Stochastic Failure"
    }
    for idx, (m_code, m_prob) in enumerate(prediction["mode_probabilities"].items()):
        with mode_cols[idx]:
            st.metric(label=f"{m_code} ({mode_descriptions[m_code]})", value=f"{m_prob * 100:.1f}%")
            st.progress(min(max(m_prob, 0.0), 1.0))

    # Sensor Physics Explanation & Domain Limitations
    st.markdown("#### 🔬 Physical Sensor Analysis & Governance")
    delta_t = val_proc - val_air
    calc_power = val_torque * (val_speed * (2 * np.pi / 60))
    calc_overstrain = val_wear * val_torque
    os_limit = 11000 if val_type == "L" else (12000 if val_type == "M" else 13000)

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.markdown("**Derived Physical Features (Domain Physics):**")
        st.markdown(f"- **Temperature Delta (Process - Air):** `{delta_t:.1f} K` (Heat Dissipation failure occurs if $< 8.6\\text{{ K}}$ and speed $< 1380\\text{{ rpm}}$)")
        st.markdown(f"- **Mechanical Spindle Power:** `{calc_power:,.0f} W` (Power failure occurs if $< 3,500\\text{{ W}}$ or $> 9,000\\text{{ W}}$)")
        st.markdown(f"- **Cumulative Overstrain (Wear × Torque):** `{calc_overstrain:,.0f} min·Nm` (Type-{val_type} limit: `{os_limit:,} min·Nm`)")
        st.markdown(f"- **Tool Wear Operating Time:** `{val_wear:.0f} min` (Replacement fatigue window: $200 - 240\\text{{ min}}$)")

    with exp_col2:
        st.markdown("**Prediction Explanation & Model Limitations:**")
        st.markdown(
            "- **Distinction of Predictions vs Causes:** High failure probabilities indicate physical telemetry is within known breakdown boundaries. "
            "However, actual physical confirmation requires technician inspection upon tool teardown.\n"
            "- **Stochastic Failure (RNF):** Random failures occur in 0.1% of synthetic instances without any sensor correlation; model predictions for RNF reflect historical base rate.\n"
            "- **Synthetic Origin Disclosure:** AI4I 2020 is a generated digital twin dataset. In a physical factory, vibrations, cutter chatter, and sensor drift will require calibrated on-machine transfer learning."
        )

# ==============================================================================
# TAB 2: COST MODEL & THRESHOLD OPTIMIZATION (Requirement B)
# ==============================================================================
with tabs[1]:
    st.subheader("Requirement B — Explicit Cost Model & Validation Threshold Optimization")
    st.markdown("Standard classifiers default to an arbitrary 0.5 probability threshold. In industrial maintenance, missed failures cause catastrophic downtime ($C_{FN}$), while false alarms cause minor inspection overhead ($C_{FP}$). Here, the decision threshold $\\tau^*$ is selected strictly to minimize total operational financial cost on validation data.")

    # Cost Parameters
    cost_c1, cost_c2, cost_c3, cost_c4 = st.columns(4)
    with cost_c1:
        new_cost_fn = st.number_input(
            "Missed Failure Cost ($C_{FN}$)",
            min_value=100.0, max_value=20000.0,
            value=float(st.session_state.cost_fn),
            step=100.0,
            help="Cost of an unexpected breakdown: line stoppage, secondary machine damage, scrapped workpieces."
        )
    with cost_c2:
        new_cost_fp = st.number_input(
            "False Alarm Cost ($C_{FP}$)",
            min_value=10.0, max_value=2000.0,
            value=float(st.session_state.cost_fp),
            step=25.0,
            help="Cost of unnecessary technician inspection when machine is nominal."
        )
    with cost_c3:
        include_tp = st.checkbox("Include Preventive Cost ($C_{TP}$)", value=True)
        new_cost_tp = st.number_input(
            "Preventive Service Cost ($C_{TP}$)",
            min_value=0.0, max_value=1000.0,
            value=float(st.session_state.cost_tp) if include_tp else 0.0,
            step=25.0,
            disabled=not include_tp,
            help="Scheduled preventive maintenance cost during planned shift window."
        )
    with cost_c4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Recalculate Threshold", use_container_width=True):
            st.session_state.cost_fn = new_cost_fn
            st.session_state.cost_fp = new_cost_fp
            st.session_state.cost_tp = new_cost_tp

    current_config = CostConfig(
        cost_fn=st.session_state.cost_fn,
        cost_fp=st.session_state.cost_fp,
        cost_tp=st.session_state.cost_tp
    )

    # Perform optimization on validation predictions
    optimal_tau, df_sweep, best_val, base_val = optimize_threshold(
        splits["y_val"], val_probs, current_config
    )
    st.session_state.active_threshold = optimal_tau

    # KPI Summary Cards
    st.markdown("#### Validation Split Financial Performance")
    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
    savings = base_val["total_cost"] - best_val["total_cost"]
    
    with kpi_c1:
        st.metric(
            label="Optimal Alert Threshold (τ*)",
            value=f"{optimal_tau:.2f}",
            delta=f"{optimal_tau - 0.50:+.2f} vs default 0.50"
        )
    with kpi_c2:
        st.metric(
            label="Total Cost at Optimal Threshold",
            value=f"${best_val['total_cost']:,.0f}"
        )
    with kpi_c3:
        st.metric(
            label="Total Cost at Baseline 0.50",
            value=f"${base_val['total_cost']:,.0f}"
        )
    with kpi_c4:
        st.metric(
            label="Validation Financial Savings",
            value=f"${savings:,.0f}",
            delta=f"{abs(savings)/base_val['total_cost']*100:.1f}% cost reduction" if base_val['total_cost'] > 0 else "0%",
            delta_color="normal"
        )

    # Interactive Plotly Curve
    st.markdown("#### Operational Cost vs. Alert Threshold Curve")
    fig = generate_cost_curve_figure(df_sweep, optimal_tau, baseline_tau=0.50)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed Cost Components Table
    st.markdown("#### Validation Cost Components Breakdown")
    comparison_df = pd.DataFrame([
        {
            "Policy": f"Optimal Threshold (τ*={optimal_tau:.2f})",
            "Missed Failures (FN)": best_val["fn"],
            "False Alarms (FP)": best_val["fp"],
            "True Alerts (TP)": best_val["tp"],
            "Recall": f"{best_val['recall']*100:.1f}%",
            "Precision": f"{best_val['precision']*100:.1f}%",
            "FN Cost ($)": f"${best_val['cost_missed']:,.0f}",
            "FP Cost ($)": f"${best_val['cost_false_alarm']:,.0f}",
            "TP Cost ($)": f"${best_val['cost_preventive']:,.0f}",
            "Total Cost ($)": f"${best_val['total_cost']:,.0f}"
        },
        {
            "Policy": "Default Baseline (τ=0.50)",
            "Missed Failures (FN)": base_val["fn"],
            "False Alarms (FP)": base_val["fp"],
            "True Alerts (TP)": base_val["tp"],
            "Recall": f"{base_val['recall']*100:.1f}%",
            "Precision": f"{base_val['precision']*100:.1f}%",
            "FN Cost ($)": f"${base_val['cost_missed']:,.0f}",
            "FP Cost ($)": f"${base_val['cost_false_alarm']:,.0f}",
            "TP Cost ($)": f"${base_val['cost_preventive']:,.0f}",
            "Total Cost ($)": f"${base_val['total_cost']:,.0f}"
        }
    ])
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)

# ==============================================================================
# TAB 3: SHIFT PLANNER (Requirement C)
# ==============================================================================
with tabs[2]:
    st.subheader("Requirement C — Maintenance Shift Planner")
    st.markdown("Upload a batch of machine readings or load the benchmark shift batch, specify available technician capacity slots ($N$), and generate an actionable shift schedule prioritized by expected failure risk.")

    plan_c1, plan_c2 = st.columns([2, 1])
    
    with plan_c1:
        batch_source = st.radio(
            "Batch Ingestion Source:",
            options=["Load Benchmark 50-Machine Shift Batch", "Upload Custom Sensor CSV"],
            horizontal=True
        )

        uploaded_df = None
        if batch_source == "Load Benchmark 50-Machine Shift Batch":
            sample_path = "data/sample_batch.csv"
            if os.path.exists(sample_path):
                uploaded_df = pd.read_csv(sample_path)
                st.caption(f"Loaded benchmark shift batch from `{sample_path}` (50 machines).")
        else:
            uploaded_file = st.file_uploader("Upload Batch CSV", type=["csv"])
            if uploaded_file is not None:
                try:
                    uploaded_df = pd.read_csv(uploaded_file)
                except Exception as e:
                    st.error(f"Failed to read CSV: {str(e)}")

    with plan_c2:
        tech_slots = st.slider(
            "Available Technician Slots (N)",
            min_value=1, max_value=25,
            value=5, step=1,
            help="Total number of maintenance technician intervention slots available in the upcoming shift."
        )
        plan_tau = st.slider(
            "Alert Decision Threshold (τ)",
            min_value=0.10, max_value=0.90,
            value=float(st.session_state.active_threshold),
            step=0.01,
            help="Threshold for identifying eligible maintenance alerts."
        )

    if uploaded_df is not None:
        try:
            # Execute Shift Plan
            shift_plan = plan_maintenance_shift(
                df_raw=uploaded_df,
                model_suite=model_suite,
                alert_threshold=plan_tau,
                technician_slots=tech_slots,
                cost_fn=st.session_state.cost_fn
            )
            
            summary = shift_plan["impact_summary"]
            selected_df = shift_plan["selected_machines"]
            deferred_df = shift_plan["deferred_machines"]
            nominal_df = shift_plan["nominal_machines"]

            # Shift Summary KPIs
            st.markdown("### Shift Allocation Overview")
            sum_c1, sum_c2, sum_c3, sum_c4 = st.columns(4)
            with sum_c1:
                st.metric("Total Batch Monitored", f"{summary['total_machines_evaluated']} machines")
            with sum_c2:
                st.metric("Eligible Failure Alerts", f"{summary['eligible_alert_machines']} machines", delta=f"{summary['eligible_alert_machines']/summary['total_machines_evaluated']*100:.1f}% of batch")
            with sum_c3:
                st.metric("Scheduled for Shift", f"{summary['machines_scheduled']} / {tech_slots} slots", delta=f"{summary['capacity_utilization_pct']}% capacity")
            with sum_c4:
                st.metric("Deferred Machines", f"{summary['machines_deferred']} machines", delta=f"-${summary['deferred_exposure_usd']:,.0f} risk", delta_color="inverse")

            # Policy Impact Narrative
            st.info(f"📋 **Shift Governance & Policy Rationale:**  \n{summary['policy_explanation']}")

            # Worklist Tabs
            sub_tab1, sub_tab2, sub_tab3 = st.tabs([
                f"🔴 Selected for Service ({len(selected_df)})",
                f"🟡 Deferred Due to Capacity ({len(deferred_df)})",
                f"🟢 Nominal Operation ({len(nominal_df)})"
            ])

            with sub_tab1:
                if selected_df.empty:
                    st.success("No machines require emergency maintenance in this batch at the selected threshold.")
                else:
                    st.markdown("**Technician Dispatch Schedule (Top-Priority Machines):**")
                    display_cols = ["Assigned_Slot", "Display_ID", "Type", "Failure_Prob", "Expected_Loss", "Predicted_Mode", "Explanation"]
                    st.dataframe(
                        selected_df[display_cols].rename(columns={
                            "Assigned_Slot": "Slot",
                            "Display_ID": "Machine ID",
                            "Failure_Prob": "Failure Prob",
                            "Expected_Loss": "Expected Loss ($)",
                            "Predicted_Mode": "Predicted Mode",
                            "Explanation": "Engineering Rationale & Telemetry Trigger"
                        }),
                        hide_index=True,
                        use_container_width=True
                    )

            with sub_tab2:
                if deferred_df.empty:
                    st.success("Zero machines deferred. All eligible alerts fit within current technician capacity.")
                else:
                    st.warning(f"⚠️ **Capacity Warning:** {len(deferred_df)} machines exceed alert threshold (τ*={plan_tau:.2f}) but cannot be serviced due to technician capacity ({tech_slots} slots). Total unaddressed risk exposure: **${summary['deferred_exposure_usd']:,.2f}**.", icon="⚠️")
                    display_cols_def = ["Display_ID", "Type", "Failure_Prob", "Expected_Loss", "Predicted_Mode", "Explanation"]
                    st.dataframe(
                        deferred_df[display_cols_def].rename(columns={
                            "Display_ID": "Machine ID",
                            "Failure_Prob": "Failure Prob",
                            "Expected_Loss": "Expected Loss ($)",
                            "Predicted_Mode": "Predicted Mode",
                            "Explanation": "Engineering Rationale & Telemetry Trigger"
                        }),
                        hide_index=True,
                        use_container_width=True
                    )

            with sub_tab3:
                st.markdown(f"**Nominal Machines ({len(nominal_df)} machines below τ*={plan_tau:.2f}):**")
                display_cols_nom = ["Display_ID", "Type", "Failure_Prob", "Tool_Wear_min", "Speed_RPM", "Torque_Nm"]
                st.dataframe(
                    nominal_df[display_cols_nom].rename(columns={
                        "Display_ID": "Machine ID",
                        "Failure_Prob": "Failure Prob",
                        "Tool_Wear_min": "Tool Wear (min)",
                        "Speed_RPM": "Speed (rpm)",
                        "Torque_Nm": "Torque (Nm)"
                    }),
                    hide_index=True,
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Shift Planning Error: {str(e)}")

# ==============================================================================
# TAB 4: MODEL EVALUATION & GOVERNANCE
# ==============================================================================
with tabs[3]:
    st.subheader("Requirement Governance — Held-Out Test Evaluation & Disclosures")
    st.markdown("All performance numbers below are evaluated strictly on the **held-out test split (15% = 1,500 samples)**, which remained completely untouched during model fitting and validation threshold tuning.")

    # Load metrics from file or computed
    metrics_summary = model_suite.metrics_summary
    if not metrics_summary and os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            metrics_summary = json.load(f)

    active_tau = float(st.session_state.active_threshold)
    test_eval = model_suite.evaluate_test_split_at_threshold(
        splits["y_test"],
        test_probs,
        selected_threshold=active_tau,
        baseline_threshold=0.50
    )
    opt_test = test_eval["selected_threshold_metrics"]
    base_test = test_eval["baseline_threshold_metrics"]

    # Compute financial costs on held-out test set for full comparison
    c_fn = float(st.session_state.cost_fn)
    c_fp = float(st.session_state.cost_fp)
    c_tp = float(st.session_state.cost_tp)
    test_cost_opt = opt_test["fn"] * c_fn + opt_test["fp"] * c_fp + opt_test["tp"] * c_tp
    test_cost_base = base_test["fn"] * c_fn + base_test["fp"] * c_fp + base_test["tp"] * c_tp

    # Top KPI Metrics Cards
    eval_c1, eval_c2, eval_c3, eval_c4, eval_c5 = st.columns(5)
    with eval_c1:
        st.metric("Test PR-AUC", f"{test_eval['pr_auc']:.4f}")
        st.caption("Primary metric for severe 3.4% imbalance.")
    with eval_c2:
        st.metric("Test ROC-AUC", f"{test_eval['roc_auc']:.4f}")
        st.caption("Threshold-independent separability.")
    with eval_c3:
        st.metric(
            f"Recall (at τ*={active_tau:.2f})",
            f"{opt_test['recall']*100:.1f}%",
            delta=f"{opt_test['recall'] - base_test['recall']:+.1%} vs τ=0.50",
            delta_color="normal"
        )
        st.caption(f"Captured {opt_test['tp']} of 51 true test failures.")
    with eval_c4:
        st.metric(
            f"Precision (at τ*={active_tau:.2f})",
            f"{opt_test['precision']*100:.1f}%",
            delta=f"{opt_test['precision'] - base_test['precision']:+.1%} vs τ=0.50",
            delta_color="normal"
        )
        st.caption(f"Only {opt_test['fp']} false alarm(s) vs {base_test['fp']} at τ=0.50.")
    with eval_c5:
        st.metric(
            f"F1-Score (at τ*={active_tau:.2f})",
            f"{opt_test['f1']:.4f}",
            delta=f"{opt_test['f1'] - base_test['f1']:+.4f} vs τ=0.50",
            delta_color="normal"
        )
        st.caption("Harmonic balance of precision and recall.")

    # Side-by-Side Held-Out Policy Comparison
    st.markdown("#### ⚖️ Held-Out Test Split Comparison: Cost-Optimal (τ*) vs. Default Baseline (τ=0.50)")
    comp_test_df = pd.DataFrame([
        {
            "Policy": f"Cost-Optimal Alert Threshold (τ*={active_tau:.2f}, Validation Tuned)",
            "Decision Threshold": f"{active_tau:.2f}",
            "Test Recall": f"{opt_test['recall']*100:.1f}%",
            "Test Precision": f"{opt_test['precision']*100:.1f}%",
            "Test F1-Score": f"{opt_test['f1']:.4f}",
            "False Alarms (FP)": opt_test["fp"],
            "Missed Failures (FN)": opt_test["fn"],
            "True Alerts (TP)": opt_test["tp"],
            "True Nominal (TN)": opt_test["tn"],
            "Total Test Cost ($)": f"${test_cost_opt:,.0f}"
        },
        {
            "Policy": "Default Standard Baseline (τ=0.50)",
            "Decision Threshold": "0.50",
            "Test Recall": f"{base_test['recall']*100:.1f}%",
            "Test Precision": f"{base_test['precision']*100:.1f}%",
            "Test F1-Score": f"{base_test['f1']:.4f}",
            "False Alarms (FP)": base_test["fp"],
            "Missed Failures (FN)": base_test["fn"],
            "True Alerts (TP)": base_test["tp"],
            "True Nominal (TN)": base_test["tn"],
            "Total Test Cost ($)": f"${test_cost_base:,.0f}"
        }
    ])
    st.dataframe(comp_test_df, hide_index=True, use_container_width=True)

    # Side-by-Side Confusion Matrices
    st.markdown("#### Held-Out Confusion Matrices (N = 1,500 Test Samples)")
    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:
        st.markdown(f"**Selected Cost-Optimal Threshold (τ* = {active_tau:.2f})**")
        cm_opt_df = pd.DataFrame(
            opt_test["confusion_matrix"],
            index=["Actual Operational (0)", "Actual Failure (1)"],
            columns=["Predicted Operational (0)", "Predicted Alert (1)"]
        )
        st.table(cm_opt_df)
        st.caption(f"TP: {opt_test['tp']} | FP: {opt_test['fp']} (slashed to {opt_test['fp']}) | FN: {opt_test['fn']} | TN: {opt_test['tn']}")

    with cm_col2:
        st.markdown("**Default Standard Baseline (τ = 0.50)**")
        cm_base_df = pd.DataFrame(
            base_test["confusion_matrix"],
            index=["Actual Operational (0)", "Actual Failure (1)"],
            columns=["Predicted Operational (0)", "Predicted Alert (1)"]
        )
        st.table(cm_base_df)
        st.caption(f"TP: {base_test['tp']} | FP: {base_test['fp']} | FN: {base_test['fn']} | TN: {base_test['tn']}")

    st.markdown("#### Evaluation Methodology & Governance Compliance")
    st.markdown(
        "1. **Validation-Only Threshold Selection:** The alert threshold $\\tau^*$ is selected strictly by sweeping validation predictions "
        "to minimize operational financial cost. The held-out test split was kept strictly untouched during threshold optimization.\n"
        "2. **Severe Class Imbalance (3.39%):** Out of 10,000 instances, only 339 are failures. Balanced class weighting was applied, "
        "and evaluation prioritizes PR-AUC and recall-precision trade-offs rather than uninformative overall accuracy.\n"
        "3. **Zero Identifier Leakage:** Identifiers `UDI` and `Product ID` are dropped prior to any feature transformation.\n"
        "4. **Preserved Baseline:** Standard 0.50 baseline metrics are maintained for objective benchmark comparison."
    )

    # Multi-Output Mode Performance
    st.markdown("#### Multi-Output Failure Mode Breakdown (Held-Out Test Set)")
    mode_metrics = metrics_summary.get("failure_mode_metrics", {})
    if mode_metrics:
        mode_rows = []
        for m, data in mode_metrics.items():
            mode_rows.append({
                "Failure Mode": m,
                "Description": mode_descriptions.get(m, m),
                "Test Support (Actual Occurrences)": data.get("support", 0),
                "Test PR-AUC": data.get("pr_auc", 0.0)
            })
        st.dataframe(pd.DataFrame(mode_rows), hide_index=True, use_container_width=True)

    st.markdown("#### 📜 Synthetic Dataset Disclosure & Licence")
    st.markdown("""
    - **Dataset Citation:** Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications with a Digital Twin*. 2020 IEEE International Conference on Artificial Intelligence and Green Energy (ICAIGE).
    - **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0).
    - **Physical Digital Twin Nature:** This dataset was synthetically generated using physical models of milling machines (heat dissipation, spindle mechanics, and cutter wear equations). While it faithfully models industrial mechanics, real-world deployment requires factory sensor calibration.
    """)
