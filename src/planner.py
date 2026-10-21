"""
Shift Planner Module (Requirement C)
Handles batch validation, risk-weighted machine prioritization,
technician capacity slot allocation, root-cause explainability, and shift impact analysis.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from src.preprocess import validate_input_dataframe, preprocess_inference_data, TARGET_MODES
from src.model import MaintenanceModelSuite


def generate_machine_explanation(row: pd.Series, mode_probs: Dict[str, float]) -> str:
    """
    Constructs an explicit, human-readable operational explanation
    connecting physical sensor readings and failure modes to the scheduling decision.
    """
    reasons = []
    
    # Sensor physical values
    air_t = row.get("Air temperature [K]", 0)
    proc_t = row.get("Process temperature [K]", 0)
    delta_t = proc_t - air_t
    speed = row.get("Rotational speed [rpm]", 0)
    torque = row.get("Torque [Nm]", 0)
    wear = row.get("Tool wear [min]", 0)
    power_w = torque * (speed * (2 * np.pi / 60))
    overstrain = wear * torque
    m_type = row.get("Type", "L")

    # 1. Heat Dissipation Check (Physical limit: delta_T < 8.6 K & speed < 1380 rpm)
    if delta_t < 8.6 and speed < 1380:
        reasons.append(f"Heat Dissipation risk: temp delta (Process - Air) = {delta_t:.1f} K (< 8.6 K threshold) at low speed {speed:.0f} rpm")
    elif mode_probs.get("HDF", 0) > 0.3:
        reasons.append(f"Elevated thermal dissipation stress (delta_T = {delta_t:.1f} K, P(HDF)={mode_probs['HDF']:.2f})")

    # 2. Power Failure Check (Physical limit: Power < 3500 W or > 9000 W)
    if power_w < 3500 or power_w > 9000:
        reasons.append(f"Spindle power anomaly: power = {power_w:,.0f} W outside safe envelope (3500 - 9000 W)")
    elif mode_probs.get("PWF", 0) > 0.3:
        reasons.append(f"Spindle power instability flagged (P(PWF)={mode_probs['PWF']:.2f})")

    # 3. Overstrain Check (Thresholds: L: 11000, M: 12000, H: 13000)
    os_thresh = 11000 if m_type == "L" else (12000 if m_type == "M" else 13000)
    if overstrain > os_thresh:
        reasons.append(f"Severe mechanical overstrain: wear * torque = {overstrain:,.0f} exceeds Type-{m_type} limit ({os_thresh:,})")
    elif mode_probs.get("OSF", 0) > 0.3:
        reasons.append(f"High cutter overstrain loading (P(OSF)={mode_probs['OSF']:.2f})")

    # 4. Tool Wear Check (Fatigue window 200 - 240 min)
    if wear >= 200:
        reasons.append(f"Critical tool wear life: {wear:.0f} min reaches tool fatigue replacement window (200 - 240 min)")
    elif mode_probs.get("TWF", 0) > 0.25:
        reasons.append(f"Tool wear replacement required (wear = {wear:.0f} min, P(TWF)={mode_probs['TWF']:.2f})")

    # 5. Stochastic / Random Failure
    if mode_probs.get("RNF", 0) > 0.15:
        reasons.append("Uncorrelated sensor drift pattern detected (random failure indicator)")

    if not reasons:
        top_m = max(mode_probs.items(), key=lambda x: x[1])
        if top_m[1] > 0.15:
            reasons.append(f"Sub-threshold degradation pattern observed in {top_m[0]} (confidence: {top_m[1]:.2f})")
        else:
            reasons.append("Elevated composite failure probability across joint telemetry")

    return " | ".join(reasons)


def plan_maintenance_shift(
    df_raw: pd.DataFrame,
    model_suite: MaintenanceModelSuite,
    alert_threshold: float,
    technician_slots: int,
    cost_fn: float = 1000.0
) -> Dict[str, Any]:
    """
    Executes shift planning on a batch of machines (Requirement C):
    1. Validates inputs.
    2. Runs model inference (failure prob + multi-output modes).
    3. Ranks candidates by expected failure loss = P(Failure) * cost_fn.
    4. Enforces capacity cap = technician_slots.
    5. Categorizes machines into Selected, Deferred, and Nominal.
    6. Generates explanations and shift impact analytics.
    """
    is_valid, err_msg = validate_input_dataframe(df_raw)
    if not is_valid:
        raise ValueError(f"Input Data Validation Failed: {err_msg}")

    if technician_slots < 0:
        raise ValueError("Technician slots cannot be negative.")

    df = df_raw.copy()
    
    # Assign display ID if not present
    if "Display_ID" not in df.columns:
        if "UDI" in df.columns:
            df["Display_ID"] = df["UDI"].astype(str)
        elif "Product ID" in df.columns:
            df["Display_ID"] = df["Product ID"].astype(str)
        else:
            df["Display_ID"] = [f"M-{i+1:03d}" for i in range(len(df))]

    # Run Inference
    X_proc = preprocess_inference_data(df, model_suite.preprocessor)
    fail_probs = model_suite.binary_model.predict_proba(X_proc)[:, 1]
    
    # Multi-output failure modes
    mode_probs_list = model_suite.modes_model.predict_proba(X_proc)
    
    # Collect results
    records = []
    for i in range(len(df)):
        f_prob = float(fail_probs[i])
        m_probs = {}
        for idx, m in enumerate(TARGET_MODES):
            prob = float(mode_probs_list[idx][i, 1]) if mode_probs_list[idx].shape[1] > 1 else float(mode_probs_list[idx][i, 0])
            m_probs[m] = round(prob, 3)

        expected_loss = f_prob * cost_fn
        explanation = generate_machine_explanation(df.iloc[i], m_probs)
        
        # Primary predicted mode
        active_m = [m for m, p in m_probs.items() if p >= 0.25]
        top_m = max(m_probs.items(), key=lambda x: x[1])[0]
        mode_str = ", ".join(active_m) if active_m else (top_m if f_prob >= alert_threshold else "Nominal")

        records.append({
            "Display_ID": df["Display_ID"].iloc[i],
            "Type": df["Type"].iloc[i],
            "Air_Temp_K": df["Air temperature [K]"].iloc[i],
            "Process_Temp_K": df["Process temperature [K]"].iloc[i],
            "Speed_RPM": df["Rotational speed [rpm]"].iloc[i],
            "Torque_Nm": df["Torque [Nm]"].iloc[i],
            "Tool_Wear_min": df["Tool wear [min]"].iloc[i],
            "Failure_Prob": round(f_prob, 4),
            "Expected_Loss": round(expected_loss, 2),
            "Meets_Threshold": f_prob >= alert_threshold,
            "Predicted_Mode": mode_str,
            "Mode_Probabilities": m_probs,
            "Explanation": explanation
        })

    results_df = pd.DataFrame(records)

    # Filter into Eligible vs Nominal
    eligible = results_df[results_df["Meets_Threshold"]].copy()
    nominal = results_df[~results_df["Meets_Threshold"]].copy()

    # Prioritization: Sort eligible by Expected Loss descending (highest breakdown risk first)
    eligible = eligible.sort_values(by=["Expected_Loss", "Tool_Wear_min"], ascending=[False, False]).reset_index(drop=True)
    nominal = nominal.sort_values(by="Failure_Prob", ascending=False).reset_index(drop=True)

    # Allocate slots (strictly <= technician_slots)
    selected = eligible.iloc[:technician_slots].copy()
    deferred = eligible.iloc[technician_slots:].copy()

    # Add slot assignment numbers to selected machines
    if not selected.empty:
        selected["Assigned_Slot"] = [f"Slot #{i+1}" for i in range(len(selected))]
    else:
        selected["Assigned_Slot"] = []

    # Shift Impact Analytics
    total_eligible = len(eligible)
    num_selected = len(selected)
    num_deferred = len(deferred)
    num_nominal = len(nominal)
    
    total_prevented_loss = float(selected["Expected_Loss"].sum()) if not selected.empty else 0.0
    deferred_unaddressed_risk = float(deferred["Expected_Loss"].sum()) if not deferred.empty else 0.0

    impact_summary = {
        "total_machines_evaluated": len(df),
        "alert_threshold_used": round(alert_threshold, 4),
        "technician_slots_available": technician_slots,
        "eligible_alert_machines": total_eligible,
        "machines_scheduled": num_selected,
        "machines_deferred": num_deferred,
        "machines_nominal": num_nominal,
        "scheduled_risk_prevented_usd": round(total_prevented_loss, 2),
        "deferred_exposure_usd": round(deferred_unaddressed_risk, 2),
        "capacity_utilization_pct": round((num_selected / technician_slots * 100), 1) if technician_slots > 0 else 0.0,
        "risk_coverage_pct": round((num_selected / total_eligible * 100), 1) if total_eligible > 0 else 100.0,
        "policy_explanation": (
            f"Alert Threshold effect: Setting tau*={alert_threshold:.2f} identified {total_eligible} machine(s) "
            f"requiring preventive intervention. Capacity limit effect: With {technician_slots} slot(s), "
            f"{num_selected} machine(s) are serviced immediately, securing ${total_prevented_loss:,.2f} in expected loss prevention. "
            + (f"However, {num_deferred} machine(s) remain deferred with ${deferred_unaddressed_risk:,.2f} of unmitigated breakdown risk. "
               "Increasing technician capacity or prioritizing secondary shift coverage is advised." if num_deferred > 0 
               else "All high-risk machines are fully covered by available technician capacity.")
        )
    }

    return {
        "selected_machines": selected,
        "deferred_machines": deferred,
        "nominal_machines": nominal,
        "impact_summary": impact_summary,
        "all_results": results_df
    }
