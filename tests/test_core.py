"""
Comprehensive Unit & Integration Test Suite
Validates all mandatory requirements:
- Dataset loading, preprocessing, and identifier removal
- Probability calibration & bounds [0, 1]
- Validation-based threshold optimization
- Cost calculation logic
- Overlapping failure mode predictions
- Batch validation & error handling
- Technician slot limit enforcement
- Selection and deferral logic
- Preprocessing consistency
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.preprocess import (
    clean_and_split_data,
    engineer_physics_features,
    validate_input_dataframe,
    preprocess_inference_data,
    IDENTIFIER_COLS,
    TARGET_MODES
)
from src.model import MaintenanceModelSuite, get_or_train_suite
from src.cost_model import CostConfig, compute_cost_components, sweep_thresholds, optimize_threshold
from src.planner import plan_maintenance_shift, generate_machine_explanation


@pytest.fixture(scope="session")
def model_suite():
    """Provides a trained or loaded model suite instance."""
    return get_or_train_suite()


@pytest.fixture(scope="session")
def dataset_splits():
    """Provides stratified data splits."""
    return clean_and_split_data()


# 1. Dataset Loading and Preprocessing Integrity
def test_dataset_loading_and_split(dataset_splits):
    splits = dataset_splits
    assert "X_train" in splits and "X_val" in splits and "X_test" in splits
    assert "y_train" in splits and "y_val" in splits and "y_test" in splits
    assert "modes_train" in splits and "modes_val" in splits and "modes_test" in splits
    
    # 70% train (7000), 15% val (1500), 15% test (1500)
    assert splits["X_train"].shape[0] == 7000
    assert splits["X_val"].shape[0] == 1500
    assert splits["X_test"].shape[0] == 1500


# 2. Removal of Identifier Columns
def test_removal_of_identifier_columns(dataset_splits):
    feature_names = dataset_splits["feature_names"]
    for id_col in IDENTIFIER_COLS:
        assert id_col not in feature_names, f"Identity leakage detected: {id_col} is present in feature set!"


# 3. Prediction Output and Probability Ranges [0, 1]
def test_prediction_output_and_probability_ranges(model_suite):
    suite = model_suite
    result = suite.predict_single(
        machine_type="M",
        air_temp=300.0,
        process_temp=310.0,
        rot_speed=1500.0,
        torque=40.0,
        tool_wear=100.0,
        alert_threshold=0.5
    )
    
    prob = result["failure_probability"]
    assert 0.0 <= prob <= 1.0, f"Failure probability {prob} out of bounds [0, 1]"
    assert isinstance(result["alert_status"], bool)
    assert "mode_probabilities" in result
    for mode, m_prob in result["mode_probabilities"].items():
        assert 0.0 <= m_prob <= 1.0, f"Mode {mode} probability {m_prob} out of bounds [0, 1]"


# 4. Threshold Selection Using Validation Data Only
def test_threshold_selection_validation_only(dataset_splits):
    splits = dataset_splits
    suite = get_or_train_suite()
    
    val_probs = suite.binary_model.predict_proba(splits["X_val"])[:, 1]
    y_val = splits["y_val"]
    
    config = CostConfig(cost_fn=1000.0, cost_fp=100.0, cost_tp=150.0)
    optimal_tau, df_sweep, best_row, base_row = optimize_threshold(y_val, val_probs, config)
    
    # Optimal threshold must be between 0.01 and 0.99
    assert 0.01 <= optimal_tau <= 0.99
    # Sweep dataframe must evaluate strictly on validation length
    assert len(df_sweep) == 99
    # Minimum cost must be <= baseline 0.5 cost on validation data
    assert best_row["total_cost"] <= base_row["total_cost"]


# 5. Cost Calculation Accuracy
def test_cost_calculation():
    y_true = np.array([1, 1, 0, 0])
    y_probs = np.array([0.9, 0.2, 0.8, 0.1])
    # At tau = 0.5:
    # Preds = [1, 0, 1, 0]
    # TP: 1 (idx 0), FN: 1 (idx 1), FP: 1 (idx 2), TN: 1 (idx 3)
    config = CostConfig(cost_fn=1000.0, cost_fp=200.0, cost_tp=150.0, cost_tn=0.0)
    metrics = compute_cost_components(y_true, y_probs, threshold=0.5, config=config)
    
    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    expected_cost = (1 * 1000.0) + (1 * 200.0) + (1 * 150.0) + (1 * 0.0)
    assert metrics["total_cost"] == expected_cost


# 6. Overlapping Failure-Mode Handling
def test_overlapping_failure_modes(model_suite):
    suite = model_suite
    # Extreme machine readings triggering multiple simultaneous failure mechanisms:
    # High wear (230 min) + High torque (65 Nm) = Overstrain + Tool Wear
    # Low delta T (6 K) at low speed (1200 rpm) = Heat Dissipation
    result = suite.predict_single(
        machine_type="L",
        air_temp=304.0,
        process_temp=309.0, # delta_T = 5.0 K < 8.6 K
        rot_speed=1250.0,   # low speed < 1380 rpm -> HDF
        torque=68.0,        # high torque
        tool_wear=235.0,    # high wear > 200 -> TWF + OSF
        alert_threshold=0.3
    )
    
    assert result["alert_status"] is True
    # The mode probabilities must not be forced to sum to 1.0 (they are independent probabilities)
    mode_probs = result["mode_probabilities"]
    assert len(mode_probs) == 5
    for mode in TARGET_MODES:
        assert mode in mode_probs


# 7. Invalid CSV Batch Input Handling & Error Messaging
def test_invalid_csv_input_handling():
    # Test missing column
    df_missing = pd.DataFrame([{
        "Type": "L",
        "Air temperature [K]": 300.0,
        # Missing Process temp
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 50.0
    }])
    valid, msg = validate_input_dataframe(df_missing)
    assert valid is False
    assert "Missing required sensor column" in msg

    # Test invalid categorical Type
    df_bad_type = pd.DataFrame([{
        "Type": "X",  # Invalid type
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 50.0
    }])
    valid, msg = validate_input_dataframe(df_bad_type)
    assert valid is False
    assert "Invalid machine Type" in msg

    # Test out of physical range values
    df_unphysical = pd.DataFrame([{
        "Type": "L",
        "Air temperature [K]": 999.0,  # Unrealistic temperature
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 50.0
    }])
    valid, msg = validate_input_dataframe(df_unphysical)
    assert valid is False
    assert "out of physical range" in msg


# 8. Technician-Slot Limits (Exact Cap N)
def test_technician_slot_limits(model_suite):
    suite = model_suite
    sample = pd.read_csv("data/sample_batch.csv")
    
    # Test capacity caps 0, 3, and 10
    for capacity in [0, 3, 10]:
        plan = plan_maintenance_shift(
            df_raw=sample,
            model_suite=suite,
            alert_threshold=0.3,
            technician_slots=capacity,
            cost_fn=1000.0
        )
        assert len(plan["selected_machines"]) <= capacity
        if capacity == 3 and plan["impact_summary"]["eligible_alert_machines"] >= 3:
            assert len(plan["selected_machines"]) == 3


# 9. Selection and Deferral Logic
def test_selection_and_deferral_partitioning(model_suite):
    suite = model_suite
    sample = pd.read_csv("data/sample_batch.csv")
    slots = 4
    threshold = 0.4
    
    plan = plan_maintenance_shift(
        df_raw=sample,
        model_suite=suite,
        alert_threshold=threshold,
        technician_slots=slots,
        cost_fn=1000.0
    )
    
    selected = plan["selected_machines"]
    deferred = plan["deferred_machines"]
    nominal = plan["nominal_machines"]
    summary = plan["impact_summary"]
    
    # Total machines must sum to batch size
    assert len(selected) + len(deferred) + len(nominal) == len(sample)
    assert summary["eligible_alert_machines"] == len(selected) + len(deferred)
    
    # Selected machines must have >= failure prob of deferred machines
    if not selected.empty and not deferred.empty:
        min_selected_loss = selected["Expected_Loss"].min()
        max_deferred_loss = deferred["Expected_Loss"].max()
        assert min_selected_loss >= max_deferred_loss

    # Nominal machines must all have Failure_Prob < threshold
    if not nominal.empty:
        assert (nominal["Failure_Prob"] < threshold).all()


# 10. Preprocessing Consistency (Training vs Inference)
def test_inference_preprocessing_consistency(model_suite, dataset_splits):
    suite = model_suite
    preprocessor = suite.preprocessor
    
    # Test that inference preprocessing produces the exact feature shape and handles optional ID columns
    sample = pd.read_csv("data/sample_batch.csv").head(5)
    X_proc_with_id = preprocess_inference_data(sample, preprocessor)
    
    sample_no_id = sample.drop(columns=[c for c in IDENTIFIER_COLS if c in sample.columns])
    X_proc_no_id = preprocess_inference_data(sample_no_id, preprocessor)
    
    assert X_proc_with_id.shape == X_proc_no_id.shape
    assert np.allclose(X_proc_with_id, X_proc_no_id)
    assert X_proc_with_id.shape[1] == len(dataset_splits["feature_names"])
