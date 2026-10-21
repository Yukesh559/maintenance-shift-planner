"""
Model Training, Persistence, and Evaluation Module
Handles binary failure prediction and multi-output failure mode identification.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.preprocess import clean_and_split_data, TARGET_MODES, TARGET_BINARY, preprocess_inference_data

ARTIFACTS_DIR = "artifacts"
BINARY_MODEL_FILE = os.path.join(ARTIFACTS_DIR, "model_binary.joblib")
MODES_MODEL_FILE = os.path.join(ARTIFACTS_DIR, "model_modes.joblib")
PREPROCESSOR_FILE = os.path.join(ARTIFACTS_DIR, "preprocessor.joblib")
METRICS_FILE = os.path.join(ARTIFACTS_DIR, "metrics_summary.json")


class MaintenanceModelSuite:
    """
    Encapsulates both the primary binary machine failure classifier
    and the multi-output failure mode classifier.
    """
    def __init__(self):
        self.binary_model: Optional[RandomForestClassifier] = None
        self.modes_model: Optional[MultiOutputClassifier] = None
        self.preprocessor = None
        self.feature_names: List[str] = []
        self.metrics_summary: Dict[str, Any] = {}

    def fit_and_evaluate(
        self,
        data_path: str = "data/ai4i2020.csv",
        random_state: int = 42,
        optimal_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Fits models on Training split only, evaluates on held-out Test split,
        and serializes artifacts.
        """
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        splits = clean_and_split_data(data_path=data_path, random_state=random_state)
        
        X_train, y_train, modes_train = splits["X_train"], splits["y_train"], splits["modes_train"]
        X_val, y_val, modes_val = splits["X_val"], splits["y_val"], splits["modes_val"]
        X_test, y_test, modes_test = splits["X_test"], splits["y_test"], splits["modes_test"]
        
        self.preprocessor = splits["preprocessor"]
        self.feature_names = splits["feature_names"]

        # 1. Train Binary Machine Failure Model
        # Address class imbalance (~3.4% failure rate) via balanced class weights
        self.binary_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
        self.binary_model.fit(X_train, y_train)

        # 2. Train Multi-Output Failure Mode Classifier
        # Overlapping modes: TWF, HDF, PWF, OSF, RNF
        base_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
        self.modes_model = MultiOutputClassifier(base_rf, n_jobs=-1)
        self.modes_model.fit(X_train, modes_train)

        # 3. Predict on Test Set (Strict Held-Out Evaluation)
        test_probs = self.binary_model.predict_proba(X_test)[:, 1]
        test_preds_at_thresh = (test_probs >= optimal_threshold).astype(int)
        test_preds_default = (test_probs >= 0.5).astype(int)

        # Evaluate Binary Metrics
        pr_auc = float(average_precision_score(y_test, test_probs))
        roc_auc = float(roc_auc_score(y_test, test_probs))
        rec_opt = float(recall_score(y_test, test_preds_at_thresh, zero_division=0))
        prec_opt = float(precision_score(y_test, test_preds_at_thresh, zero_division=0))
        f1_opt = float(f1_score(y_test, test_preds_at_thresh, zero_division=0))
        cm_opt = confusion_matrix(y_test, test_preds_at_thresh).tolist()

        # Evaluate Failure Mode Metrics on Test Set
        mode_probs_list = self.modes_model.predict_proba(X_test)
        mode_metrics = {}
        for idx, mode in enumerate(TARGET_MODES):
            # predict_proba returns list of arrays of shape (n_samples, 2)
            m_probs = mode_probs_list[idx][:, 1] if mode_probs_list[idx].shape[1] > 1 else mode_probs_list[idx][:, 0]
            m_true = modes_test[:, idx]
            m_pr_auc = float(average_precision_score(m_true, m_probs)) if m_true.sum() > 0 else 0.0
            mode_metrics[mode] = {
                "support": int(m_true.sum()),
                "pr_auc": round(m_pr_auc, 4)
            }

        self.metrics_summary = {
            "dataset_info": {
                "total_samples": 10000,
                "train_samples": int(X_train.shape[0]),
                "val_samples": int(X_val.shape[0]),
                "test_samples": int(X_test.shape[0]),
                "synthetic_note": "AI4I 2020 dataset is synthetic (Matzka, 2020). Performance does not guarantee factory deployment without on-site domain transfer."
            },
            "held_out_test_metrics": {
                "pr_auc": round(pr_auc, 4),
                "roc_auc": round(roc_auc, 4),
                "eval_threshold": round(optimal_threshold, 4),
                "recall_at_threshold": round(rec_opt, 4),
                "precision_at_threshold": round(prec_opt, 4),
                "f1_at_threshold": round(f1_opt, 4),
                "confusion_matrix": cm_opt,
                "tn": cm_opt[0][0],
                "fp": cm_opt[0][1],
                "fn": cm_opt[1][0],
                "tp": cm_opt[1][1]
            },
            "failure_mode_metrics": mode_metrics
        }

        # 4. Save Artifacts
        self.save_artifacts()
        
        return {
            "metrics": self.metrics_summary,
            "splits": splits,
            "val_probs": self.binary_model.predict_proba(X_val)[:, 1],
            "test_probs": test_probs
        }

    def evaluate_test_split_at_threshold(
        self,
        y_test: np.ndarray,
        test_probs: np.ndarray,
        selected_threshold: float,
        baseline_threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Computes held-out test evaluation metrics at both the validation-selected threshold
        and the standard 0.50 baseline threshold without retraining.
        """
        pr_auc = float(average_precision_score(y_test, test_probs))
        roc_auc = float(roc_auc_score(y_test, test_probs))

        def _metrics_for_tau(tau: float) -> Dict[str, Any]:
            preds = (test_probs >= tau).astype(int)
            cm = confusion_matrix(y_test, preds, labels=[0, 1]).tolist()
            rec = float(recall_score(y_test, preds, zero_division=0))
            prec = float(precision_score(y_test, preds, zero_division=0))
            f1 = float(f1_score(y_test, preds, zero_division=0))
            return {
                "threshold": round(float(tau), 4),
                "recall": round(rec, 4),
                "precision": round(prec, 4),
                "f1": round(f1, 4),
                "confusion_matrix": cm,
                "tn": cm[0][0],
                "fp": cm[0][1],
                "fn": cm[1][0],
                "tp": cm[1][1]
            }

        opt_metrics = _metrics_for_tau(selected_threshold)
        base_metrics = _metrics_for_tau(baseline_threshold)

        return {
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "selected_threshold_metrics": opt_metrics,
            "baseline_threshold_metrics": base_metrics
        }

    def save_artifacts(self):
        """Serializes models, preprocessor, and metrics summary."""
        joblib.dump(self.binary_model, BINARY_MODEL_FILE)
        joblib.dump(self.modes_model, MODES_MODEL_FILE)
        joblib.dump(self.preprocessor, PREPROCESSOR_FILE)
        with open(METRICS_FILE, "w") as f:
            json.dump(self.metrics_summary, f, indent=2)
        print(f"[INFO] Saved models and metrics to {ARTIFACTS_DIR}/")

    def load_artifacts(self) -> bool:
        """Loads serialized models and preprocessor if they exist."""
        if not (os.path.exists(BINARY_MODEL_FILE) and 
                os.path.exists(MODES_MODEL_FILE) and 
                os.path.exists(PREPROCESSOR_FILE)):
            return False
            
        self.binary_model = joblib.load(BINARY_MODEL_FILE)
        self.modes_model = joblib.load(MODES_MODEL_FILE)
        self.preprocessor = joblib.load(PREPROCESSOR_FILE)
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, "r") as f:
                self.metrics_summary = json.load(f)
        return True

    def predict_single(
        self,
        machine_type: str,
        air_temp: float,
        process_temp: float,
        rot_speed: float,
        torque: float,
        tool_wear: float,
        alert_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Executes prediction pipeline for a single machine reading (Requirement A).
        Returns failure probability, overlapping mode probabilities, alert status, and explanation.
        """
        if self.binary_model is None or self.preprocessor is None:
            raise RuntimeError("Model suite is not loaded or fitted.")

        input_df = pd.DataFrame([{
            "Type": machine_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rot_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }])

        X_proc = preprocess_inference_data(input_df, self.preprocessor)
        
        # Binary failure probability
        fail_prob = float(self.binary_model.predict_proba(X_proc)[0, 1])
        is_alert = bool(fail_prob >= alert_threshold)

        # Multi-output failure modes
        mode_probs_raw = self.modes_model.predict_proba(X_proc)
        mode_probs = {}
        for idx, mode in enumerate(TARGET_MODES):
            prob = float(mode_probs_raw[idx][0, 1]) if mode_probs_raw[idx].shape[1] > 1 else float(mode_probs_raw[idx][0, 0])
            mode_probs[mode] = round(prob, 4)

        # Identify predicted failure modes (handling potential overlaps)
        # Failure modes with probability >= 0.25 or top candidate
        active_modes = [m for m, p in mode_probs.items() if p >= 0.25]
        top_mode = max(mode_probs.items(), key=lambda x: x[1])[0]

        if not active_modes:
            predicted_mode_summary = "None (Nominal)" if fail_prob < alert_threshold else f"Indeterminate / Non-specific ({top_mode} candidate)"
        elif len(active_modes) == 1:
            predicted_mode_summary = active_modes[0]
        else:
            predicted_mode_summary = f"Overlapping: {', '.join(active_modes)}"

        return {
            "failure_probability": round(fail_prob, 4),
            "alert_status": is_alert,
            "alert_threshold_used": round(alert_threshold, 4),
            "predicted_mode_summary": predicted_mode_summary,
            "mode_probabilities": mode_probs,
            "active_modes": active_modes
        }


def get_or_train_suite(data_path: str = "data/ai4i2020.csv") -> MaintenanceModelSuite:
    """Helper to load suite or fit from scratch if not already saved."""
    suite = MaintenanceModelSuite()
    if not suite.load_artifacts():
        print("[INFO] Artifacts not found. Training model suite...")
        suite.fit_and_evaluate(data_path=data_path)
    return suite


if __name__ == "__main__":
    suite = MaintenanceModelSuite()
    res = suite.fit_and_evaluate()
    print("Held-out Test Metrics:")
    print(json.dumps(res["metrics"]["held_out_test_metrics"], indent=2))
    print("Failure Mode Metrics:")
    print(json.dumps(res["metrics"]["failure_mode_metrics"], indent=2))
