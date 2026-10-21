"""
Explicit Cost Model & Alert Threshold Optimization Module (Requirement B)
Implements configurable industrial cost functions, validation-based threshold optimization,
and interactive Plotly visualization.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix


@dataclass
class CostConfig:
    """
    Industrial cost assumptions.
    Units: Currency (USD) per incident.
    """
    cost_fn: float = 1000.0  # Missed failure (catastrophic tool breakdown, scrap, line stoppage)
    cost_fp: float = 100.0   # False alarm (unnecessary technician dispatch and inspection)
    cost_tp: float = 150.0   # Planned preventive service (scheduled tool swap/minor repair)
    cost_tn: float = 0.0     # Nominal operation (no action required)

    def describe(self) -> str:
        return (
            f"Missed Failure (FN): ${self.cost_fn:,.2f} | "
            f"False Alarm (FP): ${self.cost_fp:,.2f} | "
            f"Preventive Service (TP): ${self.cost_tp:,.2f} | "
            f"Nominal Operation (TN): ${self.cost_tn:,.2f}"
        )


def compute_cost_components(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float,
    config: CostConfig
) -> Dict[str, Any]:
    """
    Calculates confusion matrix counts and resulting operational financial costs
    at a specific decision threshold.
    """
    preds = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    
    cost_missed = float(fn * config.cost_fn)
    cost_false_alarm = float(fp * config.cost_fp)
    cost_preventive = float(tp * config.cost_tp)
    cost_nominal = float(tn * config.cost_tn)
    total_cost = cost_missed + cost_false_alarm + cost_preventive + cost_nominal

    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "cost_missed": cost_missed,
        "cost_false_alarm": cost_false_alarm,
        "cost_preventive": cost_preventive,
        "total_cost": total_cost,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    }


def sweep_thresholds(
    y_val: np.ndarray,
    y_val_probs: np.ndarray,
    config: CostConfig,
    n_points: int = 99
) -> pd.DataFrame:
    """
    Evaluates candidate thresholds strictly on VALIDATION data across [0.01, 0.99].
    Returns a DataFrame of metrics and costs across all thresholds.
    """
    thresholds = np.linspace(0.01, 0.99, n_points)
    rows = []
    for t in thresholds:
        metrics = compute_cost_components(y_val, y_val_probs, t, config)
        rows.append(metrics)
    return pd.DataFrame(rows)


def optimize_threshold(
    y_val: np.ndarray,
    y_val_probs: np.ndarray,
    config: CostConfig
) -> Tuple[float, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Finds the optimal alert threshold tau* that minimizes total cost on validation data.
    Also computes the baseline (threshold = 0.5) metrics for comparison.
    """
    df_sweep = sweep_thresholds(y_val, y_val_probs, config)
    
    # Select threshold with minimum total cost
    best_row_idx = df_sweep["total_cost"].idxmin()
    best_row = df_sweep.loc[best_row_idx].to_dict()
    optimal_tau = float(best_row["threshold"])

    # Baseline comparison at 0.5
    baseline_row = compute_cost_components(y_val, y_val_probs, 0.5, config)
    
    return optimal_tau, df_sweep, best_row, baseline_row


def generate_cost_curve_figure(
    df_sweep: pd.DataFrame,
    optimal_tau: float,
    baseline_tau: float = 0.5,
    title: str = "Operational Cost vs. Alert Decision Threshold (Validation Data)"
) -> go.Figure:
    """
    Creates an interactive Plotly visualization displaying:
    - Total Cost curve
    - Missed Failure (FN) cost curve
    - False Alarm (FP) cost curve
    - Preventive Service (TP) cost curve
    - Optimal threshold marker (tau*)
    - Default baseline marker (0.5)
    """
    fig = go.Figure()

    # Total Cost line
    fig.add_trace(go.Scatter(
        x=df_sweep["threshold"],
        y=df_sweep["total_cost"],
        mode="lines",
        name="Total Operational Cost",
        line=dict(color="#1f77b4", width=3.5)
    ))

    # Missed Failure Cost
    fig.add_trace(go.Scatter(
        x=df_sweep["threshold"],
        y=df_sweep["cost_missed"],
        mode="lines",
        name="Missed Failure Cost (FN)",
        line=dict(color="#d62728", width=2, dash="dash")
    ))

    # False Alarm Cost
    fig.add_trace(go.Scatter(
        x=df_sweep["threshold"],
        y=df_sweep["cost_false_alarm"],
        mode="lines",
        name="False Alarm Cost (FP)",
        line=dict(color="#ff7f0e", width=2, dash="dot")
    ))

    # Preventive Maintenance Cost
    fig.add_trace(go.Scatter(
        x=df_sweep["threshold"],
        y=df_sweep["cost_preventive"],
        mode="lines",
        name="Preventive Service Cost (TP)",
        line=dict(color="#2ca02c", width=1.5, dash="dashdot")
    ))

    # Find optimal and baseline points
    opt_row = df_sweep.loc[(df_sweep["threshold"] - optimal_tau).abs().idxmin()]
    base_row = df_sweep.loc[(df_sweep["threshold"] - baseline_tau).abs().idxmin()]

    # Highlight Optimal Threshold
    fig.add_trace(go.Scatter(
        x=[optimal_tau],
        y=[opt_row["total_cost"]],
        mode="markers+text",
        name=f"Optimal Threshold (τ*={optimal_tau:.2f})",
        marker=dict(color="#2ca02c", size=13, symbol="star"),
        text=[f"Min Cost: ${opt_row['total_cost']:,.0f}"],
        textposition="top center"
    ))

    # Highlight Baseline Threshold 0.5
    fig.add_trace(go.Scatter(
        x=[baseline_tau],
        y=[base_row["total_cost"]],
        mode="markers+text",
        name=f"Default Baseline (τ=0.50)",
        marker=dict(color="#7f7f7f", size=10, symbol="circle"),
        text=[f"Baseline: ${base_row['total_cost']:,.0f}"],
        textposition="bottom center"
    ))

    # Add vertical line for optimal tau
    fig.add_vline(
        x=optimal_tau,
        line_width=1.5,
        line_dash="dash",
        line_color="#2ca02c",
        annotation_text=f"Optimal τ*={optimal_tau:.2f}",
        annotation_position="top left"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Alert Decision Threshold (Probability of Failure)",
        yaxis_title="Total Expected Operational Cost ($)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=60, b=80)
    )

    return fig
