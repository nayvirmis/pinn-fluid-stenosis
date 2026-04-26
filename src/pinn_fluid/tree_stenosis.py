from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any, Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pinn_fluid.stenosis import (
    area_from_radius,
    integrate_pressure,
    pressure_gradient,
    reynolds_number,
    wall_shear_stress,
)


BranchName = Literal["inlet", "left_outlet", "right_outlet"]
OutletBranchName = Literal["left_outlet", "right_outlet"]


@dataclass
class TreeStenosisConfig:
    length_inlet: float = 0.6
    length_outlet: float = 0.7
    radius_inlet: float = 1.0
    radius_left: float = 0.75
    radius_right: float = 0.65
    flow_rate: float = 1.0
    viscosity: float = 0.04
    density: float = 1.06
    inlet_pressure: float = 100.0
    stenosis_width: float = 0.08
    points_per_branch: int = 600
    stenosed_branch: OutletBranchName = "left_outlet"
    severity: float = 0.55
    stenosis_center_fraction: float = 0.5


def murray_flow_split(cfg: TreeStenosisConfig) -> dict[BranchName, float]:
    left_weight = cfg.radius_left ** 3
    right_weight = cfg.radius_right ** 3
    total_weight = left_weight + right_weight
    left_flow = cfg.flow_rate * left_weight / total_weight
    right_flow = cfg.flow_rate * right_weight / total_weight
    return {
        "inlet": cfg.flow_rate,
        "left_outlet": left_flow,
        "right_outlet": right_flow,
    }


def branch_base_radius(cfg: TreeStenosisConfig, branch: BranchName) -> float:
    if branch == "inlet":
        return cfg.radius_inlet
    if branch == "left_outlet":
        return cfg.radius_left
    return cfg.radius_right


def branch_length(cfg: TreeStenosisConfig, branch: BranchName) -> float:
    return cfg.length_inlet if branch == "inlet" else cfg.length_outlet


def branch_radius_profile(
    x: np.ndarray,
    cfg: TreeStenosisConfig,
    branch: BranchName,
    stenosed_branch: OutletBranchName | None,
    severity: float,
) -> np.ndarray:
    radius = np.full_like(x, branch_base_radius(cfg, branch), dtype=float)
    if branch == stenosed_branch:
        center = cfg.stenosis_center_fraction * branch_length(cfg, branch)
        gaussian = np.exp(-0.5 * ((x - center) / cfg.stenosis_width) ** 2)
        radius = radius * (1.0 - severity * gaussian)
    return np.maximum(radius, 1e-4)


def simulate_branch(
    cfg: TreeStenosisConfig,
    branch: BranchName,
    flow_rate: float,
    start_pressure: float,
    stenosed_branch: OutletBranchName | None,
    severity: float,
) -> dict[str, np.ndarray | float]:
    x = np.linspace(0.0, branch_length(cfg, branch), cfg.points_per_branch)
    radius = branch_radius_profile(x, cfg, branch, stenosed_branch, severity)
    area = area_from_radius(radius)
    velocity = flow_rate / area
    dp_dx = pressure_gradient(flow_rate, cfg.viscosity, radius)
    pressure = integrate_pressure(start_pressure, dp_dx, x)
    local_pressure_drop = pressure[0] - pressure
    resistance = local_pressure_drop[-1] / flow_rate if flow_rate > 0 else 0.0
    return {
        "x": x,
        "radius": radius,
        "area": area,
        "velocity": velocity,
        "dp_dx": dp_dx,
        "pressure": pressure,
        "local_pressure_drop": local_pressure_drop,
        "reynolds": reynolds_number(cfg.density, velocity, radius, cfg.viscosity),
        "wall_shear_stress": wall_shear_stress(cfg.viscosity, flow_rate, radius),
        "flow_rate": float(flow_rate),
        "resistance": float(resistance),
    }


def simulate_tree(
    cfg: TreeStenosisConfig,
    stenosed_branch: OutletBranchName | None = None,
    severity: float | None = None,
) -> dict[str, Any]:
    active_branch = cfg.stenosed_branch if stenosed_branch is None else stenosed_branch
    active_severity = cfg.severity if severity is None else severity
    flows = murray_flow_split(cfg)

    inlet = simulate_branch(cfg, "inlet", flows["inlet"], cfg.inlet_pressure, None, 0.0)
    bifurcation_pressure = float(np.asarray(inlet["pressure"])[-1])
    left = simulate_branch(cfg, "left_outlet", flows["left_outlet"], bifurcation_pressure, active_branch, active_severity)
    right = simulate_branch(cfg, "right_outlet", flows["right_outlet"], bifurcation_pressure, active_branch, active_severity)

    return {
        "config": asdict(cfg),
        "flows": flows,
        "bifurcation_pressure": bifurcation_pressure,
        "branches": {
            "inlet": inlet,
            "left_outlet": left,
            "right_outlet": right,
        },
    }


def summarize_tree(
    cfg: TreeStenosisConfig,
    stenosed: dict[str, Any],
    healthy: dict[str, Any],
    stenosed_branch: OutletBranchName,
) -> dict[str, Any]:
    branches = stenosed["branches"]
    healthy_branches = healthy["branches"]
    branch_metrics: dict[str, dict[str, float | bool]] = {}
    for branch in ["inlet", "left_outlet", "right_outlet"]:
        field = branches[branch]
        healthy_field = healthy_branches[branch]
        pressure = np.asarray(field["pressure"])
        branch_metrics[branch] = {
            "flow_rate": float(field["flow_rate"]),
            "min_radius": float(np.min(field["radius"])),
            "max_velocity": float(np.max(field["velocity"])),
            "healthy_max_velocity": float(np.max(healthy_field["velocity"])),
            "total_pressure_drop": float(field["local_pressure_drop"][-1]),
            "healthy_total_pressure_drop": float(healthy_field["local_pressure_drop"][-1]),
            "max_wall_shear_stress": float(np.max(field["wall_shear_stress"])),
            "resistance": float(field["resistance"]),
            "pressure_monotone_decreasing": bool(np.all(np.diff(pressure) <= 1e-10)),
        }

    stenosed_metrics = branch_metrics[stenosed_branch]
    outlet_flow_sum = float(branch_metrics["left_outlet"]["flow_rate"] + branch_metrics["right_outlet"]["flow_rate"])
    return {
        "config": asdict(cfg),
        "stenosed_branch": stenosed_branch,
        "severity": cfg.severity,
        "branch_metrics": branch_metrics,
        "bifurcation_pressure": float(stenosed["bifurcation_pressure"]),
        "outlet_flow_sum": outlet_flow_sum,
        "inlet_flow": float(branch_metrics["inlet"]["flow_rate"]),
        "flow_conservation_error": abs(float(branch_metrics["inlet"]["flow_rate"]) - outlet_flow_sum),
        "all_pressures_monotone": all(bool(branch_metrics[b]["pressure_monotone_decreasing"]) for b in branch_metrics),
        "stenosed_branch_accelerates_flow": bool(stenosed_metrics["max_velocity"] > stenosed_metrics["healthy_max_velocity"]),
        "stenosed_branch_increases_drop": bool(stenosed_metrics["total_pressure_drop"] > stenosed_metrics["healthy_total_pressure_drop"]),
    }


def save_tree_artifacts(
    cfg: TreeStenosisConfig,
    stenosed: dict[str, Any],
    healthy: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_dir.joinpath("metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    labels = ["inlet", "left_outlet", "right_outlet"]
    fig, axes = plt.subplots(4, 3, figsize=(14, 10))
    for col, branch in enumerate(labels):
        data = stenosed["branches"][branch]
        base = healthy["branches"][branch]
        x = data["x"]
        axes[0, col].plot(x, base["radius"], label="healthy")
        axes[0, col].plot(x, data["radius"], label="stenosed")
        axes[0, col].set_title(branch)
        axes[0, col].set_ylabel("Radius")
        axes[1, col].plot(x, base["velocity"], label="healthy")
        axes[1, col].plot(x, data["velocity"], label="stenosed")
        axes[1, col].set_ylabel("Velocity")
        axes[2, col].plot(x, base["local_pressure_drop"], label="healthy")
        axes[2, col].plot(x, data["local_pressure_drop"], label="stenosed")
        axes[2, col].set_ylabel("Pressure drop")
        axes[3, col].plot(x, base["wall_shear_stress"], label="healthy")
        axes[3, col].plot(x, data["wall_shear_stress"], label="stenosed")
        axes[3, col].set_ylabel("WSS")
        axes[3, col].set_xlabel("Local x")
        for row in range(4):
            axes[row, col].grid(alpha=0.25)
        axes[0, col].legend()
    plt.tight_layout()
    plt.savefig(output_dir / "tree_forward_model.png", dpi=160)
    plt.close(fig)


def run_tree_stenosis(cfg: TreeStenosisConfig, output_dir: Path) -> dict[str, Any]:
    stenosed = simulate_tree(cfg, cfg.stenosed_branch, cfg.severity)
    healthy = simulate_tree(cfg, cfg.stenosed_branch, 0.0)
    metrics = summarize_tree(cfg, stenosed, healthy, cfg.stenosed_branch)
    save_tree_artifacts(cfg, stenosed, healthy, metrics, output_dir)
    return metrics
