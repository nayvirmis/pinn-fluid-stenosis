from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn_fluid.tree_stenosis import OutletBranchName, TreeStenosisConfig, simulate_tree


@dataclass
class InverseTreeStenosisConfig:
    true_branch: OutletBranchName = "left_outlet"
    true_severity: float = 0.55
    sensors_per_outlet: int = 12
    noise_level: float = 0.01
    epochs: int = 1200
    learning_rate: float = 0.04
    seed: int = 17
    length_outlet: float = 0.7
    radius_left: float = 0.75
    radius_right: float = 0.65
    flow_rate: float = 1.0
    viscosity: float = 0.04
    density: float = 1.06
    stenosis_width: float = 0.08
    points_per_branch: int = 600
    severity_max: float = 0.9


@dataclass
class InverseTreeRunResult:
    metrics: dict[str, Any]
    history: dict[str, list[float]]
    true_tree: dict[str, Any]
    recovered_tree: dict[str, Any]
    observations: dict[str, dict[str, np.ndarray]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tree_config_from_inverse(cfg: InverseTreeStenosisConfig, branch: OutletBranchName, severity: float) -> TreeStenosisConfig:
    return TreeStenosisConfig(
        length_outlet=cfg.length_outlet,
        radius_left=cfg.radius_left,
        radius_right=cfg.radius_right,
        flow_rate=cfg.flow_rate,
        viscosity=cfg.viscosity,
        density=cfg.density,
        stenosis_width=cfg.stenosis_width,
        points_per_branch=cfg.points_per_branch,
        stenosed_branch=branch,
        severity=severity,
    )


def _outlet_flow_rates(cfg: InverseTreeStenosisConfig) -> dict[OutletBranchName, float]:
    left_weight = cfg.radius_left ** 3
    right_weight = cfg.radius_right ** 3
    total_weight = left_weight + right_weight
    return {
        "left_outlet": cfg.flow_rate * left_weight / total_weight,
        "right_outlet": cfg.flow_rate * right_weight / total_weight,
    }


def _bounded_severity(raw: torch.Tensor, cfg: InverseTreeStenosisConfig) -> torch.Tensor:
    return cfg.severity_max * torch.sigmoid(raw)


def _torch_branch_forward(
    x: torch.Tensor,
    baseline_radius: float,
    flow_rate: float,
    severity: torch.Tensor,
    cfg: InverseTreeStenosisConfig,
) -> dict[str, torch.Tensor]:
    center = 0.5 * cfg.length_outlet
    gaussian = torch.exp(-0.5 * ((x - center) / cfg.stenosis_width) ** 2)
    radius = baseline_radius * (1.0 - severity * gaussian)
    radius = torch.clamp(radius, min=1e-4)
    area = torch.pi * radius ** 2
    velocity = flow_rate / area
    dp_dx = 8.0 * cfg.viscosity * flow_rate / (torch.pi * radius ** 4)
    dx = x[1:] - x[:-1]
    local_drop = torch.cat(
        [
            torch.zeros(1, dtype=x.dtype),
            torch.cumsum(0.5 * (dp_dx[1:] + dp_dx[:-1]) * dx, dim=0),
        ]
    )
    return {
        "radius": radius,
        "velocity": velocity,
        "local_pressure_drop": local_drop,
        "wall_shear_stress": 4.0 * cfg.viscosity * flow_rate / (torch.pi * radius ** 3),
    }


def _sensor_indices(cfg: InverseTreeStenosisConfig) -> np.ndarray:
    if cfg.sensors_per_outlet < 6:
        raise ValueError("At least 6 sensors per outlet are needed for inverse tree recovery.")
    x = np.linspace(0.0, cfg.length_outlet, cfg.points_per_branch)
    uniform = np.linspace(0, cfg.points_per_branch - 1, cfg.sensors_per_outlet // 2, dtype=int)
    center = 0.5 * cfg.length_outlet
    lesion_x = np.linspace(center - 2.25 * cfg.stenosis_width, center + 2.25 * cfg.stenosis_width, cfg.sensors_per_outlet - len(uniform))
    lesion = np.searchsorted(x, np.clip(lesion_x, 0.0, cfg.length_outlet))
    return np.unique(np.clip(np.concatenate([uniform, lesion]), 0, cfg.points_per_branch - 1)).astype(int)


def _make_observations(cfg: InverseTreeStenosisConfig, true_tree: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(cfg.seed)
    indices = _sensor_indices(cfg)
    observations: dict[str, dict[str, np.ndarray]] = {}
    for branch in ["left_outlet", "right_outlet"]:
        data = true_tree["branches"][branch]
        velocity = np.asarray(data["velocity"])[indices].copy()
        drop = np.asarray(data["local_pressure_drop"])[indices].copy()
        velocity_scale = max(float(np.max(np.abs(velocity))), 1e-8)
        drop_scale = max(float(np.max(np.abs(drop))), 1e-8)
        velocity += rng.normal(0.0, cfg.noise_level * velocity_scale, size=velocity.shape)
        drop += rng.normal(0.0, cfg.noise_level * drop_scale, size=drop.shape)
        drop[0] = max(drop[0], 0.0)
        observations[branch] = {
            "indices": indices,
            "x": np.asarray(data["x"])[indices],
            "velocity": velocity,
            "local_pressure_drop": drop,
        }
    return observations


def fit_inverse_tree_stenosis(cfg: InverseTreeStenosisConfig) -> InverseTreeRunResult:
    set_seed(cfg.seed)
    true_cfg = tree_config_from_inverse(cfg, cfg.true_branch, cfg.true_severity)
    true_tree = simulate_tree(true_cfg, cfg.true_branch, cfg.true_severity)
    observations = _make_observations(cfg, true_tree)
    flows = _outlet_flow_rates(cfg)

    x = torch.linspace(0.0, cfg.length_outlet, cfg.points_per_branch, dtype=torch.float32)
    left_idx = torch.tensor(observations["left_outlet"]["indices"], dtype=torch.long)
    right_idx = torch.tensor(observations["right_outlet"]["indices"], dtype=torch.long)
    obs_left_v = torch.tensor(observations["left_outlet"]["velocity"], dtype=torch.float32)
    obs_left_drop = torch.tensor(observations["left_outlet"]["local_pressure_drop"], dtype=torch.float32)
    obs_right_v = torch.tensor(observations["right_outlet"]["velocity"], dtype=torch.float32)
    obs_right_drop = torch.tensor(observations["right_outlet"]["local_pressure_drop"], dtype=torch.float32)

    raw_left = torch.tensor([-3.0], dtype=torch.float32, requires_grad=True)
    raw_right = torch.tensor([-3.0], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([raw_left, raw_right], lr=cfg.learning_rate)

    left_v_scale = torch.clamp(torch.max(torch.abs(obs_left_v)), min=1e-8)
    left_drop_scale = torch.clamp(torch.max(torch.abs(obs_left_drop)), min=1e-8)
    right_v_scale = torch.clamp(torch.max(torch.abs(obs_right_v)), min=1e-8)
    right_drop_scale = torch.clamp(torch.max(torch.abs(obs_right_drop)), min=1e-8)

    history = {
        "loss": [],
        "left_severity": [],
        "right_severity": [],
        "left_loss": [],
        "right_loss": [],
    }
    for _ in range(cfg.epochs):
        optimizer.zero_grad()
        left_sev = _bounded_severity(raw_left, cfg)
        right_sev = _bounded_severity(raw_right, cfg)
        left_pred = _torch_branch_forward(x, cfg.radius_left, flows["left_outlet"], left_sev, cfg)
        right_pred = _torch_branch_forward(x, cfg.radius_right, flows["right_outlet"], right_sev, cfg)

        left_loss = torch.mean(((left_pred["velocity"][left_idx] - obs_left_v) / left_v_scale) ** 2)
        left_loss = left_loss + torch.mean(((left_pred["local_pressure_drop"][left_idx] - obs_left_drop) / left_drop_scale) ** 2)
        right_loss = torch.mean(((right_pred["velocity"][right_idx] - obs_right_v) / right_v_scale) ** 2)
        right_loss = right_loss + torch.mean(((right_pred["local_pressure_drop"][right_idx] - obs_right_drop) / right_drop_scale) ** 2)
        loss = left_loss + right_loss
        loss.backward()
        optimizer.step()

        history["loss"].append(float(loss.detach()))
        history["left_loss"].append(float(left_loss.detach()))
        history["right_loss"].append(float(right_loss.detach()))
        history["left_severity"].append(float(left_sev.detach()))
        history["right_severity"].append(float(right_sev.detach()))

    recovered_left = float(_bounded_severity(raw_left, cfg).detach())
    recovered_right = float(_bounded_severity(raw_right, cfg).detach())
    recovered_branch: OutletBranchName = "left_outlet" if recovered_left >= recovered_right else "right_outlet"
    recovered_severity = recovered_left if recovered_branch == "left_outlet" else recovered_right
    recovered_cfg = tree_config_from_inverse(cfg, recovered_branch, recovered_severity)
    recovered_tree = simulate_tree(recovered_cfg, recovered_branch, recovered_severity)

    true_branch_drop = float(true_tree["branches"][cfg.true_branch]["local_pressure_drop"][-1])
    recovered_branch_drop = float(recovered_tree["branches"][cfg.true_branch]["local_pressure_drop"][-1])
    metrics: dict[str, Any] = {
        "config": asdict(cfg),
        "true_branch": cfg.true_branch,
        "recovered_branch": recovered_branch,
        "branch_recovered_correctly": recovered_branch == cfg.true_branch,
        "true_severity": cfg.true_severity,
        "recovered_severity": recovered_severity,
        "recovered_left_severity": recovered_left,
        "recovered_right_severity": recovered_right,
        "severity_abs_error": abs(recovered_severity - cfg.true_severity),
        "true_branch_pressure_drop": true_branch_drop,
        "recovered_true_branch_pressure_drop": recovered_branch_drop,
        "true_branch_pressure_drop_relative_error": abs(recovered_branch_drop - true_branch_drop) / true_branch_drop,
        "final_loss": float(history["loss"][-1]),
        "sensor_count_per_outlet": int(len(observations["left_outlet"]["indices"])),
    }
    return InverseTreeRunResult(metrics, history, true_tree, recovered_tree, observations)


def save_inverse_tree_artifacts(result: InverseTreeRunResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_dir.joinpath("metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(result.metrics, fh, indent=2)
    with output_dir.joinpath("history.json").open("w", encoding="utf-8") as fh:
        json.dump(result.history, fh, indent=2)

    epochs = np.arange(1, len(result.history["loss"]) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].semilogy(epochs, result.history["loss"], label="total")
    axes[0].semilogy(epochs, result.history["left_loss"], label="left")
    axes[0].semilogy(epochs, result.history["right_loss"], label="right")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, result.history["left_severity"], label="left recovered")
    axes[1].plot(epochs, result.history["right_severity"], label="right recovered")
    axes[1].axhline(result.metrics["true_severity"], color="black", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("Severity")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "inverse_tree_training_history.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex="col")
    for col, branch in enumerate(["left_outlet", "right_outlet"]):
        true_branch = result.true_tree["branches"][branch]
        recovered_branch = result.recovered_tree["branches"][branch]
        obs = result.observations[branch]
        x = true_branch["x"]
        axes[0, col].plot(x, true_branch["radius"], label="true")
        axes[0, col].plot(x, recovered_branch["radius"], "--", label="recovered")
        axes[0, col].set_title(branch)
        axes[0, col].set_ylabel("Radius")
        axes[1, col].plot(x, true_branch["velocity"], label="true")
        axes[1, col].plot(x, recovered_branch["velocity"], "--", label="recovered")
        axes[1, col].scatter(obs["x"], obs["velocity"], s=20, color="black", label="sensors")
        axes[1, col].set_ylabel("Velocity")
        axes[2, col].plot(x, true_branch["local_pressure_drop"], label="true")
        axes[2, col].plot(x, recovered_branch["local_pressure_drop"], "--", label="recovered")
        axes[2, col].scatter(obs["x"], obs["local_pressure_drop"], s=20, color="black", label="sensors")
        axes[2, col].set_ylabel("Pressure drop")
        axes[2, col].set_xlabel("Local x")
        for row in range(3):
            axes[row, col].grid(alpha=0.25)
        axes[0, col].legend()
    plt.tight_layout()
    plt.savefig(output_dir / "inverse_tree_reconstruction.png", dpi=160)
    plt.close(fig)


def run_inverse_tree_stenosis(cfg: InverseTreeStenosisConfig, output_dir: Path) -> dict[str, Any]:
    result = fit_inverse_tree_stenosis(cfg)
    save_inverse_tree_artifacts(result, output_dir)
    return result.metrics
