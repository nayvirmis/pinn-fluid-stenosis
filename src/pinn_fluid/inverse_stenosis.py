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

from pinn_fluid.stenosis import SyntheticStenosisConfig, simulate_vessel


@dataclass
class InverseStenosisConfig:
    true_severity: float = 0.55
    true_center: float = 0.5
    width: float = 0.08
    flow_rate: float = 1.0
    radius: float = 1.0
    viscosity: float = 0.04
    density: float = 1.06
    inlet_pressure: float = 100.0
    length: float = 1.0
    points: int = 1000
    sensors: int = 20
    noise_level: float = 0.01
    epochs: int = 1200
    learning_rate: float = 0.04
    seed: int = 11
    severity_min: float = 0.05
    severity_max: float = 0.9
    center_min: float = 0.1
    center_max: float = 0.9


@dataclass
class InverseRunResult:
    metrics: dict[str, Any]
    history: dict[str, list[float]]
    true_fields: dict[str, np.ndarray]
    recovered_fields: dict[str, np.ndarray]
    sensor_indices: np.ndarray
    sensor_observations: dict[str, np.ndarray]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _bounded_value(raw: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return lower + (upper - lower) * torch.sigmoid(raw)


def _raw_from_value(value: float, lower: float, upper: float) -> float:
    clipped = min(max(value, lower + 1e-6), upper - 1e-6)
    scaled = (clipped - lower) / (upper - lower)
    return float(np.log(scaled / (1.0 - scaled)))


def _torch_forward(
    x: torch.Tensor,
    raw_severity: torch.Tensor,
    raw_center: torch.Tensor,
    cfg: InverseStenosisConfig,
) -> dict[str, torch.Tensor]:
    severity = _bounded_value(raw_severity, cfg.severity_min, cfg.severity_max)
    center = _bounded_value(raw_center, cfg.center_min, cfg.center_max)

    gaussian = torch.exp(-0.5 * ((x - center) / cfg.width) ** 2)
    radius = cfg.radius * (1.0 - severity * gaussian)
    radius = torch.clamp(radius, min=1e-4)
    area = torch.pi * radius ** 2
    velocity = cfg.flow_rate / area
    dp_dx = 8.0 * cfg.viscosity * cfg.flow_rate / (torch.pi * radius ** 4)

    dx = x[1:] - x[:-1]
    trapezoids = 0.5 * (dp_dx[1:] + dp_dx[:-1]) * dx
    pressure_drop = torch.cat(
        [
            torch.zeros(1, dtype=x.dtype, device=x.device),
            torch.cumsum(trapezoids, dim=0),
        ]
    )
    pressure = cfg.inlet_pressure - pressure_drop
    wall_shear_stress = 4.0 * cfg.viscosity * cfg.flow_rate / (torch.pi * radius ** 3)

    return {
        "severity": severity,
        "center": center,
        "radius": radius,
        "area": area,
        "velocity": velocity,
        "pressure": pressure,
        "pressure_drop": pressure_drop,
        "wall_shear_stress": wall_shear_stress,
    }


def _numpy_forward(cfg: InverseStenosisConfig, severity: float, center: float) -> dict[str, np.ndarray]:
    vessel_cfg = SyntheticStenosisConfig(
        length=cfg.length,
        radius=cfg.radius,
        flow_rate=cfg.flow_rate,
        viscosity=cfg.viscosity,
        density=cfg.density,
        inlet_pressure=cfg.inlet_pressure,
        severity=severity,
        center=center,
        width=cfg.width,
        points=cfg.points,
    )
    fields = simulate_vessel(vessel_cfg, stenosed=True)
    fields["pressure_drop"] = fields["pressure"][0] - fields["pressure"]
    return fields


def make_sensor_indices(cfg: InverseStenosisConfig) -> np.ndarray:
    if cfg.sensors < 6:
        raise ValueError("At least 6 sensors are needed for stable inverse recovery.")

    x = np.linspace(0.0, cfg.length, cfg.points)
    uniform = np.linspace(0, cfg.points - 1, cfg.sensors // 2, dtype=int)
    lesion_x = np.linspace(cfg.true_center - 2.25 * cfg.width, cfg.true_center + 2.25 * cfg.width, cfg.sensors - len(uniform))
    lesion_x = np.clip(lesion_x, 0.0, cfg.length)
    lesion = np.searchsorted(x, lesion_x)
    indices = np.unique(np.clip(np.concatenate([uniform, lesion]), 0, cfg.points - 1))
    return indices.astype(int)


def make_sensor_observations(
    cfg: InverseStenosisConfig,
    true_fields: dict[str, np.ndarray],
    sensor_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    velocity = true_fields["velocity"][sensor_indices].copy()
    pressure_drop = true_fields["pressure_drop"][sensor_indices].copy()

    velocity_scale = max(float(np.max(np.abs(velocity))), 1e-8)
    pressure_scale = max(float(np.max(np.abs(pressure_drop))), 1e-8)
    velocity += rng.normal(0.0, cfg.noise_level * velocity_scale, size=velocity.shape)
    pressure_drop += rng.normal(0.0, cfg.noise_level * pressure_scale, size=pressure_drop.shape)
    pressure_drop[0] = max(pressure_drop[0], 0.0)

    return {
        "x": true_fields["x"][sensor_indices],
        "velocity": velocity,
        "pressure_drop": pressure_drop,
    }


def _fit_once(
    cfg: InverseStenosisConfig,
    x: torch.Tensor,
    sensor_indices: torch.Tensor,
    observed_velocity: torch.Tensor,
    observed_pressure_drop: torch.Tensor,
    initial_severity: float,
    initial_center: float,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    raw_severity = torch.tensor(
        [_raw_from_value(initial_severity, cfg.severity_min, cfg.severity_max)],
        dtype=torch.float32,
        requires_grad=True,
    )
    raw_center = torch.tensor(
        [_raw_from_value(initial_center, cfg.center_min, cfg.center_max)],
        dtype=torch.float32,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([raw_severity, raw_center], lr=cfg.learning_rate)

    velocity_scale = torch.clamp(torch.max(torch.abs(observed_velocity)), min=1e-8)
    pressure_scale = torch.clamp(torch.max(torch.abs(observed_pressure_drop)), min=1e-8)
    history = {
        "loss": [],
        "velocity_loss": [],
        "pressure_loss": [],
        "severity": [],
        "center": [],
    }

    for _ in range(cfg.epochs):
        optimizer.zero_grad()
        pred = _torch_forward(x, raw_severity, raw_center, cfg)
        pred_velocity = pred["velocity"][sensor_indices]
        pred_pressure_drop = pred["pressure_drop"][sensor_indices]

        velocity_loss = torch.mean(((pred_velocity - observed_velocity) / velocity_scale) ** 2)
        pressure_loss = torch.mean(((pred_pressure_drop - observed_pressure_drop) / pressure_scale) ** 2)
        loss = velocity_loss + pressure_loss
        loss.backward()
        optimizer.step()

        history["loss"].append(float(loss.detach()))
        history["velocity_loss"].append(float(velocity_loss.detach()))
        history["pressure_loss"].append(float(pressure_loss.detach()))
        history["severity"].append(float(pred["severity"].detach()))
        history["center"].append(float(pred["center"].detach()))

    final = _torch_forward(x, raw_severity, raw_center, cfg)
    params = {
        "loss": float(history["loss"][-1]),
        "severity": float(final["severity"].detach()),
        "center": float(final["center"].detach()),
        "initial_severity": initial_severity,
        "initial_center": initial_center,
    }
    return params, history


def fit_inverse_stenosis(cfg: InverseStenosisConfig) -> InverseRunResult:
    set_seed(cfg.seed)
    true_fields = _numpy_forward(cfg, cfg.true_severity, cfg.true_center)
    sensor_indices_np = make_sensor_indices(cfg)
    observations = make_sensor_observations(cfg, true_fields, sensor_indices_np)

    x_t = torch.linspace(0.0, cfg.length, cfg.points, dtype=torch.float32)
    sensor_indices_t = torch.tensor(sensor_indices_np, dtype=torch.long)
    observed_velocity_t = torch.tensor(observations["velocity"], dtype=torch.float32)
    observed_pressure_drop_t = torch.tensor(observations["pressure_drop"], dtype=torch.float32)

    initial_centers = [0.25, 0.5, 0.75]
    initial_severities = [0.25, 0.55]
    runs: list[tuple[dict[str, float], dict[str, list[float]]]] = []
    for center in initial_centers:
        for severity in initial_severities:
            runs.append(
                _fit_once(
                    cfg,
                    x_t,
                    sensor_indices_t,
                    observed_velocity_t,
                    observed_pressure_drop_t,
                    initial_severity=severity,
                    initial_center=center,
                )
            )

    best_params, best_history = min(runs, key=lambda item: item[0]["loss"])
    recovered_fields = _numpy_forward(cfg, best_params["severity"], best_params["center"])

    true_total_drop = float(true_fields["pressure_drop"][-1])
    recovered_total_drop = float(recovered_fields["pressure_drop"][-1])
    true_min_radius = float(np.min(true_fields["radius"]))
    recovered_min_radius = float(np.min(recovered_fields["radius"]))

    metrics: dict[str, Any] = {
        "config": asdict(cfg),
        "true_severity": cfg.true_severity,
        "recovered_severity": best_params["severity"],
        "severity_abs_error": abs(best_params["severity"] - cfg.true_severity),
        "severity_relative_error": abs(best_params["severity"] - cfg.true_severity) / cfg.true_severity,
        "true_center": cfg.true_center,
        "recovered_center": best_params["center"],
        "center_abs_error": abs(best_params["center"] - cfg.true_center),
        "true_min_radius": true_min_radius,
        "recovered_min_radius": recovered_min_radius,
        "min_radius_abs_error": abs(recovered_min_radius - true_min_radius),
        "true_total_pressure_drop": true_total_drop,
        "recovered_total_pressure_drop": recovered_total_drop,
        "total_pressure_drop_relative_error": abs(recovered_total_drop - true_total_drop) / true_total_drop,
        "final_loss": best_params["loss"],
        "best_initial_severity": best_params["initial_severity"],
        "best_initial_center": best_params["initial_center"],
        "sensor_count": int(len(sensor_indices_np)),
    }
    return InverseRunResult(metrics, best_history, true_fields, recovered_fields, sensor_indices_np, observations)


def save_inverse_artifacts(result: InverseRunResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with output_dir.joinpath("metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(result.metrics, fh, indent=2)

    with output_dir.joinpath("history.json").open("w", encoding="utf-8") as fh:
        json.dump(result.history, fh, indent=2)

    epochs = np.arange(1, len(result.history["loss"]) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].semilogy(epochs, result.history["loss"], label="total")
    axes[0].semilogy(epochs, result.history["velocity_loss"], label="velocity")
    axes[0].semilogy(epochs, result.history["pressure_loss"], label="pressure drop")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, result.history["severity"], label="severity")
    axes[1].plot(epochs, result.history["center"], label="center")
    axes[1].axhline(result.metrics["true_severity"], color="tab:blue", linestyle="--", alpha=0.5)
    axes[1].axhline(result.metrics["true_center"], color="tab:orange", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Parameter value")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "inverse_training_history.png", dpi=160)
    plt.close(fig)

    x = result.true_fields["x"]
    sx = result.sensor_observations["x"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(x, result.true_fields["radius"], label="true")
    axes[0].plot(x, result.recovered_fields["radius"], "--", label="recovered")
    axes[0].set_ylabel("Radius")
    axes[0].legend()

    axes[1].plot(x, result.true_fields["velocity"], label="true")
    axes[1].plot(x, result.recovered_fields["velocity"], "--", label="recovered")
    axes[1].scatter(sx, result.sensor_observations["velocity"], s=22, color="black", label="sensors")
    axes[1].set_ylabel("Velocity")
    axes[1].legend()

    axes[2].plot(x, result.true_fields["pressure_drop"], label="true")
    axes[2].plot(x, result.recovered_fields["pressure_drop"], "--", label="recovered")
    axes[2].scatter(sx, result.sensor_observations["pressure_drop"], s=22, color="black", label="sensors")
    axes[2].set_ylabel("Pressure drop")
    axes[2].legend()

    axes[3].plot(x, result.true_fields["wall_shear_stress"], label="true")
    axes[3].plot(x, result.recovered_fields["wall_shear_stress"], "--", label="recovered")
    axes[3].set_ylabel("WSS")
    axes[3].set_xlabel("x")
    axes[3].legend()

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.axvline(result.metrics["true_center"], color="tab:green", linestyle=":", alpha=0.75)
        ax.axvline(result.metrics["recovered_center"], color="tab:red", linestyle=":", alpha=0.75)

    plt.tight_layout()
    plt.savefig(output_dir / "inverse_reconstruction.png", dpi=160)
    plt.close(fig)


def run_inverse_stenosis(cfg: InverseStenosisConfig, output_dir: Path) -> dict[str, Any]:
    result = fit_inverse_stenosis(cfg)
    save_inverse_artifacts(result, output_dir)
    return result.metrics
