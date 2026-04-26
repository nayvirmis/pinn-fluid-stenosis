from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SyntheticStenosisConfig:
    length: float = 1.0
    radius: float = 1.0
    flow_rate: float = 1.0
    viscosity: float = 0.04
    density: float = 1.06
    inlet_pressure: float = 100.0
    severity: float = 0.55
    center: float = 0.5
    width: float = 0.08
    points: int = 1000


def stenosis_profile(x: np.ndarray, cfg: SyntheticStenosisConfig) -> np.ndarray:
    gaussian = np.exp(-0.5 * ((x - cfg.center) / cfg.width) ** 2)
    radius = cfg.radius * (1.0 - cfg.severity * gaussian)
    return np.maximum(radius, 1e-4)


def healthy_profile(x: np.ndarray, cfg: SyntheticStenosisConfig) -> np.ndarray:
    return np.full_like(x, cfg.radius)


def area_from_radius(radius: np.ndarray) -> np.ndarray:
    return np.pi * radius ** 2


def velocity_from_area(flow_rate: float, area: np.ndarray) -> np.ndarray:
    return flow_rate / area


def pressure_gradient(flow_rate: float, viscosity: float, radius: np.ndarray) -> np.ndarray:
    return 8.0 * viscosity * flow_rate / (np.pi * radius ** 4)


def integrate_pressure(inlet_pressure: float, dp_dx: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    pressure = np.empty_like(x)
    pressure[0] = inlet_pressure
    pressure[1:] = inlet_pressure - np.cumsum(0.5 * (dp_dx[1:] + dp_dx[:-1]) * dx)
    return pressure


def reynolds_number(density: float, velocity: np.ndarray, radius: np.ndarray, viscosity: float) -> np.ndarray:
    diameter = 2.0 * radius
    return density * velocity * diameter / viscosity


def wall_shear_stress(viscosity: float, flow_rate: float, radius: np.ndarray) -> np.ndarray:
    return 4.0 * viscosity * flow_rate / (np.pi * radius ** 3)


def lesion_window_mask(x: np.ndarray, cfg: SyntheticStenosisConfig, half_width_multiplier: float = 2.0) -> np.ndarray:
    window = half_width_multiplier * cfg.width
    return np.abs(x - cfg.center) <= window


def simulate_vessel(cfg: SyntheticStenosisConfig, stenosed: bool = True) -> dict[str, np.ndarray]:
    x = np.linspace(0.0, cfg.length, cfg.points)
    radius = stenosis_profile(x, cfg) if stenosed else healthy_profile(x, cfg)
    area = area_from_radius(radius)
    velocity = velocity_from_area(cfg.flow_rate, area)
    dp_dx = pressure_gradient(cfg.flow_rate, cfg.viscosity, radius)
    pressure = integrate_pressure(cfg.inlet_pressure, dp_dx, x)
    reynolds = reynolds_number(cfg.density, velocity, radius, cfg.viscosity)
    wss = wall_shear_stress(cfg.viscosity, cfg.flow_rate, radius)
    return {
        "x": x,
        "radius": radius,
        "area": area,
        "velocity": velocity,
        "dp_dx": dp_dx,
        "pressure": pressure,
        "reynolds": reynolds,
        "wall_shear_stress": wss,
    }


def summarize_simulation(cfg: SyntheticStenosisConfig, stenosed: dict[str, np.ndarray], healthy: dict[str, np.ndarray]) -> dict[str, float | bool | dict[str, float]]:
    mask = lesion_window_mask(stenosed["x"], cfg)
    lesion_drop = float(stenosed["pressure"][mask][0] - stenosed["pressure"][mask][-1])
    total_drop = float(stenosed["pressure"][0] - stenosed["pressure"][-1])
    healthy_total_drop = float(healthy["pressure"][0] - healthy["pressure"][-1])

    metrics = {
        "config": asdict(cfg),
        "min_radius": float(np.min(stenosed["radius"])),
        "radius_reduction_fraction": float(1.0 - np.min(stenosed["radius"]) / cfg.radius),
        "max_velocity": float(np.max(stenosed["velocity"])),
        "healthy_velocity": float(np.max(healthy["velocity"])),
        "velocity_gain_vs_healthy": float(np.max(stenosed["velocity"]) / np.max(healthy["velocity"])),
        "outlet_pressure": float(stenosed["pressure"][-1]),
        "total_pressure_drop": total_drop,
        "healthy_total_pressure_drop": healthy_total_drop,
        "pressure_drop_gain_vs_healthy": float(total_drop / healthy_total_drop),
        "lesion_pressure_drop": lesion_drop,
        "lesion_drop_fraction_of_total": float(lesion_drop / total_drop) if total_drop > 0 else 0.0,
        "max_wall_shear_stress": float(np.max(stenosed["wall_shear_stress"])),
        "healthy_wall_shear_stress": float(np.max(healthy["wall_shear_stress"])),
        "max_reynolds": float(np.max(stenosed["reynolds"])),
        "pressure_monotone_decreasing": bool(np.all(np.diff(stenosed["pressure"]) <= 1e-10)),
        "stenosis_accelerates_flow": bool(np.max(stenosed["velocity"]) > np.max(healthy["velocity"])),
        "stenosis_increases_drop": bool(total_drop > healthy_total_drop),
    }
    return metrics


def save_stenosis_artifacts(
    cfg: SyntheticStenosisConfig,
    stenosed: dict[str, np.ndarray],
    healthy: dict[str, np.ndarray],
    metrics: dict[str, float | bool | dict[str, float]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with output_dir.joinpath("metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    x = stenosed["x"]
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(x, stenosed["radius"])
    axes[0].set_ylabel("Radius")
    axes[0].set_title("Synthetic Stenosed Vessel Forward Model")

    axes[1].plot(x, stenosed["area"])
    axes[1].set_ylabel("Area")

    axes[2].plot(x, stenosed["velocity"])
    axes[2].set_ylabel("Velocity")

    axes[3].plot(x, stenosed["pressure"])
    axes[3].set_ylabel("Pressure")

    axes[4].plot(x, stenosed["wall_shear_stress"])
    axes[4].set_ylabel("WSS")
    axes[4].set_xlabel("x")

    for ax in axes:
        ax.axvline(cfg.center, color="tab:red", linestyle="--", alpha=0.7)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_dir / "stenosis_forward_model.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes[0, 0].plot(x, healthy["radius"], label="healthy")
    axes[0, 0].plot(x, stenosed["radius"], label="stenosed")
    axes[0, 0].set_ylabel("Radius")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(x, healthy["velocity"], label="healthy")
    axes[0, 1].plot(x, stenosed["velocity"], label="stenosed")
    axes[0, 1].set_ylabel("Velocity")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(x, healthy["pressure"], label="healthy")
    axes[1, 0].plot(x, stenosed["pressure"], label="stenosed")
    axes[1, 0].set_ylabel("Pressure")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(x, healthy["wall_shear_stress"], label="healthy")
    axes[1, 1].plot(x, stenosed["wall_shear_stress"], label="stenosed")
    axes[1, 1].set_ylabel("WSS")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_to_healthy.png", dpi=160)
    plt.close(fig)


def run_synthetic_stenosis(cfg: SyntheticStenosisConfig, output_dir: Path) -> dict[str, float | bool | dict[str, float]]:
    stenosed = simulate_vessel(cfg, stenosed=True)
    healthy = simulate_vessel(cfg, stenosed=False)
    metrics = summarize_simulation(cfg, stenosed, healthy)
    save_stenosis_artifacts(cfg, stenosed, healthy, metrics, output_dir)
    return metrics
