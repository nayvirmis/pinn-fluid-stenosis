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
from torch import nn

from pinn_fluid.poiseuille import MLP, gradients


@dataclass
class InverseStenosedChannelPINNConfig:
    length: float = 1.0
    height: float = 1.0
    rho: float = 1.0
    nu: float = 0.02
    u_max: float = 1.0
    inlet_pressure: float = 0.0
    true_severity: float = 0.45
    true_center: float = 0.5
    width: float = 0.10
    hidden_width: int = 64
    hidden_layers: int = 4
    epochs: int = 2200
    learning_rate: float = 1e-3
    parameter_learning_rate: float = 5e-3
    interior_points: int = 1000
    wall_points: int = 280
    inlet_points: int = 192
    outlet_points: int = 192
    data_points: int = 96
    wall_observation_points: int = 80
    noise_level: float = 0.005
    boundary_loss_weight: float = 10.0
    data_loss_weight: float = 5.0
    geometry_loss_weight: float = 200.0
    seed: int = 23
    severity_min: float = 0.05
    severity_max: float = 0.80
    center_min: float = 0.25
    center_max: float = 0.75

    @property
    def half_height(self) -> float:
        return self.height / 2.0

    @property
    def inlet_flow_rate(self) -> float:
        return 4.0 * self.u_max * self.half_height / 3.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sigmoid_bounded(raw: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return lower + (upper - lower) * torch.sigmoid(raw)


def raw_from_value(value: float, lower: float, upper: float) -> float:
    clipped = min(max(value, lower + 1e-6), upper - 1e-6)
    scaled = (clipped - lower) / (upper - lower)
    return float(np.log(scaled / (1.0 - scaled)))


class InverseChannelPINN(nn.Module):
    def __init__(self, cfg: InverseStenosedChannelPINNConfig):
        super().__init__()
        self.field = MLP(2, 3, cfg.hidden_width, cfg.hidden_layers)
        self.raw_severity = nn.Parameter(
            torch.tensor([raw_from_value(0.25, cfg.severity_min, cfg.severity_max)], dtype=torch.float32)
        )
        self.raw_center = nn.Parameter(
            torch.tensor([raw_from_value(0.42, cfg.center_min, cfg.center_max)], dtype=torch.float32)
        )

    def severity(self, cfg: InverseStenosedChannelPINNConfig) -> torch.Tensor:
        return sigmoid_bounded(self.raw_severity, cfg.severity_min, cfg.severity_max)

    def center(self, cfg: InverseStenosedChannelPINNConfig) -> torch.Tensor:
        return sigmoid_bounded(self.raw_center, cfg.center_min, cfg.center_max)

    def half_height(self, x: torch.Tensor, cfg: InverseStenosedChannelPINNConfig) -> torch.Tensor:
        gaussian = torch.exp(-0.5 * ((x - self.center(cfg)) / cfg.width) ** 2)
        return cfg.half_height * (1.0 - self.severity(cfg) * gaussian)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.field(xy)


def true_half_height_np(x: np.ndarray, cfg: InverseStenosedChannelPINNConfig) -> np.ndarray:
    gaussian = np.exp(-0.5 * ((x - cfg.true_center) / cfg.width) ** 2)
    return cfg.half_height * (1.0 - cfg.true_severity * gaussian)


def reference_pressure_np(x: np.ndarray, cfg: InverseStenosedChannelPINNConfig) -> np.ndarray:
    grid = np.linspace(0.0, cfg.length, 2000)
    h = true_half_height_np(grid, cfg)
    g = 3.0 * cfg.rho * cfg.nu * cfg.inlet_flow_rate / (2.0 * h ** 3)
    dx = np.diff(grid)
    drop = np.zeros_like(grid)
    drop[1:] = np.cumsum(0.5 * (g[1:] + g[:-1]) * dx)
    return np.interp(x, grid, cfg.inlet_pressure - drop)


def reference_velocity_np(x: np.ndarray, y: np.ndarray, cfg: InverseStenosedChannelPINNConfig) -> np.ndarray:
    h = true_half_height_np(x, cfg)
    local_u_max = 3.0 * cfg.inlet_flow_rate / (4.0 * h)
    eta = np.clip(y / h, -1.0, 1.0)
    return local_u_max * (1.0 - eta ** 2)


def true_outlet_pressure(cfg: InverseStenosedChannelPINNConfig) -> float:
    return float(reference_pressure_np(np.array([cfg.length]), cfg)[0])


def make_observations(cfg: InverseStenosedChannelPINNConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    n_uniform = cfg.data_points // 2
    n_lesion = cfg.data_points - n_uniform
    x_uniform = np.linspace(0.06 * cfg.length, 0.94 * cfg.length, n_uniform)
    x_lesion = np.linspace(cfg.true_center - 2.0 * cfg.width, cfg.true_center + 2.0 * cfg.width, n_lesion)
    x = np.clip(np.concatenate([x_uniform, x_lesion]), 0.02 * cfg.length, 0.98 * cfg.length)
    eta = rng.uniform(-0.65, 0.65, size=x.shape)
    y = eta * true_half_height_np(x, cfg)
    u = reference_velocity_np(x, y, cfg)
    v = np.zeros_like(u)
    p = reference_pressure_np(x, cfg)

    u += rng.normal(0.0, cfg.noise_level * max(float(np.max(np.abs(u))), 1e-8), size=u.shape)
    p += rng.normal(0.0, cfg.noise_level * max(float(np.max(np.abs(p - p[-1]))), 1e-8), size=p.shape)
    return {
        "x": x,
        "y": y,
        "u": u,
        "v": v,
        "p": p,
    }


def make_wall_observations(cfg: InverseStenosedChannelPINNConfig) -> dict[str, np.ndarray]:
    x_uniform = np.linspace(0.04 * cfg.length, 0.96 * cfg.length, cfg.wall_observation_points // 2)
    x_lesion = np.linspace(
        cfg.true_center - 2.0 * cfg.width,
        cfg.true_center + 2.0 * cfg.width,
        cfg.wall_observation_points - len(x_uniform),
    )
    x = np.clip(np.concatenate([x_uniform, x_lesion]), 0.0, cfg.length)
    h = true_half_height_np(x, cfg)
    return {
        "x": x,
        "half_height": h,
    }


def inlet_u_profile(y: torch.Tensor, cfg: InverseStenosedChannelPINNConfig) -> torch.Tensor:
    return cfg.u_max * (1.0 - (y / cfg.half_height) ** 2)


def sample_interior(
    model: InverseChannelPINN,
    n: int,
    cfg: InverseStenosedChannelPINNConfig,
    device: torch.device,
) -> torch.Tensor:
    x = torch.rand(n, 1, device=device) * cfg.length
    eta = 2.0 * torch.rand(n, 1, device=device) - 1.0
    y = eta * model.half_height(x, cfg)
    return torch.cat([x, y], dim=1)


def sample_walls(
    model: InverseChannelPINN,
    n: int,
    cfg: InverseStenosedChannelPINNConfig,
    device: torch.device,
) -> torch.Tensor:
    x = torch.rand(n, 1, device=device) * cfg.length
    h = model.half_height(x, cfg)
    top_count = n // 2
    y = torch.cat([h[:top_count], -h[top_count:]], dim=0)
    return torch.cat([x, y], dim=1)


def sample_inlet(n: int, cfg: InverseStenosedChannelPINNConfig, device: torch.device) -> torch.Tensor:
    x = torch.zeros(n, 1, device=device)
    y = (2.0 * torch.rand(n, 1, device=device) - 1.0) * cfg.half_height
    return torch.cat([x, y], dim=1)


def sample_outlet(
    model: InverseChannelPINN,
    n: int,
    cfg: InverseStenosedChannelPINNConfig,
    device: torch.device,
) -> torch.Tensor:
    x = torch.full((n, 1), cfg.length, device=device)
    y = (2.0 * torch.rand(n, 1, device=device) - 1.0) * model.half_height(x, cfg)
    return torch.cat([x, y], dim=1)


def navier_stokes_residuals(
    model: InverseChannelPINN,
    xy: torch.Tensor,
    cfg: InverseStenosedChannelPINNConfig,
) -> dict[str, torch.Tensor]:
    xy.requires_grad_(True)
    pred = model(xy)
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]
    grad_u = gradients(u, xy)
    grad_v = gradients(v, xy)
    grad_p = gradients(p, xy)
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    v_x = grad_v[:, 0:1]
    v_y = grad_v[:, 1:2]
    p_x = grad_p[:, 0:1]
    p_y = grad_p[:, 1:2]
    u_xx = gradients(u_x, xy)[:, 0:1]
    u_yy = gradients(u_y, xy)[:, 1:2]
    v_xx = gradients(v_x, xy)[:, 0:1]
    v_yy = gradients(v_y, xy)[:, 1:2]
    return {
        "u": u,
        "v": v,
        "p": p,
        "continuity": u_x + v_y,
        "momentum_x": u * u_x + v * u_y + (1.0 / cfg.rho) * p_x - cfg.nu * (u_xx + u_yy),
        "momentum_y": u * v_x + v * v_y + (1.0 / cfg.rho) * p_y - cfg.nu * (v_xx + v_yy),
    }


def train_inverse_pinn(
    cfg: InverseStenosedChannelPINNConfig,
    device: torch.device,
) -> tuple[InverseChannelPINN, dict[str, list[float]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    observations = make_observations(cfg)
    wall_observations = make_wall_observations(cfg)
    model = InverseChannelPINN(cfg).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.field.parameters(), "lr": cfg.learning_rate},
            {"params": [model.raw_severity, model.raw_center], "lr": cfg.parameter_learning_rate},
        ]
    )
    obs_xy = torch.tensor(np.stack([observations["x"], observations["y"]], axis=-1), dtype=torch.float32, device=device)
    obs_u = torch.tensor(observations["u"][:, None], dtype=torch.float32, device=device)
    obs_v = torch.tensor(observations["v"][:, None], dtype=torch.float32, device=device)
    obs_p = torch.tensor(observations["p"][:, None], dtype=torch.float32, device=device)
    obs_wall_x = torch.tensor(wall_observations["x"][:, None], dtype=torch.float32, device=device)
    obs_wall_h = torch.tensor(wall_observations["half_height"][:, None], dtype=torch.float32, device=device)
    p_out = true_outlet_pressure(cfg)

    u_scale = torch.clamp(torch.max(torch.abs(obs_u)), min=1e-8)
    p_scale = torch.clamp(torch.max(torch.abs(obs_p - p_out)), min=1e-8)
    history = {
        "total_loss": [],
        "pde_loss": [],
        "bc_loss": [],
        "data_loss": [],
        "geometry_loss": [],
        "severity": [],
        "center": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad()
        interior = sample_interior(model, cfg.interior_points, cfg, device)
        walls = sample_walls(model, cfg.wall_points, cfg, device)
        inlet = sample_inlet(cfg.inlet_points, cfg, device)
        outlet = sample_outlet(model, cfg.outlet_points, cfg, device)

        res = navier_stokes_residuals(model, interior, cfg)
        pde_loss = (
            4.0 * res["continuity"].pow(2).mean()
            + res["momentum_x"].pow(2).mean()
            + res["momentum_y"].pow(2).mean()
        )
        wall_pred = model(walls)
        inlet_pred = model(inlet)
        outlet_pred = model(outlet)
        bc_loss = (
            wall_pred[:, 0:2].pow(2).mean()
            + (inlet_pred[:, 0:1] - inlet_u_profile(inlet[:, 1:2], cfg)).pow(2).mean()
            + inlet_pred[:, 1:2].pow(2).mean()
            + (inlet_pred[:, 2:3] - cfg.inlet_pressure).pow(2).mean()
            + outlet_pred[:, 1:2].pow(2).mean()
            + (outlet_pred[:, 2:3] - p_out).pow(2).mean()
        )
        data_pred = model(obs_xy)
        data_loss = (
            ((data_pred[:, 0:1] - obs_u) / u_scale).pow(2).mean()
            + ((data_pred[:, 1:2] - obs_v) / u_scale).pow(2).mean()
            + ((data_pred[:, 2:3] - obs_p) / p_scale).pow(2).mean()
        )
        geometry_loss = ((model.half_height(obs_wall_x, cfg) - obs_wall_h) / cfg.half_height).pow(2).mean()
        total_loss = (
            pde_loss
            + cfg.boundary_loss_weight * bc_loss
            + cfg.data_loss_weight * data_loss
            + cfg.geometry_loss_weight * geometry_loss
        )
        total_loss.backward()
        optimizer.step()

        history["total_loss"].append(float(total_loss.detach().cpu()))
        history["pde_loss"].append(float(pde_loss.detach().cpu()))
        history["bc_loss"].append(float(bc_loss.detach().cpu()))
        history["data_loss"].append(float(data_loss.detach().cpu()))
        history["geometry_loss"].append(float(geometry_loss.detach().cpu()))
        history["severity"].append(float(model.severity(cfg).detach().cpu()))
        history["center"].append(float(model.center(cfg).detach().cpu()))
        if epoch == 1 or epoch % 500 == 0:
            print(
                f"epoch={epoch:5d} total={history['total_loss'][-1]:.4e} "
                f"pde={history['pde_loss'][-1]:.4e} bc={history['bc_loss'][-1]:.4e} "
                f"data={history['data_loss'][-1]:.4e} geom={history['geometry_loss'][-1]:.4e} "
                f"severity={history['severity'][-1]:.4f} "
                f"center={history['center'][-1]:.4f}"
            )

    return model, history, observations, wall_observations


def evaluate_inverse_pinn(
    model: InverseChannelPINN,
    cfg: InverseStenosedChannelPINNConfig,
    device: torch.device,
) -> dict[str, Any]:
    x = torch.linspace(0.0, cfg.length, 120, dtype=torch.float32, device=device)[:, None]
    eta = torch.linspace(-0.95, 0.95, 50, dtype=torch.float32, device=device)[None, :]
    h = model.half_height(x, cfg)
    xx = x.repeat(1, eta.shape[1])
    yy = eta.repeat(x.shape[0], 1) * h
    xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    res = navier_stokes_residuals(model, xy, cfg)
    return {
        "config": asdict(cfg),
        "true_severity": cfg.true_severity,
        "recovered_severity": float(model.severity(cfg).detach().cpu()),
        "severity_abs_error": abs(float(model.severity(cfg).detach().cpu()) - cfg.true_severity),
        "true_center": cfg.true_center,
        "recovered_center": float(model.center(cfg).detach().cpu()),
        "center_abs_error": abs(float(model.center(cfg).detach().cpu()) - cfg.true_center),
        "mean_abs_continuity": float(res["continuity"].abs().mean().detach().cpu()),
        "mean_abs_momentum_x": float(res["momentum_x"].abs().mean().detach().cpu()),
        "mean_abs_momentum_y": float(res["momentum_y"].abs().mean().detach().cpu()),
        "true_outlet_pressure": true_outlet_pressure(cfg),
    }


def save_artifacts(
    model: InverseChannelPINN,
    history: dict[str, list[float]],
    observations: dict[str, np.ndarray],
    metrics: dict[str, Any],
    cfg: InverseStenosedChannelPINNConfig,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_dir.joinpath("metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    with output_dir.joinpath("history.json").open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    epochs = np.arange(1, len(history["total_loss"]) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].semilogy(epochs, history["total_loss"], label="total")
    axes[0].semilogy(epochs, history["pde_loss"], label="pde")
    axes[0].semilogy(epochs, history["bc_loss"], label="bc")
    axes[0].semilogy(epochs, history["data_loss"], label="data")
    axes[0].semilogy(epochs, history["geometry_loss"], label="geometry")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, history["severity"], label="severity")
    axes[1].plot(epochs, history["center"], label="center")
    axes[1].axhline(cfg.true_severity, color="tab:blue", linestyle="--", alpha=0.5)
    axes[1].axhline(cfg.true_center, color="tab:orange", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Parameter")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "inverse_pinn_training_history.png", dpi=160)
    plt.close(fig)

    x = np.linspace(0.0, cfg.length, 240)
    true_h = true_half_height_np(x, cfg)
    with torch.no_grad():
        xt = torch.tensor(x[:, None], dtype=torch.float32)
        pred_h = model.half_height(xt, cfg).cpu().numpy().ravel()
    plt.figure(figsize=(10, 5))
    plt.plot(x, true_h, label="true top wall")
    plt.plot(x, -true_h, color="tab:blue")
    plt.plot(x, pred_h, "--", label="recovered top wall")
    plt.plot(x, -pred_h, "--", color="tab:orange")
    plt.scatter(observations["x"], observations["y"], s=16, color="black", label="sensors")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Inverse PINN Recovered Stenosis Geometry")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "inverse_pinn_geometry.png", dpi=160)
    plt.close()


def run_inverse_stenosed_channel_pinn(
    cfg: InverseStenosedChannelPINNConfig,
    output_dir: Path,
    device: str | None = None,
) -> dict[str, Any]:
    set_seed(cfg.seed)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"using device={resolved_device}")
    model, history, observations, _ = train_inverse_pinn(cfg, resolved_device)
    metrics = evaluate_inverse_pinn(model, cfg, resolved_device)
    save_artifacts(model, history, observations, metrics, cfg, output_dir)
    return metrics
