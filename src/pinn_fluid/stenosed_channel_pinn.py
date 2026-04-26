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
class StenosedChannelPINNConfig:
    length: float = 1.0
    height: float = 1.0
    rho: float = 1.0
    nu: float = 0.02
    u_max: float = 1.0
    inlet_pressure: float = 0.0
    outlet_pressure: float = -0.25
    severity: float = 0.45
    center: float = 0.5
    width: float = 0.10
    hidden_width: int = 96
    hidden_layers: int = 5
    epochs: int = 3500
    learning_rate: float = 1e-3
    interior_points: int = 1800
    wall_points: int = 480
    inlet_points: int = 256
    outlet_points: int = 256
    reference_points: int = 512
    eval_x_points: int = 140
    eval_y_points: int = 90
    lesion_sample_fraction: float = 0.25
    near_wall_sample_fraction: float = 0.20
    continuity_loss_weight: float = 6.0
    momentum_loss_weight: float = 4.0
    wall_loss_weight: float = 10.0
    inlet_loss_weight: float = 10.0
    outlet_loss_weight: float = 10.0
    reference_loss_weight: float = 0.5
    geometry_aware_coordinates: bool = False
    hard_wall_velocity: bool = False
    seed: int = 19

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


def channel_half_height_torch(x: torch.Tensor, cfg: StenosedChannelPINNConfig) -> torch.Tensor:
    gaussian = torch.exp(-0.5 * ((x - cfg.center) / cfg.width) ** 2)
    return cfg.half_height * (1.0 - cfg.severity * gaussian)


def channel_half_height_np(x: np.ndarray, cfg: StenosedChannelPINNConfig) -> np.ndarray:
    gaussian = np.exp(-0.5 * ((x - cfg.center) / cfg.width) ** 2)
    return cfg.half_height * (1.0 - cfg.severity * gaussian)


def inlet_u_profile(y: torch.Tensor, cfg: StenosedChannelPINNConfig) -> torch.Tensor:
    return cfg.u_max * (1.0 - (y / cfg.half_height) ** 2)


class GeometryAwareStenosedChannelPINN(nn.Module):
    """PINN using normalized axial coordinates and local wall coordinate eta."""

    def __init__(self, cfg: StenosedChannelPINNConfig):
        super().__init__()
        self.field = MLP(2, 3, cfg.hidden_width, cfg.hidden_layers)

    def forward(self, xy: torch.Tensor, cfg: StenosedChannelPINNConfig) -> torch.Tensor:
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        h = channel_half_height_torch(x, cfg)
        eta = y / h
        x_scaled = 2.0 * x / cfg.length - 1.0
        raw = self.field(torch.cat([x_scaled, eta], dim=1))
        if not cfg.hard_wall_velocity:
            return raw
        wall_factor = 1.0 - eta.pow(2)
        return torch.cat(
            [
                wall_factor * raw[:, 0:1],
                wall_factor * raw[:, 1:2],
                raw[:, 2:3],
            ],
            dim=1,
        )


def model_forward(
    model: nn.Module,
    xy: torch.Tensor,
    cfg: StenosedChannelPINNConfig,
) -> torch.Tensor:
    if isinstance(model, GeometryAwareStenosedChannelPINN):
        return model(xy, cfg)
    return model(xy)


def sample_axial_locations(
    n: int,
    cfg: StenosedChannelPINNConfig,
    device: torch.device,
    lesion_fraction: float | None = None,
) -> torch.Tensor:
    lesion_fraction = cfg.lesion_sample_fraction if lesion_fraction is None else lesion_fraction
    lesion_count = int(round(n * lesion_fraction))
    uniform_count = n - lesion_count
    xs = []
    if uniform_count:
        xs.append(torch.rand(uniform_count, 1, device=device) * cfg.length)
    if lesion_count:
        lesion_x = cfg.center + cfg.width * torch.randn(lesion_count, 1, device=device)
        xs.append(torch.clamp(lesion_x, 0.0, cfg.length))
    x = torch.cat(xs, dim=0)
    return x[torch.randperm(n, device=device)]


def sample_interior(n: int, cfg: StenosedChannelPINNConfig, device: torch.device) -> torch.Tensor:
    x = sample_axial_locations(n, cfg, device)
    h = channel_half_height_torch(x, cfg)
    near_wall_count = int(round(n * cfg.near_wall_sample_fraction))
    bulk_count = n - near_wall_count
    etas = []
    if bulk_count:
        etas.append(2.0 * torch.rand(bulk_count, 1, device=device) - 1.0)
    if near_wall_count:
        sign = torch.where(
            torch.rand(near_wall_count, 1, device=device) < 0.5,
            -torch.ones(near_wall_count, 1, device=device),
            torch.ones(near_wall_count, 1, device=device),
        )
        distance_from_wall = 0.30 * torch.rand(near_wall_count, 1, device=device).pow(2)
        etas.append(sign * (1.0 - distance_from_wall))
    eta = torch.cat(etas, dim=0)
    eta = eta[torch.randperm(n, device=device)]
    y = eta * h
    return torch.cat([x, y], dim=1)


def sample_walls(n: int, cfg: StenosedChannelPINNConfig, device: torch.device) -> torch.Tensor:
    x = sample_axial_locations(n, cfg, device, lesion_fraction=0.55)
    h = channel_half_height_torch(x, cfg)
    top_count = n // 2
    y = torch.cat([h[:top_count], -h[top_count:]], dim=0)
    return torch.cat([x, y], dim=1)


def sample_inlet(n: int, cfg: StenosedChannelPINNConfig, device: torch.device) -> torch.Tensor:
    x = torch.zeros(n, 1, device=device)
    y = (2.0 * torch.rand(n, 1, device=device) - 1.0) * cfg.half_height
    return torch.cat([x, y], dim=1)


def sample_outlet(n: int, cfg: StenosedChannelPINNConfig, device: torch.device) -> torch.Tensor:
    x = torch.full((n, 1), cfg.length, device=device)
    h = channel_half_height_torch(x, cfg)
    y = (2.0 * torch.rand(n, 1, device=device) - 1.0) * h
    return torch.cat([x, y], dim=1)


def reference_pressure_drop_np(x: np.ndarray, cfg: StenosedChannelPINNConfig) -> np.ndarray:
    grid = np.linspace(0.0, cfg.length, 2000)
    h = channel_half_height_np(grid, cfg)
    gradient = 3.0 * cfg.rho * cfg.nu * cfg.inlet_flow_rate / (2.0 * h ** 3)
    dx = np.diff(grid)
    drop = np.zeros_like(grid)
    drop[1:] = np.cumsum(0.5 * (gradient[1:] + gradient[:-1]) * dx)
    target_drop = cfg.inlet_pressure - cfg.outlet_pressure
    if drop[-1] > 1e-12:
        drop *= target_drop / drop[-1]
    return np.interp(x, grid, drop)


def reference_velocity_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: StenosedChannelPINNConfig,
) -> torch.Tensor:
    h = channel_half_height_torch(x, cfg)
    local_u_max = 3.0 * cfg.inlet_flow_rate / (4.0 * h)
    eta = torch.clamp(y / h, -1.0, 1.0)
    return local_u_max * (1.0 - eta.pow(2))


def sample_reference(
    n: int,
    cfg: StenosedChannelPINNConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xy = sample_interior(n, cfg, device)
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    u = reference_velocity_torch(x, y, cfg)
    v = torch.zeros_like(u)
    drop = reference_pressure_drop_np(x.detach().cpu().numpy().ravel(), cfg)
    p = cfg.inlet_pressure - torch.tensor(drop[:, None], dtype=torch.float32, device=device)
    return xy, u, v, p


def navier_stokes_residuals(
    model: nn.Module,
    xy: torch.Tensor,
    cfg: StenosedChannelPINNConfig,
) -> dict[str, torch.Tensor]:
    xy.requires_grad_(True)
    pred = model_forward(model, xy, cfg)
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

    continuity = u_x + v_y
    momentum_x = u * u_x + v * u_y + (1.0 / cfg.rho) * p_x - cfg.nu * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + (1.0 / cfg.rho) * p_y - cfg.nu * (v_xx + v_yy)
    return {
        "u": u,
        "v": v,
        "p": p,
        "continuity": continuity,
        "momentum_x": momentum_x,
        "momentum_y": momentum_y,
    }


def train_stenosed_channel_pinn(
    cfg: StenosedChannelPINNConfig,
    device: torch.device,
) -> tuple[nn.Module, dict[str, list[float]]]:
    if cfg.geometry_aware_coordinates:
        model: nn.Module = GeometryAwareStenosedChannelPINN(cfg).to(device)
    else:
        model = MLP(2, 3, cfg.hidden_width, cfg.hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    history = {
        "total_loss": [],
        "pde_loss": [],
        "wall_loss": [],
        "inlet_loss": [],
        "outlet_loss": [],
        "reference_loss": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad()
        interior = sample_interior(cfg.interior_points, cfg, device)
        walls = sample_walls(cfg.wall_points, cfg, device)
        inlet = sample_inlet(cfg.inlet_points, cfg, device)
        outlet = sample_outlet(cfg.outlet_points, cfg, device)
        ref_xy, ref_u, ref_v, ref_p = sample_reference(cfg.reference_points, cfg, device)

        res = navier_stokes_residuals(model, interior, cfg)
        pde_loss = (
            cfg.continuity_loss_weight * res["continuity"].pow(2).mean()
            + cfg.momentum_loss_weight
            * (res["momentum_x"].pow(2).mean() + res["momentum_y"].pow(2).mean())
        )

        wall_pred = model_forward(model, walls, cfg)
        wall_loss = wall_pred[:, 0:2].pow(2).mean()

        inlet_pred = model_forward(model, inlet, cfg)
        inlet_loss = (
            (inlet_pred[:, 0:1] - inlet_u_profile(inlet[:, 1:2], cfg)).pow(2).mean()
            + inlet_pred[:, 1:2].pow(2).mean()
            + (inlet_pred[:, 2:3] - cfg.inlet_pressure).pow(2).mean()
        )

        outlet_pred = model_forward(model, outlet, cfg)
        outlet_loss = (
            outlet_pred[:, 1:2].pow(2).mean()
            + (outlet_pred[:, 2:3] - cfg.outlet_pressure).pow(2).mean()
        )

        ref_pred = model_forward(model, ref_xy, cfg)
        pressure_scale = max(abs(cfg.inlet_pressure - cfg.outlet_pressure), 1e-8)
        reference_loss = (
            ((ref_pred[:, 0:1] - ref_u) / cfg.u_max).pow(2).mean()
            + ((ref_pred[:, 1:2] - ref_v) / cfg.u_max).pow(2).mean()
            + ((ref_pred[:, 2:3] - ref_p) / pressure_scale).pow(2).mean()
        )

        total_loss = (
            pde_loss
            + cfg.wall_loss_weight * wall_loss
            + cfg.inlet_loss_weight * inlet_loss
            + cfg.outlet_loss_weight * outlet_loss
            + cfg.reference_loss_weight * reference_loss
        )
        total_loss.backward()
        optimizer.step()

        history["total_loss"].append(float(total_loss.detach().cpu()))
        history["pde_loss"].append(float(pde_loss.detach().cpu()))
        history["wall_loss"].append(float(wall_loss.detach().cpu()))
        history["inlet_loss"].append(float(inlet_loss.detach().cpu()))
        history["outlet_loss"].append(float(outlet_loss.detach().cpu()))
        history["reference_loss"].append(float(reference_loss.detach().cpu()))

        if epoch == 1 or epoch % 500 == 0:
            print(
                f"epoch={epoch:5d} "
                f"total={history['total_loss'][-1]:.4e} "
                f"pde={history['pde_loss'][-1]:.4e} "
                f"wall={history['wall_loss'][-1]:.4e} "
                f"inlet={history['inlet_loss'][-1]:.4e} "
                f"outlet={history['outlet_loss'][-1]:.4e} "
                f"ref={history['reference_loss'][-1]:.4e}"
            )

    return model, history


def evaluate_model(
    model: nn.Module,
    cfg: StenosedChannelPINNConfig,
    device: torch.device,
) -> dict[str, Any]:
    x = np.linspace(0.0, cfg.length, cfg.eval_x_points)
    y = np.linspace(-cfg.half_height, cfg.half_height, cfg.eval_y_points)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    hh = channel_half_height_np(xx, cfg)
    mask = np.abs(yy) <= hh
    xy = np.stack([xx[mask], yy[mask]], axis=-1)
    xy_t = torch.tensor(xy, dtype=torch.float32, device=device, requires_grad=True)
    res = navier_stokes_residuals(model, xy_t, cfg)
    pred = {key: value.detach().cpu().numpy().ravel() for key, value in res.items()}

    x_values = xy[:, 0]
    throat_window = np.abs(x_values - cfg.center) <= 0.03 * cfg.length
    inlet_window = x_values <= 0.03 * cfg.length
    outlet_window = x_values >= 0.97 * cfg.length
    inlet_pressure = float(np.mean(pred["p"][inlet_window]))
    outlet_pressure = float(np.mean(pred["p"][outlet_window]))

    throat_max_velocity = float(np.max(np.sqrt(pred["u"][throat_window] ** 2 + pred["v"][throat_window] ** 2)))
    inlet_max_velocity = float(np.max(np.sqrt(pred["u"][inlet_window] ** 2 + pred["v"][inlet_window] ** 2)))
    centerline_idx = np.abs(x_values - cfg.center).argmin()

    return {
        "config": asdict(cfg),
        "final_mean_abs_continuity": float(np.mean(np.abs(pred["continuity"]))),
        "final_mean_abs_momentum_x": float(np.mean(np.abs(pred["momentum_x"]))),
        "final_mean_abs_momentum_y": float(np.mean(np.abs(pred["momentum_y"]))),
        "max_velocity": float(np.max(np.sqrt(pred["u"] ** 2 + pred["v"] ** 2))),
        "throat_max_velocity": throat_max_velocity,
        "inlet_max_velocity": inlet_max_velocity,
        "throat_velocity_gain_vs_inlet": throat_max_velocity / max(inlet_max_velocity, 1e-8),
        "mean_inlet_pressure": inlet_pressure,
        "mean_outlet_pressure": outlet_pressure,
        "pressure_drop": inlet_pressure - outlet_pressure,
        "min_half_height": float(np.min(channel_half_height_np(x, cfg))),
        "centerline_pressure_near_throat": float(pred["p"][centerline_idx]),
        "evaluated_points": int(len(xy)),
    }


def save_artifacts(
    model: nn.Module,
    history: dict[str, list[float]],
    metrics: dict[str, Any],
    cfg: StenosedChannelPINNConfig,
    device: torch.device,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_dir.joinpath("metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    epochs = np.arange(1, len(history["total_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    for key in ["total_loss", "pde_loss", "wall_loss", "inlet_loss", "outlet_loss", "reference_loss"]:
        plt.semilogy(epochs, history[key], label=key.replace("_", " "))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("2D Stenosed Channel PINN Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "stenosed_channel_training_history.png", dpi=160)
    plt.close()

    x = np.linspace(0.0, cfg.length, cfg.eval_x_points)
    y = np.linspace(-cfg.half_height, cfg.half_height, cfg.eval_y_points)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    hh = channel_half_height_np(xx, cfg)
    mask = np.abs(yy) <= hh
    xy = np.stack([xx[mask], yy[mask]], axis=-1)
    xy_t = torch.tensor(xy, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model_forward(model, xy_t, cfg).cpu().numpy()

    speed = np.full_like(xx, np.nan, dtype=float)
    pressure = np.full_like(xx, np.nan, dtype=float)
    speed[mask] = np.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2)
    pressure[mask] = pred[:, 2]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    im0 = axes[0].imshow(speed, origin="lower", aspect="auto", extent=[0, cfg.length, -cfg.half_height, cfg.half_height])
    axes[0].plot(x, channel_half_height_np(x, cfg), color="white", linewidth=1.5)
    axes[0].plot(x, -channel_half_height_np(x, cfg), color="white", linewidth=1.5)
    axes[0].set_ylabel("y")
    axes[0].set_title("Velocity magnitude")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pressure, origin="lower", aspect="auto", extent=[0, cfg.length, -cfg.half_height, cfg.half_height])
    axes[1].plot(x, channel_half_height_np(x, cfg), color="white", linewidth=1.5)
    axes[1].plot(x, -channel_half_height_np(x, cfg), color="white", linewidth=1.5)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Pressure")
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(output_dir / "stenosed_channel_fields.png", dpi=160)
    plt.close(fig)


def run_stenosed_channel_pinn(
    cfg: StenosedChannelPINNConfig,
    output_dir: Path,
    device: str | None = None,
) -> dict[str, Any]:
    set_seed(cfg.seed)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"using device={resolved_device}")
    model, history = train_stenosed_channel_pinn(cfg, resolved_device)
    metrics = evaluate_model(model, cfg, resolved_device)
    save_artifacts(model, history, metrics, cfg, resolved_device, output_dir)
    return metrics
