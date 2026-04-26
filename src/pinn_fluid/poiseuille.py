from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import random
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


@dataclass
class PoiseuilleConfig:
    length: float = 1.0
    height: float = 1.0
    rho: float = 1.0
    nu: float = 0.02
    u_max: float = 1.0
    p0: float = 0.0
    hidden_width: int = 64
    hidden_layers: int = 4
    epochs: int = 3000
    learning_rate: float = 1e-3
    interior_points: int = 1024
    wall_points: int = 256
    inlet_points: int = 256
    outlet_points: int = 256
    eval_x_points: int = 101
    eval_y_points: int = 81
    seed: int = 7

    @property
    def half_height(self) -> float:
        return self.height / 2.0

    @property
    def dp_dx(self) -> float:
        return -2.0 * self.rho * self.nu * self.u_max / (self.half_height ** 2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_width: int, hidden_layers: int):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_width, hidden_width), nn.Tanh()])
        layers.append(nn.Linear(hidden_width, out_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)


def exact_solution(x: torch.Tensor, y: torch.Tensor, cfg: PoiseuilleConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u = cfg.u_max * (1.0 - (y / cfg.half_height) ** 2)
    v = torch.zeros_like(u)
    p = cfg.p0 + cfg.dp_dx * x
    return u, v, p


def sample_interior(n: int, cfg: PoiseuilleConfig, device: torch.device) -> torch.Tensor:
    x = torch.rand(n, 1, device=device) * cfg.length
    y = (torch.rand(n, 1, device=device) - 0.5) * cfg.height
    return torch.cat([x, y], dim=1)


def sample_walls(n: int, cfg: PoiseuilleConfig, device: torch.device) -> torch.Tensor:
    x = torch.rand(n, 1, device=device) * cfg.length
    top = torch.full((n // 2, 1), cfg.half_height, device=device)
    bottom = torch.full((n - n // 2, 1), -cfg.half_height, device=device)
    y = torch.cat([top, bottom], dim=0)
    return torch.cat([x, y], dim=1)


def sample_inlet(n: int, cfg: PoiseuilleConfig, device: torch.device) -> torch.Tensor:
    x = torch.zeros(n, 1, device=device)
    y = (torch.rand(n, 1, device=device) - 0.5) * cfg.height
    return torch.cat([x, y], dim=1)


def sample_outlet(n: int, cfg: PoiseuilleConfig, device: torch.device) -> torch.Tensor:
    x = torch.full((n, 1), cfg.length, device=device)
    y = (torch.rand(n, 1, device=device) - 0.5) * cfg.height
    return torch.cat([x, y], dim=1)


def gradients(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def navier_stokes_residuals(model: nn.Module, xy: torch.Tensor, cfg: PoiseuilleConfig) -> Dict[str, torch.Tensor]:
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


def relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(pred - target) / np.linalg.norm(target))


def train_poiseuille_pinn(cfg: PoiseuilleConfig, device: torch.device) -> Tuple[nn.Module, Dict[str, List[float]]]:
    model = MLP(2, 3, cfg.hidden_width, cfg.hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    history: Dict[str, List[float]] = {
        "total_loss": [],
        "pde_loss": [],
        "wall_loss": [],
        "inlet_loss": [],
        "outlet_loss": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad()

        interior = sample_interior(cfg.interior_points, cfg, device)
        walls = sample_walls(cfg.wall_points, cfg, device)
        inlet = sample_inlet(cfg.inlet_points, cfg, device)
        outlet = sample_outlet(cfg.outlet_points, cfg, device)

        interior_res = navier_stokes_residuals(model, interior, cfg)
        pde_loss = (
            interior_res["continuity"].pow(2).mean()
            + interior_res["momentum_x"].pow(2).mean()
            + interior_res["momentum_y"].pow(2).mean()
        )

        wall_pred = model(walls)
        wall_loss = wall_pred[:, 0:2].pow(2).mean()

        inlet_pred = model(inlet)
        inlet_u, inlet_v, _ = exact_solution(inlet[:, 0:1], inlet[:, 1:2], cfg)
        inlet_loss = (
            (inlet_pred[:, 0:1] - inlet_u).pow(2).mean()
            + (inlet_pred[:, 1:2] - inlet_v).pow(2).mean()
        )

        outlet_pred = model(outlet)
        _, outlet_v, outlet_p = exact_solution(outlet[:, 0:1], outlet[:, 1:2], cfg)
        outlet_loss = (
            (outlet_pred[:, 1:2] - outlet_v).pow(2).mean()
            + (outlet_pred[:, 2:3] - outlet_p).pow(2).mean()
        )

        total_loss = 1.0 * pde_loss + 10.0 * wall_loss + 10.0 * inlet_loss + 10.0 * outlet_loss
        total_loss.backward()
        optimizer.step()

        history["total_loss"].append(float(total_loss.detach().cpu()))
        history["pde_loss"].append(float(pde_loss.detach().cpu()))
        history["wall_loss"].append(float(wall_loss.detach().cpu()))
        history["inlet_loss"].append(float(inlet_loss.detach().cpu()))
        history["outlet_loss"].append(float(outlet_loss.detach().cpu()))

        if epoch % 500 == 0 or epoch == 1:
            print(
                f"epoch={epoch:5d} "
                f"total={history['total_loss'][-1]:.4e} "
                f"pde={history['pde_loss'][-1]:.4e} "
                f"wall={history['wall_loss'][-1]:.4e} "
                f"inlet={history['inlet_loss'][-1]:.4e} "
                f"outlet={history['outlet_loss'][-1]:.4e}"
            )

    return model, history


def evaluate_model(model: nn.Module, cfg: PoiseuilleConfig, device: torch.device) -> Dict[str, object]:
    x = np.linspace(0.0, cfg.length, cfg.eval_x_points)
    y = np.linspace(-cfg.half_height, cfg.half_height, cfg.eval_y_points)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    xy = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    xy_t = torch.tensor(xy, dtype=torch.float32, device=device, requires_grad=True)

    res = navier_stokes_residuals(model, xy_t, cfg)
    pred_u = res["u"].detach().cpu().numpy().reshape(cfg.eval_y_points, cfg.eval_x_points)
    pred_v = res["v"].detach().cpu().numpy().reshape(cfg.eval_y_points, cfg.eval_x_points)
    pred_p = res["p"].detach().cpu().numpy().reshape(cfg.eval_y_points, cfg.eval_x_points)

    exact_u_t, exact_v_t, exact_p_t = exact_solution(
        xy_t[:, 0:1],
        xy_t[:, 1:2],
        cfg,
    )
    exact_u = exact_u_t.detach().cpu().numpy().reshape(cfg.eval_y_points, cfg.eval_x_points)
    exact_v = exact_v_t.detach().cpu().numpy().reshape(cfg.eval_y_points, cfg.eval_x_points)
    exact_p = exact_p_t.detach().cpu().numpy().reshape(cfg.eval_y_points, cfg.eval_x_points)

    metrics = {
        "relative_l2_u": relative_l2(pred_u, exact_u),
        "relative_l2_v": float(np.linalg.norm(pred_v - exact_v)),
        "relative_l2_p": relative_l2(pred_p, exact_p) if np.linalg.norm(exact_p) > 0 else float(np.linalg.norm(pred_p - exact_p)),
        "max_abs_v": float(np.max(np.abs(pred_v))),
        "mean_abs_continuity": float(res["continuity"].abs().mean().detach().cpu()),
        "mean_abs_momentum_x": float(res["momentum_x"].abs().mean().detach().cpu()),
        "mean_abs_momentum_y": float(res["momentum_y"].abs().mean().detach().cpu()),
        "pred_u_centerline": pred_u[cfg.eval_y_points // 2, :].tolist(),
        "exact_u_centerline": exact_u[cfg.eval_y_points // 2, :].tolist(),
        "pred_p_centerline": pred_p[cfg.eval_y_points // 2, :].tolist(),
        "exact_p_centerline": exact_p[cfg.eval_y_points // 2, :].tolist(),
        "grid_shape": [cfg.eval_y_points, cfg.eval_x_points],
    }
    return metrics


def save_artifacts(
    history: Dict[str, List[float]],
    metrics: Dict[str, object],
    model: nn.Module,
    cfg: PoiseuilleConfig,
    device: torch.device,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with output_dir.joinpath("metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    epochs = np.arange(1, len(history["total_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.semilogy(epochs, history["total_loss"], label="total")
    plt.semilogy(epochs, history["pde_loss"], label="pde")
    plt.semilogy(epochs, history["wall_loss"], label="wall")
    plt.semilogy(epochs, history["inlet_loss"], label="inlet")
    plt.semilogy(epochs, history["outlet_loss"], label="outlet")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Poiseuille PINN Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=160)
    plt.close()

    x = np.linspace(0.0, cfg.length, cfg.eval_x_points)
    y = np.linspace(-cfg.half_height, cfg.half_height, cfg.eval_y_points)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xy = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    xy_t = torch.tensor(xy, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(xy_t)
    pred_u = pred[:, 0].cpu().numpy().reshape(cfg.eval_y_points, cfg.eval_x_points)
    pred_p = pred[:, 2].cpu().numpy().reshape(cfg.eval_y_points, cfg.eval_x_points)
    exact_u_t, _, exact_p_t = exact_solution(xy_t[:, 0:1], xy_t[:, 1:2], cfg)
    exact_u = exact_u_t.cpu().numpy().reshape(cfg.eval_y_points, cfg.eval_x_points)
    exact_p = exact_p_t.cpu().numpy().reshape(cfg.eval_y_points, cfg.eval_x_points)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    im0 = axes[0, 0].imshow(pred_u, origin="lower", aspect="auto", extent=[0, cfg.length, -cfg.half_height, cfg.half_height])
    axes[0, 0].set_title("Predicted u(x, y)")
    plt.colorbar(im0, ax=axes[0, 0])
    im1 = axes[0, 1].imshow(exact_u, origin="lower", aspect="auto", extent=[0, cfg.length, -cfg.half_height, cfg.half_height])
    axes[0, 1].set_title("Exact u(x, y)")
    plt.colorbar(im1, ax=axes[0, 1])
    axes[1, 0].plot(y, pred_u[:, cfg.eval_x_points // 2], label="pred")
    axes[1, 0].plot(y, exact_u[:, cfg.eval_x_points // 2], "--", label="exact")
    axes[1, 0].set_title("Mid-channel Velocity Profile")
    axes[1, 0].set_xlabel("y")
    axes[1, 0].legend()
    axes[1, 1].plot(x, pred_p[cfg.eval_y_points // 2, :], label="pred")
    axes[1, 1].plot(x, exact_p[cfg.eval_y_points // 2, :], "--", label="exact")
    axes[1, 1].set_title("Centerline Pressure")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].legend()
    plt.tight_layout()
    plt.savefig(output_dir / "poiseuille_solution.png", dpi=160)
    plt.close(fig)


def run_baseline(cfg: PoiseuilleConfig, output_dir: Path, device: str | None = None) -> Dict[str, object]:
    set_seed(cfg.seed)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"using device={resolved_device}")
    print(f"dp_dx={cfg.dp_dx:.6f}")

    model, history = train_poiseuille_pinn(cfg, resolved_device)
    metrics = evaluate_model(model, cfg, resolved_device)
    save_artifacts(history, metrics, model, cfg, resolved_device, output_dir)
    return metrics
