#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MPLCONFIGDIR = ROOT / ".mplconfig"
XDG_CACHE_HOME = ROOT / ".cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_fluid.stenosed_channel_pinn import StenosedChannelPINNConfig, run_stenosed_channel_pinn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 2D PINN on a stenosed channel geometry.")
    parser.add_argument("--severity", type=float, default=0.45)
    parser.add_argument("--center", type=float, default=0.5)
    parser.add_argument("--width", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=3500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "stenosed_channel_pinn")
    parser.add_argument("--continuity-threshold", type=float, default=0.08)
    parser.add_argument("--momentum-threshold", type=float, default=0.12)
    parser.add_argument("--pressure-drop-threshold", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = StenosedChannelPINNConfig(
        severity=args.severity,
        center=args.center,
        width=args.width,
        epochs=args.epochs,
        learning_rate=args.lr,
    )
    metrics = run_stenosed_channel_pinn(cfg, args.output_dir, device=args.device)
    print(json.dumps(metrics, indent=2))

    checks = {
        "continuity": metrics["final_mean_abs_continuity"] < args.continuity_threshold,
        "momentum_x": metrics["final_mean_abs_momentum_x"] < args.momentum_threshold,
        "momentum_y": metrics["final_mean_abs_momentum_y"] < args.momentum_threshold,
        "positive_pressure_drop": metrics["pressure_drop"] > args.pressure_drop_threshold,
        "positive_throat_gain": metrics["throat_velocity_gain_vs_inlet"] > 0.9,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(f"\n2D stenosed channel verification failed: {', '.join(failed)}")
        return 1

    print("\n2D stenosed channel verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
