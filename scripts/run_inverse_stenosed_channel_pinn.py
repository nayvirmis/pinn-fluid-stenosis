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

from pinn_fluid.inverse_stenosed_channel_pinn import (
    InverseStenosedChannelPINNConfig,
    run_inverse_stenosed_channel_pinn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover 2D stenosis parameters with an inverse PINN.")
    parser.add_argument("--true-severity", type=float, default=0.45)
    parser.add_argument("--true-center", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=2200)
    parser.add_argument("--noise-level", type=float, default=0.005)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "inverse_stenosed_channel_pinn")
    parser.add_argument("--severity-error-threshold", type=float, default=0.08)
    parser.add_argument("--center-error-threshold", type=float, default=0.08)
    parser.add_argument("--continuity-threshold", type=float, default=0.16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = InverseStenosedChannelPINNConfig(
        true_severity=args.true_severity,
        true_center=args.true_center,
        epochs=args.epochs,
        noise_level=args.noise_level,
    )
    metrics = run_inverse_stenosed_channel_pinn(cfg, args.output_dir, device=args.device)
    print(json.dumps(metrics, indent=2))

    checks = {
        "severity_error": metrics["severity_abs_error"] < args.severity_error_threshold,
        "center_error": metrics["center_abs_error"] < args.center_error_threshold,
        "continuity": metrics["mean_abs_continuity"] < args.continuity_threshold,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(f"\nInverse 2D PINN verification failed: {', '.join(failed)}")
        return 1

    print("\nInverse 2D PINN verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
