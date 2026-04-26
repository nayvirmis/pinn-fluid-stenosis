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

from pinn_fluid.inverse_stenosis import InverseStenosisConfig, run_inverse_stenosis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover stenosis location and severity from sparse synthetic measurements.")
    parser.add_argument("--true-severity", type=float, default=0.55)
    parser.add_argument("--true-center", type=float, default=0.5)
    parser.add_argument("--width", type=float, default=0.08)
    parser.add_argument("--flow-rate", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--viscosity", type=float, default=0.04)
    parser.add_argument("--density", type=float, default=1.06)
    parser.add_argument("--sensors", type=int, default=20)
    parser.add_argument("--noise-level", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--severity-error-threshold", type=float, default=0.03)
    parser.add_argument("--center-error-threshold", type=float, default=0.02)
    parser.add_argument("--pressure-drop-error-threshold", type=float, default=0.08)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "inverse_stenosis",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = InverseStenosisConfig(
        true_severity=args.true_severity,
        true_center=args.true_center,
        width=args.width,
        flow_rate=args.flow_rate,
        radius=args.radius,
        viscosity=args.viscosity,
        density=args.density,
        sensors=args.sensors,
        noise_level=args.noise_level,
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
    )
    metrics = run_inverse_stenosis(cfg, args.output_dir)
    print(json.dumps(metrics, indent=2))

    checks = {
        "severity_error": metrics["severity_abs_error"] < args.severity_error_threshold,
        "center_error": metrics["center_abs_error"] < args.center_error_threshold,
        "pressure_drop_error": metrics["total_pressure_drop_relative_error"] < args.pressure_drop_error_threshold,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(f"\nInverse stenosis verification failed: {', '.join(failed)}")
        return 1

    print("\nInverse stenosis verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
