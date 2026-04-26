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

from pinn_fluid.stenosis import SyntheticStenosisConfig, run_synthetic_stenosis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synthetic stenosed-vessel forward model.")
    parser.add_argument("--severity", type=float, default=0.55)
    parser.add_argument("--center", type=float, default=0.5)
    parser.add_argument("--width", type=float, default=0.08)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--flow-rate", type=float, default=1.0)
    parser.add_argument("--viscosity", type=float, default=0.04)
    parser.add_argument("--density", type=float, default=1.06)
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "synthetic_stenosis",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = SyntheticStenosisConfig(
        severity=args.severity,
        center=args.center,
        width=args.width,
        radius=args.radius,
        flow_rate=args.flow_rate,
        viscosity=args.viscosity,
        density=args.density,
        points=args.points,
    )
    metrics = run_synthetic_stenosis(cfg, args.output_dir)

    print(json.dumps(metrics, indent=2))

    checks = {
        "pressure_monotone_decreasing": metrics["pressure_monotone_decreasing"],
        "stenosis_accelerates_flow": metrics["stenosis_accelerates_flow"],
        "stenosis_increases_drop": metrics["stenosis_increases_drop"],
        "radius_reduction_fraction_positive": metrics["radius_reduction_fraction"] > 0.0,
        "lesion_drop_fraction_positive": metrics["lesion_drop_fraction_of_total"] > 0.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(f"\nSynthetic stenosis verification failed: {', '.join(failed)}")
        return 1

    print("\nSynthetic stenosis verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
