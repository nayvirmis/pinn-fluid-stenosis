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

from pinn_fluid.inverse_tree_stenosis import InverseTreeStenosisConfig, run_inverse_tree_stenosis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover stenosed outlet branch and severity in a Y-bifurcation tree.")
    parser.add_argument("--true-branch", choices=["left_outlet", "right_outlet"], default="left_outlet")
    parser.add_argument("--true-severity", type=float, default=0.55)
    parser.add_argument("--sensors-per-outlet", type=int, default=12)
    parser.add_argument("--noise-level", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "inverse_tree_stenosis")
    parser.add_argument("--severity-error-threshold", type=float, default=0.03)
    parser.add_argument("--pressure-drop-error-threshold", type=float, default=0.08)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = InverseTreeStenosisConfig(
        true_branch=args.true_branch,
        true_severity=args.true_severity,
        sensors_per_outlet=args.sensors_per_outlet,
        noise_level=args.noise_level,
        epochs=args.epochs,
        learning_rate=args.lr,
    )
    metrics = run_inverse_tree_stenosis(cfg, args.output_dir)
    print(json.dumps(metrics, indent=2))

    checks = {
        "branch_recovered_correctly": metrics["branch_recovered_correctly"],
        "severity_error": metrics["severity_abs_error"] < args.severity_error_threshold,
        "pressure_drop_error": metrics["true_branch_pressure_drop_relative_error"] < args.pressure_drop_error_threshold,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(f"\nInverse tree stenosis verification failed: {', '.join(failed)}")
        return 1

    print("\nInverse tree stenosis verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
