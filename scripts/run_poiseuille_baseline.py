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

from pinn_fluid.poiseuille import PoiseuilleConfig, run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 2D Poiseuille PINN baseline.")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "poiseuille_baseline",
    )
    parser.add_argument("--u-threshold", type=float, default=0.08)
    parser.add_argument("--p-threshold", type=float, default=0.12)
    parser.add_argument("--continuity-threshold", type=float, default=0.08)
    parser.add_argument("--momentum-threshold", type=float, default=0.08)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = PoiseuilleConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
    )
    metrics = run_baseline(cfg, args.output_dir, device=args.device)

    print("\nVerification metrics:")
    print(json.dumps(metrics, indent=2))

    checks = {
        "relative_l2_u": metrics["relative_l2_u"] < args.u_threshold,
        "relative_l2_p": metrics["relative_l2_p"] < args.p_threshold,
        "mean_abs_continuity": metrics["mean_abs_continuity"] < args.continuity_threshold,
        "mean_abs_momentum_x": metrics["mean_abs_momentum_x"] < args.momentum_threshold,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(f"\nBaseline verification failed: {', '.join(failed)}")
        return 1

    print("\nBaseline verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
