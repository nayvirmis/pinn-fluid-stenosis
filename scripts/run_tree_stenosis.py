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

from pinn_fluid.tree_stenosis import TreeStenosisConfig, run_tree_stenosis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed Y-bifurcation stenosis forward model.")
    parser.add_argument("--stenosed-branch", choices=["left_outlet", "right_outlet"], default="left_outlet")
    parser.add_argument("--severity", type=float, default=0.55)
    parser.add_argument("--flow-rate", type=float, default=1.0)
    parser.add_argument("--points-per-branch", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "tree_stenosis")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = TreeStenosisConfig(
        stenosed_branch=args.stenosed_branch,
        severity=args.severity,
        flow_rate=args.flow_rate,
        points_per_branch=args.points_per_branch,
    )
    metrics = run_tree_stenosis(cfg, args.output_dir)
    print(json.dumps(metrics, indent=2))

    checks = {
        "stenosed_branch_accelerates_flow": metrics["stenosed_branch_accelerates_flow"],
        "stenosed_branch_increases_drop": metrics["stenosed_branch_increases_drop"],
        "all_pressures_monotone": metrics["all_pressures_monotone"],
        "flow_conservation": metrics["flow_conservation_error"] < 1e-10,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(f"\nTree stenosis verification failed: {', '.join(failed)}")
        return 1

    print("\nTree stenosis verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
