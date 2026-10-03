#!/usr/bin/env python3
"""24-hour BMA inference for one location/variable.

The module is intentionally small so the backend/orchestrator can import it.
It accepts deterministic AIFS and GFS F24 point forecasts and returns a
2-dimensional JSON-friendly array:

    [[value, probability_mass], ...]

The probability column sums to 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bma_core import FORECAST_HOUR, LOCATIONS, distribution_2d, load_models


def fuse_distribution(
    *,
    variable: str,
    location: str,
    aifs_value: float,
    gfs_value: float,
    model_file: str | Path,
    bins: int = 80,
) -> list[list[float]]:
    models = load_models(model_file)
    if variable not in models:
        raise KeyError(f"Variable {variable!r} is not present in {model_file}")
    if location not in models[variable]:
        raise KeyError(f"Location {location!r} is not present in {model_file}")
    model = models[variable][location]
    if model.forecast_hour != FORECAST_HOUR:
        raise ValueError("Only forecast hour 24 is supported.")
    return distribution_2d(model, float(aifs_value), float(gfs_value), bins=bins)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fuse AIFS + GFS into a 24-hour BMA probability distribution.")
    p.add_argument("--variable", choices=("temperature", "precipitation"), required=True)
    p.add_argument("--location", choices=tuple(LOCATIONS), required=True)
    p.add_argument("--aifs", type=float, required=True, help="AIFS forecast value at F24.")
    p.add_argument("--gfs", type=float, required=True, help="GFS forecast value at F24.")
    p.add_argument("--model-file", default="./models/bma_models_2025.json")
    p.add_argument("--bins", type=int, default=80)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        distribution = fuse_distribution(
            variable=args.variable,
            location=args.location,
            aifs_value=args.aifs,
            gfs_value=args.gfs,
            model_file=args.model_file,
            bins=args.bins,
        )
        payload = {
            "forecast_hour": FORECAST_HOUR,
            "variable": args.variable,
            "location": args.location,
            "distribution": distribution,
        }
        print(json.dumps(payload, separators=(",", ":")))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
