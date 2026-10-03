#!/usr/bin/env python3
"""ClimaFuse forecast orchestrator for the 10-location prototype.

This is NOT a FastAPI/backend implementation.  It is an importable Python
orchestration layer that a backend can call, or it can be executed as a CLI
that prints JSON to stdout.

Contract
--------
For each requested variable, ``forecast()`` returns exactly:

    {
      "New Delhi": [[value, probability], ...],
      "Mumbai": [[value, probability], ...],
      ...
    }

The output is React-friendly: every location maps directly to a 2D array.
The 2D rows are [x, probability_mass].

Only forecast hour 24 is supported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from bma_core import FORECAST_HOUR, LOCATIONS, VARIABLES, load_models
from bma_fusion import fuse_distribution


def _coord_name(ds: xr.Dataset, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
    raise ValueError(f"Missing coordinates {tuple(candidates)}; found {list(ds.coords)}")


def _data_var(ds: xr.Dataset, variable: str) -> str:
    if variable in ds.data_vars:
        return variable
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    aliases = {
        "temperature": ("t2m", "temperature"),
        "precipitation": ("tp", "apcp", "precipitation"),
    }
    for name in aliases[variable]:
        if name in ds.data_vars:
            return name
    raise ValueError(f"Could not find {variable} in dataset variables={list(ds.data_vars)}")


def _lon_for_grid(lon: float, ds: xr.Dataset, lon_name: str) -> float:
    values = np.asarray(ds[lon_name].values, dtype=float)
    if values.size and float(np.nanmin(values)) >= 0 and float(np.nanmax(values)) > 180 and lon < 0:
        return lon % 360
    return lon


def _extract_f24_values(path: Path, variable: str) -> tuple[pd.Timestamp, dict[str, float]]:
    with xr.open_dataset(path) as ds:
        if "forecast_hour" not in ds.coords and "forecast_hour" not in ds.dims:
            raise ValueError("Missing forecast_hour")
        selected = ds.sel(forecast_hour=FORECAST_HOUR)
        if "valid_time" in selected.coords:
            valid_time = pd.Timestamp(np.asarray(selected["valid_time"].values).reshape(-1)[0])
        elif "init_time" in selected.coords:
            valid_time = pd.Timestamp(selected["init_time"].values) + pd.Timedelta(hours=FORECAST_HOUR)
        else:
            raise ValueError("Missing valid_time/init_time")

        variable_name = _data_var(selected, variable)
        da = selected[variable_name]
        lat_name = _coord_name(selected, ("latitude", "lat"))
        lon_name = _coord_name(selected, ("longitude", "lon"))
        values = {}
        for location, (lat, lon) in LOCATIONS.items():
            target_lon = _lon_for_grid(lon, selected, lon_name)
            values[location] = float(da.sel({lat_name: lat, lon_name: target_lon}, method="nearest").load().values)
        return valid_time.tz_localize(None), values


def _find_files(root: Path, variable: str) -> list[Path]:
    files = sorted(root.rglob(f"*_{variable}.nc"))
    if not files:
        raise FileNotFoundError(f"No '*_{variable}.nc' files under {root}")
    return files


def _latest_common_forecast(
    aifs_dir: Path,
    gfs_dir: Path,
    variable: str,
    *,
    requested_valid_time: str | None = None,
) -> tuple[pd.Timestamp, dict[str, float], dict[str, float]]:
    aifs_map: dict[pd.Timestamp, tuple[Path, dict[str, float]]] = {}
    gfs_map: dict[pd.Timestamp, tuple[Path, dict[str, float]]] = {}

    for path in _find_files(aifs_dir, variable):
        try:
            vt, values = _extract_f24_values(path, variable)
            aifs_map[vt] = (path, values)
        except Exception:
            continue
    for path in _find_files(gfs_dir, variable):
        try:
            vt, values = _extract_f24_values(path, variable)
            gfs_map[vt] = (path, values)
        except Exception:
            continue

    common = sorted(set(aifs_map).intersection(gfs_map))
    if requested_valid_time:
        desired = pd.Timestamp(requested_valid_time).tz_localize(None)
        common = [vt for vt in common if vt == desired]
    if not common:
        raise RuntimeError("No common AIFS/GFS F24 valid time found.")
    vt = common[-1]
    return vt, aifs_map[vt][1], gfs_map[vt][1]


def forecast(
    *,
    variable: str,
    model_file: str | Path = "./models/bma_models_2025.json",
    aifs_dir: str | Path = "./data/regridded/aifs_2025",
    gfs_dir: str | Path = "./data/regridded/gfs_2025",
    valid_time: str | None = None,
    bins: int = 80,
) -> dict[str, list[list[float]]]:
    """Return the exact React-facing 10-location JSON shape for one variable."""
    if variable not in VARIABLES:
        raise ValueError(f"Supported variables: {VARIABLES}")
    models = load_models(model_file)
    if variable not in models:
        raise ValueError(f"No trained BMA models for {variable} in {model_file}")
    if set(models[variable]) != set(LOCATIONS):
        missing = sorted(set(LOCATIONS) - set(models[variable]))
        raise ValueError(f"Model artifact missing required locations: {missing}")

    _, aifs_values, gfs_values = _latest_common_forecast(
        Path(aifs_dir).expanduser().resolve(),
        Path(gfs_dir).expanduser().resolve(),
        variable,
        requested_valid_time=valid_time,
    )

    output: dict[str, list[list[float]]] = {}
    for location in LOCATIONS:
        output[location] = fuse_distribution(
            variable=variable,
            location=location,
            aifs_value=aifs_values[location],
            gfs_value=gfs_values[location],
            model_file=model_file,
            bins=bins,
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Orchestrate 24-hour ClimaFuse BMA forecasts for 10 fixed locations.")
    p.add_argument("--variable", choices=VARIABLES, required=True)
    p.add_argument("--model-file", default="./models/bma_models_2025.json")
    p.add_argument("--aifs-dir", default="./data/regridded/aifs_2025")
    p.add_argument("--gfs-dir", default="./data/regridded/gfs_2025")
    p.add_argument("--valid-time", help="Optional exact common valid time, e.g. 2026-09-30T12:00:00")
    p.add_argument("--bins", type=int, default=80)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = forecast(
            variable=args.variable,
            model_file=args.model_file,
            aifs_dir=args.aifs_dir,
            gfs_dir=args.gfs_dir,
            valid_time=args.valid_time,
            bins=args.bins,
        )
        # Stdout is JSON-only so a backend can invoke the CLI and parse it directly.
        print(json.dumps(payload, separators=(",", ":")))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
