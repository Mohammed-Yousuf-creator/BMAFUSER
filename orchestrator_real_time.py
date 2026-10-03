#!/usr/bin/env python3
"""Real-time ClimaFuse orchestrator.

Drop-in replacement for the existing local-file-only orchestrator.

On a live forecast request it:
1. determines a recent UTC window containing published 00/06/12/18Z cycles;
2. calls the existing ECMWF AIFS fetcher;
3. calls the existing NOAA GFS fetcher;
4. finds the latest common F24 valid time;
5. extracts the 10 fixed city values from each model;
6. applies the already-trained BMA model;
7. returns the same city -> [[value, probability_mass], ...] contract.

The model is never retrained at runtime.

Default fetchers:
  ./ecmwf_aifs_fetcher.py
  ./noaa_gfs_fetcher.py

Default live cache:
  ./data/realtime/aifs
  ./data/realtime/gfs

For speed, runtime inference extracts the 10 city values directly from the
newly fetched model grids using nearest native grid cells. Full-domain xESMF
regridding is intentionally not performed on every request.

Examples:
  python orchestrator_realtime.py --variable temperature
  python orchestrator_realtime.py --variable precipitation
  python orchestrator_realtime.py --variable temperature --metadata
  python orchestrator_realtime.py --variable temperature --offline
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from bma_core import FORECAST_HOUR, LOCATIONS, VARIABLES, load_models
from bma_fusion import fuse_distribution

DEFAULT_MODEL = "./models/bma_models_2025.json"
DEFAULT_AIFS_FETCHER = "./ecmwf_aifs_fetcher.py"
DEFAULT_GFS_FETCHER = "./noaa_gfs_fetcher.py"
DEFAULT_AIFS_CACHE = "./data/realtime/aifs"
DEFAULT_GFS_CACHE = "./data/realtime/gfs"
CYCLES = (0, 6, 12, 18)


def _load_module(path: Path, name: str):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Fetcher not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _coord_name(ds: xr.Dataset, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
        if name in ds.variables and ds[name].ndim >= 1:
            return name
    raise ValueError(f"Missing coordinates {tuple(candidates)}")


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
    raise ValueError(f"Could not find {variable}; variables={list(ds.data_vars)}")


def _lon_for_grid(lon: float, ds: xr.Dataset, lon_name: str) -> float:
    values = np.asarray(ds[lon_name].values, dtype=float)
    if values.size and np.nanmin(values) >= 0 and np.nanmax(values) > 180 and lon < 0:
        return lon % 360
    return lon


def _extract_f24(path: Path, variable: str):
    with xr.open_dataset(path) as ds:
        if "forecast_hour" not in ds.coords and "forecast_hour" not in ds.dims:
            raise ValueError("Missing forecast_hour")
        try:
            selected = ds.sel(forecast_hour=FORECAST_HOUR)
        except Exception as exc:
            raise ValueError("Dataset does not contain forecast_hour=24") from exc

        if "valid_time" in selected.coords:
            vt = pd.Timestamp(np.asarray(selected["valid_time"].values).reshape(-1)[0])
        elif "init_time" in selected.coords:
            vt = pd.Timestamp(np.asarray(selected["init_time"].values).reshape(-1)[0]) + pd.Timedelta(hours=24)
        else:
            raise ValueError("Missing valid_time/init_time")

        da = selected[_data_var(selected, variable)]
        lat_name = _coord_name(selected, ("latitude", "lat"))
        lon_name = _coord_name(selected, ("longitude", "lon"))

        values = {}
        for location, (lat, lon) in LOCATIONS.items():
            target_lon = _lon_for_grid(lon, selected, lon_name)
            value = float(da.sel({lat_name: lat, lon_name: target_lon}, method="nearest").load().values)
            if not np.isfinite(value):
                raise ValueError(f"Non-finite {variable} value for {location}")
            values[location] = value
        return vt.tz_localize(None), values


def _files(root: Path, variable: str):
    if not root.exists():
        return []
    return sorted(root.rglob(f"*_{variable}.nc"))


def latest_common(aifs_dir: Path, gfs_dir: Path, variable: str):
    aifs = {}
    gfs = {}
    for path in _files(aifs_dir, variable):
        try:
            vt, vals = _extract_f24(path, variable)
            aifs[vt] = (path, vals)
        except Exception as exc:
            print(f"WARNING: skipping AIFS {path}: {exc}", file=sys.stderr)
    for path in _files(gfs_dir, variable):
        try:
            vt, vals = _extract_f24(path, variable)
            gfs[vt] = (path, vals)
        except Exception as exc:
            print(f"WARNING: skipping GFS {path}: {exc}", file=sys.stderr)

    common = sorted(set(aifs) & set(gfs))
    if not common:
        raise RuntimeError("No common AIFS/GFS F24 valid time is available.")
    vt = common[-1]
    return vt, aifs[vt][1], gfs[vt][1], aifs[vt][0], gfs[vt][0]


def _fetch_live(
    variable: str,
    aifs_fetcher: Path,
    gfs_fetcher: Path,
    aifs_dir: Path,
    gfs_dir: Path,
    lookback_hours: int,
    publication_lag_hours: float,
):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    end = now - timedelta(hours=publication_lag_hours)
    start = end - timedelta(hours=lookback_hours)

    aifs_mod = _load_module(aifs_fetcher, "climafuse_aifs_rt")
    gfs_mod = _load_module(gfs_fetcher, "climafuse_gfs_rt")

    if variable == "temperature":
        aifs_fxx = (24,)
        gfs_fxx = (24,)
    elif variable == "precipitation":
        # AIFS precipitation is cumulative upstream; the supplied fetcher
        # converts the F18/F24 support fields into a 6-hour F24 total.
        aifs_fxx = (18, 24)
        gfs_fxx = (24,)
    else:
        raise ValueError(variable)

    print(
        f"Fetching candidate cycles from {start:%Y-%m-%d %H:%M} to "
        f"{end:%Y-%m-%d %H:%M} UTC",
        file=sys.stderr,
    )

    # The uploaded AIFS fetcher already includes transient-error retries,
    # mirror switching, throttling cooldown and request pacing.
    aifs_mod.fetch_aifs(
        start,
        end,
        variable,
        cycles=CYCLES,
        fxx=aifs_fxx,
        priority=["azure", "aws", "ecmwf", "google"],
        save_dir=aifs_dir,
        overwrite=False,
        fail_fast=False,
        retries=4,
        pause=3.0,
        cooldown_min=10.0,
        max_cooldowns=3,
        max_consecutive_failures=3,
        debug=False,
    )

    # Keep the existing working GFS fetcher unchanged.
    gfs_mod.fetch_gfs(
        start,
        end,
        variable,
        cycles=CYCLES,
        fxx=gfs_fxx,
        priority=["aws", "google", "nomads", "azure"],
        save_dir=gfs_dir,
        overwrite=False,
        fail_fast=False,
    )


def forecast_with_metadata(
    *,
    variable: str,
    model_file: str | Path = DEFAULT_MODEL,
    aifs_dir: str | Path = DEFAULT_AIFS_CACHE,
    gfs_dir: str | Path = DEFAULT_GFS_CACHE,
    aifs_fetcher: str | Path = DEFAULT_AIFS_FETCHER,
    gfs_fetcher: str | Path = DEFAULT_GFS_FETCHER,
    bins: int = 80,
    live: bool = True,
    lookback_hours: int = 36,
    publication_lag_hours: float = 3.0,
):
    if variable not in VARIABLES:
        raise ValueError(f"Supported variables: {VARIABLES}")

    model_path = Path(model_file).expanduser().resolve()
    aifs_path = Path(aifs_dir).expanduser().resolve()
    gfs_path = Path(gfs_dir).expanduser().resolve()

    models = load_models(model_path)
    if variable not in models:
        raise ValueError(f"No trained model for {variable} in {model_path}")
    missing = sorted(set(LOCATIONS) - set(models[variable]))
    if missing:
        raise ValueError(f"Missing locations in model artifact: {missing}")

    aifs_path.mkdir(parents=True, exist_ok=True)
    gfs_path.mkdir(parents=True, exist_ok=True)

    if live:
        _fetch_live(
            variable,
            Path(aifs_fetcher),
            Path(gfs_fetcher),
            aifs_path,
            gfs_path,
            lookback_hours,
            publication_lag_hours,
        )

    valid_time, aifs_vals, gfs_vals, aifs_file, gfs_file = latest_common(
        aifs_path, gfs_path, variable
    )

    locations = {}
    for location in LOCATIONS:
        locations[location] = fuse_distribution(
            variable=variable,
            location=location,
            aifs_value=aifs_vals[location],
            gfs_value=gfs_vals[location],
            model_file=model_path,
            bins=bins,
        )

    return {
        "forecast_hour": FORECAST_HOUR,
        "valid_time": valid_time.isoformat(),
        "variable": variable,
        "source": {
            "aifs": "ECMWF AIFS",
            "gfs": "NOAA GFS",
            "aifs_file": str(aifs_file),
            "gfs_file": str(gfs_file),
        },
        "locations": locations,
    }


def forecast(
    *,
    variable: str,
    model_file: str | Path = DEFAULT_MODEL,
    aifs_dir: str | Path = DEFAULT_AIFS_CACHE,
    gfs_dir: str | Path = DEFAULT_GFS_CACHE,
    bins: int = 80,
):
    # Same return shape as the old orchestrator.
    return forecast_with_metadata(
        variable=variable,
        model_file=model_file,
        aifs_dir=aifs_dir,
        gfs_dir=gfs_dir,
        bins=bins,
        live=True,
    )["locations"]


def main():
    p = argparse.ArgumentParser(description="Real-time ClimaFuse orchestrator")
    p.add_argument("--variable", choices=VARIABLES, required=True)
    p.add_argument("--model-file", default=DEFAULT_MODEL)
    p.add_argument("--aifs-dir", default=DEFAULT_AIFS_CACHE)
    p.add_argument("--gfs-dir", default=DEFAULT_GFS_CACHE)
    p.add_argument("--aifs-fetcher", default=DEFAULT_AIFS_FETCHER)
    p.add_argument("--gfs-fetcher", default=DEFAULT_GFS_FETCHER)
    p.add_argument("--bins", type=int, default=80)
    p.add_argument("--lookback-hours", type=int, default=36)
    p.add_argument("--publication-lag-hours", type=float, default=3.0)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--metadata", action="store_true")
    args = p.parse_args()

    try:
        payload = forecast_with_metadata(
            variable=args.variable,
            model_file=args.model_file,
            aifs_dir=args.aifs_dir,
            gfs_dir=args.gfs_dir,
            aifs_fetcher=args.aifs_fetcher,
            gfs_fetcher=args.gfs_fetcher,
            bins=args.bins,
            live=not args.offline,
            lookback_hours=args.lookback_hours,
            publication_lag_hours=args.publication_lag_hours,
        )
        if not args.metadata:
            payload = payload["locations"]
        print(json.dumps(payload, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
