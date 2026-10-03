#!/usr/bin/env python3
"""Train 24-hour BMA models for the ClimaFuse prototype.

Inputs
------
AIFS and GFS NetCDF files produced by the fetchers (or the xESMF-remapped
copies), plus IMDLIB NetCDF observations for 2025.

Only forecast_hour=24 is used. No PyMC is used: SciPy maximum likelihood
optimization estimates the two calibrated Gaussian component kernels and the
BMA mixture weight.

Temperature target
------------------
IMDLIB archive has daily Tmin/Tmax, not an hourly temperature observation.
For this prototype the verification value is the daily midpoint:
    tmean_proxy = (tmin + tmax) / 2
for the valid_time's calendar day.

Precipitation target
--------------------
The current fetchers' ``precipitation`` NetCDF at F24 represents the available
precipitation accumulation field ending at forecast hour 24. This script uses
that F24 predictor directly and pairs it with IMDLIB daily rain on the valid
date as a prototype verification target. For a research-grade setup, use an
observation accumulation window exactly matching the forecast accumulation
window.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from bma_core import FORECAST_HOUR, LOCATIONS, VARIABLES, fit_bma, save_models


@dataclass
class ForecastRecord:
    valid_time: pd.Timestamp
    location: str
    value: float


def _coord_name(ds: xr.Dataset, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
    raise ValueError(f"Could not find coordinate among {tuple(candidates)}. Coordinates: {list(ds.coords)}")


def _data_var(ds: xr.Dataset, preferred: str) -> str:
    if preferred in ds.data_vars:
        return preferred
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    aliases = {
        "temperature": ("t2m", "temperature"),
        "precipitation": ("tp", "apcp", "precipitation"),
    }
    for name in aliases.get(preferred, ()):
        if name in ds.data_vars:
            return name
    raise ValueError(f"Could not identify {preferred}; data variables={list(ds.data_vars)}")


def _fxx24_slice(ds: xr.Dataset, variable: str) -> tuple[xr.Dataset, pd.Timestamp]:
    if "forecast_hour" not in ds.coords and "forecast_hour" not in ds.dims:
        raise ValueError("Dataset does not contain forecast_hour coordinate/dimension.")

    # Fetchers may write forecast_hour as either a dimension (e.g. [24]) or a
    # scalar coordinate (e.g. 24). Support both representations.
    if "forecast_hour" in ds.coords and ds["forecast_hour"].ndim == 0:
        try:
            fxx_value = float(ds["forecast_hour"].values)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid scalar forecast_hour coordinate.") from exc
        if not np.isclose(fxx_value, FORECAST_HOUR):
            raise ValueError("Dataset does not contain forecast_hour=24.")
        selected = ds
    else:
        try:
            selected = ds.sel(forecast_hour=FORECAST_HOUR)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Dataset does not contain forecast_hour=24.") from exc

    if "valid_time" in selected.coords:
        value = selected["valid_time"].values
        ts = pd.Timestamp(np.asarray(value).reshape(-1)[0])
    elif "init_time" in selected.coords:
        ts = pd.Timestamp(np.asarray(selected["init_time"].values).reshape(-1)[0]) + pd.Timedelta(hours=FORECAST_HOUR)
    else:
        raise ValueError("Dataset lacks both valid_time and init_time.")

    name = _data_var(selected, variable)
    if name != variable:
        selected = selected.rename({name: variable})
    return selected, ts


def _normalize_longitude_for_dataset(lon: float, ds: xr.Dataset, lon_name: str) -> float:
    values = np.asarray(ds[lon_name].values, dtype=float)
    if values.size == 0:
        return lon
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if lo >= 0.0 and hi > 180.0 and lon < 0.0:
        return lon % 360.0
    return lon


def load_forecast_records(root: Path, variable: str, model_name: str) -> list[ForecastRecord]:
    if not root.exists():
        raise FileNotFoundError(root)
    files = sorted(root.rglob(f"*_{variable}.nc"))
    if not files:
        raise FileNotFoundError(f"No '*_{variable}.nc' files under {root}")

    records: list[ForecastRecord] = []
    for path in files:
        try:
            with xr.open_dataset(path) as ds:
                ds24, valid_time = _fxx24_slice(ds, variable)
                lat_name = _coord_name(ds24, ("latitude", "lat"))
                lon_name = _coord_name(ds24, ("longitude", "lon"))
                da = ds24[variable]
                for location, (lat, lon) in LOCATIONS.items():
                    target_lon = _normalize_longitude_for_dataset(lon, ds24, lon_name)
                    value = float(da.sel({lat_name: lat, lon_name: target_lon}, method="nearest").load().values)
                    if np.isfinite(value):
                        records.append(ForecastRecord(valid_time, location, value))
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: skipping {model_name} file {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if not records:
        raise RuntimeError(f"No usable {model_name} F24 records found under {root}")
    return records


def records_to_frame(records: list[ForecastRecord], column: str) -> pd.DataFrame:
    return pd.DataFrame({
        "valid_time": [r.valid_time for r in records],
        "location": [r.location for r in records],
        column: [r.value for r in records],
    })


def _load_imd_dataset(path: Path, expected_var: str) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(path)
    ds = xr.open_dataset(path)
    if expected_var not in ds.data_vars:
        if len(ds.data_vars) == 1:
            ds = ds.rename({next(iter(ds.data_vars)): expected_var})
        else:
            ds.close()
            raise ValueError(f"Expected {expected_var} in {path}; found {list(ds.data_vars)}")
    return ds


def _select_imd_spatial_cell(
    da: xr.DataArray,
    location: str,
    dates: list[pd.Timestamp],
    *,
    min_coverage: float = 0.50,
) -> tuple[str, str, float, float, float]:
    """Select one fixed, nearby IMD grid cell with usable temporal coverage.

    IMDLIB historical rainfall is land-masked. A city can therefore map to a
    nearest grid cell that is entirely missing (this occurs for Chennai). We
    first look for the nearest grid cell with at least ``min_coverage`` valid
    observations over the requested dates. If none exists, we fall back to the
    cell with the highest valid coverage, breaking ties by distance.
    """
    lat_name = _coord_name(da.to_dataset(name=da.name or "obs"), ("lat", "latitude"))
    lon_name = _coord_name(da.to_dataset(name=da.name or "obs"), ("lon", "longitude"))
    target_lat, target_lon = LOCATIONS[location]
    target_lon = _normalize_longitude_for_dataset(float(target_lon), da.to_dataset(name="obs"), lon_name)

    available_time = pd.DatetimeIndex(pd.to_datetime(da["time"].values)).normalize()
    requested = sorted({pd.Timestamp(d).normalize() for d in dates})
    requested_set = set(requested)
    time_mask = np.array([t in requested_set for t in available_time], dtype=bool)
    if not time_mask.any():
        raise RuntimeError(f"No IMD observation dates available for {location}/{da.name}.")

    time_indices = np.flatnonzero(time_mask)
    sampled = da.isel(time=time_indices).load()
    valid = np.isfinite(sampled) & (sampled > -900.0)
    if da.name == "rain":
        valid &= sampled >= 0.0
    coverage = valid.mean(dim="time")

    lat_values = np.asarray(da[lat_name].values, dtype=float)
    lon_values = np.asarray(da[lon_name].values, dtype=float)
    lat2, lon2 = np.meshgrid(lat_values, lon_values, indexing="ij")
    distance2 = (lat2 - float(target_lat)) ** 2 + (lon2 - float(target_lon)) ** 2
    cov_values = np.asarray(coverage.values, dtype=float)

    eligible = np.isfinite(cov_values) & (cov_values >= min_coverage)
    if eligible.any():
        score = np.where(eligible, distance2, np.inf)
        idx = np.unravel_index(np.argmin(score), score.shape)
    else:
        # No nearby cell has the requested coverage; use the most complete cell,
        # with distance as the tie-breaker.
        best_cov = np.nanmax(np.where(np.isfinite(cov_values), cov_values, np.nan))
        candidates = np.isfinite(cov_values) & np.isclose(cov_values, best_cov, rtol=0.0, atol=1e-12)
        score = np.where(candidates, distance2, np.inf)
        idx = np.unravel_index(np.argmin(score), score.shape)

    selected_lat = float(lat_values[idx[0]])
    selected_lon = float(lon_values[idx[1]])
    selected_coverage = float(cov_values[idx])
    return lat_name, lon_name, selected_lat, selected_lon, selected_coverage


def _extract_imd_series(
    da: xr.DataArray,
    location: str,
    dates: list[pd.Timestamp],
    *,
    min_coverage: float = 0.50,
) -> dict[pd.Timestamp, float]:
    lat_name, lon_name, selected_lat, selected_lon, coverage = _select_imd_spatial_cell(
        da, location, dates, min_coverage=min_coverage
    )
    series = da.sel({lat_name: selected_lat, lon_name: selected_lon}, method="nearest").load()
    output: dict[pd.Timestamp, float] = {}
    for date in sorted({pd.Timestamp(d).normalize() for d in dates}):
        matches = np.flatnonzero(pd.DatetimeIndex(pd.to_datetime(series["time"].values)).normalize() == date)
        if len(matches) != 1:
            continue
        value = float(np.asarray(series.isel(time=int(matches[0])).values))
        if np.isfinite(value) and value > -900.0 and (series.name != "rain" or value >= 0.0):
            output[date] = value
    print(
        f"IMD {location}/{series.name}: selected grid ({selected_lat:.2f}, {selected_lon:.2f}) "
        f"with {coverage:.1%} valid coverage.",
        file=sys.stderr,
    )
    return output


def load_observations(imd_dir: Path, variable: str, dates: list[pd.Timestamp]) -> dict[tuple[str, pd.Timestamp], float]:
    if variable == "precipitation":
        path = imd_dir / "rain" / "imd_rain_2025_2025.nc"
        if not path.exists():
            matches = sorted((imd_dir / "rain").glob("*.nc"))
            if not matches:
                raise FileNotFoundError(f"No rainfall NetCDF under {imd_dir / 'rain'}")
            path = matches[0]
        ds = _load_imd_dataset(path, "rain")
        try:
            da = ds["rain"]
            output: dict[tuple[str, pd.Timestamp], float] = {}
            for location in LOCATIONS:
                series = _extract_imd_series(da, location, dates)
                for date, value in series.items():
                    output[(location, date)] = value
            return output
        finally:
            ds.close()

    tmin_path = imd_dir / "tmin" / "imd_tmin_2025_2025.nc"
    tmax_path = imd_dir / "tmax" / "imd_tmax_2025_2025.nc"
    if not tmin_path.exists():
        matches = sorted((imd_dir / "tmin").glob("*.nc"))
        if not matches:
            raise FileNotFoundError(f"No tmin NetCDF under {imd_dir / 'tmin'}")
        tmin_path = matches[0]
    if not tmax_path.exists():
        matches = sorted((imd_dir / "tmax").glob("*.nc"))
        if not matches:
            raise FileNotFoundError(f"No tmax NetCDF under {imd_dir / 'tmax'}")
        tmax_path = matches[0]

    tmin_ds = _load_imd_dataset(tmin_path, "tmin")
    tmax_ds = _load_imd_dataset(tmax_path, "tmax")
    try:
        tmin_da = tmin_ds["tmin"]
        tmax_da = tmax_ds["tmax"]
        output = {}
        tmin_series_by_location = {}
        tmax_series_by_location = {}
        for location in LOCATIONS:
            tmin_series_by_location[location] = _extract_imd_series(tmin_da, location, dates)
            tmax_series_by_location[location] = _extract_imd_series(tmax_da, location, dates)

        for location in LOCATIONS:
            common_dates = set(tmin_series_by_location[location]) & set(tmax_series_by_location[location])
            for date in common_dates:
                output[(location, date)] = 0.5 * (
                    tmin_series_by_location[location][date] + tmax_series_by_location[location][date]
                )
        return output
    finally:
        tmin_ds.close()
        tmax_ds.close()


def train_variable(
    aifs_dir: Path,
    gfs_dir: Path,
    imd_dir: Path,
    variable: str,
    *,
    training_start: str,
    training_end: str,
) -> dict[str, object]:
    aifs = records_to_frame(load_forecast_records(aifs_dir, variable, "AIFS"), "aifs")
    gfs = records_to_frame(load_forecast_records(gfs_dir, variable, "GFS"), "gfs")
    aifs["valid_time"] = pd.to_datetime(aifs["valid_time"]).dt.tz_localize(None)
    gfs["valid_time"] = pd.to_datetime(gfs["valid_time"]).dt.tz_localize(None)
    paired = aifs.merge(gfs, on=["location", "valid_time"], how="inner")
    paired = paired[paired["valid_time"].dt.year == 2025].copy()
    if paired.empty:
        raise RuntimeError(f"No overlapping AIFS/GFS F24 samples for {variable}.")

    # ``inner`` is intentional: an AIFS/GFS case is usable only when both model
    # predictors exist. Missing GFS cases for a location are therefore skipped,
    # rather than causing the entire location/model to fail.
    obs = load_observations(imd_dir, variable, paired["valid_time"].tolist())
    paired["observed"] = [obs.get((row.location, row.valid_time.normalize()), np.nan) for row in paired.itertuples()]
    paired = paired.dropna(subset=["aifs", "gfs", "observed"])

    models = {}
    report = {}
    for location in LOCATIONS:
        rows = paired[paired["location"] == location]
        if rows.empty:
            raise RuntimeError(f"No paired samples for {location}/{variable}.")
        model = fit_bma(
            rows["aifs"].to_numpy(),
            rows["gfs"].to_numpy(),
            rows["observed"].to_numpy(),
            variable=variable,
            location=location,
            forecast_hour=FORECAST_HOUR,
            training_start=training_start,
            training_end=training_end,
        )
        models[location] = model
        report[location] = {
            "samples": model.n_samples,
            "weight_aifs": model.weight_aifs,
            "weight_gfs": model.weight_gfs,
            "aifs_sigma": model.aifs_sigma,
            "gfs_sigma": model.gfs_sigma,
        }
    return {"models": models, "report": report}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ClimaFuse BMA models using only forecast hour 24 and 2025 data.")
    p.add_argument("--aifs-dir", default="./data/regridded/aifs_2025")
    p.add_argument("--gfs-dir", default="./data/regridded/gfs_2025")
    p.add_argument("--imd-dir", default="./data/imd_2025")
    p.add_argument("--variable", choices=VARIABLES + ("all",), default="all")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--output", default="./models/bma_models_2025.json")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.start != "2025-01-01" or args.end != "2025-12-31":
        print("ERROR: This prototype trainer is intentionally fixed to all of 2025.", file=sys.stderr)
        return 2

    variables = VARIABLES if args.variable == "all" else (args.variable,)
    all_models = {}
    reports = {}
    try:
        for variable in variables:
            print(f"Training {variable} using F24 only...", file=sys.stderr)
            trained = train_variable(
                Path(args.aifs_dir).expanduser().resolve(),
                Path(args.gfs_dir).expanduser().resolve(),
                Path(args.imd_dir).expanduser().resolve(),
                variable,
                training_start=args.start,
                training_end=args.end,
            )
            all_models[variable] = trained["models"]
            reports[variable] = trained["report"]

        save_models(
            all_models,
            args.output,
            metadata={
                "training_period": [args.start, args.end],
                "forecast_hour": FORECAST_HOUR,
                "locations": {k: list(v) for k, v in LOCATIONS.items()},
                "temperature_observation_proxy": "(IMD tmin + IMD tmax) / 2 for valid_time calendar day",
                "precipitation_observation": "IMD daily rain for valid_time calendar day",
                "method": "Two-component Gaussian-mixture BMA fit by SciPy maximum likelihood; precipitation uses log1p transform.",
                "report": reports,
            },
        )
        print(f"WROTE MODEL ARTIFACT: {Path(args.output).resolve()}", file=sys.stderr)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
