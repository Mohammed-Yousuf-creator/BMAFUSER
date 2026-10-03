#!/usr/bin/env python3
"""Core 2-model BMA implementation for the ClimaFuse prototype.

This module intentionally uses NumPy + SciPy only.  It implements a practical
prototype of Bayesian Model Averaging for two deterministic predictors:
ECMWF AIFS and NOAA GFS.

The model is fit independently for each variable/location using a Gaussian
mixture in an appropriate working space:

* temperature: Celsius space
* precipitation: log1p(mm) space

For component k:
    y | f_k ~ Normal(a_k + b_k f_k, sigma_k)

and
    p(y) = w_1 p_1(y) + w_2 p_2(y),    w_1 + w_2 = 1.

This is a prototype implementation, not a claim of statistical optimality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

FORECAST_HOUR = 24

LOCATIONS: dict[str, tuple[float, float]] = {
    "New Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Ahmedabad": (23.0225, 72.5714),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
}

VARIABLES = ("temperature", "precipitation")


@dataclass(frozen=True)
class BMAModel:
    variable: str
    location: str
    forecast_hour: int
    transform: str
    weight_aifs: float
    weight_gfs: float
    aifs_intercept: float
    aifs_slope: float
    aifs_sigma: float
    gfs_intercept: float
    gfs_slope: float
    gfs_sigma: float
    n_samples: int
    training_start: str
    training_end: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BMAModel":
        return cls(**dict(value))


def _working_transform(variable: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if variable == "temperature":
        return values
    if variable == "precipitation":
        return np.log1p(np.clip(values, 0.0, None))
    raise ValueError(f"Unsupported variable: {variable}")


def _inverse_transform(variable: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if variable == "temperature":
        return values
    if variable == "precipitation":
        return np.expm1(values)
    raise ValueError(f"Unsupported variable: {variable}")


def _initial_component(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    X = np.column_stack([np.ones_like(x), x])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        intercept, slope = float(beta[0]), float(beta[1])
    except np.linalg.LinAlgError:
        intercept, slope = float(np.mean(y)), 1.0
    residual = y - (intercept + slope * x)
    sigma = float(np.std(residual))
    if not np.isfinite(sigma) or sigma < 0.05:
        sigma = max(float(np.std(y)), 0.5)
    return intercept, slope, sigma


def _fit_objective(theta: np.ndarray, x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> float:
    a1, b1, log_s1, a2, b2, log_s2, logit_w1 = theta
    s1 = np.exp(log_s1)
    s2 = np.exp(log_s2)
    w1 = float(expit(logit_w1))
    w2 = 1.0 - w1

    mu1 = a1 + b1 * x1
    mu2 = a2 + b2 * x2
    lp1 = np.log(w1) + norm.logpdf(y, loc=mu1, scale=s1)
    lp2 = np.log(w2) + norm.logpdf(y, loc=mu2, scale=s2)
    log_mix = np.logaddexp(lp1, lp2)
    if not np.all(np.isfinite(log_mix)):
        return 1e100
    return float(-np.sum(log_mix))


def fit_bma(
    aifs: np.ndarray,
    gfs: np.ndarray,
    observed: np.ndarray,
    *,
    variable: str,
    training_start: str,
    training_end: str,
    location: str,
    forecast_hour: int = FORECAST_HOUR,
    n_starts: int = 8,
    seed: int = 42,
) -> BMAModel:
    if forecast_hour != FORECAST_HOUR:
        raise ValueError("This prototype is hard-coded to forecast hour 24 only.")
    if variable not in VARIABLES:
        raise ValueError(variable)

    aifs = np.asarray(aifs, dtype=float)
    gfs = np.asarray(gfs, dtype=float)
    observed = np.asarray(observed, dtype=float)
    mask = np.isfinite(aifs) & np.isfinite(gfs) & np.isfinite(observed)
    if variable == "precipitation":
        mask &= aifs >= 0.0
        mask &= gfs >= 0.0
        mask &= observed >= 0.0
    aifs, gfs, observed = aifs[mask], gfs[mask], observed[mask]
    if len(observed) < 30:
        raise ValueError(f"Only {len(observed)} usable samples for {location}/{variable}; at least 30 are required.")

    x1 = _working_transform(variable, aifs)
    x2 = _working_transform(variable, gfs)
    y = _working_transform(variable, observed)
    i1 = _initial_component(x1, y)
    i2 = _initial_component(x2, y)

    rng = np.random.default_rng(seed)
    bounds = [
        (-20.0, 20.0),
        (0.0, 3.0),
        (np.log(0.03), np.log(30.0)),
        (-20.0, 20.0),
        (0.0, 3.0),
        (np.log(0.03), np.log(30.0)),
        (-8.0, 8.0),
    ]
    if variable == "precipitation":
        bounds[0] = (-5.0, 5.0)
        bounds[1] = (0.0, 3.0)
        bounds[3] = (-5.0, 5.0)
        bounds[4] = (0.0, 3.0)
        bounds[2] = (np.log(0.03), np.log(4.0))
        bounds[5] = (np.log(0.03), np.log(4.0))

    base = np.array([
        i1[0], np.clip(i1[1], 0.0, 3.0), np.log(i1[2]),
        i2[0], np.clip(i2[1], 0.0, 3.0), np.log(i2[2]), 0.0,
    ], dtype=float)

    starts = [base]
    for _ in range(max(0, n_starts - 1)):
        trial = base.copy()
        trial[0] += rng.normal(0.0, 0.5)
        trial[1] = np.clip(trial[1] + rng.normal(0.0, 0.15), 0.0, 3.0)
        trial[3] += rng.normal(0.0, 0.5)
        trial[4] = np.clip(trial[4] + rng.normal(0.0, 0.15), 0.0, 3.0)
        trial[6] = rng.normal(0.0, 1.5)
        starts.append(trial)

    best = None
    for start in starts:
        result = minimize(
            _fit_objective,
            start,
            args=(x1, x2, y),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-7},
        )
        if not result.success and not np.isfinite(result.fun):
            continue
        if best is None or result.fun < best.fun:
            best = result

    if best is None or not np.isfinite(best.fun):
        raise RuntimeError(f"BMA optimizer failed for {location}/{variable}.")

    a1, b1, log_s1, a2, b2, log_s2, logit_w1 = best.x
    w1 = float(expit(logit_w1))
    w2 = 1.0 - w1
    return BMAModel(
        variable=variable,
        location=location,
        forecast_hour=FORECAST_HOUR,
        transform="identity" if variable == "temperature" else "log1p",
        weight_aifs=w1,
        weight_gfs=w2,
        aifs_intercept=float(a1),
        aifs_slope=float(b1),
        aifs_sigma=float(np.exp(log_s1)),
        gfs_intercept=float(a2),
        gfs_slope=float(b2),
        gfs_sigma=float(np.exp(log_s2)),
        n_samples=len(observed),
        training_start=training_start,
        training_end=training_end,
    )


def _component_parameters(model: BMAModel, aifs: float, gfs: float):
    x1 = float(_working_transform(model.variable, np.array([aifs]))[0])
    x2 = float(_working_transform(model.variable, np.array([gfs]))[0])
    mu1 = model.aifs_intercept + model.aifs_slope * x1
    mu2 = model.gfs_intercept + model.gfs_slope * x2
    return mu1, model.aifs_sigma, mu2, model.gfs_sigma


def mixture_cdf_latent(model: BMAModel, latent: np.ndarray, aifs: float, gfs: float) -> np.ndarray:
    latent = np.asarray(latent, dtype=float)
    mu1, s1, mu2, s2 = _component_parameters(model, aifs, gfs)
    return model.weight_aifs * norm.cdf(latent, mu1, s1) + model.weight_gfs * norm.cdf(latent, mu2, s2)


def mixture_quantile_latent(model: BMAModel, q: float, aifs: float, gfs: float) -> float:
    q = float(np.clip(q, 1e-10, 1.0 - 1e-10))
    mu1, s1, mu2, s2 = _component_parameters(model, aifs, gfs)
    lo = min(mu1 - 10.0 * s1, mu2 - 10.0 * s2)
    hi = max(mu1 + 10.0 * s1, mu2 + 10.0 * s2)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if mixture_cdf_latent(model, np.array([mid]), aifs, gfs)[0] < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def distribution_2d(
    model: BMAModel,
    aifs: float,
    gfs: float,
    *,
    bins: int = 80,
) -> list[list[float]]:
    """Return ``[[value, probability_mass], ...]`` for React.

    The output is a discrete approximation of the continuous BMA mixture. The
    second column sums to 1 (up to floating-point error).
    """
    if bins < 10:
        raise ValueError("bins must be >= 10")
    if not np.isfinite(aifs) or not np.isfinite(gfs):
        raise ValueError("Forecast values must be finite.")
    if model.forecast_hour != FORECAST_HOUR:
        raise ValueError("Loaded BMA model is not a forecast-hour-24 model.")

    latent_lo = mixture_quantile_latent(model, 1e-6, aifs, gfs)
    latent_hi = mixture_quantile_latent(model, 1.0 - 1e-6, aifs, gfs)
    edges_latent = np.linspace(latent_lo, latent_hi, bins + 1)
    cdf_edges = mixture_cdf_latent(model, edges_latent, aifs, gfs)
    mass = np.diff(cdf_edges)
    mass = np.clip(mass, 0.0, None)
    total = float(mass.sum())
    if total <= 0.0:
        raise RuntimeError("BMA distribution has zero probability mass.")
    mass /= total

    centers_latent = 0.5 * (edges_latent[:-1] + edges_latent[1:])
    centers = _inverse_transform(model.variable, centers_latent)
    if model.variable == "precipitation":
        centers = np.maximum(centers, 0.0)

    output = [[float(v), float(p)] for v, p in zip(centers, mass, strict=True)]
    # Normalize once more so JSON consumers see an exact-ish probability sum.
    s = sum(row[1] for row in output)
    for row in output:
        row[1] /= s
    return output


def save_models(models: Mapping[str, Mapping[str, BMAModel]], path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "format_version": 1,
        "forecast_hour": FORECAST_HOUR,
        "variables": {
            variable: {location: model.to_dict() for location, model in per_location.items()}
            for variable, per_location in models.items()
        },
    }
    if metadata:
        payload["metadata"] = metadata
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_models(path: str | Path) -> dict[str, dict[str, BMAModel]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("forecast_hour") != FORECAST_HOUR:
        raise ValueError("Model artifact is not for forecast hour 24.")
    result: dict[str, dict[str, BMAModel]] = {}
    for variable, locations in payload.get("variables", {}).items():
        result[variable] = {loc: BMAModel.from_dict(model) for loc, model in locations.items()}
    return result

