#!/usr/bin/env python3
"""Download IMD gridded archive observations through IMDLIB.

Supported archive variables:
    rain -> daily rainfall (0.25 degree grid)
    tmin -> daily minimum temperature (1 degree grid)
    tmax -> daily maximum temperature (1 degree grid)

This intentionally does NOT implement wind speed: IMDLIB's documented archive
API supports rainfall, minimum temperature, and maximum temperature only.

The script downloads the original IMD .grd files year-by-year via imdlib and
also writes a NetCDF copy for each requested variable, which is convenient for
xarray/xESMF processing later.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VARIABLES = ("rain", "tmin", "tmax")


def require_imdlib():
    try:
        import imdlib as imd
    except ImportError as exc:
        raise RuntimeError(
            "imdlib is not installed. Install it with `python -m pip install imdlib`."
        ) from exc
    return imd


def parse_year(value: str) -> int:
    try:
        year = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid year: {value!r}") from exc
    if year < 1900 or year > 2100:
        raise argparse.ArgumentTypeError("Year must be between 1900 and 2100.")
    return year


def download_variable(imd, variable: str, start_year: int, end_year: int, output_dir: Path,
                      overwrite_nc: bool = False):
    variable_dir = output_dir / variable
    variable_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== IMD {variable}: {start_year}-{end_year} ===")
    print(f"Raw GRD directory: {variable_dir}")

    # IMDLIB documents get_data(..., fn_format='yearwise', file_dir=...,
    # sub_dir=True) and returns an IMD object. Using one top-level output
    # directory keeps rain/tmin/tmax in separate subdirectories.
    data = imd.get_data(
        variable,
        start_year,
        end_year,
        fn_format="yearwise",
        file_dir=str(output_dir),
        sub_dir=True,
    )

    nc_path = variable_dir / f"imd_{variable}_{start_year}_{end_year}.nc"
    if nc_path.exists() and not overwrite_nc:
        print(f"NetCDF exists; not overwriting: {nc_path}")
        return nc_path

    print(f"Writing NetCDF: {nc_path}")
    data.to_netcdf(nc_path.name, str(variable_dir))
    print(f"WROTE: {nc_path}")
    return nc_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download IMD gridded archive observations via IMDLIB. "
            "Supported variables: rain, tmin, tmax. Wind speed is intentionally excluded."
        )
    )
    parser.add_argument("--start", required=True, type=parse_year, help="First archive year, e.g. 2025")
    parser.add_argument("--end", required=True, type=parse_year, help="Last archive year, e.g. 2025")
    parser.add_argument(
        "--variable",
        choices=VARIABLES + ("all",),
        default="all",
        help="Variable to download. Default: all (rain, tmin, tmax).",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/imd_2025",
        help="Directory containing rain/tmin/tmax subdirectories.",
    )
    parser.add_argument(
        "--overwrite-nc",
        action="store_true",
        help="Overwrite an existing combined NetCDF output.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.start > args.end:
        print("ERROR: --start must be <= --end.", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    imd = require_imdlib()

    variables = VARIABLES if args.variable == "all" else (args.variable,)
    for variable in variables:
        try:
            download_variable(
                imd,
                variable,
                args.start,
                args.end,
                output_dir,
                overwrite_nc=args.overwrite_nc,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"ERROR downloading {variable}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1

    print("\nIMD download complete.")
    print("Wind speed is intentionally not downloaded; it will be handled later by the ClimaFuse fill model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
