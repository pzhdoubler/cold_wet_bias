import xarray as xr
import numpy as np
from pathlib import Path
import dask
from dask.diagnostics import ProgressBar
import time

# ── Paths ──────────────────────────────────────────────────────────────────────
HOURLY_DIR   = Path("/ocean/projects/ees210011p/shared/ERA5_land/hourly")
DIR_3H       = Path("/ocean/projects/ees210011p/shared/ERA5_land/3hourly")
DAILY_DIR    = Path("/ocean/projects/ees210011p/shared/ERA5_land/daily")

# ── Config ─────────────────────────────────────────────────────────────────────
PRECIP_VAR   = "tp"
TEMP_VAR     = "t2m"
PRECIP_SCALE = 1000  # m → mm
TIME_DIM     = "valid_time"


def fix_precip_accumulation(tp: xr.DataArray) -> xr.DataArray:
    """
    ERA5-Land tp is accumulated since the start of each forecast run.
    Runs reset at 01 UTC and 13 UTC each day.
    Differences consecutive steps, fixes resets, and preserves original NaNs.
    """
    tp_diff = tp.diff(dim=TIME_DIM)

    is_reset    = tp_diff < 0
    raw_shifted = tp.isel({TIME_DIM: slice(1, None)})
    tp_fixed    = xr.where(is_reset, raw_shifted, tp_diff)

    # Restore NaNs: if either the current or previous timestep was NaN, keep NaN
    nan_mask = (
        np.isnan(tp.isel({TIME_DIM: slice(1, None)})) |
        np.isnan(tp.isel({TIME_DIM: slice(None, -1)}))
    )
    tp_fixed = tp_fixed.where(~nan_mask)

    tp_fixed.attrs = tp.attrs
    tp_fixed.attrs["units"] = "mm"
    tp_fixed.attrs["long_name"] = "Hourly precipitation"

    return tp_fixed


def process_temperature_3h(t2m: xr.DataArray) -> xr.DataArray:
    t2m_3h = t2m.resample({TIME_DIM: "3h"}).mean()
    t2m_3h.attrs = t2m.attrs
    t2m_3h.attrs["long_name"] = "3-hourly mean 2m temperature"
    return t2m_3h


def process_precip_3h(tp_hourly: xr.DataArray) -> xr.DataArray:
    tp_3h = tp_hourly.resample({TIME_DIM: "3h"}).sum(skipna=False)
    tp_3h.attrs = tp_hourly.attrs
    tp_3h.attrs["long_name"] = "3-hourly total precipitation"
    return tp_3h


def process_temperature_daily(t2m: xr.DataArray) -> xr.DataArray:
    t2m_daily = t2m.resample({TIME_DIM: "1D"}).mean()
    t2m_daily.attrs = t2m.attrs
    t2m_daily.attrs["long_name"] = "Daily mean 2m temperature"
    return t2m_daily


def process_precip_daily(tp_hourly: xr.DataArray) -> xr.DataArray:
    tp_daily = tp_hourly.resample({TIME_DIM: "1D"}).sum(skipna=False)
    tp_daily.attrs = tp_hourly.attrs
    tp_daily.attrs["long_name"] = "Daily total precipitation"
    return tp_daily

def process_file(fpath: Path):
    print(f"\nProcessing: {fpath.name}")

    # Step 1: open with native on-disk chunks to avoid misalignment warnings
    ds = xr.open_dataset(fpath, chunks={})

    ds = ds.unify_chunks()

    # Step 2: inspect native chunk sizes
    tp_native = ds[PRECIP_VAR]
    print(f"  Native chunks: {dict(tp_native.chunksizes)}")

    # Step 3: rechunk to dask-friendly sizes — keep time whole, chunk spatial dims
    # Adjust lat/lon chunk sizes if your grid is coarser/finer than 0.1deg ERA5-Land
    ds = ds.chunk({"latitude": 200, "longitude": 200, TIME_DIM: -1})

    tp_hourly = fix_precip_accumulation(ds[PRECIP_VAR]) * PRECIP_SCALE
    t2m       = ds[TEMP_VAR]
    stem      = fpath.stem

    # ── 3-Hourly ──────────────────────────────────────────────────────────────
    ds_3h = xr.Dataset(
        {
            TEMP_VAR:   process_temperature_3h(t2m),
            PRECIP_VAR: process_precip_3h(tp_hourly),
        },
        attrs={**ds.attrs, "frequency": "3-hourly"},
    )

    # ── Daily ─────────────────────────────────────────────────────────────────
    ds_daily = xr.Dataset(
        {
            TEMP_VAR:   process_temperature_daily(t2m),
            PRECIP_VAR: process_precip_daily(tp_hourly),
        },
        attrs={**ds.attrs, "frequency": "daily"},
    )

    # ── Write both outputs in parallel ────────────────────────────────────────
    out_3h    = DIR_3H    / f"{stem}_3h.nc"
    out_daily = DAILY_DIR / f"{stem}_daily.nc"

    write_3h    = ds_3h.to_netcdf(out_3h,    compute=False)
    write_daily = ds_daily.to_netcdf(out_daily, compute=False)

    with ProgressBar():
        dask.compute(write_3h, write_daily)

    print(f"  → {out_3h}")
    print(f"  → {out_daily}")
    ds.close()

# ── Main ───────────────────────────────────────────────────────────────────────
files = sorted(HOURLY_DIR.glob("ERA_land_*.nc"))

if not files:
    print(f"No files found in {HOURLY_DIR}")
else:
    print(f"Found {len(files)} files to process")
    initial = True
    for f in files:
        s = time.time()
        process_file(f)
        e = time.time()
        if initial:
            print(f"File took {e-s} seconds")
            print(f"Expect completeion in {(e-s) * len(files)} seconds")
            initial = False
    print("\nDone!")
