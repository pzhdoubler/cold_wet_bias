import xarray as xr
import numpy as np
from pathlib import Path
import glob

# ── Paths ──────────────────────────────────────────────────────────────────────
HOURLY_DIR  = Path("/ocean/projects/ees210011p/shared/ERA5_land/hourly")
DIR_3H      = Path("/ocean/projects/ees210011p/shared/ERA5_land/3hourly")
DAILY_DIR   = Path("/ocean/projects/ees210011p/shared/ERA5_land/daily")

# ── Config ─────────────────────────────────────────────────────────────────────
PRECIP_VAR  = "tp"   # total precipitation
TEMP_VAR    = "t2m"  # 2m temperature
PRECIP_SCALE = 1000  # m → mm


def fix_precip_accumulation(tp: xr.DataArray) -> xr.DataArray:
    """
    ERA5-Land tp is accumulated since the start of each forecast run.
    Runs reset at 01 UTC and 13 UTC each day.
    This function differences consecutive steps and masks reset artifacts.
    """
    tp_diff = tp.diff(dim="valid_time")

    # Negative diffs = forecast reset → replace with the raw value at that step
    # (the first step of each run IS the accumulation from 0, so it's valid as-is)
    raw_shifted = tp.isel(valid_time=slice(1, None))
    tp_fixed = xr.where(tp_diff < 0, raw_shifted, tp_diff)
    tp_fixed.attrs = tp.attrs
    tp_fixed.attrs["units"] = "mm"
    tp_fixed.attrs["long_name"] = "Hourly precipitation"

    return tp_fixed


def process_temperature_3h(t2m: xr.DataArray) -> xr.DataArray:
    """Resample 2m temperature to 3-hourly means."""
    t2m_3h = t2m.resample(valid_time="3h").mean()
    t2m_3h.attrs = t2m.attrs
    t2m_3h.attrs["long_name"] = "3-hourly mean 2m temperature"
    return t2m_3h


def process_precip_3h(tp_hourly: xr.DataArray) -> xr.DataArray:
    """Sum corrected hourly precip to 3-hourly totals."""
    tp_3h = tp_hourly.resample(valid_time="3h").sum()
    tp_3h.attrs = tp_hourly.attrs
    tp_3h.attrs["long_name"] = "3-hourly total precipitation"
    return tp_3h


def process_temperature_daily(t2m: xr.DataArray) -> xr.DataArray:
    """Resample 2m temperature to daily means."""
    t2m_daily = t2m.resample(valid_time="1D").mean()
    t2m_daily.attrs = t2m.attrs
    t2m_daily.attrs["long_name"] = "Daily mean 2m temperature"
    return t2m_daily


def process_precip_daily(tp_hourly: xr.DataArray) -> xr.DataArray:
    """Sum corrected hourly precip to daily totals."""
    tp_daily = tp_hourly.resample(valid_time="1D").sum()
    tp_daily.attrs = tp_hourly.attrs
    tp_daily.attrs["long_name"] = "Daily total precipitation"
    return tp_daily


def process_file(fpath: Path):
    print(f"Processing: {fpath.name}")

    ds = xr.open_dataset(fpath)

    # ── Fix precipitation accumulation ────────────────────────────────────────
    tp_hourly = fix_precip_accumulation(ds[PRECIP_VAR]) * PRECIP_SCALE
    t2m       = ds[TEMP_VAR]
    print("fixed precip...")

    # ── Build output stem (e.g. ERA_land_1980_01) ─────────────────────────────
    stem = fpath.stem

    # ── 3-Hourly output ───────────────────────────────────────────────────────
    ds_3h = xr.Dataset(
        {
            TEMP_VAR:  process_temperature_3h(t2m),
            PRECIP_VAR: process_precip_3h(tp_hourly),
        },
        attrs=ds.attrs,
    )
    ds_3h.attrs["frequency"] = "3-hourly"
    out_3h = DIR_3H / f"{stem}_3h.nc"
    ds_3h.to_netcdf(out_3h)
    print(f"  → Saved 3-hourly: {out_3h}")

    # ── Daily output ──────────────────────────────────────────────────────────
    ds_daily = xr.Dataset(
        {
            TEMP_VAR:  process_temperature_daily(t2m),
            PRECIP_VAR: process_precip_daily(tp_hourly),
        },
        attrs=ds.attrs,
    )
    ds_daily.attrs["frequency"] = "daily"
    out_daily = DAILY_DIR / f"{stem}_daily.nc"
    ds_daily.to_netcdf(out_daily)
    print(f"  → Saved daily:    {out_daily}")

    ds.close()


# ── Main ───────────────────────────────────────────────────────────────────────
files = sorted(HOURLY_DIR.glob("ERA_land_*.nc"))

files = files[:5]

if not files:
    print(f"No files found in {HOURLY_DIR}")
else:
    print(f"Found {len(files)} files to process\n")
    for f in files:
        process_file(f)
    print("\nDone!")
