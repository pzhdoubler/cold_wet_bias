import xarray as xr
import numpy as np
from pathlib import Path
import dask
from dask.diagnostics import ProgressBar
import time
import pandas as pd
import traceback

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

try:
    # ── Main ───────────────────────────────────────────────────────────────────────
    # Step 1: open with native on-disk chunks to avoid misalignment warnings
    full_ds = xr.open_mfdataset(
        "/ocean/projects/ees210011p/shared/ERA5_land/hourly/*.nc", 
        concat_dim="valid_time", 
        combine="nested",
        data_vars="minimal", 
        coords="minimal",
        # compact="override",
        parallel=True,
        chunks={"valid_time": 744, "latitude": 85, "longitude": 181},
        engine="h5netcdf"
    )


    months = pd.period_range(
        start=full_ds.valid_time.values[0],
        end=full_ds.valid_time.values[-1],
        freq="M"
    )

    print("Data ingested")

    for month in months:
        print(f"Working on {month}")

        # tp_hourly = fix_precip_accumulation(full_ds[PRECIP_VAR].sel(valid_time=str(month))) * PRECIP_SCALE
        # t2m       = full_ds[TEMP_VAR].sel(valid_time=str(month))

        # # ── 3-Hourly ──────────────────────────────────────────────────────────────
        # ds_3h = xr.Dataset(
        #     {
        #         TEMP_VAR:   process_temperature_3h(t2m),
        #         PRECIP_VAR: process_precip_3h(tp_hourly),
        #     },
        #     attrs={**full_ds.attrs, "frequency": "3-hourly"},
        # )

        # # ── Daily ─────────────────────────────────────────────────────────────────
        # ds_daily = xr.Dataset(
        #     {
        #         TEMP_VAR:   process_temperature_daily(t2m),
        #         PRECIP_VAR: process_precip_daily(tp_hourly),
        #     },
        #     attrs={**full_ds.attrs, "frequency": "daily"},
        # )

        # # ── Write both outputs in parallel ────────────────────────────────────────
        # out_3h    = DIR_3H    / f"ERA_land_{month.year}_{month.month:02d}.nc"
        # out_daily = DAILY_DIR / f"ERA_land_{month.year}_{month.month:02d}.nc"

        # write_3h    = ds_3h.to_netcdf(out_3h,    compute=False)
        # write_daily = ds_daily.to_netcdf(out_daily, compute=False)

        # with ProgressBar():
        #     dask.compute(write_3h)
        # print(f"  → {out_3h}")
        # with ProgressBar():
        #     dask.compute(write_daily)
        # print(f"  → {out_daily}")
        # print()

        tp_hourly = fix_precip_accumulation(
            full_ds[PRECIP_VAR].sel(valid_time=str(month))
        ) * PRECIP_SCALE
        t2m = full_ds[TEMP_VAR].sel(valid_time=str(month))

        out_3h    = DIR_3H    / f"ERA_land_{month.year}_{month.month:02d}.nc"
        out_daily = DAILY_DIR / f"ERA_land_{month.year}_{month.month:02d}.nc"
        
        # ── 3-Hourly ──────────────────────────────────────────────────────────────
        ds_3h = xr.Dataset(
            {
                TEMP_VAR:   process_temperature_3h(t2m),
                PRECIP_VAR: process_precip_3h(tp_hourly),
            },
            attrs={**full_ds.attrs, "frequency": "3-hourly"},
        ).chunk({"valid_time": -1,  "latitude": 85, "longitude": 181})

        ds_3h.to_netcdf(out_3h)
        ds_3h.close()
        print("Saved 3hr")
        # ── Daily ─────────────────────────────────────────────────────────────────
        ds_daily = xr.Dataset(
            {
                TEMP_VAR:   process_temperature_daily(t2m),
                PRECIP_VAR: process_precip_daily(tp_hourly),
            },
            attrs={**full_ds.attrs, "frequency": "daily"},
        ).chunk({"valid_time": -1,  "latitude": 85, "longitude": 181})
        ds_daily.to_netcdf(out_daily)
        ds_daily.close()
        print("Saved daily")
        
        tp_hourly.close()
        t2m.close()
        print()
except Exception:
    traceback.print_exc()
