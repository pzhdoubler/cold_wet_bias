import xarray as xr
import numpy as np
from pathlib import Path
import dask
from dask.diagnostics import ProgressBar
import time
import pandas as pd
import xesmf as xe
import os
import hashlib

###################################################
################# Helpers #################
###################################################
class CMIP_group:
    def __init__(self, file_string):
        tokens = file_string.split("_")
        if len(tokens) < 6:
            print("Invalid file string")
        self.variable = tokens[0]
        self.freq = tokens[1]
        self.model = tokens[2]
        self.run = tokens[3]
        self.variant = tokens[4]
        self.grid = tokens[5]
        if len(tokens) > 6:
            self.end_label = "_".join(tokens[6:])
        else:
            self.end_label = ""

        self.base_string = f"{self.variable}_{self.freq}_{self.model}_{self.run}_{self.variant}_{self.grid}"

    def __str__(self):
        return self.base_string

    def get_string_base(self):
        return self.base_string
        
    def make_file_path(self, BASE_PATH, end_label):
        base = Path(BASE_PATH) / self.model
        os.makedirs(base, exist_ok=True)
        return base / f"{self.base_string}_{end_label}.nc"

def get_cwrf_regridder(in_grid, interp_method):
    # setup target domain
    proj = xr.open_dataset("/jet/home/hdoubler/fixed_cwrf_domain.nc")
    save_dir = "/ocean/projects/ees210011p/hdoubler/cwrf_regridders"
    os.makedirs(save_dir, exist_ok=True)

    lat_bytes = in_grid["lat"].values.tobytes()
    lon_bytes = in_grid["lon"].values.tobytes()
    h = hashlib.sha256(lat_bytes + lon_bytes).hexdigest()

    rg_path = f"{save_dir}/{interp_method}_{h}.nc"

    try:
        rg = xe.Regridder(in_grid, proj, method=interp_method, weights=rg_path)
    except:
        rg = xe.Regridder(in_grid, proj, method=interp_method)
        rg.to_netcdf(rg_path)
    
    return rg

###################################################
################# Readers #################
###################################################

def read_ERA5_obs(var):
    ERA5_ds = xr.open_mfdataset(
        "/ocean/projects/ees210011p/shared/ERA5_land/regrid/daily/*.nc",
        concat_dim="time", 
        combine="nested",
        data_vars="minimal", 
        coords="minimal",
        compat="override",
        parallel=True,
        engine="h5netcdf"
    ).sel(time=slice("1980", "2014"))

    # add more vars here if needed
    if var == "pr":
        da = ERA5_ds["tp"]
    if var == "tas":
        da = ERA5_ds["t2m"]
    
    ERA5_ds.close()
    return da
    
def read_Daymet_obs(var):
    # add more vars here if needed
    if var == "pr":
        Daymet_ds = xr.open_mfdataset(
            "/ocean/projects/ees210011p/hdoubler/regrid_daily/OBS_PRAVG*.nc",
            concat_dim="time", 
            combine="nested",
            data_vars="minimal", 
            coords="minimal",
            compat="override",
            parallel=True,
            engine="h5netcdf"
        ).sel(time=slice("1980", "2014"))
        da = Daymet_ds["PRAVG"]
        Daymet_ds.close()
    if var == "tas":
        Daymet_ds = xr.open_mfdataset(
            "/ocean/projects/ees210011p/hdoubler/regrid_daily/OBS_AT2M*.nc",
            concat_dim="time", 
            combine="nested",
            data_vars="minimal", 
            coords="minimal",
            compat="override",
            parallel=True,
            engine="h5netcdf"
        ).sel(time=slice("1980", "2014"))
        da = Daymet_ds["AT2M"]
        Daymet_ds.close()

    return da

def do_CMIP_regrid(cmip_group):
    CMIP_ds = xr.open_mfdataset(
        f"/ocean/projects/ees210011p/shared/CMIP6/daily/{cmip_group.model}/{cmip_group.get_string_base()}*.nc", 
        concat_dim="time", 
        combine="nested",
        data_vars="minimal", 
        coords="minimal",
        compat="override",
        parallel=True,
        engine="h5netcdf"
        ).chunk({"time": 366})

    cmip_times = pd.DatetimeIndex([
        pd.Timestamp(str(t)) for t in CMIP_ds['time'].values
    ])
    CMIP_ds['time'] = cmip_times
    
    CMIP_ds = CMIP_ds.sel(time=slice("1980", "2014"))

    # add more vars here if needed
    if cmip_group.variable == "pr":
        regridder = get_cwrf_regridder(CMIP_ds[cmip_group.variable], "conservative")
        CMIP_da = (regridder(CMIP_ds[cmip_group.variable]) * 86400)
    if cmip_group.variable == "tas":
        regridder = get_cwrf_regridder(CMIP_ds[cmip_group.variable], "bilinear")
        CMIP_da = (regridder(CMIP_ds[cmip_group.variable]))

    CMIP_ds.close()
    
    return CMIP_da

###################################################
################# Analysis #################
###################################################

def do_bias_compare(cmip_group, cmip_da, obs, SAVE_DIR, end_label):
    # align times
    common_times = np.intersect1d(cmip_da["time"].values, obs["time"].values)
    bias = cmip_da.sel(time=common_times) - obs.sel(time=common_times)
    # do bias climatology
    bias.name = f"{cmip_group.variable}-bias"
    cmip_ds = xr.Dataset({cmip_group.variable : cmip_da})
    bias_climatology =  cmip_ds.groupby("time.month").mean()
    write_job = bias_climatology.to_netcdf(cmip_group.make_file_path(SAVE_DIR, end_label), parallel=True, compute=False)
    # Write with progress bar and encoding to reduce file size
    with ProgressBar():
        write_job.compute()

# def do_quantile_compare(quantile, cmip_group, cmip_da, obs, SAVE_DIR, end_label):
#     pass

###################################################
################# Main #################
###################################################

if __name__ == "__main__":
    experiment = "CMIP6"
    var = "pr"
    obs_group = "Daymet"
    SAVE_DIR = Path(f"/ocean/projects/ees210011p/hdoubler/cold_wet_bias/CMIP_bias/{experiment}")
    TEMP_DIR = Path(f"/ocean/projects/ees210011p/hdoubler/cold_wet_bias/temp/{experiment}")

    # open obs
    if obs_group == "ERA5":
        obs = read_ERA5_obs(var).chunk({"time": -1, "lat": 138, "lon": 195}).persist()
    if obs_group == "Daymet":
        obs = read_Daymet_obs(var).chunk({"time": -1, "lat": 138, "lon": 195}).persist()

    print(f"{var} obs read")

    # main CMIP6
    cmip_models_loc = Path(f"/ocean/projects/ees210011p/shared/{experiment}/daily")
    models = os.listdir(cmip_models_loc)

    models = models[3:4]

    print(f"Found following models:")
    print(models)


    for m, model in enumerate(models):
        print(f"Working on {model} ({m+1}/{len(models)})...")
        # read cmip_groups and only use those that match target var
        model_loc = cmip_models_loc / model
        cmip_groups = [CMIP_group(g) for g in list(set(["_".join(f.split("_")[:-1]) for f in os.listdir(model_loc)]))] #crazy line btw
        cmip_groups = [group for group in cmip_groups if group.variable == var]
        
        for g, cmip_group in enumerate(cmip_groups):
            print(f"working on group {cmip_group.get_string_base()} ({g+1}/{len(cmip_groups)})...")
            cmip_da = do_CMIP_regrid(cmip_group)
            cmip_da = cmip_da.chunk({"time": -1, "lat": 138, "lon": 195})
            # do whatever analysis needed here
            do_bias_compare(cmip_group, cmip_da, obs, SAVE_DIR, f"{obs_group}_bias")

        print(f"{model} done.")
        print()
