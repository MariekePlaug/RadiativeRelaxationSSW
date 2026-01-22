import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pyarts3 as pyarts
import netCDF4 as netcdf4


def atm(year, timestep):
    ds = xr.open_dataset(f"data/{year}_data.nc", engine="netcdf4")

    atm = xr.Dataset(
        data_vars={
            "co2": (("pressure"), ds.co2[timestep,:].data),
            "no2": (("pressure"), ds.no2[timestep,:].data),
            "no": (("pressure"), ds.no[timestep,:].data),
            "o3": (("pressure"), ds.o3[timestep,:].data),
            "t" : (("pressure"), ds.t[timestep,:].data),
            "u": (("pressure"), ds.u[timestep,:].data),
            "rh" : (("pressure"), ds.rh[timestep,:].data),},
        coords={"pressure" : ds.pressure.data}
    )
    return atm

atm1 = atm(2010, 0)
print(atm1)


def atm_co2doubling(year, timestep):
    ds = xr.open_dataset(f"data/{year}_data.nc", engine="netcdf4")

    atm_co2doubling = xr.Dataset(
        data_vars={
            "co2": (("pressure"), ds.co2[timestep,:].data * 2),
            "no2": (("pressure"), ds.no2[timestep,:].data),
            "no": (("pressure"), ds.no[timestep,:].data),
            "o3": (("pressure"), ds.o3[timestep,:].data),
            "t" : (("pressure"), ds.t[timestep,:].data),
            "u": (("pressure"), ds.u[timestep,:].data),
            "rh" : (("pressure"), ds.rh[timestep,:].data),},
        coords={"pressure" : ds.pressure.data}
    )
    return atm_co2doubling

atm_co2 = atm_co2doubling(2010, 0)
print(atm_co2)