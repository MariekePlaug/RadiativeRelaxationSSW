import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pyarts3 as pyarts
import netCDF4 as netcdf4

years = [2004,2006,2007,2008,2009,2010,2013]

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


