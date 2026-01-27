# imports

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pyarts3 as pyarts
import netCDF4 as netcdf4
import typhon as ty


def get_atm(year, timestep):
    """
    Reads in the atmospheric profile for a certain day of an SSW event.

    Inputs:
    ----------
    year : year that the event occurs in, possible years: 2004, 2006, 2007, 2008, 2009, 2010, 2013
    timestep : day after the start of the event

    Returns:
    ----------
    atm : atmospheric profile ready to use in ARTS 3
    """
    ds = xr.open_dataset(f"data/{year}_data.nc", engine="netcdf4")

    # convert RH to water vapor VMR:
    x_profile = ty.physics.relative_humidity2vmr(ds.rh.isel(time=timestep).data, ds.pressure.data, ds.t.isel(time=timestep).data)

    # convert mass mixing ratios to VMR:
    co2_profile = ty.physics.mixing_ratio2vmr(ds.co2.isel(time=timestep).data)
    no2_profile = ty.physics.mixing_ratio2vmr(ds.no2.isel(time=timestep).data)
    no_profile = ty.physics.mixing_ratio2vmr(ds.no.isel(time=timestep).data)
    o3_profile = ty.physics.mixing_ratio2vmr(ds.o3.isel(time=timestep).data)

    # convert pressure from hPa to Pa:
    p_profile = ds.pressure.data * 100

    # convert pressure to height:
    z_profile = ty.physics.pressure2height(p_profile)

    atm = xr.Dataset(
        data_vars={
            "CO2": ("alt", co2_profile),
            "NO2": ("alt", no2_profile),
            "NO": ("alt", no_profile),
            "O3": ("alt", o3_profile),
            "t": ("alt", ds.t[timestep,:].data),
            "p": ("alt", p_profile),
            "wind_u": ("alt", ds.u[timestep,:].data),
            "H2O" : ("alt", x_profile),
            "O2": ("alt", np.ones_like(ds.co2[timestep,:].data) * 0.21),
            "N2": ("alt", np.ones_like(ds.co2[timestep,:].data) * 0.78),
        },
        coords={"alt" : z_profile, "lat": 75, "lon": 0},
    )

    atm["CO2"].attrs = {
        "units": "mol/mol",
        "long_name": "CO2 volume mixing ratio",
    }
    atm["NO2"].attrs = {
        "units": "mol/mol",
        "long_name": "NO2 volume mixing ratio",
    }
    atm["NO"].attrs = {
        "units": "mol/mol",
        "long_name": "NO volume mixing ratio",
    }
    atm["O3"].attrs = {
        "units": "mol/mol",
        "long_name": "Ozone volume mixing ratio",
    }
    atm["t"].attrs = {
        "units": "K",
        "long_name": "Temperature",
    }
    atm["p"].attrs = {
        "units": "Pa",
        "long_name": "Pressure",
    }
    atm["wind_u"].attrs = {
        "units": "m/s",
        "long_name": "zonal mean zonal wind at 10 hPa and 60°N",
    }
    atm["H2O"].attrs = {
        "units": "mol/mol",
        "long_name": "Water vapor volume mixing ratio",
    }
    atm["O2"].attrs = {
        "units": "mol/mol",
        "long_name": "Oxygen volume mixing ratio",
    }
    atm["N2"].attrs = {
        "units": "mol/mol",
        "long_name": "Nitrogen volume mixing ratio",
    }
    atm["alt"].attrs = {
        "units": "m",
        "long_name": "Geometric altitude",
    }

    return atm

