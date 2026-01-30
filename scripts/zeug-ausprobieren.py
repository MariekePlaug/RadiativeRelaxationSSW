# %% imports

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pyarts3 as pyarts
import netCDF4 as netcdf4
import typhon as ty
from pyarts3.arts import AtmPoint

from src.SSW_functions import get_atm, get_atm_test

pyarts.data.download()

# %% variables:

years = [2004,2006,2007,2008,2009,2010,2013]
kayser_grid = np.linspace(1, 2300, 2000)  # Wavenumbers in cm-1
f_grid = pyarts.arts.convert.kaycm2freq(kayser_grid)
species = ["H2O", "H2O-ForeignContCKDMT400", "H2O-SelfContCKDMT400", "CO2", "O3"]


# %% test atmosphere

atm_test = get_atm(2010, 0)

fop = pyarts.recipe.AtmosphericFlux(
    species=["H2O-161", "O2-66", "N2-44", "CO2-626", "O3-XFIT"],
    remove_lines_percentile={"H2O": 70},
)
solar, thermal, altitude = fop(atmospheric_profile=atm_test)

# %%
for key, value in atm_test.items():
    try:
        print(key, len(value))
    except TypeError:
        print(key, "hat keine Länge:", type(value))

print(atm_test.keys())

# %%
ds = xr.open_dataset(f"data/{2010}_data.nc", engine="netcdf4")

p_profile = ds.pressure.data * 100  # hPa -> Pa
t_profile = ds.t.isel(time=0).data
# Erstelle ein minimales Test-Dictionary:
test_atm = {
    pyarts.arts.AtmKey.p: pyarts.arts.Vector(p_profile),
    pyarts.arts.AtmKey.t: pyarts.arts.Vector(t_profile),
}

print("Länge p:", len(test_atm[pyarts.arts.AtmKey.p]))
print("Länge t:", len(test_atm[pyarts.arts.AtmKey.t]))

# Teste mit fop:
solar, thermal, altitude = fop(atmospheric_profile=test_atm)


# %%

import pyarts3 as pyarts
import typhon as ty
import numpy as np
import xarray as xr

pyarts.data.download()

def get_atm_nochmal(year, timestep):
    ds = xr.open_dataset(f"data/{year}_data.nc", engine="netcdf4")

    # Alle Konvertierungen
    x_profile = ty.physics.relative_humidity2vmr(
        ds.rh.isel(time=timestep).data,
        ds.pressure.data,
        ds.t.isel(time=timestep).data
    )
    co2_profile = ty.physics.mixing_ratio2vmr(ds.co2.isel(time=timestep).data)
    no2_profile = ty.physics.mixing_ratio2vmr(ds.no2.isel(time=timestep).data)
    no_profile = ty.physics.mixing_ratio2vmr(ds.no.isel(time=timestep).data)
    o3_profile = ty.physics.mixing_ratio2vmr(ds.o3.isel(time=timestep).data)

    p_profile = ds.pressure.data * 100  # hPa -> Pa
    t_profile = ds.t.isel(time=timestep).data

    # Erstelle ArrayOfAtmPoint von Hand
    atm_points = pyarts.arts.ArrayOfAtmPoint()

    for i in range(len(p_profile)):
        point = pyarts.arts.AtmPoint()
        point.pressure = float(p_profile[i])
        point.temperature = float(t_profile[i])

        # Setze die Spezies
        point[pyarts.arts.SpeciesEnum.Water] = float(x_profile[i])
        point[pyarts.arts.SpeciesEnum.CarbonDioxide] = float(co2_profile[i])
        point[pyarts.arts.SpeciesEnum.NitrogenDioxide] = float(no2_profile[i])
        point[pyarts.arts.SpeciesEnum.NitricOxide] = float(no_profile[i])
        point[pyarts.arts.SpeciesEnum.Ozone] = float(o3_profile[i])

        atm_points.append(point)

        atm_dict = pyarts.arts.ArrayOfAtmPoint.to_dict(atm_points)

    return atm_dict, atm_points


# %% nochmal atmosphäre von get_atm anschauen:

atmosphäre_versuch10000_dict, atmosphäre_versuch10000_points = get_atm_nochmal(2010, 0)

fop = pyarts.recipe.AtmosphericFlux(
    species=["H2O-161", "O2-66", "N2-44", "CO2-626", "O3-XFIT"],
    remove_lines_percentile={"H2O": 70},
)
solar, thermal, altitude = fop(atmospheric_profile=atmosphäre_versuch10000_dict)

# %%
for key, value in atmosphäre_versuch10000_dict.items():
    try:
        print(key, len(value))
    except TypeError:
        print(key, "hat keine Länge:", type(value))

print(type(atmosphäre_versuch10000_points))
print(len(atmosphäre_versuch10000_points))

# %%

solar, thermal, altitude = fop(atmospheric_profile=atmosphäre_versuch10000)

# %%

def get_atm_nocheinversuch(year, timestep):
    ds = xr.open_dataset(f"data/{year}_data.nc", engine="netcdf4")

    # Alle Konvertierungen
    x_profile = ty.physics.relative_humidity2vmr(
        ds.rh.isel(time=timestep).data,
        ds.pressure.data,
        ds.t.isel(time=timestep).data
    )
    co2_profile = ty.physics.mixing_ratio2vmr(ds.co2.isel(time=timestep).data)
    o3_profile = ty.physics.mixing_ratio2vmr(ds.o3.isel(time=timestep).data)

    p_profile = ds.pressure.data * 100  # hPa -> Pa
    t_profile = ds.t.isel(time=timestep).data
    z_profile = ty.physics.pressure2height(p_profile)

    # O2 und N2 konstant
    o2_profile = np.full(len(p_profile), 0.209)
    n2_profile = np.full(len(p_profile), 0.781)

    # Dictionary mit PyARTS-Keys und PyARTS-Vectors!
    atm = {
        pyarts.arts.AtmKey.p: pyarts.arts.Vector(p_profile),
        pyarts.arts.AtmKey.t: pyarts.arts.Vector(t_profile),
        pyarts.arts.SpeciesEnum.Water: pyarts.arts.Vector(x_profile),
        pyarts.arts.SpeciesEnum.Oxygen: pyarts.arts.Vector(o2_profile),
        pyarts.arts.SpeciesEnum.Nitrogen: pyarts.arts.Vector(n2_profile),
        pyarts.arts.SpeciesEnum.CarbonDioxide: pyarts.arts.Vector(co2_profile),
        pyarts.arts.SpeciesEnum.Ozone: pyarts.arts.Vector(o3_profile),
    }

    return atm

# %%
atm_test = get_atm_nocheinversuch(2010, 0)
print("Dict keys:", list(atm_test.keys()))
print("Type of first key:", type(list(atm_test.keys())[0]))
print("Type of first value:", type(list(atm_test.values())[0]))

# solar, thermal, altitude = fop(atmospheric_profile=atm_test)