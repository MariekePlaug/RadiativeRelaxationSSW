# %% imports

import os

import numpy as np
import pyarts3 as pa
import xarray as xr

from src.idealised_single_column_atm import single_column_atmosphere as sca



pa.data.download()

# %% Define functions

def get_atm(Ts=290, Tcp=200, RH=0.8, co2_conc=4e-4):
    p, T_profile, x_profile, z, p_tp = sca(Ts=Ts, Tcp=Tcp, RH=RH, N=100)

    atm = xr.Dataset(
        {
            "t": ("alt", T_profile),
            "p": ("alt", p),
            "H2O": ("alt", x_profile),
            "O2": ("alt", np.ones_like(p) * 0.21),
            "N2": ("alt", np.ones_like(p) * 0.78),
            "O3": ("alt", np.ones_like(p) * 1e-6),
            "CO2": ("alt", np.ones_like(p) * co2_conc),
        },
        coords={"alt": z, "lat": 0, "lon": 0},
    )

    atm["t"].attrs = {
        "units": "K",
        "long_name": "Temperature",
    }
    atm["p"].attrs = {
        "units": "Pa",
        "long_name": "Pressure",
    }
    atm["H2O"].attrs = {
        "units": "mol/mol",
        "long_name": "Water vapor volume mixing ratio",
    }
    atm["O3"].attrs = {
        "units": "mol/mol",
        "long_name": "Ozone volume mixing ratio",
    }
    atm["O2"].attrs = {
        "units": "mol/mol",
        "long_name": "Oxygen volume mixing ratio",
    }
    atm["CO2"].attrs = {
        "units": "mol/mol",
        "long_name": "CO2 volume mixing ratio",
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


def calculate_olr(atm, f_grid, species):
    """
    Calculates the outgoing longwave radiation (OLR) at the top of the atmosphere (TOA) for a given atmospheric profile.

    Parameters:
    -----------
    atm : xarray.Dataset
        Atmospheric profile.
        Coordinates:
        - alt : Geometric altitude in meters.
        - lat : Latitude in degrees.
        - lon : Longitude in degrees.
        MUST hold the following species profiles:
        - t : Temperature profile in Kelvin.
        - p : Pressure profile in Pa.
        CAN hold all other species known to arts, like H2O, O2, N2, O3, CO2, etc.

    f_grid : array
        Frequency grid in Hz.

    species : list
        List of species to be considered in the radiative transfer calculation.


    Returns:
    --------
    LW_arr : xarray.DataArray
        Longwave radiation at TOA in W cm / m^2
    """
    # Set some default parameters
    NQuad = 16
    max_level_step = 1e3
    cutoff = ["ByLine", 750e9]
    remove_lines_percentile = 70
    planet = "Earth"

    # Create a pyarts workspace
    ws = pa.Workspace()
    ws.frequency_grid = f_grid

    # Set the atmospheric profile and find according absorption species
    ws.atmospheric_field = pa.data.to_atmospheric_field(atm)
    ws.absorption_species = species
    ws.ReadCatalogData(ignore_missing=True)
    ws.propagation_matrix_agendaAuto(T_extrapolfac=1e9)

    # Specify cutoffs for absorption bands to save computational time
    for band in ws.absorption_bands:
        ws.absorption_bands[band].cutoff = cutoff[0]
        ws.absorption_bands[band].cutoff_value = cutoff[1]

    # Remove lines with low absorption
    ws.absorption_bands.keep_hitran_s(remove_lines_percentile)

    # Set the surface properties
    ws.surface_fieldPlanet(option=planet)
    ws.surface_field["t"] = atm["t"].sel(alt=0).values

    # Set disort settings
    ws.disort_quadrature_dimension = NQuad
    ws.disort_fourier_mode_dimension = 1
    ws.disort_legendre_polynomial_dimension = 1

    # Set the ray path
    ws.ray_pathGeometricDownlooking(
        latitude=atm["lat"].values,
        longitude=atm["lon"].values,
        max_step=max_level_step,
    )

    # Set up geometry of observation
    pos = [100e3, 0, 0]
    los = [180.0, 0.0]
    ws.ray_pathGeometric(pos=pos, los=los, max_step=1000.0)
    ws.spectral_radianceClearskyEmission()

    # Extract the OLR and build a xarray.DataArray
    LW_up = (
        ws.spectral_radiance[:, 0]
        * 3e10
        * np.pi  # Convert from W/m2/Hz/sr to W/m2/cm-1
    )
    kayser = pa.arts.convert.freq2kaycm(ws.frequency_grid)
    LW_arr = xr.DataArray(LW_up, dims=["wavenum"], coords={"wavenum": kayser})
    LW_arr["wavenum"].attrs = {
        "units": "cm-1",
        "long_name": "Wavenumber",
    }
    LW_arr.attrs = {
        "units": "W cm m-2",
        "long_name": "Upwelling LW flux at TOA",
    }

    return LW_arr