# imports

import os

import numpy as np
import matplotlib.pyplot as plt
import pyarts3
from scipy.interpolate import interp1d
import xarray as xr
import pyarts3 as pyarts
import netCDF4 as netcdf4
import typhon as ty

from src.atm_flux_recipe_mod import AtmosphericFlux


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
    x_profile = ty.physics.relative_humidity2vmr(ds.rh.isel(time=timestep).data / 1e2, ds.pressure.data * 100, ds.t.isel(time=timestep).data)

    # convert mass mixing ratios to VMR:
    co2_profile = ty.physics.mixing_ratio2vmr(ds.co2.isel(time=timestep).data)
    no2_profile = ty.physics.mixing_ratio2vmr(ds.no2.isel(time=timestep).data)
    no_profile = ty.physics.mixing_ratio2vmr(ds.no.isel(time=timestep).data)
    o3_profile = ty.physics.mixing_ratio2vmr(ds.o3.isel(time=timestep).data)

    # convert pressure from hPa to Pa:
    p_profile = ds.pressure.data * 100

    # convert pressure to height:
    z_profile = ty.physics.pressure2height(p_profile, ds.t.isel(time=timestep).data)

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

def duration(year):
    """
    Calculates the duration of the SSW event in a specific year.

    Inputs:
    ----------
    year: year of SSW event

    Returns:
    ----------
    duration_ssw: duration of the SSW event in days as integer
    """
    ds = xr.open_dataset(f"data/{year}_data.nc", engine="netcdf4")
    duration_ssw = len(ds.time)
    return duration_ssw

def calculate_fluxes(year, species_list=["H2O-161", "O2-66", "N2-44", "CO2-626", "O3-XFIT"],
                     force_recalculate=False, output_dir='flux_data'):
    """
    Calculate or load fluxes for all timesteps in a year.

    Parameters:
    -----------
    year : int
        Year to calculate fluxes for
    species_list : list
        List of species to include
    force_recalculate : bool
        Force recalculation even if saved file exists (default: False)
    output_dir : str
        Directory to save/load files (default: 'flux_data')

    Returns:
    --------
    solar_fluxes, thermal_fluxes, altitudes, net_thermal, net_solar, net_total
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'fluxes_{year}.npz')

    # Prüfe ob Datei existiert und lade sie
    if os.path.exists(filename) and not force_recalculate:
        print(f"Loading fluxes from {filename}")
        data = np.load(filename, allow_pickle=True)

        # Rekonstruiere die Flux-Objekte
        solar_fluxes = [
            type('Flux', (), {
                'up': data['solar_up'][i],
                'direct_down': data['solar_direct_down'][i],
                'diffuse_down': data['solar_diffuse_down'][i],
                'down': data['solar_direct_down'][i] + data['solar_diffuse_down'][i]
            })()
            for i in range(len(data['solar_up']))
        ]

        thermal_fluxes = [
            type('Flux', (), {
                'up': data['thermal_up'][i],
                'down': data['thermal_down'][i]
            })()
            for i in range(len(data['thermal_up']))
        ]

        altitudes = list(data['altitudes'])
        net_thermal = list(data['net_thermal'])
        net_solar = list(data['net_solar'])
        net_total = list(data['net_total'])

        return solar_fluxes, thermal_fluxes, altitudes, net_thermal, net_solar, net_total

    # Sonst berechne neu
    print(f"Calculating fluxes for year {year}")
    pyarts.data.download()
    num_timesteps = duration(year)

    fop = AtmosphericFlux(
        species=species_list,
        remove_lines_percentile={"H2O": 70},
    )

    solar_fluxes = []
    thermal_fluxes = []
    altitudes = []
    net_thermal = []
    net_solar = []
    net_total = []

    for timestep in range(num_timesteps):
        atm_profile = get_atm(year, timestep)

        solar, thermal, altitude = fop(
            atmospheric_profile=atm_profile,
            surface_temperature=atm_profile.t[0]
        )
        net_lw = thermal.up - thermal.down
        net_sw = solar.up - solar.down
        net_flux = net_lw + net_sw

        solar_fluxes.append(solar)
        thermal_fluxes.append(thermal)
        altitudes.append(altitude)
        net_thermal.append(net_lw)
        net_solar.append(net_sw)
        net_total.append(net_flux)

    # Speichern als Object Arrays (erlaubt unterschiedliche Längen)
    # Verwende dtype=object für Arrays mit unterschiedlichen Längen
    solar_up_array = np.empty(len(solar_fluxes), dtype=object)
    solar_direct_down_array = np.empty(len(solar_fluxes), dtype=object)
    solar_diffuse_down_array = np.empty(len(solar_fluxes), dtype=object)
    thermal_up_array = np.empty(len(thermal_fluxes), dtype=object)
    thermal_down_array = np.empty(len(thermal_fluxes), dtype=object)
    altitudes_array = np.empty(len(altitudes), dtype=object)
    net_thermal_array = np.empty(len(net_thermal), dtype=object)
    net_solar_array = np.empty(len(net_solar), dtype=object)
    net_total_array = np.empty(len(net_total), dtype=object)

    for i in range(len(solar_fluxes)):
        solar_up_array[i] = solar_fluxes[i].up
        solar_direct_down_array[i] = solar_fluxes[i].direct_down
        solar_diffuse_down_array[i] = solar_fluxes[i].diffuse_down
        thermal_up_array[i] = thermal_fluxes[i].up
        thermal_down_array[i] = thermal_fluxes[i].down
        altitudes_array[i] = altitudes[i]
        net_thermal_array[i] = net_thermal[i]
        net_solar_array[i] = net_solar[i]
        net_total_array[i] = net_total[i]

    np.savez(filename,
             solar_up=solar_up_array,
             solar_direct_down=solar_direct_down_array,
             solar_diffuse_down=solar_diffuse_down_array,
             thermal_up=thermal_up_array,
             thermal_down=thermal_down_array,
             altitudes=altitudes_array,
             net_thermal=net_thermal_array,
             net_solar=net_solar_array,
             net_total=net_total_array,
             year=year,
             num_timesteps=num_timesteps)

    print(f"Fluxes saved to {filename}")

    return solar_fluxes, thermal_fluxes, altitudes, net_thermal, net_solar, net_total

def observed_temperature(year, pressure_level_strat):
    """
    Extracts the observed temperature evolution with time of an SSW event in a specific year.

    Inputs:
    ----------
    year: year of SSW event
    duration_of_SSW: duration of the SSW event in days as integer
    pressure_level_strat: pressure level where the temperature evolution should be extracted

    Returns:
    ----------
    temp_strat: observed temperature evolution as a numpy array
    pressure_level: pressure level where the temperature evolution should be extracted
    """
    duration_of_SSW = duration(year)
    temp_strat = []
    pressure_level = 0.
    for timestep in range(duration_of_SSW):
        atmosphere = get_atm(year, timestep)
        pressure_level =+ atmosphere.p[pressure_level_strat]
        temp_strat.append(atmosphere.t[pressure_level_strat])
    return np.array(temp_strat), pressure_level.item()

# def calculate_heating_rate(solar, thermal, altitude):
#     """
#     Calculate heating rate in K/day from fluxes using altitude gradient.
#
#     Parameters:
#     -----------
#     solar : Flux object with .up, .direct_down, .diffuse_down
#     thermal : Flux object with .up and .diffuse_down (or .down)
#     altitude : array of altitude levels in meters
#
#     Returns:
#     --------
#     heating_rate_net, heating_rate_thermal, heating_rate_solar : arrays of heating rates in K/day
#     """
#
#     def compute_heating_rate_from_flux(flux, altitude, rho, cp, seconds_per_day):
#         """Helper function to compute heating rate from a flux array."""
#         flux_divergence = np.zeros_like(flux)
#
#         # Central differences for interior points
#         for i in range(1, len(flux) - 1):
#             flux_divergence[i] = (flux[i + 1] - flux[i - 1]) / (altitude[i + 1] - altitude[i - 1])
#
#         # Boundary conditions
#         flux_divergence[0] = (flux[1] - flux[0]) / (altitude[1] - altitude[0])
#         flux_divergence[-1] = (flux[-1] - flux[-2]) / (altitude[-1] - altitude[-2])
#
#         # Calculate heating rate
#         return -(1 / (rho * cp)) * flux_divergence * seconds_per_day
#
#     # Constants
#     cp = 1004  # J/(kg·K)
#     rho = 1.225  # kg/m³
#     seconds_per_day = 86400
#
#     # Calculate net fluxes at each level (W/m²)
#     solar_net = solar.up - solar.direct_down - solar.diffuse_down
#     thermal_net = thermal.up - thermal.down
#     net_flux = solar_net + thermal_net
#
#     # Calculate heating rates for all flux types
#     heating_rate_net = compute_heating_rate_from_flux(net_flux, altitude, rho, cp, seconds_per_day)
#     heating_rate_thermal = compute_heating_rate_from_flux(thermal_net, altitude, rho, cp, seconds_per_day)
#     heating_rate_solar = compute_heating_rate_from_flux(solar_net, altitude, rho, cp, seconds_per_day)
#
#     return heating_rate_net, heating_rate_thermal, heating_rate_solar


def calculate_heating_rate_with_density(solar, thermal, altitude, atm_profile):
    """
    Calculate heating rate with altitude-dependent air density.
    """
    cp = 1004  # J/(kg·K)
    seconds_per_day = 86400
    p_profile_37 = atm_profile.p
    t_profile_37 = atm_profile.t
    z_profile_37 = atm_profile.alt

    f_logp = interp1d(
        z_profile_37,
        np.log(p_profile_37),
        kind="linear",
        bounds_error=False,
        fill_value='extrapolate',
    )
    p_profile = np.exp(f_logp(altitude))

    f_t = interp1d(
        z_profile_37,
        t_profile_37,
        kind="linear",
        bounds_error=False,
        fill_value='extrapolate',
    )
    t_profile = f_t(altitude)

    # Get density from atmospheric profile (pressure and temperature)
    # rho = p / (R * T), where R = 287 J/(kg·K) for dry air
    R = 287  # J/(kg·K)
    rho = p_profile / (R * t_profile)  # kg/m³

    # Calculate net fluxes
    solar_net = solar.up - solar.down
    thermal_net = thermal.up - thermal.down
    net_flux = solar_net + thermal_net

    # Calculate flux divergence (dF/dz)
    solar_divergence = np.zeros_like(solar_net)
    thermal_divergence = np.zeros_like(thermal_net)
    flux_divergence = np.zeros_like(net_flux)

    for i in range(1, len(net_flux) - 1):
        # Total flux divergence
        flux_divergence[i] = (net_flux[i + 1] - net_flux[i - 1]) / (altitude[i + 1] - altitude[i - 1])
        # Solar (shortwave) flux divergence
        solar_divergence[i] = (solar_net[i + 1] - solar_net[i - 1]) / (altitude[i + 1] - altitude[i - 1])
        # Thermal (longwave) flux divergence
        thermal_divergence[i] = (thermal_net[i + 1] - thermal_net[i - 1]) / (altitude[i + 1] - altitude[i - 1])

    # Boundary conditions
    flux_divergence[0] = (net_flux[1] - net_flux[0]) / (altitude[1] - altitude[0])
    flux_divergence[-1] = (net_flux[-1] - net_flux[-2]) / (altitude[-1] - altitude[-2])

    solar_divergence[0] = (solar_net[1] - solar_net[0]) / (altitude[1] - altitude[0])
    solar_divergence[-1] = (solar_net[-1] - solar_net[-2]) / (altitude[-1] - altitude[-2])

    thermal_divergence[0] = (thermal_net[1] - thermal_net[0]) / (altitude[1] - altitude[0])
    thermal_divergence[-1] = (thermal_net[-1] - thermal_net[-2]) / (altitude[-1] - altitude[-2])

    # Calculate heating rates with variable density
    heating_rate = -(1 / (rho * cp)) * flux_divergence * seconds_per_day
    sw_heating_rate = -(1 / (rho * cp)) * solar_divergence * seconds_per_day
    lw_cooling_rate = -(1 / (rho * cp)) * thermal_divergence * seconds_per_day

    return heating_rate, sw_heating_rate, lw_cooling_rate
