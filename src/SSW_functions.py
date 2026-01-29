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
            surface_temperature=atm_profile.t[0],
            max_level_step=None
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

def map_fluxes_on_atm_profile(year, net_thermal, net_total, altitudes):

    timesteps = duration(year)

    net_thermal_37_list = []
    net_total_37_list = []

    for timestep in range(timesteps):
        atm_profile = get_atm(year, timestep)

        thermal = interp1d(
            altitudes[timestep],
            np.log(net_thermal[timestep]),
            kind='linear',
            bounds_error=False,
            fill_value="extrapolate"
        )

        net_thermal_37 = np.exp(thermal(atm_profile.alt))
        net_thermal_37_list.append(net_thermal_37)

        total = interp1d(
            altitudes[timestep],
            np.log(net_total[timestep]),
            kind='linear',
            bounds_error=False,
            fill_value="extrapolate"
        )

        net_total_37 = np.exp(total(atm_profile.alt))
        net_total_37_list.append(net_total_37)

    return net_thermal_37_list, net_total_37_list

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

# def calculate_heating_rate_with_density(solar, thermal, altitude, atm_profile):
#     """
#     Calculate heating rate with altitude-dependent air density.
#     """
#     cp = 1004  # J/(kg·K)
#     seconds_per_day = 86400
#     p_profile_37 = atm_profile.p
#     t_profile_37 = atm_profile.t
#     z_profile_37 = atm_profile.alt
#
#     R = 287  # J/(kg·K)
#     rho = p_profile_37 / (R * t_profile_37)  # kg/m³
#
#     # Calculate net fluxes
#     solar_net = solar.up - solar.down
#     thermal_net = thermal.up - thermal.down
#     net_flux = solar_net + thermal_net
#
#     # Calculate flux divergence (dF/dz)
#     solar_divergence = np.zeros_like(solar_net)
#     thermal_divergence = np.zeros_like(thermal_net)
#     flux_divergence = np.zeros_like(net_flux)
#
#     for i in range(1, len(net_flux) - 1):
#         # Total flux divergence
#         flux_divergence[i] = (net_flux[i + 1] - net_flux[i - 1]) / (altitude[i + 1] - altitude[i - 1])
#         # Solar (shortwave) flux divergence
#         solar_divergence[i] = (solar_net[i + 1] - solar_net[i - 1]) / (altitude[i + 1] - altitude[i - 1])
#         # Thermal (longwave) flux divergence
#         thermal_divergence[i] = (thermal_net[i + 1] - thermal_net[i - 1]) / (altitude[i + 1] - altitude[i - 1])
#
#
#     # Boundary conditions
#     flux_divergence[0] = (net_flux[1] - net_flux[0]) / (altitude[1] - altitude[0])
#     flux_divergence[-1] = (net_flux[-1] - net_flux[-2]) / (altitude[-1] - altitude[-2])
#
#     solar_divergence[0] = (solar_net[1] - solar_net[0]) / (altitude[1] - altitude[0])
#     solar_divergence[-1] = (solar_net[-1] - solar_net[-2]) / (altitude[-1] - altitude[-2])
#
#     thermal_divergence[0] = (thermal_net[1] - thermal_net[0]) / (altitude[1] - altitude[0])
#     thermal_divergence[-1] = (thermal_net[-1] - thermal_net[-2]) / (altitude[-1] - altitude[-2])
#
#     # Calculate heating rates with variable density
#     heating_rate = -(1 / (rho * cp)) * flux_divergence * seconds_per_day
#     sw_heating_rate = -(1 / (rho * cp)) * solar_divergence * seconds_per_day
#     lw_cooling_rate = -(1 / (rho * cp)) * thermal_divergence * seconds_per_day
#
#     return heating_rate, sw_heating_rate, lw_cooling_rate, t_profile_37, p_profile_37, z_profile_37


def calculate_heating_rate_with_density(thermal_net, atm_profile, use_geometric_mean_p=True):
    """
    Calculate heating rate with altitude-dependent air density.

    Parameters:
    -----------
    solar : object with .up and .down attributes
        Solar flux data at layer centers (36 values)
    thermal : object with .up and .down attributes
        Thermal flux data at layer centers (36 values)
    altitude : array-like
        Altitude at layer centers (36 values)
    atm_profile : object
        Atmospheric profile with .p, .t, and .alt attributes at levels (37 values)
    use_geometric_mean_p : bool, optional
        Use geometric mean for pressure (more accurate). Default True.

    Returns:
    --------
    heating_rate : array (36 values)
        Total heating rate at layer centers (K/day)
    sw_heating_rate : array (36 values)
        Shortwave heating rate at layer centers (K/day)
    lw_cooling_rate : array (36 values)
        Longwave cooling rate at layer centers (K/day)
    t_profile_layers : array (36 values)
        Temperature at layer centers (K)
    p_profile_layers : array (36 values)
        Pressure at layer centers (Pa)
    z_profile_layers : array (36 values)
        Altitude at layer centers (m)
    """
    cp = 1004  # J/(kg·K)
    seconds_per_day = 86400
    R = 287  # J/(kg·K)

    # Get level values (37 values)
    p_profile_37 = atm_profile.p
    t_profile_37 = atm_profile.t
    z_profile = atm_profile.alt

    # Calculate density at layer boarders (37 values)
    rho = p_profile_37 / (R * t_profile_37)  # kg/m³

    rho = np.asarray(rho)

    if hasattr(rho, 'values'):
        rho = rho.values

    rho_layer_profile = np.sqrt(rho[:-1] * rho[1:])[::-1]

    print(rho_layer_profile)
    print(len(rho_layer_profile))

    # Calculate flux divergence (dF/dz)
    thermal_divergence = np.zeros_like(thermal_net)


    for i in range(1, len(thermal_net) - 1):
        thermal_divergence[i] = (thermal_net[i + 1] - thermal_net[i-1]) / (z_profile[i + 1] - z_profile[i-1])

    # Boundary conditions
    thermal_divergence[0] = (thermal_net[1] - thermal_net[0]) / (z_profile[1] - z_profile[0])
    thermal_divergence[-1] = (thermal_net[-1] - thermal_net[-2]) / (z_profile[-1] - z_profile[-2])

    # Calculate heating rates with variable density
    lw_cooling_rate = -(1 / (rho * cp)) * thermal_divergence * seconds_per_day

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(thermal_net, z_profile, 'o-')
    axes[0, 0].set_xlabel('Net Flux (W/m²)')
    axes[0, 0].set_ylabel('Altitude (km)')
    axes[0, 0].set_title('Net Flux Profile')
    axes[0, 0].grid(True)

    axes[0, 1].plot(thermal_divergence, z_profile, 'o-')
    axes[0, 1].set_xlabel('Flux Divergence (W/m³)')
    axes[0, 1].set_ylabel('Altitude (km)')
    axes[0, 1].set_title('Flux Divergence')
    axes[0, 1].grid(True)

    axes[1, 0].plot(rho, z_profile, 'o-')
    axes[1, 0].set_xlabel('Density (kg/m³)')
    axes[1, 0].set_ylabel('Altitude (km)')
    axes[1, 0].set_title('Air Density')
    axes[1, 0].grid(True)

    axes[1, 1].plot(lw_cooling_rate, z_profile, 'o-')
    axes[1, 1].set_xlabel('Heating Rate (K/day)')
    axes[1, 1].set_ylabel('Altitude (km)')
    axes[1, 1].set_title('Heating Rate')
    axes[1, 1].grid(True)
    axes[1, 1].axvline(0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

    return lw_cooling_rate, z_profile, t_profile_37, p_profile_37

def calculate_expected_temperature(heating_rates, temperature_profiles, time_step_hours=24):
    """
    Calculate expected temperature profile based on heating rates.

    Parameters:
    -----------
    heating_rates : array-like, shape (n_days, n_levels)
        Heating rates in K/day for each day and altitude level
    temperature_profiles : array-like, shape (n_days, n_levels)
        Temperature profiles in K for each day and altitude level
    time_step_hours : float, optional
        Time step in hours (default: 24 for daily heating rates)

    Returns:
    --------
    expected_temps : array-like, shape (n_days, n_levels)
        Expected temperature profiles based on heating rate integration
    """
    heating_rates = np.array(heating_rates, dtype=object)
    temperature_profiles = np.array(temperature_profiles, dtype=object)

    n_days, n_levels = heating_rates.shape
    expected_temps = np.zeros_like(temperature_profiles)

    # Initialize with the first day's actual temperature
    expected_temps[0] = temperature_profiles[0]

    # Convert time step to fraction of a day
    time_step_days = time_step_hours / 24.0

    # Integrate heating rates forward in time
    for day in range(1, n_days):
        # Apply heating rate from previous day to get next day's temperature
        # dT/dt = heating_rate (in K/day)
        # T(t+dt) = T(t) + heating_rate * dt
        expected_temps[day] = expected_temps[day - 1] + heating_rates[day - 1] * time_step_days

    return expected_temps


def calculate_expected_temperature_with_diagnostics(heating_rates, temperature_profiles,
                                                    time_step_hours=24):
    """
    Calculate expected temperature profile with diagnostic information.

    Parameters:
    -----------
    heating_rates : array-like, shape (n_days, n_levels)
        Heating rates in K/day for each day and altitude level
    temperature_profiles : array-like, shape (n_days, n_levels)
        Actual temperature profiles in K for each day and altitude level
    time_step_hours : float, optional
        Time step in hours (default: 24 for daily heating rates)

    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'expected_temps': Expected temperature profiles
        - 'actual_temps': Actual temperature profiles (for comparison)
        - 'temperature_bias': Difference between expected and actual (expected - actual)
        - 'rmse': Root mean square error for each day
        - 'mean_bias': Mean bias for each day
    """
    heating_rates = np.array(heating_rates)
    temperature_profiles = np.array(temperature_profiles)

    n_days, n_levels = heating_rates.shape
    expected_temps = np.zeros_like(temperature_profiles)

    # Initialize with the first day's actual temperature
    expected_temps[0] = temperature_profiles[0]

    # Convert time step to fraction of a day
    time_step_days = time_step_hours / 24.0

    # Integrate heating rates forward in time
    for day in range(1, n_days):
        expected_temps[day] = expected_temps[day - 1] + heating_rates[day - 1] * time_step_days

    # Calculate diagnostics
    temperature_bias = expected_temps - temperature_profiles

    # RMSE for each day
    rmse = np.sqrt(np.mean(temperature_bias ** 2, axis=1))

    # Mean bias for each day
    mean_bias = np.mean(temperature_bias, axis=1)

    results = {
        'expected_temps': expected_temps,
        'actual_temps': temperature_profiles,
        'temperature_bias': temperature_bias,
        'rmse': rmse,
        'mean_bias': mean_bias
    }

    return results


def calculate_expected_temperature_multilevel(heating_rates, temperature_profiles,
                                              time_step_hours=24, method='forward_euler'):
    """
    Calculate expected temperature with different integration methods.

    Parameters:
    -----------
    heating_rates : array-like, shape (n_days, n_levels)
        Heating rates in K/day for each day and altitude level
    temperature_profiles : array-like, shape (n_days, n_levels)
        Temperature profiles in K for each day and altitude level
    time_step_hours : float, optional
        Time step in hours (default: 24 for daily heating rates)
    method : str, optional
        Integration method: 'forward_euler', 'backward_euler', or 'trapezoidal'

    Returns:
    --------
    expected_temps : array-like, shape (n_days, n_levels)
        Expected temperature profiles
    """
    heating_rates = np.array(heating_rates)
    temperature_profiles = np.array(temperature_profiles)

    n_days, n_levels = heating_rates.shape
    expected_temps = np.zeros_like(temperature_profiles)

    # Initialize with the first day's actual temperature
    expected_temps[0] = temperature_profiles[0]

    # Convert time step to fraction of a day
    dt = time_step_hours / 24.0

    if method == 'forward_euler':
        # Forward Euler: T(n+1) = T(n) + HR(n) * dt
        for day in range(1, n_days):
            expected_temps[day] = expected_temps[day - 1] + heating_rates[day - 1] * dt

    elif method == 'backward_euler':
        # Backward Euler: T(n+1) = T(n) + HR(n+1) * dt
        for day in range(1, n_days):
            expected_temps[day] = expected_temps[day - 1] + heating_rates[day] * dt

    elif method == 'trapezoidal':
        # Trapezoidal rule: T(n+1) = T(n) + (HR(n) + HR(n+1)) / 2 * dt
        for day in range(1, n_days):
            expected_temps[day] = (expected_temps[day - 1] +
                                   0.5 * (heating_rates[day - 1] + heating_rates[day]) * dt)
    else:
        raise ValueError(f"Unknown method: {method}")

    return expected_temps