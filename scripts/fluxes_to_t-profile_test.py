#%% imports

import os
import matplotlib.pyplot as plt
import netCDF4 as netcdf4
import numpy as np
import pyarts3 as pyarts
import typhon as ty
import xarray as xr
from fontTools.ttLib.tables.ttProgram import tt_instructions_error

import src.SSW_functions as ssw
from scripts.test import atm_profile

# %% constants and arrays

years = [2004, 2006, 2007, 2008, 2009, 2010, 2013]
year = 2007
length = ssw.duration(year)
timesteps = np.linspace(0, length-1, length, dtype=int)
K_per_day = np.linspace(-2.5, 2.5, 100)


# %% calculate fluxes

solar_2007, thermal_2007, altitude_2007, net_thermal_2007, net_solar_2007, net_total_2007 = ssw.calculate_fluxes(year)

# %%

print(len(net_solar_2007[0]))

# %% map to 37 levels

thermal_37, total_37 = ssw.map_fluxes_on_atm_profile(year, net_thermal_2007, net_total_2007, altitude_2007)


# %% plot fluxes

for timestep in timesteps:

    atm_profile = ssw.get_atm(year, timestep)
    solar = solar_2007[timestep]
    thermal = thermal_2007[timestep]
    altitude = altitude_2007[timestep]
    net_sw = net_solar_2007[timestep]
    net_lw = net_thermal_2007[timestep]
    net_flux = net_total_2007[timestep]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    ax1.plot(solar.up, altitude / 1e3)
    ax1.plot(solar.down, altitude / 1e3)
    ax1.plot(net_sw, altitude / 1e3)
    ax1.legend(["up", "down", "net"])
    ax1.set_ylabel("Altitude [km]")
    ax1.set_xlabel("Flux [W / m$^2$]")
    ax1.set_title(f"Solar flux - Day {timestep}")

    ax2.plot(thermal.up, altitude / 1e3)
    ax2.plot(thermal.down, altitude / 1e3)
    ax2.plot(net_lw, altitude / 1e3)
    ax2.plot(thermal_37[timestep], atm_profile.alt / 1e3, linestyle="dashed")
    ax2.legend(["up", "down", "net", "mapped"])
    ax2.set_ylabel("Altitude [km]")
    ax2.set_xlabel("Flux [W / m$^2$]")
    ax2.set_title(f"Thermal flux - Day {timestep}")

    ax3.plot(net_sw, altitude / 1e3)
    ax3.plot(net_lw, altitude / 1e3)
    ax3.plot(net_flux, altitude / 1e3)
    ax3.legend(["net_sw", "net_lw", "net"])
    ax3.set_ylabel("Altitude [km]")
    ax3.set_xlabel("Flux [W / m$^2$]")
    ax3.set_title(f"Total flux - Day {timestep}")

    plt.tight_layout()
    plt.show()
# %% calculate heating rates

lw_cooling_rates_list = []
z_profiles_list = []
t_profiles_list = []
p_profiles_list = []

for timestep in range(length):
    lw_cooling_rate, z_profile, t_profile, p_profile = ssw.calculate_heating_rate_with_density(
        thermal_37[timestep],
        atm_profile=ssw.get_atm(year, timestep)
    )

    lw_cooling_rates_list.append(lw_cooling_rate)
    z_profiles_list.append(z_profile)
    t_profiles_list.append(t_profile)
    p_profiles_list.append(p_profile)

# %% plot heating rates for all days

fig, axes = plt.subplots(1, len(timesteps), figsize=(30, 6))

for i, (ax, day) in enumerate(zip(axes, timesteps)):
    ax.plot(lw_cooling_rates_list[day], z_profiles_list[day] / 1e3)
    ax.legend(["LW cooling rate"])
    ax.set_ylabel("altitude [km]")
    ax.set_xlabel("Heating Rate [K/day]")
    ax.set_title(f"Heating Rate {year} day {day}")
    ax.axvline(x=0, color="grey", linestyle="--")

plt.tight_layout()
plt.show()

# %% calculate expected temperatures

expected_temps_test = ssw.calculate_expected_temperature(
    lw_cooling_rates_list,
    t_profiles_list,
)

# %%

print(expected_temps_test)

fig, ax = plt.subplots(1, figsize=(8, 12))

ax.plot(expected_temps_test[0], p_profiles_list[0] / 1e2)
ax.plot(expected_temps_test[1], p_profiles_list[1] / 1e2)
ax.plot(expected_temps_test[2], p_profiles_list[2] / 1e2)
ax.plot(expected_temps_test[3], p_profiles_list[3] / 1e2)
ax.plot(expected_temps_test[4], p_profiles_list[4] / 1e2)
ax.plot(expected_temps_test[5], p_profiles_list[5] / 1e2)
ax.set_yscale("log")
ax.set_ylim(ax.get_ylim()[::-1])
ax.legend(["day 0", "day 1", "day 2", "day 3", "day 4", "day 5"])
ax.set_xlabel("Temperature [K]")
ax.set_ylabel("pressure [hPa")
ax.set_title(f"Expedcted temperature profile from heating rates")

plt.tight_layout()
plt.show()

# %%
print(p_profiles_list[0])