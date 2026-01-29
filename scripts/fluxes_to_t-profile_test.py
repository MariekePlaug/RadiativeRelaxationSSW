#%% imports

import os
import matplotlib.pyplot as plt
import netCDF4 as netcdf4
import numpy as np
import pyarts3 as pyarts
import typhon as ty
import xarray as xr

import src.SSW_functions as ssw


# %% constants and arrays

year = 2007
length = ssw.duration(year)
timesteps = [0, 5]
K_per_day = np.linspace(-2.5, 2.5, 100)
print(length)
# %% calculate fluxes

solar_2007, thermal_2007, altitude_2007, net_thermal_2007, net_solar_2007, net_total_2007 = ssw.calculate_fluxes(year)

# %% plot fluxes

for timestep in timesteps:
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
    ax2.legend(["up", "down", "net"])
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

net_heating_rates_list = []
thermal_rates_list = []
solar_rates_list = []


for timestep in range(length):
    heating_rate_net, heating_rate_thermal, heating_rate_solar = ssw.calculate_heating_rate(solar_2007[timestep],
                                                                                            thermal_2007[timestep],
                                                                                            altitude_2007[timestep])
    net_heating_rates_list.append(heating_rate_net)
    thermal_rates_list.append(heating_rate_thermal)
    solar_rates_list.append(heating_rate_solar)

# %%

fig, axes = plt.subplots(1, len(timesteps), figsize=(10, 6))

for i, (ax, day) in enumerate(zip(axes, timesteps)):
    ax.plot(net_heating_rates_list[day], altitude_2007[day] / 1e3)
    ax.plot(thermal_rates_list[day], altitude_2007[day] / 1e3)
    ax.plot(solar_rates_list[day], altitude_2007[day] / 1e3)
    ax.legend(["net heating rate", "LW cooling rate", "SW heating rate"])
    ax.set_ylabel("altitude [km]")
    ax.set_xlabel("Heating Rate [K/day]")
    ax.set_title(f"Heating Rate {year} day {day}")
    ax.axvline(x=0, color="grey", linestyle="--")

plt.tight_layout()
plt.show()

# %%

heating_rate_test = ssw.calculate_heating_rate_with_density(
    solar_2007[0],
    thermal_2007[0],
    altitude_2007[0],
    atm_profile=ssw.get_atm(2007,0))

# %% plot heating rates

fig, ax = plt.subplots(1, 1, figsize=(5, 6))
ax.plot(heating_rate_test, altitude_2007[0] / 1e3)
ax.axvline(x = 0, color = "black", linestyle = "dashed")
ax.legend(["heating rate"])
ax.set_ylabel("Altitude [km]")
ax.set_xlabel("heating rate [K / dt]")
ax.set_title("Heating rate")
plt.show()