#%% imports

import os

import matplotlib.pyplot as plt
import netCDF4 as netcdf4
import numpy as np
import pyarts3 as pyarts
import typhon as ty
import xarray as xr

import src.SSW_functions as ssw

#%%

year = 2006
length = ssw.duration(year)
timesteps = [0, 5, 10, 15, 20, 26]
all_fluxes = {'solar': [], 'thermal': [], 'altitude': [], 'net_thermal': [], 'net_solar': [], 'net_total': []}
K_per_day = np.linspace(-2.5, 2.5, 100)

# %%

solar_2006, thermal_2006, altitude_2006, net_thermal_2006, net_solar_2006, net_total_2006 = ssw.calculate_fluxes(year)



# %%

net_heating_rates_list = []
thermal_rates_list = []
solar_rates_list = []


for timestep in range(length):
    heating_rate_net, heating_rate_thermal, heating_rate_solar = ssw.calculate_heating_rate(solar_2006[timestep],
                                                                                            thermal_2006[timestep],
                                                                                            altitude_2006[timestep])
    net_heating_rates_list.append(heating_rate_net)
    thermal_rates_list.append(heating_rate_thermal)
    solar_rates_list.append(heating_rate_solar)

# %%

fig, axes = plt.subplots(1, len(timesteps), figsize=(24, 10))

for i, (ax, day) in enumerate(zip(axes, timesteps)):
    ax.plot(net_heating_rates_list[day], altitude_2006[0] / 1e3)
    ax.plot(thermal_rates_list[day], altitude_2006[0] / 1e3)
    ax.plot(solar_rates_list[day], altitude_2006[0] / 1e3)
    ax.legend(["net heating rate", "LW cooling rate", "SW heating rate"])
    ax.set_ylabel("altitude [km]")
    ax.set_xlabel("Heating Rate [K/day]")
    ax.set_title(f"Heating Rate 2006 day {day}")
    ax.axvline(x=0, color="grey", linestyle="--")

plt.tight_layout()
plt.show()