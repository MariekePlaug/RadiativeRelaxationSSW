# %%
import os

import matplotlib.pyplot as plt
import pyarts3 as pyarts
import numpy as np
from src.atm_flux_recipe_mod import AtmosphericFlux

# Download catalogs
pyarts.data.download()
from src.SSW_functions import get_atm, rad_heating_rate, duration, observed_temperature

# %% plot observed temperature evolution in the atmosphere:
temp_obs_test, pressure_strat = observed_temperature(2010, duration(2010), 23)

plt.plot(range(0,duration(2010)), temp_obs_test)
plt.xlabel("Day after wind reversal")
plt.ylabel("Temperature [K]")
plt.title(f"observed temperature evolution \n at pressure level of {pressure_strat/100} hPa")
plt.show()


# %% calculate expected temperature evolution from heating rate:
fop = AtmosphericFlux(
    species=["H2O-161", "O2-66", "N2-44", "CO2-626", "O3-XFIT", "NO2", "NO", ],
    remove_lines_percentile={"H2O": 70},
)
fluxes = []
initial_temp = []
heating_rates = []
temp_exp = initial_temp
for timestep in range(duration(2010)):
    atm_profile = get_atm(2010, timestep)
    if timestep == 0:
        initial_temp.append(atm_profile.t)
    solar, thermal, altitude = fop(atmospheric_profile=atm_profile)

    net_lw = thermal.down - thermal.up
    net_sw = solar.down - solar.up
    net_flux = net_lw + net_sw
    fluxes.append(net_flux)

    heating_rate = rad_heating_rate(altitude_vec=altitude, flux_vec=net_flux)
    heating_rates.append(heating_rate)

    temperature = temp_exp[-1] + heating_rates[timestep-1]
    temp_exp.append(temperature)


print(len(temp_exp))