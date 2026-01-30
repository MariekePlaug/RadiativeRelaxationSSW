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
temp_obs_test, pressure_strat = observed_temperature(2006, duration(2006), 23)

plt.plot(range(0,duration(2006)), temp_obs_test)
plt.xlabel("Day after wind reversal")
plt.ylabel("Temperature [K]")
plt.title(f"observed temperature evolution \n at pressure level of {pressure_strat/100} hPa")
plt.show()


# %% calculate expected temperature evolution from heating rate:
fop = AtmosphericFlux(
    species=["H2O-161", "O2-66", "N2-44", "CO2-626", "O3-XFIT", "NO2", "NO", ],
    remove_lines_percentile={"H2O": 70},
)
print(duration(2006))
fluxes = []
initial_temp = []
heating_rates = []
temp_exp = []
delta_T = []
for timestep in range(0, duration(2006)-1):
    atm_profile = get_atm(2006, timestep)
    if timestep == 0:
        temp_exp.append(atm_profile.t[23])

    solar, thermal, altitude = fop(atmospheric_profile=atm_profile)

    net_lw = thermal.down - thermal.up
    net_sw = solar.down - solar.up
    net_flux = net_lw + net_sw
    fluxes.append(net_flux[34])

    heating_rate = rad_heating_rate(altitude_vec=altitude, flux_vec=net_flux)
    heating_rates.append(heating_rate[34])

    dT = heating_rate[34] * 86400
    delta_T.append(dT)

    temperature = temp_exp[timestep - 1] + delta_T[timestep - 1]
    temp_exp.append(temperature)


print(len(temp_exp))

# %% plot expected and observed temperature
fig, ax = plt.subplots(1,1, figsize = (8,6))
ax.plot(range(0,duration(2006)), temp_exp, color = "red", label = "temperature evolution from fluxes")
ax.plot(range(0,duration(2006)), temp_obs_test, color = "blue", label = "observed temperature evolution")
ax.set_xlabel("Day after wind reversal")
ax.set_ylabel("Temperature [K]")
ax.set_title("Observed temperature and temperature calculated from fluxes")
ax.legend()
plt.show()
