"""Atmospheric flux operator"""

# %%
import os
import typhon as ty
import matplotlib.pyplot as plt
import pyarts3 as pyarts
import numpy as np
from scipy.interpolate import interp1d

from src.atm_flux_recipe_mod import AtmosphericFlux

# Download catalogs
pyarts.data.download()
from src.SSW_functions import get_atm, rad_heating_rate


# %% Initialize the operator
fop = AtmosphericFlux(
    species=["H2O-161", "O2-66", "N2-44", "CO2-626", "O3-XFIT", "NO2", "NO", ],
    remove_lines_percentile={"H2O": 70},
)


# %% Get the atmosphere (optional)
# The atmosphere is the full atmospheric field of ARTS as a dictionary,
# which is likely more than you wish to change.  You may change only part
# of the atmosphere by simply creating a dictionary that only contains the
# fields that you want to change.

atm_profile = get_atm(2006, 26)
print(np.where(atm_profile.p == 17500.))
print(atm_profile.alt[23])
print(atm_profile.t[0])
# %% Get the profile flux for the given `atm`
# Passing `atm` is optional, if not passed the operator will use the current atmosphere,
# which is the atmosphere that was set with the last call to `__call__`, or the constructor
# default if no call to `__call__` has been made.

solar, thermal, altitude = fop(atmospheric_profile=atm_profile, surface_temperature=atm_profile.t[0])


# %% net fluxes
net_lw = thermal.down - thermal.up
net_sw = solar.down - solar.up
net_flux = net_lw + net_sw
print(net_flux.shape)

# %% Plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

ax1.plot(solar.up, altitude / 1e3)
ax1.plot(solar.down, altitude / 1e3)
ax1.plot(net_sw, altitude / 1e3)
ax1.legend(["up", "down", "net"])
ax1.set_ylabel("Altitude [km]")
ax1.set_xlabel("Flux [W / m$^2$]")
ax1.set_title("Solar flux")

ax2.plot(thermal.up, altitude / 1e3)
ax2.plot(thermal.down, altitude / 1e3)
ax2.plot(net_lw, altitude / 1e3)
ax2.legend(["up", "down", "net"])
ax2.set_ylabel("Altitude [km]")
ax2.set_xlabel("Flux [W / m$^2$]")
ax2.set_title("Thermal flux")

ax3.plot(net_sw, altitude / 1e3)
ax3.plot(net_lw, altitude / 1e3)
ax3.plot(net_flux, altitude / 1e3)
ax3.legend(["net_sw", "net_lw", "net"])
ax3.set_ylabel("Altitude [km]")
ax3.set_xlabel("Flux [W / m$^2$]")
ax3.set_title("Total flux")

if "ARTS_HEADLESS" not in os.environ:
    plt.show()

# %% calculate heating / cooling rates
heating_rate = rad_heating_rate(altitude_vec=altitude, flux_vec = net_flux)

# %% plot heating rate:
fig, ax = plt.subplots(1, 1, figsize=(5, 6))
ax.plot(heating_rate, altitude[:-1] / 1e3)
ax.axvline(x = 0, color = "black", linestyle = "dashed")
ax.legend(["heating rate"])
ax.set_ylabel("Altitude [km]")
ax.set_xlabel("heating rate [K / dt]")
ax.set_title("Heating rate")
plt.show()