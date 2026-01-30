# %% imports

import os
import matplotlib.pyplot as plt
import netCDF4 as netcdf4
import numpy as np
import pyarts3 as pyarts
import typhon as ty
import xarray as xr
from fontTools.ttLib.tables.ttProgram import tt_instructions_error

import src.SSW_functions as ssw

output_dir = "output_plots"
os.makedirs(output_dir, exist_ok=True)


# %% constants and arrays

years = [2004, 2006, 2007, 2008, 2009, 2010, 2013]
K_per_day = np.linspace(-2.5, 2.5, 100)

# %% Process all years

# Dictionaries to store results for all years
all_solar = {}
all_thermal = {}
all_altitude = {}
all_net_thermal = {}
all_net_solar = {}
all_net_total = {}
all_thermal_37 = {}
all_total_37 = {}
all_lw_cooling_rates = {}
all_z_profiles = {}
all_t_profiles = {}
all_p_profiles = {}
all_expected_temps = {}

for year in years:
    print(f"\n{'=' * 60}")
    print(f"Processing year {year}")
    print(f"{'=' * 60}")

    length = ssw.duration(year)
    timesteps = np.linspace(0, length - 1, length, dtype=int)

    # Calculate fluxes
    print(f"Calculating fluxes for {year}...")
    solar, thermal, altitude, net_thermal, net_solar, net_total = ssw.calculate_fluxes(year)

    all_solar[year] = solar
    all_thermal[year] = thermal
    all_altitude[year] = altitude
    all_net_thermal[year] = net_thermal
    all_net_solar[year] = net_solar
    all_net_total[year] = net_total

    print(f"Flux array length for day 0: {len(net_solar[0])}")

    # Map to 37 levels
    print(f"Mapping fluxes to 37 levels...")
    thermal_37, total_37 = ssw.map_fluxes_on_atm_profile(year, net_thermal, net_total, altitude)

    all_thermal_37[year] = thermal_37
    all_total_37[year] = total_37

    # Calculate heating rates
    print(f"Calculating heating rates...")
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

    all_lw_cooling_rates[year] = lw_cooling_rates_list
    all_z_profiles[year] = z_profiles_list
    all_t_profiles[year] = t_profiles_list
    all_p_profiles[year] = p_profiles_list

    # Calculate expected temperatures
    print(f"Calculating expected temperatures...")
    expected_temps = ssw.calculate_expected_temperature(
        lw_cooling_rates_list,
        t_profiles_list,
    )

    all_expected_temps[year] = expected_temps

    print(f"Completed processing for {year}")

print(f"\n{'=' * 60}")
print("All years processed successfully!")
print(f"{'=' * 60}\n")

# %% Plot fluxes for all years

for year in years:
    length = ssw.duration(year)
    timesteps = np.linspace(0, length - 1, length, dtype=int)

    print(f"Plotting fluxes for {year}...")

    for timestep in timesteps:
        atm_profile = ssw.get_atm(year, timestep)
        solar = all_solar[year][timestep]
        thermal = all_thermal[year][timestep]
        altitude = all_altitude[year][timestep]
        net_sw = all_net_solar[year][timestep]
        net_lw = all_net_thermal[year][timestep]
        net_flux = all_net_total[year][timestep]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

        ax1.plot(solar.up, altitude / 1e3)
        ax1.plot(solar.down, altitude / 1e3)
        ax1.plot(net_sw, altitude / 1e3)
        ax1.legend(["up", "down", "net"])
        ax1.set_ylabel("Altitude [km]")
        ax1.set_xlabel("Flux [W / m$^2$]")
        ax1.set_title(f"Solar flux - {year} Day {timestep}")

        ax2.plot(thermal.up, altitude / 1e3)
        ax2.plot(thermal.down, altitude / 1e3)
        ax2.plot(net_lw, altitude / 1e3)
        ax2.plot(all_thermal_37[year][timestep], atm_profile.alt / 1e3, linestyle="dashed")
        ax2.legend(["up", "down", "net", "mapped"])
        ax2.set_ylabel("Altitude [km]")
        ax2.set_xlabel("Flux [W / m$^2$]")
        ax2.set_title(f"Thermal flux - {year} Day {timestep}")

        ax3.plot(net_sw, altitude / 1e3)
        ax3.plot(net_lw, altitude / 1e3)
        ax3.plot(net_flux, altitude / 1e3)
        ax3.legend(["net_sw", "net_lw", "net"])
        ax3.set_ylabel("Altitude [km]")
        ax3.set_xlabel("Flux [W / m$^2$]")
        ax3.set_title(f"Total flux - {year} Day {timestep}")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/fluxes_{year}_day{timestep}.png", dpi=150, bbox_inches='tight')
        plt.close()

# %% Plot heating rates for all years with shared axes - Save heating rates as individual PDFs

import os

output_dir = "heating_rate_plots"
os.makedirs(output_dir, exist_ok=True)

for year in years:
    length = ssw.duration(year)
    timesteps = np.linspace(0, length - 1, length, dtype=int)

    print(f"Saving heating rate plots for {year}...")

    # Create year subdirectory
    year_dir = os.path.join(output_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)

    for day in timesteps:
        fig, ax = plt.subplots(1, 1, figsize=(7, 9))

        ax.plot(all_lw_cooling_rates[year][day], all_z_profiles[year][day] / 1e3,
                linewidth=3, color='C0', marker='o', markersize=5,
                markerfacecolor='white', markeredgewidth=1.5)
        ax.axvline(x=0, color='grey', linestyle='--', alpha=0.6, linewidth=2)
        ax.set_xlabel("Heating Rate [K/day]", fontsize=14, fontweight='bold')
        ax.set_ylabel("Altitude [km]", fontsize=14, fontweight='bold')
        ax.set_title(f"LW Cooling Rate - Year {year}, Day {day}",
                     fontsize=16, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.4, linewidth=0.8)
        ax.set_ylim(0, 50)
        ax.tick_params(labelsize=12)

        # Add text box with statistics
        hr_mean = np.mean(all_lw_cooling_rates[year][day])
        hr_max = np.max(all_lw_cooling_rates[year][day])
        hr_min = np.min(all_lw_cooling_rates[year][day])
        textstr = f'Max: {hr_max:.2f} K/day\nMin: {hr_min:.2f} K/day'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                                                   facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(year_dir, f"heating_rate_day{day:02d}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved {len(timesteps)} plots to {year_dir}")

# %% Plot expected temperatures for all years

for year in years:
    expected_temps = all_expected_temps[year]
    p_profiles = all_p_profiles[year]
    n_days = len(expected_temps)

    print(f"Plotting expected temperatures for {year}...")

    fig, ax = plt.subplots(1, figsize=(8, 12))

    for i in range(n_days):
        ax.plot(expected_temps[i], p_profiles[i] / 1e2, label=f"day {i}")

    ax.set_yscale("log")
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.legend()
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [hPa]")
    ax.set_title(f"Expected temperature profile from heating rates in {year}")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/expected_temps_{year}.png", dpi=150, bbox_inches='tight')
    plt.close()

# %% Comparison plot - Expected temperatures for all years

fig, axes = plt.subplots(2, 4, figsize=(20, 16))
axes = axes.flatten()

for idx, year in enumerate(years):
    ax = axes[idx]
    expected_temps = all_expected_temps[year]
    p_profiles = all_p_profiles[year]
    n_days = len(expected_temps)

    for i in range(n_days):
        ax.plot(expected_temps[i], p_profiles[i] / 1e2, label=f"day {i}")

    ax.set_yscale("log")
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.legend(fontsize=8)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [hPa]")
    ax.set_title(f"Year {year}")
    ax.grid(True, alpha=0.3)

# Hide the last empty subplot
axes[-1].axis('off')

plt.suptitle("Expected Temperature Profiles from Heating Rates - All Years", fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig(f"{output_dir}/expected_temps_all_years.png", dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("All plots completed!")
print("=" * 60)