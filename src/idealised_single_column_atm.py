import numpy as np
import matplotlib.pyplot as plt

# Physical constants
g = 9.81  # m/s^2, gravitational acceleration
e0 = 611.21  # Pa, reference saturation vapor pressure
T0 = 273.15  # K, reference temperature
Lv = 2.26e6  # J/kg, latent heat of vaporization
Rv = 461.5  # J/(kg·K), gas constant for water vapor
Rd = 287.0  # J/(kg·K), gas constant for dry air
omega = Rd / Rv  # ratio of gas constants
cpd = 1005.0  # J/(kg·K), specific heat at constant pressure for dry air


def saturation_vapor_pressure(T):
    """
    Calculate saturation vapor pressure using Clausius-Clapeyron equation.

    Parameters:
    T : float or array, temperature in K

    Returns:
    es : float or array, saturation vapor pressure in Pa
    """
    es = e0 * np.exp(Lv / Rv * (1 / T0 - 1 / T))
    return es


def moist_adiabatic_lapse_rate(T, p, rs):
    """
    Calculate moist adiabatic lapse rate.

    Parameters:
    T : float, temperature in K
    p : float, pressure in Pa
    rs : float, saturation mass mixing ratio

    Returns:
    gamma_m : float, moist adiabatic lapse rate in K/m
    """
    numerator = g * (1 + Lv * rs / (Rd * T))
    denominator = cpd + Lv ** 2 * rs * omega / (Rv * T ** 2)
    gamma_m = numerator / denominator
    return gamma_m


def single_column_atmosphere(Ts, Tcp, RH, N=100):
    """
    Create an idealized single-column atmosphere model.

    Parameters:
    Ts : float, surface temperature in K
    Tcp : float, cold-point tropopause temperature in K
    RH : float, relative humidity (0 to 1)
    N : int, number of pressure levels

    Returns:
    p : array, pressure levels in Pa
    T : array, temperature profile in K
    x : array, water vapor volume mixing ratio
    p_tp : float, tropopause pressure in Pa
    """

    # Step 1: Define pressure levels (logarithmically spaced)
    p_min = 100  # 1 hPa in Pa
    p_max = 100000  # 1000 hPa in Pa
    p = np.logspace(np.log10(p_max), np.log10(p_min), N)

    # Initialize arrays
    T = np.zeros(N)
    x = np.zeros(N)
    z = np.zeros(N)

    # Step 2: Set surface temperature
    T[0] = Ts
    z[0] = 0.0

    # Initialize tropopause pressure
    p_tp = None
    i_tp = None

    # Steps 3-8: Integrate upward through the troposphere
    for i in range(N - 1):
        # Step 3: Evaluate saturation quantities
        es_i = saturation_vapor_pressure(T[i])
        rs_i = omega * es_i / (p[i] - es_i)

        # Step 4: Compute vapor partial pressure and specific humidity
        e_i = RH * es_i
        r_i = omega * e_i / (p[i] - e_i)
        q_i = r_i / (1 + r_i)

        # Step 5: Compute moist adiabatic lapse rate
        gamma_m_i = moist_adiabatic_lapse_rate(T[i], p[i], rs_i)

        # Step 6: Compute virtual temperature
        Tv_i = T[i] * (1 + (Rv / Rd - 1) * q_i)

        # Step 7: Compute height increment and update temperature
        dln_p = np.log(p[i + 1]) - np.log(p[i])
        dz = -Rd * Tv_i / g * dln_p
        z[i + 1] = z[i] + dz

        T[i + 1] = T[i] - gamma_m_i * dz

        # Step 8: Check if we've reached the tropopause
        if T[i + 1] <= Tcp:
            T[i + 1] = Tcp
            p_tp = p[i + 1]
            i_tp = i + 1
            break

    # If tropopause not reached, use the top level
    if i_tp is None:
        i_tp = N - 1
        p_tp = p[i_tp]
        print(f"Warning: Tropopause temperature {Tcp} K not reached. Using top level.")

    # Step 9: Set stratospheric temperature and build volume mixing ratio
    # Stratosphere: constant temperature
    T[i_tp:] = Tcp

    # Build volume mixing ratio
    # Troposphere: compute volume mixing ratio from RH
    for i in range(i_tp + 1):
        es_i = saturation_vapor_pressure(T[i])
        e_i = RH * es_i
        x[i] = e_i / p[i]

    # Stratosphere: constant volume mixing ratio
    x_tp = x[i_tp]
    x[i_tp:] = x_tp

    # Continue calculating heights for stratosphere
    for i in range(i_tp, N - 1):
        # Use constant mixing ratio from tropopause
        q_i = x_tp / (1 + x_tp)  # Convert VMR to specific humidity
        Tv_i = Tcp * (1 + (Rv / Rd - 1) * q_i)
        dln_p = np.log(p[i + 1]) - np.log(p[i])
        dz = -Rd * Tv_i / g * dln_p
        z[i + 1] = z[i] + dz

    return p, T, x, z, p_tp


# Example usage
if __name__ == "__main__":
    # Example parameters
    Ts = 290  # Surface temperature: 16.85°C
    Tcp = 200.0  # Tropopause temperature: -73°C
    RH = 0.8  # 80% relative humidity

    # Run the model
    p, T, x, z, p_tp = single_column_atmosphere(Ts, Tcp, RH)

    # print(z) to check the z-heights profile,

    # Convert pressure to hPa for plotting
    p_hPa = p / 100
    p_tp_hPa = p_tp / 100

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Temperature profile
    axes[0].plot(T, p_hPa, 'b-', linewidth=2)
    axes[0].axhline(p_tp_hPa, color='r', linestyle='--', label=f'Tropopause ({p_tp_hPa:.1f} hPa)')
    axes[0].set_xlabel('Temperature (K)', fontsize=12)
    axes[0].set_ylabel('Pressure (hPa)', fontsize=12)
    axes[0].set_title('Temperature Profile', fontsize=14)
    axes[0].invert_yaxis()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Water vapor volume mixing ratio
    axes[1].plot(x * 1e6, p_hPa, 'g-', linewidth=2)
    axes[1].axhline(p_tp_hPa, color='r', linestyle='--', label=f'Tropopause ({p_tp_hPa:.1f} hPa)')
    axes[1].set_xlabel('Volume Mixing Ratio (ppmv)', fontsize=12)
    axes[1].set_ylabel('Pressure (hPa)', fontsize=12)
    axes[1].set_title('Water Vapor VMR', fontsize=14)
    axes[1].invert_yaxis()
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Temperature in Celsius
    axes[2].plot(T - 273.15, p_hPa, 'r-', linewidth=2)
    axes[2].axhline(p_tp_hPa, color='r', linestyle='--', label=f'Tropopause ({p_tp_hPa:.1f} hPa)')
    axes[2].set_xlabel('Temperature (°C)', fontsize=12)
    axes[2].set_ylabel('Pressure (hPa)', fontsize=12)
    axes[2].set_title('Temperature Profile (°C)', fontsize=14)
    axes[2].invert_yaxis()
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    # Print some key information
    print(f"Surface Temperature: {Ts:.2f} K ({Ts - 273.15:.2f} °C)")
    print(f"Tropopause Temperature: {Tcp:.2f} K ({Tcp - 273.15:.2f} °C)")
    print(f"Tropopause Pressure: {p_tp_hPa:.2f} hPa")
    print(f"Relative Humidity: {RH * 100:.0f}%")
    print(f"Stratospheric H2O VMR: {x[np.where(p <= p_tp)[0][0]] * 1e6:.2f} ppmv")