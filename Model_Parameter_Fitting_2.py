import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

# Define zone-specific population
zone_population = {
    "North West": 36481142412,
    "South West": 25531531064,
    "North Central": 29180823484,
    "South South": 21881459466,
    "North East": 21881558382,
    "South East": 18231702663,
}

# -------------------------------
# CUSTOM INITIAL CONDITIONS
# -------------------------------
custom_initial_conditions = {
    "North West": {"S0": 48942307, "E0": 2074166, "T0": 761118, "R0": 1105628, "I0": 1930183, "D0": 182750},
    "South West": {"S0": 46706662, "E0": 2045599, "T0": 745796, "R0": 1154534, "I0": 1890906, "D0": 182094},
    "North Central": {"S0": 29252408, "E0": 1954642, "T0": 776790, "R0": 1116099, "I0": 1877308, "D0": 1116099},
    "South South": {"S0": 28829288, "E0": 1967426, "T0": 740835, "R0": 1139483, "I0": 1858195, "D0": 190367},
    "North East": {"S0": 26263866, "E0": 2020937, "T0": 752219, "R0": 1125244, "I0": 1907831, "D0": 181019},
    "South East": {"S0": 21955414, "E0": 2030175, "T0": 782614, "R0": 1101223, "I0": 1810160, "D0": 180010},
}

# -------------------------------
# COVID MODEL DEFINITION
# -------------------------------
def covid_model(y, t, t_knot, beta, theta, rho, tau, sigma, gamma, delta, alpha, omega, f, a, b, N):
    S, E, T, R, I, D = y
    # Logistic effect modeling the lockdown impact (corrected multiplication)
    Logistic_effect = 1 / (1 + np.exp(-((1 - rho) * (t - t_knot))))
    beta_effective = beta * Logistic_effect

    dSdt = -beta_effective * S * I * (1 - f) / N + theta * T + omega * R
    dEdt = beta_effective * S * I * (1 - f) / N - alpha * E
    dTdt = alpha * E - sigma * T - theta * T
    dRdt = gamma * I - omega * R
    dIdt = sigma * T - gamma * I - delta * I
    dDdt = delta * I
    return dSdt, dEdt, dTdt, dRdt, dIdt, dDdt

# -------------------------------
# MODEL SIMULATION FUNCTION
# -------------------------------
def simulate_model(params, S0, E0, T0, R0, I0, D0, N, t, t_knot):
    # params order: beta, theta, rho, tau, sigma, gamma, delta, alpha, omega, f, a, b
    y0 = (S0, E0, T0, R0, I0, D0)
    return odeint(covid_model, y0, t, args=(t_knot, *params, N))

# -------------------------------
# OBJECTIVE FUNCTION FOR PARAMETER FITTING
# -------------------------------
def objective(params, t, t_knot, observed_I, S0, E0, T0, R0, I0, D0, N):
    result = simulate_model(params, S0, E0, T0, R0, I0, D0, N, t, t_knot)
    S, E, T, R, I, D = result.T
    return np.sum((I - observed_I) ** 2)

# -------------------------------
# SIMULATION PARAMETERS
# -------------------------------
t = np.linspace(0, 365, 366)  # Simulate for 365 days
t_knot = 50  # Example value for the time when lockdown effects start

# Define “true” parameters for simulation (example values for 12 parameters)
# Order: beta, theta, rho, tau, sigma, gamma, delta, alpha, omega, f, a, b
true_params = [0.87, 0.5, 0.64, 0.1, 0.55, 0.7, 0.013, 0.1, 0.02, 1.5, 1.0, 0.1]

# Dictionary to hold fitted results for each zone
fitted_results = {}

# -------------------------------
# LOOP OVER EACH ZONE
# -------------------------------
for zone, pop in zone_population.items():
    # Use custom initial conditions if provided; otherwise, use default fractions of the population
    if zone in custom_initial_conditions:
        ic = custom_initial_conditions[zone]
        S0 = ic["S0"]
        E0 = ic["E0"]
        T0 = ic["T0"]
        R0 = ic["R0"]
        I0 = ic["I0"]
        D0 = ic["D0"]
    else:
        S0 = 0.99 * pop   # Susceptible
        E0 = 0.005 * pop  # Exposed
        T0 = 0.002 * pop  # Treated (or another compartment)
        R0 = 0.001 * pop  # Recovered
        I0 = 0.002 * pop  # Infected
        D0 = 0.0005 * pop # Deceased

    # Simulate synthetic “true” data using the known parameters
    sim = simulate_model(true_params, S0, E0, T0, R0, I0, D0, pop, t, t_knot)
    S_true, E_true, T_true, R_true, I_true, D_true = sim.T

    # Add noise to the infected data to mimic real-world observations
    noise = np.random.normal(0, 0.01 * pop, size=I_true.shape)
    observed_I = I_true + noise

    # Initial parameter guess for optimization
    initial_guess = [0.2, 0.4, 0.4, 0.05, 0.03, 0.1, 0.00015, 0.05, 0.01, 1.0, 0.8, 0.05]
    # Bounds for parameters (adjust as needed)
    bounds = [(0, 1)] * 9 + [(0, 2), (0, 2), (0, 1)]
    
    # Fit parameters by minimizing the objective function
    res = minimize(objective, initial_guess, args=(t, t_knot, observed_I, S0, E0, T0, R0, I0, D0, pop), bounds=bounds)
    fitted_params = res.x
    fitted_results[zone] = fitted_params

    # Simulate the model with the fitted parameters
    sim_fit = simulate_model(fitted_params, S0, E0, T0, R0, I0, D0, pop, t, t_knot)
    S_fit, E_fit, T_fit, R_fit, I_fit, D_fit = sim_fit.T

    # Visualization for the current zone
    plt.figure(figsize=(10, 6))
    plt.scatter(t, observed_I, label="Observed Infected Cases", color="red", marker="o", s=10)
    plt.plot(t, I_fit, label="Fitted Model Infected Cases", color="blue", linewidth=2)
    plt.xlabel("Time (Days)")
    plt.ylabel("Number of Infected Individuals")
    plt.title(f"COVID-19 Model Fit for {zone}")
    plt.legend()
    plt.show()

    print(f"Zone: {zone}")
    param_names = ["Beta", "Theta", "Rho", "Tau", "Sigma", "Gamma", "Delta", "Alpha", "Omega", "f", "a", "b"]
    for name, value in zip(param_names, fitted_params):
        print(f"{name}: {value:.4f}")
    print("\n" + "="*50 + "\n")
