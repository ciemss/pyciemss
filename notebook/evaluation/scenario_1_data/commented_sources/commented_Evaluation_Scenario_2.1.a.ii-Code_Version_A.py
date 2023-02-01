from scipy.integrate import odeint
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

def SIDARTHE_model(y, t, alpha, beta, gamma, delta, epsilon, mu, zeta, lamb, eta, rho, theta, kappa, nu, xi, sigma, tau):
    """
    The SIDARTHE_model function calculates the derivatives of the model's state variables with respect to time.
    
    Parameters:
    y (tuple of float): A tuple representing the current values of the state variables:
       S - SUSCEPTIBLE
       I - INFECTED
       D - DIAGNOSED
       A - AILING
       R - RECOGNISED
       T - THREATENED
       H - HEALED
       E - EXTINCT

    t (float): The current time
    
    (function): 
        Functions representing the rates at which the state variables change
    
    Returns:
    tuple of float: A tuple representing the derivatives of the state variables with respect to time (dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt)
    """
    
    # Unpack the state variables
    S, I, D, A, R, T, H, E = y
    
    # Calculate the derivative of susceptible individuals (dSdt)
    dSdt = -S*(alpha(t)*I + beta(t)*D + gamma(t)*A + delta(t)*R)
    
    # Calculate the derivative of infected individuals (dIdt)
    dIdt = S*(alpha(t)*I + gamma(t)*D + beta(t)*A + delta(t)*R) - (zeta(t) + lamb(t))*I
    
    # Calculate the derivative of dead individuals (dDdt)
    dDdt = epsilon(t)/3*I - (eta(t))*D
    
    # Calculate the derivative of asymptomatic individuals (dAdt)
    dAdt = zeta(t)*I - (theta(t) + mu(t) + kappa(t))*A
    
    # Calculate the derivative of recovered individuals (dRdt)
    dRdt = eta(t)*D + theta(t)*A - (nu(t) + xi(t))*R
    
    # Calculate the derivative of treated individuals (dTdt)
    dTdt = mu(t)*A + nu(t)*R - sigma(t)*T + tau(t)*T
    
    # Calculate the derivative of hospitalized individuals (dHdt)
    dHdt = lamb(t)*I + sigma(t)*D + xi(t)*R + kappa(t)*T
    
    # Calculate the derivative of exposed individuals (dEdt)
    dEdt = -tau(t)*T
    
    # Return the derivatives of all state variables
    return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt
    
# Example Simulation and Plot

def alpha(t): return np.piecewise(t, [t>=0], [0.75])
def beta(t): return np.piecewise(t, [t>=0], [0.1])
def delta(t): return np.piecewise(t, [t>=0], [0.05])
def gamma(t): return np.piecewise(t, [t>=0], [0.2])

def epsilon(t): return np.piecewise(t, [t>=0], [0.1])
def theta(t): return np.piecewise(t, [t>=0], [0.1])

def zeta(t): return np.piecewise(t, [t>=0], [0.05])
def eta(t): return np.piecewise(t, [t>=0], [0.05])

def mu(t): return np.piecewise(t, [t>=0], [0.05])
def nu(t): return np.piecewise(t, [t>=0], [0.05])
def lamb(t): return np.piecewise(t, [t>=0], [0.05])
def rho(t): return np.piecewise(t, [t>=0], [0.05])

def kappa(t): return np.piecewise(t, [t>=0], [0.01])
def xi(t): return np.piecewise(t, [t>=0], [0.01])
def sigma(t): return np.piecewise(t, [t>=0], [0.01])

def tau(t): return np.piecewise(t, [t>=0], [0.05])

N0 = 1e6
I0, D0, A0, R0, T0, H0, E0 = 50/N0, 50/N0, 5/N0, 1/N0, 0, 0, 0
S0 = 1-I0-D0-A0-R0-T0-H0-E0
y0 = S0, I0, D0, A0, R0, T0, H0, E0 # Initial conditions vector

dt = 2
tstart = 0
tend = 40
tvect = np.arange(tstart, tend, dt) 

sim = odeint(SIDARTHE_model, y0, tvect, args=(alpha, beta, gamma, delta, epsilon, mu, zeta, lamb, eta, rho, theta, kappa, nu, xi, sigma, tau))
S, I, D, A, R, T, H, E = sim.T

f, ax = plt.subplots(1,1,figsize=(10,4))
ax.plot(tvect, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
ax.plot(tvect, I, 'r', alpha=0.7, linewidth=2, label='Infected (Asymptomatic, Infected, Undetected)')
ax.plot(tvect, D, 'r.', alpha=0.7, linewidth=2, label='Diagnosed (Asymptomatic, Infected, Detected)')
ax.plot(tvect, A, 'r:', alpha=0.7, linewidth=2, label='Ailing (Symptomatic, Infected, Undetected)')
ax.plot(tvect, R, 'r--', alpha=0.7, linewidth=2, label='Recognized (Symptomatic, Infected, Detected)')
ax.plot(tvect, T, 'r-.', alpha=0.7, linewidth=2, label='Threatened (Acutely Symptomatic)')
ax.plot(tvect, H, 'g', alpha=0.7, linewidth=2, label='Healed')
ax.plot(tvect, E, 'k', alpha=0.7, linewidth=2, label='Extinct (Dead)')

ax.set_xlabel('Time (days)')
ax.set_ylabel('Fraction of population')

ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)

plt.show();