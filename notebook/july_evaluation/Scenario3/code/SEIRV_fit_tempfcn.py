import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat
from scipy.integrate import odeint
import pandas as pd

# Seed the random number generator
np.random.seed(0)

# Load data
data = loadmat('data.mat')
cRNA2 = np.array(data['cRNA2'])
F2 = np.array(data['F2'])
V = cRNA2 * F2
split = 78
V = V[:split]
tspan = np.arange(1, len(V) + 1)

# Curve-fitting
beta_fixed = 4.48526e7
lb = [0, 51, beta_fixed, 10]
ub = [1e-4, 796, beta_fixed, 5000]
p0 = [9.06e-08, 360, beta_fixed, 1182]

# Define objective function
from scipy.integrate import solve_ivp
import numpy as np

def obj_fun(param, tspan, data):
    """"""
    traveltime = 18 # hours
    k = getDecay(1) # use first time point

    eta = 1 - np.exp(-k*traveltime)

    # total population served by DITP
    N0 = 2300000

    E0 = param[3]
    I0 = data[0] / (param[1] * param[2] * (1-eta))
    R0 = 0
    S0 = N0 - (E0 + I0 + R0)
    V0 = data[0] # use first data point
    ICs  = [S0, E0, I0, R0, V0, E0]

    # You need to have the function SEIRV defined somewhere, as you did in MATLAB
    sol = solve_ivp(lambda t, y: SEIRV(t, y, param[0:3]), [tspan[0], tspan[-1]], ICs)

    # get daily virus
    cumVirus = sol.y[4]
    dailyVirus = np.diff(cumVirus)

    temp = np.log10(data[1:]) - np.log10(np.abs(dailyVirus))
    adiff = temp[~np.isnan(temp)]

    err = np.sum((adiff)**2)
    return err


# Perform optimization
bounds = opt.Bounds(lb, ub)
result = opt.differential_evolution(lambda x: obj_fun(x, tspan, V), bounds=bounds)

best_params = result.x
SSE = result.fun

# Create data frame with parameters and their estimated values
parameter = ["lambda", "alpha", "beta", "E(0)", "SSE"]
estimated_val = np.append(best_params, SSE)
df = pd.DataFrame({'parameter': parameter, 'estimated_val': estimated_val})
print(df)

# Simulate with best params
alpha = best_params[1]
beta = best_params[2]

traveltime = 18  # hours
k = getDecay(1)  # use first time point
eta = 1 - np.exp(-k * traveltime)
N0 = 2300000  # total population served by DITP
E0 = best_params[3]
I0 = V[0] / (alpha * beta * (1 - eta))
R0 = 0
S0 = N0 - (E0 + I0 + R0)
V0 = V[0]
ICs = [S0, E0, I0, R0, V0, E0]

# Define SEIRV model function
def SEIRV(ICs, t, best_params):
    # Define SEIRV code here...

T, Y = odeint(SEIRV, ICs, np.arange(1, len(cRNA2) + 1), args=(best_params,))

# Plot results
import matplotlib.pyplot as plt
import datetime
from scipy.stats import linregress

# Conversion of MATLAB's datetime to Python's datetime
start_date = datetime.datetime(2020, 9, 30)
time = [start_date + datetime.timedelta(days=i) for i in range(len(cRNA2))]

# Create plot
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# First subplot
axes[0].plot(time[1:], np.log10(np.diff(Y[:,4])), linewidth=2)
axes[0].scatter(time[1:], np.log10(cRNA2[1:] * F2[1:]), color='r', s=20, linewidth=2)
axes[0].axvline(x=time[split - 1], linestyle='--', linewidth=2, color='g')
axes[0].set_ylim([13.5, None])
axes[0].set_xlim([time[18 - 1], time[116 - 1]])
axes[0].set_ylabel(r'$\log_{10}$ viral RNA copies')

# Second subplot
axes[1].plot(time[1:], np.log10(np.diff(Y[:,5])), linewidth=2)
axes[1].plot(time[1:], np.log10(newRepCases2[1:]), linewidth=2, color='r')
axes[1].set_ylabel(r'$\log_{10}$ Daily Incidence')

index1 = np.argmax(np.diff(Y[:,5]))
index2 = np.argmax(newRepCases2)
axes[1].axvline(x=time[index1 + 1], linestyle='--', linewidth=2, color='b')
axes[1].axvline(x=time[index2], linestyle='--', linewidth=2, color='r')
axes[1].legend(['Model', 'Data'], loc='upper left')
axes[1].set_ylim([2.379, 4.5])
axes[1].set_xlim([time[1], time[118]])

plt.tight_layout()

# Save figure
plt.savefig('fitting_with_temperature.pdf', dpi=600)
plt.clf()

# Second figure
plt.figure()
plt.box(True)
y = np.diff(Y[:,5])
x = newRepCases2[1:]
X = np.column_stack((np.ones(len(x)), x))
b = np.linalg.lstsq(X, y, rcond=None)[0]
yCalc2 = X @ b

plt.scatter(x, y, color='k', s=20, linewidth=2)
plt.plot(x, yCalc2, color='r', linewidth=2)
plt.ylim([0, None])
plt.ylabel('Predicted cases')
plt.xlabel('Reported cases')

# Calculate R2
Rsq2 = 1 - np.sum((y - yCalc2)**2) / np.sum((y - np.mean(y))**2)
R = np.corrcoef(x, y)

plt.savefig('corr_1.pdf', dpi=600)




import numpy as np

def getDecay(t):
    # compute temperature-adjusted decay rate of viral RNA
    
    # high titer -> tau0 = 0.99 days * 24 hours/day = 23.76
    # low titer  -> tau0 = 7.9 days * 24 hours/day  = 189.6

    tau0 = 189.6 #23.76
    Q0 = 2.5
    T0 = 20

    # get current temperature using best-fit sine function
    A = 3.624836409841919
    B = 0.020222716119084
    C = 4.466530666828714
    D = 16.229757918464635

    T = A * np.sin(B * t - C) + D

    tau = tau0 * Q0 ** (-(T - T0) / 10)

    k = np.log(2) / tau

    return k

import numpy as np
from scipy.integrate import odeint

def SEIRV(y, t, param):
    # parameters to be fit
    lambda_ = param[0]
    alpha = param[1]
    beta = param[2]

    S, E, I, R, V, cumulative_cases = y

    traveltime = 18  # hours
    k = getDecay(t)

    eta = 1 - np.exp(-k*traveltime)

    sigma = 1/3
    gamma = 1/8

    dS = -lambda_ * S * I
    dE = lambda_ * S * I - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    dV = alpha * beta * (1 - eta) * I
    dcumulative_cases = lambda_ * S * I  # track cumulative cases

    dy = [dS, dE, dI, dR, dV, dcumulative_cases]

    return dy

# Note: to use odeint in scipy to integrate this system of differential equations, you would do something like the following:
# y0 = [S0, E0, I0, R0, V0, cumulative_cases0]
# t = np.linspace(start_time, end_time, num_timepoints)
# result = odeint(SEIRV, y0, t, args=(param,))

