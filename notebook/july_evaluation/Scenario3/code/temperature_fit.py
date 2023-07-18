import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy

# Set default text interpreter
plt.rc('text', usetex=True)
plt.rc('font', size=16) 

bl = '#0072BD'
br = '#D95319'

# Load data
data = scipy.io.loadmat('data.mat')
xdata = np.arange(len(data['tempData2'][0][:-1]))
ydata = data['tempData2'][0][:-1]

# Fit sinusoidal model f(x) = A*sin(Bx - C) + D
def fun(x, A, B, C, D):
    return A * np.sin(B * x - C) + D

#     A    B      C   D
p0 = [4, 0.02, 0, 15]
params, _ = curve_fit(fun, xdata, ydata, p0)

print(params)

plt.figure()
plt.plot(data['flowDate2'][0][:-1], ydata, '.', markersize=20, color=br)
plt.plot(xdata, fun(xdata, *params), linewidth=4, color=bl)
plt.ylabel('Temperature ($^\circ$C)')
plt.tight_layout()
plt.show()
