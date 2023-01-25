######## Written by: Anirban Chaudhuri (Oden Institute; UT Austin)
import numpy as np
import matplotlib.pyplot as plt

from CIEMSS.model import ODESVIIvR
from CIEMSS.control import Control

# # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
# beta, betaV, gamma, gammaV, nu = 0.2, 0.15, 1./10, 1.5/10, 0.005 

sampled_state = np.array([[0.2, 0.15, 1./10, 1.5/10]])
ii=0
control=Control().vaccination_rate(rate=0.005)
print(control.nu)
tspan = np.array([0, 160])
S, V, I, Iv, R, t = ODESVIIvR(sampled_state[ii,:], control, sim_times=tspan, dt=1., N=1000.)

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, V/1000, 'b--', alpha=0.5, lw=2, label='SucceptibleVaccinated')
ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, Iv/1000, 'r--', alpha=0.5, lw=2, label='InfectedVaccinated')
ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
	ax.spines[spine].set_visible(False)
plt.show()