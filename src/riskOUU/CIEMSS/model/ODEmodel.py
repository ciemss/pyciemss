######## Written by: Anirban Chaudhuri (Oden Institute; UT Austin)
### Modified version from Jeremy Zucker
import numpy as np
import scipy.integrate


def ODESVIIvR(theta, control, sim_times=np.array([60]), dt=1., N=100000.):
	"""
		##################################################
		Inputs:
		theta:    Vector containing ODE random variable realizations
		control:  Control object
		sim_times:  Range of times to simulate
		dt: timestep
		N: Total population
		##################################################
		Outputs:
		S, V, I, Iv, R:    ODE solution at each time step
		t:      simulation times
	"""
	# unpack input vector into contact rate, beta, and mean recovery rate, gamma, (in 1/days).
	beta, betaV, gamma, gammaV = theta
	# TODO: apply beta, gamma at end of day
	
	# TODO: additional initial conditions as random variables
	# Initial number of infected and recovered individuals, I0 and R0.
	V0, I0, Iv0, R0,  = 0., 1., 0., 0.
	# Everyone else, S0, is susceptible to infection initially.
	S0 = N - I0 - Iv0 - V0 - R0
	
	# grid of time points (using timestep dt)
	# NOTE: assume last time in sim_times is final, and we start at t=0
	sim_times = np.array(sim_times) # enforce array of start and end times
	t = np.linspace(0, int(sim_times[-1]), int(sim_times[-1]/dt)+1)

	nu = control.nu

	# Initial conditions vector
	y0 = S0, V0, I0, Iv0, R0
	# Integrate the SVIIvR equations over the time grid, t.
	out_soln = scipy.integrate.odeint(SVIIvR, y0, t, args=(N, beta, betaV, gamma, gammaV, nu))
	S, V, I, Iv, R = out_soln.T

	return S, V, I, Iv, R, t


# SVIIvR model differential equations
def SVIIvR(y, t, N, beta, betaV, gamma, gammaV, nu):
	S, V, I, Iv, R = y
	dSdt = -beta  * S * I  / N - beta   * S * Iv / N - nu * S 
	dVdt = -betaV * V * Iv / N - betaV  * V * I  / N + nu * S
	dIdt =  beta  * S * I  / N  + beta  * S * Iv / N - gamma  * I 
	dIvdt = betaV * V * I / N   + betaV * V * Iv / N - gammaV * Iv 
	dRdt =  gamma * I + gammaV * Iv
	return dSdt, dVdt, dIdt, dIvdt, dRdt