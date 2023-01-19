######## Written by: Anirban Chaudhuri (Oden Institute; UT Austin)
# import sys
# sys.path.append('/storage/achaudhuri/covid_askem/')
import numpy as np
from riskOUU.CIEMSS.digitaltwin import ODEState
from riskOUU.CIEMSS.model import ODESVIIvR
from riskOUU.CIEMSS.control import Control, Control_pyro

import torch
from torchdiffeq import odeint
from pyro.infer import Predictive
from pyciemss.ODE.models import SVIIvR
from pyro import do

import time

class ODETwin:
	def __init__(self, initial_state=ODEState().prior(), forwardmodel=ODESVIIvR, control=Control().vaccination_rate()):
		self.state = initial_state
		self.forward = forwardmodel
		self.control = control

	def forwardUQ(self, num_samples=1000, t0=0, tf=160, dt=1., rseed=None):
		# Run Monte Carlo samples through model
		S_MC = [[] for i in range(num_samples)]
		V_MC = [[] for i in range(num_samples)]
		I_MC = [[] for i in range(num_samples)]
		Iv_MC = [[] for i in range(num_samples)]
		R_MC = [[] for i in range(num_samples)]
		tspan = np.array([t0, tf])
		# Sample random variables
		if rseed is None:
			rseed=1
		sampled_state = self.state.sample(num_samples=num_samples, rseed=rseed)
		# Run model at the samples to get outputs
		for ii in range(num_samples):
			if ii == 0:
				S_MC[ii], V_MC[ii], I_MC[ii], Iv_MC[ii], R_MC[ii], t = self.forward(sampled_state[ii,:], self.control, sim_times=tspan, dt=dt)
			else:
				S_MC[ii], V_MC[ii], I_MC[ii], Iv_MC[ii], R_MC[ii], _ = self.forward(sampled_state[ii,:], self.control, sim_times=tspan, dt=dt)

		return {'S': np.array(S_MC), 'V': np.array(V_MC), 'I': np.array(I_MC), 'Iv': np.array(Iv_MC), 'R': np.array(R_MC)}, sampled_state, t


	def forwardUQ_pyro(self, num_samples=1000, t0=0, tf=90, guide=None, dt=1., initial_state=None, rseed=None):
		# Run Monte Carlo samples through model
		# Total population, N.
		torch.manual_seed(1)
		N = 100000.0
		# Initial number of infected and recovered individuals, I0 and R0.
		V0, I0, Iv0, R0 = 0.0, 1.0, 0.0, 0.0
		# Everyone else, S0, is susceptible to infection initially.
		S0 = N - I0 - Iv0 - V0 - R0
		initial_state = tuple(torch.as_tensor(s) for s in  (S0, V0, I0, Iv0, R0))
		# tspan = self.get_tspan(t0, tf, tf+1)
		tspan = torch.linspace(float(t0), float(tf), tf+1)
		ode_model = SVIIvR(N)

		# TODO: automate the control selection
		controlled_model = do(ode_model, {"nu": torch.from_numpy(np.squeeze(self.control.nu)), "noise_var": 1e-5})

		if guide is None:
			out_prediction = Predictive(controlled_model, num_samples=num_samples)(initial_state, tspan)
		else:
			out_prediction = Predictive(controlled_model, guide=guide, num_samples=num_samples)(initial_state, tspan)
		S_MC = out_prediction["S_obs"].detach().numpy()
		V_MC = out_prediction["V_obs"].detach().numpy()
		Itot_MC = out_prediction["I_obs"].detach().numpy()
		R_MC = out_prediction["R_obs"].detach().numpy()

		sampled_state = None
		t = tspan.detach().numpy()

		return {'S': S_MC, 'V': V_MC, 'Itot': Itot_MC,'R': R_MC}, sampled_state, t


	def forwardUQsv_pyro(self, num_samples=1000, t0=0, tf=90, guide=None, dt=1., initial_state=None, rseed=None):
		# Run Monte Carlo samples through model
		# Total population, N.
		torch.manual_seed(1)
		N = 100000.0
		# Initial number of infected and recovered individuals, I0 and R0.
		V0, I0, Iv0, R0 = 0.0, 1.0, 0.0, 0.0
		# Everyone else, S0, is susceptible to infection initially.
		S0 = N - I0 - Iv0 - V0 - R0
		initial_state = tuple(torch.as_tensor(s) for s in  (S0, V0, I0, Iv0, R0))
		# tspan = self.get_tspan(t0, tf, tf+1)
		tspan = torch.linspace(float(t0), float(tf), tf+1)
		ode_model = SVIIvR(N)

		# TODO: automate the control selection
		control_dict = self.scheduled_intervention("SV_flux", torch.from_numpy(np.squeeze(self.control.svflux)), tspan)
		control_dict.update({"noise_var": 1e-5})
		controlled_model = do(ode_model, control_dict)

		if guide is None:
			out_prediction = Predictive(controlled_model, num_samples=num_samples)(initial_state, tspan)
		else:
			out_prediction = Predictive(controlled_model, guide=guide, num_samples=num_samples)(initial_state, tspan)
		S_MC = out_prediction["S_obs"].detach().numpy()
		V_MC = out_prediction["V_obs"].detach().numpy()
		Itot_MC = out_prediction["I_obs"].detach().numpy()
		R_MC = out_prediction["R_obs"].detach().numpy()

		sampled_state = None
		t = tspan.detach().numpy()

		return {'S': S_MC, 'V': V_MC, 'Itot': Itot_MC,'R': R_MC}, sampled_state, t

	# def get_tspan(start, end, steps):
	# 	return torch.linspace(float(start), float(end), steps)

	def scheduled_intervention(self, name, intervention_assignment, tspan):
	    index = lambda t : torch.floor(torch.tensor(t-0.0001) / 30.).int()
	    return {name + " %f" % (t): intervention_assignment[index(t)] for t in tspan}

	# TODO: add inverse problem solution
	def assimilate(self, observation, method="mcmc", num_samples=5000, savedir=None):
		return
