######## Written by: Anirban Chaudhuri (Oden Institute; UT Austin)
import numpy as np
from CIEMSS.digitaltwin import ODEState
from CIEMSS.model import ODESVIIvR
from CIEMSS.control import Control


class ODETwin:
	def __init__(self, initial_state=ODEState().prior(), forwardmodel=ODESVIIvR, control=Control().vaccination_rate()):
		self.state = initial_state
		self.forward = forwardmodel
		self.control = control
		# self.obs_noise_std = 2e9

	def forward_UQ(self, num_samples=1000, t0=0, tf=160, dt=1., rseed=None):
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

	# TODO: add inverse problem solution
	def assimilate(self, observation, method="mcmc", num_samples=5000, savedir=None):
		return