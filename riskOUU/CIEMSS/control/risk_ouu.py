######## Written by: Anirban Chaudhuri (Oden Institute; UT Austin)
import numpy as np
# from scipy.optimize import minimize
from scipy.optimize import basinhopping

# from CIEMSS.digitaltwin import Twin, State
from CIEMSS.control.risk_measures import Risk
from CIEMSS.control import Control

class OptProblem():
	'''
	Risk-based optimization formulations
	'''
	def __init__(self, twin, decision_metric='infections', risk_metric='alpha-superquantile', risk_params=0.95, nu_bounds=np.array([0., 1.]), 
		t0=0, tf=90, num_samples=10000, dt=1., maxfeval=100, maxiter=10):
		self.twin = twin
		self.decision_metric = decision_metric
		self.risk_metric = risk_metric
		self.risk_params = risk_params
		self.nu_bounds = nu_bounds
		self.samples = []
		self.t0 = t0
		self.tf = tf
		self.dt = dt
		self.num_samples = num_samples
		self.maxfeval = maxfeval
		self.maxiter = maxiter

	def d_metric(self, ndays=7):
		'''
		Output: Samples of the desired decision metric
		Decision metrics:
		'infections': ndays-average for total infections at given time tf
		'''
		outputs_MC, samples_MC, t = self.twin.forward_UQ(num_samples=self.num_samples, tf=self.tf, rseed=1, dt=self.dt)   # Run forward UQ
		S = outputs_MC['S']
		V = outputs_MC['V']
		I = outputs_MC['I']
		Iv = outputs_MC['Iv']
		R = outputs_MC['R']
		# returns distribution of 7-day average of total infections at given time tf
		if self.decision_metric == 'infections':
			# Estimate n-day average of cases
			I_ndays = I[:,int(self.tf/self.dt)-ndays+1:int(self.tf/self.dt)+1]
			Iv_ndays = Iv[:,int(self.tf/self.dt)-ndays+1:int(self.tf/self.dt)+1]
			samples = np.mean(I_ndays + Iv_ndays, axis=1)
		# if self.decision_metric == 'infections':
		# 	samples = I[:,int(self.tf/self.dt)] + Iv[:,int(self.tf/self.dt)]

		return samples

	
	def _vrate(self, vrate):
		'''
		Vaccination rate
		'''
		return vrate


	def _decision_risk(self, vrate):
		"""
		Compute the risk associated with control action for decision metric
		"""
		self.twin.control = Control().vaccination_rate(vrate)    # Set control with prescribed vaccination rate
		samples = self.d_metric()  # Get samples for desired decision metric
		risk = Risk(samples)
		risk_value = getattr(risk, self.risk_metric)(self.risk_params)  # Risk associated
		return risk_value

	
	def solve_minvrate(self, risk_bound=None):
		'''solve optimization problem for finiding minimum vaccination with threshold on the risk constraint'''
		if risk_bound is None:
			risk_bound = 10.

		# Define constraints
		cons = (
			# risk constraint
			{'type': 'ineq', 'fun': lambda x: risk_bound - self._decision_risk(x)},

			# check if in bounds
			{'type': 'ineq', 'fun': lambda x: x - self.nu_bounds[0]},
			{'type': 'ineq', 'fun': lambda x: self.nu_bounds[1] - x}
		)

		# class RandomDisplacementBounds(object):
		# 	"""random displacement with bounds"""
		# 	def __init__(self, xmin, xmax, stepsize=0.25):
		# 		self.xmin = xmin
		# 		self.xmax = xmax
		# 		self.stepsize = stepsize

		# 	def __call__(self, x):
		# 		"""take a random step but ensure the new position is within the bounds"""
		# 		xnew = np.clip(x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x)), self.xmin, self.xmax)
		# 		return xnew

		# take_step = RandomDisplacementBounds(self.nu_bounds[0], self.nu_bounds[1])
		v_init = 0.005  # initial guess
		minimizer_kwargs = dict(constraints=cons, method='COBYLA', 
								tol=1e-5, options={'disp': False, 'maxiter':  self.maxfeval})
		# result = basinhopping(self._vrate, v_init, stepsize=0.25, 
							# niter=self.maxiter, minimizer_kwargs=minimizer_kwargs, take_step=take_step)
		result = basinhopping(self._vrate, v_init, stepsize=0.25, T=1.5, 
							niter=self.maxiter, minimizer_kwargs=minimizer_kwargs, interval=2) 

		return result