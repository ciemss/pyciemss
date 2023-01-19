######## Written by: Anirban Chaudhuri (Oden Institute; UT Austin)
import numpy as np
# from scipy.optimize import minimize
from scipy.optimize import basinhopping

# from riskOUU.CIEMSS.digitaltwin import Twin, State
from riskOUU.CIEMSS.control.risk_measures import Risk
from riskOUU.CIEMSS.control import Control_pyro

class OptProblem():
	'''
	Risk-based optimization formulations
	'''
	def __init__(self, twin, quantity_of_interest='infections', risk_measure='alpha-superquantile', risk_params=0.95,
		guide=None, control_name='nu', u_bounds=np.array([0., 1.]), 
		t0=0, tf=90, num_samples=5000, dt=1., maxfeval=100, maxiter=10, stepsize=None):
		self.twin = twin
		self.qoi = quantity_of_interest
		self.risk_measure = risk_measure
		self.risk_params = risk_params
		self.u_bounds = u_bounds
		self.samples = []
		self.t0 = t0
		self.tf = tf
		self.dt = dt
		self.num_samples = num_samples
		self.maxfeval = maxfeval
		self.maxiter = maxiter
		self.guide = guide
		self.control_name = control_name
		self.stepsize = stepsize

	def d_metric(self, ndays=7):
		'''
		Output: Samples of the desired decision metric
		Decision metrics:
		'infections': ndays-average for total infections at given time tf
		'''
		outputs_MC, _, _ = self.twin.forwardUQ_pyro(num_samples=self.num_samples, tf=self.tf, guide=self.guide, rseed=1, dt=self.dt)   # Run forward UQ
		Itot = outputs_MC['Itot']
		# returns distribution of 7-day average of total infections at given time tf
		if self.qoi == 'infections':
			# Estimate n-day average of cases
			Itot_ndays = Itot[:,int(self.tf/self.dt)-ndays+1:int(self.tf/self.dt)+1]
			# Iv_ndays = Iv[:,int(self.tf/self.dt)-ndays+1:int(self.tf/self.dt)+1]
			samples = np.mean(Itot_ndays, axis=1)
		# if self.qoi == 'infections':
		# 	samples = I[:,int(self.tf/self.dt)] + Iv[:,int(self.tf/self.dt)]

		return samples


	def d_metricsv(self, ndays=7):
		'''
		Output: Samples of the desired decision metric
		Decision metrics:
		'infections': ndays-average for total infections at given time tf
		'''
		outputs_MC, _, _ = self.twin.forwardUQsv_pyro(num_samples=self.num_samples, tf=self.tf, guide=self.guide, rseed=1, dt=self.dt)   # Run forward UQ
		Itot = outputs_MC['Itot']
		# returns distribution of 7-day average of total infections at given time tf
		# TODO: needs to be fixed for dt != 1
		if self.qoi == 'infections':
			# Estimate n-day average of cases
			Itot_ndays = Itot[:,int(self.tf/self.dt)-ndays+1:int(self.tf/self.dt)+1]
			# Iv_ndays = Iv[:,int(self.tf/self.dt)-ndays+1:int(self.tf/self.dt)+1]
			samples = np.mean(Itot_ndays, axis=1)
		# if self.qoi == 'infections':
		# 	samples = I[:,int(self.tf/self.dt)] + Iv[:,int(self.tf/self.dt)]

		return samples

	
	def _vrate(self, vrate):
		'''
		Vaccination rate
		'''
		return np.sum(np.array([vrate]))


	def _decision_risk(self, vrate):
		"""
		Compute the risk associated with control action for decision metric
		"""
		self.twin.control = Control_pyro().vaccination_rate(vrate)    # Set control with prescribed vaccination rate
		samples = self.d_metric()  # Get samples for desired decision metric
		risk = Risk(samples)
		risk_value = getattr(risk, self.risk_measure)(self.risk_params)  # Risk associated
		return risk_value


	def _decision_risk_sv(self, vrate):
		"""
		Compute the risk associated with control action for decision metric
		"""
		self.twin.control = Control_pyro().scheduled_vaccination(vrate)    # Set control with prescribed vaccination rate
		samples = self.d_metricsv()  # Get samples for desired decision metric
		risk = Risk(samples)
		risk_value = getattr(risk, self.risk_measure)(self.risk_params)  # Risk associated
		return risk_value

	
	def solve_minvrate(self, risk_bound=None, u_init=None):
		'''solve optimization problem for finiding minimum vaccination with threshold on the risk constraint'''
		if risk_bound is None:
			risk_bound = 10.

		# Define constraints
		if self.control_name == 'nu':
			cons = (
				# risk constraint
				{'type': 'ineq', 'fun': lambda x: risk_bound - self._decision_risk(x)},

				# check if in bounds
				{'type': 'ineq', 'fun': lambda x: x - self.u_bounds[0]},
				{'type': 'ineq', 'fun': lambda x: self.u_bounds[1] - x}
			)
			if u_init is None:
				u_init = 0.005  # initial guess for nu
			if self.stepsize is None:
				stepsize = 0.25
			else:
				stepsize = self.stepsize
		elif self.control_name == 'svflux':
			cons = (
				# risk constraint
				{'type': 'ineq', 'fun': lambda x: risk_bound - self._decision_risk_sv(x)},

				# check if in bounds
				{'type': 'ineq', 'fun': lambda x: x - self.u_bounds[0]},
				{'type': 'ineq', 'fun': lambda x: self.u_bounds[1] - x}
			)
			if u_init is None:
				u_init = np.array([500.,500.,1.])  # initial guess for scheduled intervention
			if self.stepsize is None:
				stepsize = 100.
			else:
				stepsize = self.stepsize
		else:
			print(self.control_name + "is not a supported control.")

		class RandomDisplacementBounds(object):
			"""random displacement with bounds"""
			def __init__(self, xmin, xmax, stepsize=0.25):
				self.xmin = xmin
				self.xmax = xmax
				self.stepsize = stepsize

			def __call__(self, x):
				"""take a random step but ensure the new position is within the bounds"""
				xnew = np.clip(x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x)), self.xmin, self.xmax)
				return xnew

		minimizer_kwargs = dict(constraints=cons, method='COBYLA', 
								tol=1e-5, options={'disp': False, 'maxiter':  self.maxfeval})
		take_step = RandomDisplacementBounds(self.u_bounds[0], self.u_bounds[1], stepsize=stepsize)
		result = basinhopping(self._vrate, u_init, stepsize=stepsize, T=1.5, 
							niter=self.maxiter, minimizer_kwargs=minimizer_kwargs, take_step=take_step, interval=2)
		# result = basinhopping(self._vrate, u_init, stepsize=stepsize, T=1.5, 
		# 					niter=self.maxiter, minimizer_kwargs=minimizer_kwargs, interval=2, disp=True) 

		return result