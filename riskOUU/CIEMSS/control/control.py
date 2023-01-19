######## Written by: Anirban Chaudhuri (Oden Institute; UT Austin)
import numpy as np

class Control():
	def __init__(self):
		self.name = "custom policy"
		self.nu=None
		self.beta=None
		self.betaV=None

	def vaccination_rate(self, rate=0.005):
		# Vaccination rate controlled through parameter nu (/day)
		self.name = "Vaccination rate: nu"        
		self.nu = rate
		return self

	def masking_policy(self, rate=0.3):
		# Masking policy rate controlled through parameters beta and betav
		# TODO: fix how beta is assigned given masking policy rate
		self.name = "Masking policy: beta"
		self.beta = rate
		self.betav = rate
		return self


class Control_pyro():
	def __init__(self):
		self.name = "custom policy"
		self.nu=None
		self.beta=None
		self.betaV=None

	def vaccination_rate(self, rate=0.005):
		# Vaccination rate controlled through parameter nu (/day)
		self.name = "Vaccination rate parmeter: nu"        
		self.nu = np.array([rate])
		return self

	def scheduled_vaccination(self, rate=[100.,50.,20.]):
		# Vaccination rate controlled through parameter nu (/day)
		self.name = "Vaccination rate: svlux"        
		self.svflux = np.array([rate])
		return self

	def masking_policy(self, rate=0.3):
		# Masking policy rate controlled through parameters beta and betav
		# TODO: fix how beta is assigned given masking policy rate
		self.name = "Masking policy: beta"
		self.beta = rate
		self.betav = rate
		return self