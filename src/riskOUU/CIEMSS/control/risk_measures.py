######## Written by: Anirban Chaudhuri (Oden Institute; UT Austin)
import numpy as np

class Risk():
	'''
	Given a set of samples estimate different risk measures
	to be used for decision-making under uncertainty
	'''
	def __init__(self, samples):
		# self.mean = None
		# self.variance = None
		# self.robust_comb = None
		# self.pof = None
		# self.bpof = None
		# self.qtile = None
		# self.squantile = None
		if type(samples) == list:
			self.samples = np.array(samples)
		else:
			self.samples = samples
		self.nsamples = float(self.samples.shape[0])

	def sample_mean(self):
		# sample mean of vector of samples
		self.mean = np.mean(self.samples)
		return self.mean

	def sample_variance(self):
		# sample variance of vector of samples
		self.variance = np.var(self.samples)
		return self.variance

	def robust(self, eta=2.):
		self.robust_comb = np.mean(self.samples) + eta*np.std(self.samples)
		return self.robust_comb

	def probability_failure(self, threshold=0.):
		# probability of exceeding a threshold
		self.pof = np.sum(self.samples >= threshold)/self.nsamples
		return self.pof

	def buffered_pof(self, threshold=0.):
		# buffered probability of exceeding a threshold
		sorted_samples = np.sort(self.samples)[::-1]    # sort samples in descending order
		squantile = sorted_samples[0]
		prob_level = 1
		for i in range(int(self.nsamples)-1):
			prob_level = prob_level - 1./self.nsamples
			squantile = (squantile*i + sorted_samples[i+1])/float(i+1)
			if squantile < threshold:
				break
		# qtile = sorted_samples[int(prob_level*self.nsamples)]
		self.bpof = 1-prob_level
		return self.bpof

	def alpha_quantile(self, alpha=0.95):
		'''
		upper alpha-quantile for a given alpha in [0,1]
		a.k.a value-at-risk (VaR)

		'''
		self.qtile = np.quantile(self.samples, alpha)
		# # Estimate MC error in estimating quantile
		# b_edges = np.histogram_bin_edges(self.samples, bins='fd')
		# id_qtile = np.argwhere(b_edges<self.qtile)[0][0]
		# pdf_qtile = np.sum(b_edges[id_qtile-1]<=self.samples<=b_edges[id_qtile])/(self.nsamples*(b_edges[id_qtile]-b_edges[id_qtile-1]))
		# qtile_err = np.sqrt(alpha*(1-alpha)/self.nsamples)/pdf_qtile
		return self.qtile

	def alpha_superquantile(self, alpha=0.95):
		'''
		upper alpha-superquantile for a given alpha in [0,1]
		a.k.a conditional value-at-risk (CVaR), expected shortfall, average value-at-risk
		'''
		# Splitting atoms
		sorted_samples = np.sort(self.samples)[::-1]    # sort samples in descending order
		ka = int(np.ceil(self.nsamples*(1-alpha)))  # index for alpha-quantile
		qtile = sorted_samples[ka-1]
		# self.squantile = (1-(ka-1.)/(self.nsamples*(1-alpha)))*qtile + 1./(self.nsamples*(1-alpha))*np.sum(sorted_samples[0:ka-1])
		self.squantile = qtile + 1./(self.nsamples*(1-alpha))*np.sum(sorted_samples[0:ka-1]-qtile)

		# # Estimate MC error in estimating superquantile
		# sqtile_err = np.sqrt(1/(1-alpha)**2 * np.var(sorted_samples[0:ka])-qtile)
		return self.squantile
