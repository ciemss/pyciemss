######## Written by: Anirban Chaudhuri (Oden Institute; UT Austin)
import os
import datetime
import time
import pickle
# import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import ticker
from multiprocessing import set_start_method, get_context

from CIEMSS.digitaltwin import ODETwin, ODEState
from CIEMSS.model import ODESVIIvR
from CIEMSS.control import Control
from CIEMSS.control.risk_ouu import OptProblem
from CIEMSS.control.risk_measures import Risk


def decision_metric_samples(I, Iv, tf=90, metric='infections', ndays=7, dt=1.):
	# returns distribution of ndays average of total infections at given time tf
	if metric == 'infections':
		# Estimate n-day average of cases
		I_ndays = I[:,int(tf/dt)-ndays+1:int(tf/dt)+1]
		Iv_ndays = Iv[:,int(tf/dt)-ndays+1:int(tf/dt)+1]
		samples = np.mean(I_ndays + Iv_ndays,axis=1)
		samp = samples

	return samples



def plot_forward_UQ(samples, times, ptiles=[0, 100], lab='prior', ax=None, **kwargs):
	'''
	Plot forward UQ (Uncertainty Propagation)
	INPUT:
		- samples: Monte Carlo results
		- times: simulation times
		- ptiles: Percentiles to plot for UP
	'''
	if ax is None:
		ax = plt.gca()

	# ODE solutions at low, median, high
	samples = np.array(samples)  # ensure NumPy array
	median = np.median(samples, axis=0)
	ptiles_lo = np.percentile(samples, ptiles[0], axis=0)
	ptiles_hi = np.percentile(samples, ptiles[1], axis=0)

	# Plot solutions
	sideaxis(ax)
	ax.plot(times, median, linewidth=2, label=lab, **kwargs)
	ax.plot(times, ptiles_lo, linewidth=1, linestyle='--', **kwargs)
	ax.plot(times, ptiles_hi, linewidth=1, linestyle='--', **kwargs)

	# Add shading between percentiles
	ax.fill_between(times, ptiles_lo, ptiles_hi, alpha=0.15, **kwargs)

	# Adjust MPL style
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif', size=18.)

	# Set axis labels, legend
	ax.set_xlabel('Time (days)', fontsize=20)
	ax.set_ylabel('Number infected', fontsize=20)
	ax.legend()
	ax.ticklabel_format(axis='y', style='sci')
	return ax


def sideaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	return

def sideaxishist(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	# For y-axis
	ax.yaxis.set_major_locator(ticker.NullLocator())
	ax.tick_params(axis='x', labelsize=12)  # change fontsize for x-axis tick labels
	ax.xaxis.major.formatter._useMathText = True
	return


def run_svivr_risk():
	#######################################
	# Initializations
	#######################################
	start_time = time.time()
	SAVE_PLOTS = True
	# co = ['#377eb8', '#ff7f00', '#e41a1c', '#4daf4a', '#984ea3', '#ffd92f', '#a65628']
	co = ['#377eb8', '#ff7f00', '#984ea3', '#ffd92f', '#a65628']
	# co = ['#d17145', '#7f64b9', '#c36785']
	rlabel = [r'$Q_{\alpha}$-based OUU', r'$\bar{Q}_{\alpha}$-based OUU']
	# rlabel = [r'$Q_{\alpha}$', r'$\bar{Q}_{\alpha}$']

	risk_metrics=['alpha_quantile', 'alpha_superquantile']
	risk_params=[0.95, 0.95]

	Nfwd_samples = int(1e4)

	FIGSIZE = (24, 6)

	# ptiles = [5,95]
	ptiles = [0, 100]
	# Parameter names
	pnames = [r'$\beta$', r'$\beta_V$',
			  r'$\gamma$', r'$\gamma_V$']

	#################### Generic inputs
	decision_metric = 'infections'
	tf = 90
	N_SAMPLES_inv = int(1e5)
	N_SAMPLES_fwd = int(1e4)
	N_SAMPLES_ouu = int(5e3)
	N_SAMPLES=[N_SAMPLES_inv, N_SAMPLES_fwd, N_SAMPLES_ouu]
	maxfeval=50
	maxiter=10

	#######################################
	# Simulation Studies
	#######################################
	cwd = os.getcwd()
	OUTDIR = os.path.join(cwd, 'results/ouu/')
	os.makedirs(OUTDIR, exist_ok=True)
	FORMAT = 'pdf'
	fext = 'Query2_' + decision_metric
	fext1 = fext + '_vrate05'
	fpriorposttraj = fext1 + '_trajectory.'
	fpriorposthist = fext1 + '_rvsDist.'
	fname_out = fext + '_ouu5e3_'+ str(maxfeval) + '_' + str(maxiter) + '.pkl'

	############### Initialize twin
	twin = ODETwin()
	twin.control = Control().vaccination_rate(0.005)

	# TODO
	#######################################
	# Generate observations
	#######################################
	#######################################
	# Solve inverse problem
	#######################################
	
	# Use priors to run the analysis
	print('Performing forward UQ with priors')
	outputs_MC, samples_MC, t = twin.forward_UQ(num_samples=N_SAMPLES_fwd, tf=tf, rseed=1)
	S = outputs_MC['S']
	V = outputs_MC['V']
	I = outputs_MC['I']
	Iv = outputs_MC['Iv']
	R = outputs_MC['R']

	#######################################
	# Plot infections
	#######################################
	# Plot only once since it is the same forall thresholds
	fig1 = plt.figure(figsize=FIGSIZE)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif', size=18.)
	# plt.title('Prior vs posterior (SOC)')
	ax = plt.gca()
	plot_forward_UQ(I, t, color=co[0], lab=r'Infections', ptiles=ptiles, ax=ax)
	plot_forward_UQ(Iv, t, color=co[1], lab=r'Infections vaccinated', ptiles=ptiles, ax=ax)
	fig1.savefig(os.path.join(OUTDIR, fpriorposttraj+FORMAT), format=FORMAT, bbox_inches='tight')
	plt.close()

	# Histograms for random variables
	bins_hist=50
	fig, ax = plt.subplots(1, len(pnames), figsize=FIGSIZE)
	ax = np.hstack(ax)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif', size=6.)

	# Build Subplot
	for i in range(len(pnames)):
		cax = ax[i]
		sideaxishist(cax)
		cax.hist(samples_MC[:, i], color=co[0], bins=bins_hist, histtype='stepfilled', alpha=0.5, label='prior')
		# cax.hist(samples_MC_posterior[:, i], color=colors[1], bins=bins_hist, histtype='stepfilled', alpha=0.5, label='posterior')
		cax.set_xlabel(pnames[i], usetex=True, size=20)
	ax[-1].legend(loc='upper right', prop={'size': 20})
	fig.savefig(os.path.join(OUTDIR, fpriorposthist+FORMAT), format=FORMAT, bbox_inches='tight')
	plt.close()

	#######################################
	# Estimate risk
	#######################################
	# Get samples of total infections for each day
	Itotal_samples = decision_metric_samples(I, Iv, tf=tf, metric='infections', dt=1.)

	# # Change control if desired (helpful for query 1 comparison)
	# twin.control = Control().vaccination_rate(0.005)            
	# outputs_MC, samples_MC, t = twin.forward_UQ(num_samples=N_SAMPLES_fwd, tf=tf, rseed=1)
	# S = outputs_MC['S']
	# V = outputs_MC['V']
	# I = outputs_MC['I']
	# Iv = outputs_MC['Iv']
	# R = outputs_MC['R']
	# # Get samples of total infections for each day
	# Itotal_samples = decision_metric_samples(I, Iv, tf=tf, metric='infections', dt=1.)

	# Estimate risk
	meanval = np.mean(Itotal_samples)
	stdval = np.std(Itotal_samples)
	qbarval = getattr(Risk(Itotal_samples),'alpha_superquantile')(0.95)
	qval = getattr(Risk(Itotal_samples),'alpha_quantile')(0.95)
	print(meanval, stdval, qval, qbarval)

	#### plot total infections distribution at 90 days
	bins_hist=50
	fig1 = plt.figure()
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif', size=18.)
	cax = plt.gca()
	sideaxishist(cax)
	cax.hist(Itotal_samples, color=co[2], bins=bins_hist, histtype='stepfilled', alpha=0.5, label='total infections')
	miny = min(cax.get_ylim())
	maxy = max(cax.get_ylim())
	cax.vlines(qval, miny, maxy, linestyle='--', linewidth=2.5, label='alpha-quantile', color=co[0])
	cax.vlines(qbarval, miny, maxy, linestyle='--', linewidth=2.5, label='alpha-superquantile', color=co[1])
	cax.set_xlabel('Total infections after 90 days (7-day average)', usetex=True, size=20)
	cax.legend(loc='upper right', prop={'size': 20})
	fig1.savefig(os.path.join(OUTDIR, fext+'_90days_hist.'+FORMAT), format=FORMAT, bbox_inches='tight')
	plt.close()

	#######################################
	# Risk-based OUU
	#######################################
	print('Performing risk-based OUU with priors')
	if os.path.exists(os.path.join(OUTDIR,fname_out)):
		with open(os.path.join(OUTDIR,fname_out),"rb") as f:
			data_dict=pickle.load(f)
		risk_res = data_dict['risk_res']
		risk_metrics = data_dict['risk_metrics']
		risk_params = data_dict['risk_params']
		N_SAMPLES = data_dict['N_SAMPLES']
	else:
		risk_res = [[] for i in range(len(risk_metrics))] # list to store results for each risk with iterating over thresholds
		for jj, risk_met in enumerate(risk_metrics):
			print(risk_met)
			start_time1 = time.time()
			optprob = OptProblem(twin, decision_metric='infections', risk_metric=risk_met, risk_params=risk_params[jj],
				t0=0, tf=90, nu_bounds=np.array([0., 1.]), num_samples=N_SAMPLES_ouu, 
				maxfeval=maxfeval, maxiter=maxiter)
			
			#############Minimizing dosage for constraint on risk
			risk_bound = 10.
			risk_res[jj] = optprob.solve_minvrate(risk_bound=risk_bound)

			print('Best vaccination rate under ' + risk_met + ':')
			print(risk_res[jj].x)
			print(risk_res[jj])    # prints the entire summary of the optimization
			# print('Best objective function value under ' + risk_met + ':' + str(risk_res[jj].fun))
			print('--- %s seconds for OUU ---' % (time.time() - start_time1))


		#######################################
		# Write results to file
		#######################################
		print('----------------------')
		print('Saving results to file:\t' + os.path.join(OUTDIR,fname_out))
		print('----------------------')
		results = {
			'risk_res': risk_res,
			'risk_metrics': risk_metrics,
			'risk_params': risk_params,
			'N_SAMPLES': N_SAMPLES,
			'maxfeval': maxfeval,
			'maxiter': maxiter
		}
		
		with open(os.path.join(OUTDIR, fname_out),'wb') as f:
			pickle.dump(results, f)

	#######################################
	## Post-process OUU results
	print('Post-processing OUU results...')
	out_MC = [[] for i in range(len(risk_metrics))]
	t_all = [[] for i in range(len(risk_metrics))]
	samples = [[] for i in range(len(risk_metrics))]
	qval = [[] for i in range(len(risk_metrics))]      # 95-quantiles
	qbarval = [[] for i in range(len(risk_metrics))]   # 95-superquantile

	for jj, risk_met in enumerate(risk_metrics):
		twin.control = Control().vaccination_rate(risk_res[jj].x)            
		out_MC[jj], _, t_all[jj] = twin.forward_UQ(num_samples=N_SAMPLES_fwd, tf=tf, rseed=1)
		# Get samples of total infections for each day
		samples[jj] = decision_metric_samples(out_MC[jj]['I'], out_MC[jj]['Iv'], tf=tf, metric='infections', dt=1.)
		# Estimate risk
		qval[jj] = getattr(Risk(samples[jj]),'alpha_quantile')(0.95)
		qbarval[jj] = getattr(Risk(samples[jj]),'alpha_superquantile')(0.95)
		print(risk_met, qval[jj], qbarval[jj])

	########## Plot I_total distribution
	fig1 = plt.figure()
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif', size=18.)
	cax = plt.gca()
	sideaxishist(cax)
	for jj, risk_met in enumerate(risk_metrics):
		cax.hist(samples[jj], color=co[jj+1], bins=50, histtype='stepfilled', alpha=0.5, label=rlabel[jj])
	cax.set_xlabel('Total infections after 90 days (7-day average)', usetex=True, size=20)
	cax.legend(loc="upper right", prop={'size': 14})
	plt.tight_layout()
	fig1.savefig(os.path.join(OUTDIR, fext+'_infectionsdist_ouu.'+FORMAT), format=FORMAT, bbox_inches='tight')
	plt.close()

	print('--- %s seconds ---' % (time.time() - start_time))

	return


####################################################
if __name__ == '__main__':
	run_svivr_risk()