# %% [markdown]
# # Load Dependencies
# 

# %%
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import copy

from pyciemss.PetriNetODE.interfaces import setup_model, intervene, sample, calibrate, optimize
from pyciemss.utils import get_tspan
from pyciemss.utils import setup_ax, plot_predictive, plot_trajectory, plot_intervention_line, plot_ouu_risk

from pyciemss.risk.risk_measures import alpha_quantile, alpha_superquantile
from pyciemss.risk.ouu import computeRisk
from pyciemss.risk.qoi import scenario2dec_nday_average

import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # Setup parameters
# 

# %%
full_tspan = torch.tensor([float(x) for x in list(range(1,90))])

# Total population, N.
N = 100000.0
# Initial number of infected and recovered individuals, I0 and R0.
V0, I0, Iv0, R0 = 0., 81.0, 0., 0. #may want to revisit this choice, consider setting I0 to 1 (there will not be zero recovered people when there are 81 infectious people)
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - Iv0 - V0 - R0

# 18 - 24 year olds
I_obs_data = [81.47, 84.3, 86.44, 89.66, 93.32, 94.1, 96.31] #these numbers make no sense, why not use whole numbers?
plot_I_obs = dict(I_obs=torch.tensor(I_obs_data))
I_obs = [(float(i+1), dict(I_obs=obs/N)) for i, obs in enumerate(I_obs_data)]
observed_tspan = torch.tensor([float(x) for x in list(range(1,len(I_obs)+1))])

# %% [markdown]
# # Probabilistic forecasting - ignoring any observed data

# %% [markdown]
# ### Load the model

# %%
from pyciemss.PetriNetODE.models import MIRA_I_obs_with_scaled_Gaussian_noise
model_json = '../../test/models/SVIIvR_mira.json'
petri_net_ode_model = MIRA_I_obs_with_scaled_Gaussian_noise.from_mira(model_json)
petri_net_ode_model.total_population = N
petri_net_ode_model.data_reliability = 4.0
petri_net_ode_model

# %% [markdown]
# ### Initialize model

# %%
initialized_petri_net_ode_model = setup_model(petri_net_ode_model, start_time=0.0, start_state=dict(S=S0, V=V0, I=I0, I_v=Iv0, R=R0))

# %% [markdown]
# ## Q: "What likely future outcomes does our model imply while ignoring observed Data?"

# %% [markdown]
# ### Sample a single trajectory from the prior distribution

# %%
single_prior = sample(initialized_petri_net_ode_model, timepoints=full_tspan, num_samples=1)
single_prior['I_obs'] = single_prior['I_sol'] + single_prior['I_v_sol']

# %% [markdown]
# ### Sample 100 trajectories from the prior distribution

# %%
prior_prediction = sample(initialized_petri_net_ode_model, timepoints=full_tspan, num_samples=100)
prior_prediction['I_obs'] = prior_prediction['I_sol'] + prior_prediction['I_v_sol']

# %% [markdown]
# ### Plot trajectories using prior information only

# %%
ax = plot_trajectory(single_prior, full_tspan, color="blue", label="Before Seeing Data - Single Forecast (MIRA)", marker='', lw=1.)
ax = plot_predictive(prior_prediction, full_tspan, ax=ax, label="Before Seeing Data - Probabilistic Forecast (MIRA)", color="red", alpha=0.2)

# %% [markdown]
# ## Probabibilistic forecasting - incorporating observed data

# %% [markdown]
# ### Calibration

# %%
data = [(float(i+1), dict(I_obs=obs)) for i, obs in enumerate([81.47, 84.3, 86.44, 89.66, 93.32, 94.1, 96.31])]
calibrated_parameters = calibrate(initialized_petri_net_ode_model, data=data, verbose=True)

# %% [markdown]
# ### Samples from the calibrated parameters

# %%
posterior_prediction = sample(initialized_petri_net_ode_model,  inferred_parameters=calibrated_parameters, timepoints=full_tspan, num_samples=100)
posterior_prediction['I_obs'] = posterior_prediction['I_sol'] + posterior_prediction['I_v_sol']
single_posterior = sample(initialized_petri_net_ode_model,  inferred_parameters =calibrated_parameters, timepoints=full_tspan, num_samples=1)
single_posterior['I_obs'] = single_posterior['I_sol'] + single_posterior['I_v_sol']

# %% [markdown]
# ## Q: "What future outcomes are likely given the data we've seen?"

# %%
ax = plot_trajectory(plot_I_obs, get_tspan(1, len(I_obs), len(I_obs)), ax=setup_ax())
ax = plot_trajectory(single_prior, full_tspan, ax=ax, color="blue", marker='', lw=1., label="Before Seeing Data - Single Forecast")
ax = plot_predictive(prior_prediction, full_tspan, ax=ax, label="Before Seeing Data - Probabilistic Forecast", color="red")
ax = plot_trajectory(single_posterior, full_tspan, ax=ax, marker='', lw=1., label="After Seeing Data - Single Forecast")
ax = plot_predictive(posterior_prediction, full_tspan, ax=ax, label="After Seeing Data - Probabilistic Forecast")
ax = plot_trajectory(plot_I_obs, get_tspan(1,len(I_obs), len(I_obs)), ax=setup_ax())
ax = plot_predictive(posterior_prediction, full_tspan, ax=ax, label="After Seeing Data - Probabilistic Forecast (MIRA)")

# %% [markdown]
# # Probabilistic forecasting - exploring possible interventions
# 

# %% [markdown]
# 
# ## Q: "What would happen if we set the vaccination rate parameter, $\nu$, to 0.01 after 7.5 days?"
# 

# %%
# note that we cannot intervene at a previously existing timepoint, or odeint will complain.
rate_parameter_intervention = [(7.5, "nu", 0.01)]
num_samples = 100
intervened_parameter_model = intervene(initialized_petri_net_ode_model, rate_parameter_intervention)
intervened_parameter_prediction = sample(intervened_parameter_model, full_tspan, num_samples, calibrated_parameters)
intervened_parameter_prediction['I_obs'] = intervened_parameter_prediction['I_sol'] + intervened_parameter_prediction['I_v_sol']

# %%
ax = setup_ax()
ax = plot_trajectory(plot_I_obs, observed_tspan, ax=ax)
ax = plot_predictive(posterior_prediction, full_tspan, ax=ax, label="No Intervention")
ax = plot_predictive(intervened_parameter_prediction, full_tspan, tmin=7.5, ax=ax, color='red', label="Vaccination Rate Parameter Intervention")
ax = plot_intervention_line(7.5, ax=ax)

# %% [markdown]
# # Risk-based optimization under uncertainty (OUU)
# ## Q: "What is the minimal vaccination policy that results in less than 10 infected individuals after 90 days?"
# 
# ### Problem Formulation
# * **Quantity of interest**: 7-day average of total infections after 90 days
#     $$M(\mathbf{u}, \theta) = \frac{1}{7}\sum_{i=0}^6 I(t=90-i;\mathbf{u},\theta)+I_\text{V}(t=90-i;\mathbf{u},\theta)$$
# 
# * **Control**: $\mathbf{u}\in \mathcal{U} \subseteq \mathbb{R}^{n_u}$
#     * Vaccination rate parameter: $u=\nu$
# 

# %% [markdown]
# 
# * **Risk-based optimization under uncertainty problem formulation**
#     * Objective Function - Minimize the vaccination rate.
#     * Constraint - Risk of infections, $\mathcal{R}( M(\mathbf{u}, \theta))$, exceeding the prescribed threshold of 10 is below the acceptable risk threshold, $\mathcal{R}_\text{threshold}$.
# 
# \begin{equation} 
# \begin{split} 
# \mathbf{u}^*= \underset{\mathbf{u}\in\mathcal{U}}{\arg\min}\ & \lVert \mathbf{u} \rVert_1 \\ \text{s.t.}\ & \mathcal{R}( M(\mathbf{u}, \theta)) \le \mathcal{R}_\text{threshold} 
# \end{split} 
# \end{equation}

# %% [markdown]
# ## Comparing risk measures

# %% [markdown]
# ![table_risk.png](figures/table_risk.png)

# %% [markdown]
# #### Adavantages of using alpha-superquantile
# * Considers magnitude of infections exceeding the threshold:
#     * Overcome limitations of hard thresholding
#     * Desirable data-informed conservativeness
# * Preserves properties of underlying quantities of interest, such as convexity

# %% [markdown]
# ### Exploring the intervention on $\nu$ to highlight the difference between Quantiles and Superquantiles

# %%
control_model = copy.deepcopy(initialized_petri_net_ode_model)
INTERVENTION= {"intervention1": [7.5, "nu"]} # Control action / intervention
QOI = lambda y: scenario2dec_nday_average(y, contexts=["I_obs"], ndays=7)
POLICY = 0.01
N_SAMPLES = 100
RISK = computeRisk(model=control_model,
                   interventions=INTERVENTION,
                   qoi=QOI,
                   risk_measure=alpha_superquantile,
                   num_samples=N_SAMPLES,
                   tspan=full_tspan,
                   guide=calibrated_parameters)

start_time = time.time()
sq_dataCube = RISK.propagate_uncertainty(POLICY)
end_time = time.time()
forward_time = end_time - start_time
time_per_eval = forward_time / N_SAMPLES
print(f"Forward UQ took {forward_time:.2f} seconds total ({forward_time/N_SAMPLES:.2e} seconds per model evaluation).")
# Estimate QoI
sq_qoi = RISK.qoi(sq_dataCube)
# Estimate superquantile risk
sq_sv = RISK.risk_measure(sq_qoi)
# Estimate quantile risk
RISK.risk_measure = alpha_quantile
q_sv = RISK.risk_measure(sq_qoi)
print('quantile: ', q_sv, '\nalpha-superquantile: ', sq_sv)

# %%
risk_results = {"risk": [q_sv, sq_sv], "samples": sq_dataCube, "qoi": sq_qoi, "tspan": RISK.tspan}
ax1 = plot_ouu_risk(risk_results, color=['#377eb8', '#ff7f00'], label=['alpha-quantile','alpha-superquantile'], tmin=7.5)

# %% [markdown]
# ### Setup and run OUU problem
# **NOTE:** This is a demonstration of the interface. The optimizer can be run with higher maximum iterations, maximum function evaluations, and number of samples to accurately estimate the risk to achieve better convergence of the optimization results.

# %%
OBJFUN = lambda x: np.abs(x)
INTERVENTION= {"VaccinationParam": [7.5, "nu"]}
QOI = lambda y: scenario2dec_nday_average(y, contexts=["I_obs"], ndays=7)
timepoints_qoi = range(83,90)
ouu_policy = optimize(initialized_petri_net_ode_model,
                   timepoints=timepoints_qoi,
                   interventions=INTERVENTION,
                   qoi=QOI,
                   risk_bound=10.,
                   objfun=OBJFUN,
                   initial_guess=0.05,
                   bounds=[[0.],[3.]],
                   n_samples_ouu=int(2e2),
                   maxiter=0,
                   maxfeval=30,
                   inferred_parameters=calibrated_parameters)

# %% [markdown]
# ## Assess the effect of the optimal policy under uncertainty
# ### Optimum value for $\nu^*=0.012$

# %%
ouu_policy["risk"] = [ouu_policy["risk"]]
ax = plot_ouu_risk(ouu_policy, color=['#ff7f00'], tmin=7.5)
ax[0].set_title(r"Optimal $\nu$ parameter intervention"+ "\n" + r"with risk-based OUU", size=14)


