import pytest
from pytest_bdd import scenario, given, when, then
import torch
from pyciemss.ODE.models import SIDARTHE
from pyciemss.utils import get_tspan
from pyciemss.ODE.askem_primitives import sample
import pyro.distributions as dist


@scenario('evaluation_scenario_2.feature', 'Unit test 1')
def test_UnitTest1():
    pass


@given("initial conditions", target_fixture="initial_conditions")
def initial_conditions():
    num_samples = 100
    N = 1
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, D0, A0, R0, T0, H0, E0 = 200/60e6, 20/60e6, 1/60e6, 2/60e6, 0, 0, 0
    # Everyone else
    S0 = N - I0 - D0 - A0 - R0 - T0 - H0 - E0
    return dict(N=1,
                initial_state=tuple(torch.as_tensor(s) for s in  (S0, I0, D0, A0, R0, T0, H0, E0)),
                final_observed_state = tuple(torch.as_tensor(s) for s in  (S0, I0, D0, A0, R0, T0, H0, E0)))

@given("parameters", target_fixture="parameters")
def parameters():
    return dict(alpha_prior=dist.Delta(torch.tensor (0.570)),
                beta_prior=dist.Delta(torch.tensor (0.011)) ,
                 gamma_prior=dist.Delta(torch.tensor (0.456)) ,
                 delta_prior=dist.Delta(torch.tensor (0.011)) ,
                 epsilon_prior=dist.Delta(torch.tensor (0.171)) ,
                 lamb_prior =dist.Delta(torch.tensor (0.034)) ,
                 zeta_prior=dist.Delta(torch.tensor (0.125)) ,
                 eta_prior=dist.Delta(torch.tensor (0.125)) ,
                 kappa_prior=dist.Delta(torch.tensor (0.017)) ,
                 theta_prior=dist.Delta(torch.tensor (0.371)) ,
                 rho_prior=dist.Delta(torch.tensor (0.034)) ,
                 xi_prior=dist.Delta(torch.tensor (0.017)) ,
                 sigma_prior=dist.Delta(torch.tensor (0.017)) ,
                 mu_prior=dist.Delta(torch.tensor (0.017)) ,
                 nu_prior=dist.Delta(torch.tensor (0.027)) ,
                 tau_prior=dist.Delta(torch.tensor (0.01)) )


@given("SIDARTHE model", target_fixture="initialize_SIDARTHE_model")
def initialize_SIDARTHE_model(initial_conditions, parameters):
    return SIDARTHE(initial_conditions["N"],**parameters)


@when("simulating the model for 100 days", target_fixture="simulate_for_days")
def simulate_for_days(initialize_SIDARTHE_model,initial_conditions,days=100):
    return sample(initialize_SIDARTHE_model, 1, initial_conditions["initial_state"], get_tspan(1, days, days))


@then("peak of infection is around day 47")
def day_around_47(simulate_for_days):
    assert torch.abs(torch.argmax(simulate_for_days["I_total_obs"]) - torch.tensor(47)) <= 5
