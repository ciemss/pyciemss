import pytest
from pytest_bdd import scenario, given, when, then
import torch
import pyro.distributions as dist
from pyciemss.utils import get_tspan
from pyciemss.ODE.models import SIDARTHE
from pyciemss.ODE.askem_primitives import sample, intervene
from pyciemss.ODE.interventions import time_and_state_dependent_intervention_builder
from pyciemss.ODE.base import get_name


@scenario("evaluation_scenario_2.feature", "Unit test 2")
def test_UnitTest2():
    pass


@given("initial conditions", target_fixture="initial_conditions")
def initial_conditions():
    num_samples = 100
    N = 1
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, D0, A0, R0, T0, H0, E0 = 200 / 60e6, 20 / 60e6, 1 / 60e6, 2 / 60e6, 0, 0, 0
    # Everyone else
    S0 = N - I0 - D0 - A0 - R0 - T0 - H0 - E0
    return dict(
        N=1,
        initial_state=tuple(
            torch.as_tensor(s) for s in (S0, I0, D0, A0, R0, T0, H0, E0)
        ),
        final_observed_state=tuple(
            torch.as_tensor(s) for s in (S0, I0, D0, A0, R0, T0, H0, E0)
        ),
    )


@given("parameters", target_fixture="parameters")
def parameters():
    return dict(
        alpha_prior=dist.Delta(torch.tensor(0.570)),
        beta_prior=dist.Delta(torch.tensor(0.011)),
        gamma_prior=dist.Delta(torch.tensor(0.456)),
        delta_prior=dist.Delta(torch.tensor(0.011)),
        epsilon_prior=dist.Delta(torch.tensor(0.171)),
        lamb_prior=dist.Delta(torch.tensor(0.034)),
        zeta_prior=dist.Delta(torch.tensor(0.125)),
        eta_prior=dist.Delta(torch.tensor(0.125)),
        kappa_prior=dist.Delta(torch.tensor(0.017)),
        theta_prior=dist.Delta(torch.tensor(0.371)),
        rho_prior=dist.Delta(torch.tensor(0.034)),
        xi_prior=dist.Delta(torch.tensor(0.017)),
        sigma_prior=dist.Delta(torch.tensor(0.017)),
        mu_prior=dist.Delta(torch.tensor(0.017)),
        nu_prior=dist.Delta(torch.tensor(0.027)),
        tau_prior=dist.Delta(torch.tensor(0.01)),
    )


@given("SIDARTHE model", target_fixture="initialize_SIDARTHE_model")
def initialize_SIDARTHE_model(initial_conditions, parameters):
    return SIDARTHE(initial_conditions["N"], **parameters)


@given("interventions", target_fixture="interventions")
def define_interventions():
    def SI_alpha_intervention(t, flux):
        alpha = 0.570
        if t < 4:
            return flux
        elif t < 22:
            return flux / alpha * 0.422
        elif t < 28:
            return flux / alpha * 0.360
        else:
            return flux / alpha * 0.210

    def SD_beta_intervention(t, flux):
        beta = 0.011
        if t < 4:
            return flux
        elif t < 22:
            return flux / beta * 0.0057
        else:
            return flux / beta * 0.005

    def SA_gamma_intervention(t, flux):
        gamma = 0.456
        if t < 4:
            return flux
        elif t < 22:
            return flux / gamma * 0.285
        elif t < 28:
            return flux / gamma * 0.200
        else:
            return flux / gamma * 0.110

    def SR_delta_intervention(t, flux):
        delta = 0.011
        if t < 4:
            return flux
        elif t < 22:
            return flux / delta * 0.0057
        else:
            return flux / delta * 0.0057

    def ID_epsilon_intervention(t, flux):
        epsilon = 0.171
        if t < 12:
            return flux
        elif t < 38:
            return flux / epsilon * 0.143
        else:
            return flux / epsilon * 0.200

    def IA_zeta_intervention(t, flux):
        zeta = 0.125
        if t < 22:
            return flux
        elif t < 38:
            return flux / zeta * 0.034
        else:
            return flux / zeta * 0.025

    def DR_eta_intervention(t, flux):
        eta = 0.125
        if t < 22:
            return flux
        elif t < 38:
            return flux / eta * 0.034
        else:
            return flux / eta * 0.025

    def AT_mu_intervention(t, flux):
        mu = 0.017
        if t < 22:
            return flux
        else:
            return flux / mu * 0.008

    def RT_nu_intervention(t, flux):
        nu = 0.027
        if t < 22:
            return flux
        else:
            return flux / nu * 0.015

    def IH_lamb_intervention(t, flux):
        lamb = 0.034
        if t < 22:
            return flux
        else:
            return flux / lamb * 0.08

    def DH_rho_intervention(t, flux):
        rho = 0.034
        if t < 22:
            return flux
        elif t < 38:
            return flux / rho * 0.017
        else:
            return flux / rho * 0.020

    def AH_kappa_intervention(t, flux):
        kappa = 0.017
        if t < 38:
            return flux
        else:
            return flux / kappa * 0.020

    def RH_xi_intervention(t, flux):
        xi = 0.017
        if t < 38:
            return flux
        else:
            return flux / xi * 0.020

    def TH_sigma_intervention(t, flux):
        sigma = 0.017
        if t < 38:
            return flux
        else:
            return flux / sigma * 0.010

    interventions = {
        "alpha": SI_alpha_intervention,
        "beta": SD_beta_intervention,
        "gamma": SA_gamma_intervention,
        "delta": SR_delta_intervention,
        "epsilon": ID_epsilon_intervention,
        "lambda": IH_lamb_intervention,
        "zeta": IA_zeta_intervention,
        "eta": DR_eta_intervention,
        "kappa": AH_kappa_intervention,
        # "theta": AR_theta_intervention,
        "rho": DH_rho_intervention,
        "xi": RH_xi_intervention,
        "sigma": TH_sigma_intervention,
        "mu": AT_mu_intervention,
        "nu": RT_nu_intervention,
    }

    parameter_intervention = {}
    for param in interventions:
        parameter_intervention.update(
            time_and_state_dependent_intervention_builder(
                "flux_" + get_name(param2transition[param]),
                interventions[param],
                full_tspan,
            )
        )

    return parameter_intervention


@when("applying all interventions", target_fixture="intervened_model")
def apply_intervention(initialize_SIDARTHE_model, parameter_intervention):
    return intervene(initialize_SIDARTHE_model, parameter_intervention)


@when(
    "simulating the intervened model for 100 days", target_fixture="simulate_for_days"
)
def simulate_for_days(intervened_model, initial_conditions, days=100):
    return sample(
        intervened_model,
        1,
        initial_conditions["initial_state"],
        get_tspan(1, days, days),
    )


@then("peak of infection is around day 50")
def day_around_50(simulate_for_days):
    assert (
        torch.abs(torch.argmax(simulate_for_days["I_total_obs"]) - torch.tensor(50))
        <= 3
    )


@then("percent infected is around 0.2%")
def percent_around_002(simulate_for_days):
    assert (
        torch.abs(torch.max(simulate_for_days["I_total_obs"]) - torch.tensor(0.2))
        <= 0.1
    )
