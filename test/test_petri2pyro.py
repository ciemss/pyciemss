import os

import torch
import pyro
import pytest

import numpy as np
import pyro.distributions as dist
from pyro.poutine import trace, replay, block
from pyro.infer.autoguide.guides import AutoDelta, AutoNormal
from pyro.infer import SVI, Trace_ELBO, Predictive

from pyciemss.ODE.base import PetriNetODESystem


@pytest.mark.parametrize("filename", [
    "CHIME-SVIIvR/model_petri.json",
    "CHIME-SIR/model_petri.json",
    "Bucky/model_petri.json",
])
def test_load_starterkit_scenarios_from_json(filename):
    filename = os.path.join("test/models/starter_kit_examples", filename)
    model = PetriNetODESystem.from_mira(filename)
    assert len(model.var_order) > 0


@pytest.mark.xfail(reason="mira cannot load these without annotations")
@pytest.mark.parametrize("filename", [
    "sir_age3.json",
    "sir_age16.json",
    "sir.json",
    "sird.json",
    "sirh.json",
    "sirhd_renew_vax_age16.json",
    "sirhd_renew_vax.json",
    "sirhd_renew.json",
    "sirhd_vax_age16.json",
    "sirhd_vax.json",
    "sirhd.json",
])
def test_load_stratified_acset_evaluation_examples_from_json(filename):
    filename = os.path.join("test/models/evaluation_examples", filename)
    model = PetriNetODESystem.from_mira(filename)
    assert len(model.var_order) > 0


@pytest.mark.parametrize("filename", [
    "scenario1_all_ages.json",
    "scenario1_sir.json",
    "scenario1_three_ages.json",
])
def test_load_evaluation_scenario1_from_json(filename):
    filename = os.path.join("test/models/evaluation_examples", filename)
    model = PetriNetODESystem.from_mira(filename)
    assert len(model.var_order) > 0
    if hasattr(model, "default_initial_state"):
        tspan = torch.linspace(0, 100, 100)
        solution, _ = model(model.default_initial_state, tspan)
        assert len(solution) == len(model.default_initial_state)
        assert solution[0].shape[0] == tspan.shape[0]


@pytest.mark.parametrize("filename", [
    "scenario2_sidarthe.json",
    "scenario2_sidarthe_v.json",
])
def test_load_evaluation_scenario2_from_json(filename):
    filename = os.path.join("test/models/evaluation_examples", filename)
    model = PetriNetODESystem.from_mira(filename)
    assert len(model.var_order) > 0
    assert hasattr(model, "default_initial_state")
    tspan = torch.linspace(0, 100, 100)
    solution, _ = model(model.default_initial_state, tspan)
    assert len(solution) == len(model.default_initial_state)
    assert solution[0].shape[0] == tspan.shape[0]


@pytest.mark.parametrize("filename", [
    "scenario3_biomd958.json",
    "scenario3_biomd960.json",
])
def test_load_evaluation_scenario3_from_json(filename):
    filename = os.path.join("test/models/evaluation_examples", filename)
    model = PetriNetODESystem.from_mira(filename)
    assert len(model.var_order) > 0
    if hasattr(model, "default_initial_state"):
        tspan = torch.linspace(0, 100, 100)
        solution, _ = model(model.default_initial_state, tspan)
        assert len(solution) == len(model.default_initial_state)
        assert solution[0].shape[0] == tspan.shape[0]
