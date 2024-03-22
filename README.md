# PyCIEMSS

PyCIEMSS is a library for causal and probabilistic reasoning with continuous time dynamical systems. Specifically, PyCIEMSS provides high-level interfaces for building dynamical systems models from [standardized JSON templates](https://github.com/DARPA-ASKEM/Model-Representations), and automating a small collection of standard modeling tasks using them. PyCIEMSS is built on the more general-purpose causal probabilistic programming language [ChiRho](https://basisresearch.github.io/chirho/getting_started.html).

# Overview

PyCIEMSS automates the following standard high-level modeling tasks.

`sample`: Sample simulated trajectories from a probabilistic dynamical systems model. Optionally, add causal interventions.

`calibrate`: Update dynamical systems parameters from data. Optionally, add causal interventions.

`optimize`: Find the intervention that best achieves some objective subject to (probabilistic) constraints.

`ensemble_sample`: Sample simulated trajectories from an ensemble of probabilistic dynamical systems models.

`ensemble_calibrate`: Update dynamical systems parameters and mixture weights for an ensemble of probabilistic dynamical systems models.

# Getting Started

Install PyCIEMSS following our [installation instructions](./INSTALL.md).

See our [interfaces notebook](./docs/source/interfaces.ipynb) for example usage of all interfaces.

# Intermediate Usage

While PyCIEMSS does not require any direct interaction with ChiRho, there are some use-cases where it makes sense to (i) compile models from the [standardized JSON templates](https://github.com/DARPA-ASKEM/Model-Representations) using PyCIEMSS and then (ii) use ChiRho directly to answer advanced probabilistic and causal questions. For example, one may wish to add an uncertain intervention as described in ChiRho's [dynamical systems tutorial](https://basisresearch.github.io/chirho/dynamical_intro.html).


```
from chirho.dynamical.handlers import StaticIntervention
from pyciemss.compiled_dynamics import CompiledDynamics

import pyro
import torch

sir_model_path = ... # TODO: provide a model JSON path

sir_model = CompiledDynamics.load(sir_model_path)

start_time = torch.tensor(0.0)
end_time = torch.tensor(10.0)

intervention_time = torch.tensor(10.0)
intervention_assignment = pyro.sample("intervention_assignment", pyro.distributions.Uniform(10., 20.))

with StaticIntervention(intervention_time, {"S": intervention_assignment}):
    end_state = sir_model(start_time, end_time)
```

