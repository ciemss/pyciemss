import numpy as np
from scipy.optimize import basinhopping
import contextlib
import pyro
import torch
from chirho.dynamical.handlers import (
    InterruptionEventLoop,
    LogTrajectory,
    StaticIntervention,
)
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.ops import State
from chirho.observational.handlers import condition
from pyro.contrib.autoname import scope

# from pyciemss.PetriNetODE.events import LoggingEvent, StaticParameterInterventionEvent
from pyciemss.ouu.risk_measures import alpha_superquantile

# from typing import List, Optional, Tuple, Union, Dict
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from tqdm import tqdm


class RandomDisplacementBounds:
    """
    Callable to take random displacement step within bounds
    """

    def __init__(self, xmin, xmax, stepsize=0.25):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        return np.clip(
            x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x)),
            self.xmin,
            self.xmax,
        )


class computeRisk:
    """
    Implements necessary forward uncertainty propagation, quantity of interest and risk measure computation.
    """

    def __init__(
        self,
        model: Callable,
        interventions: Dict[float, Dict[str, torch.Tensor]],
        qoi: Callable,
        end_time: float,
        logging_step_size: float,
        *,
        start_time: float = 0.0,
        # tspan: np.ndarray,
        risk_measure: Callable = lambda z: alpha_superquantile(z, alpha=0.95),
        num_samples: int = 1000,
        guide=None,
        solver_method: str = "dopri5",
        solver_options: Dict[str, Any] = {},
    ):
        self.model = model
        self.interventions = interventions
        self.qoi = qoi
        self.risk_measure = risk_measure
        self.num_samples = num_samples
        # self.tspan = tspan
        self.start_time = start_time
        self.end_time = end_time
        self.guide = guide
        # logging_events = [LoggingEvent(timepoint) for timepoint in self.tspan]
        # self.model.load_events(logging_events)
        self.solver_method = solver_method
        self.solver_options = solver_options
        self.timespan = torch.arange(
            start_time + logging_step_size, end_time, logging_step_size
        )

    def __call__(self, x):
        # Apply intervention and perform forward uncertainty propagation
        samples = self.propagate_uncertainty(x)
        # Compute quanity of interest
        sample_qoi = self.qoi(samples)
        # Estimate risk measure
        return self.risk_measure(sample_qoi)

    def propagate_uncertainty(self, x):
        """
        Perform forward uncertainty propagation.
        """
        pyro.set_rng_seed(0)
        # # TODO: generalize for more sophisticated interventions.
        # x = np.atleast_1d(x)
        # interventions = []
        # count = 0
        # for k in self.interventions:
        #     interventions.append(StaticParameterInterventionEvent(k[0], k[1], x[count]))
        #     count = count + 1
        # # Apply intervention to model
        # self.model.load_events(interventions)

        # TODO: update interventions
        static_intervention_handlers = []
        count = 0
        # for k in self.interventions:
        static_intervention_handlers = [
            StaticIntervention(time, State(**static_intervention_assignment))
            for time, static_intervention_assignment in self.interventions.items()
        ]

        def wrapped_model():
            with LogTrajectory(self.timespan) as lt:
                with InterruptionEventLoop():
                    with contextlib.ExitStack() as stack:
                        for handler in static_intervention_handlers:
                            stack.enter_context(handler)
                        self.model(
                            torch.as_tensor(self.start_time),
                            torch.as_tensor(self.end_time),
                            TorchDiffEq(
                                method=self.solver_method, options=self.solver_options
                            ),
                        )

            trajectory = self.model.add_observables(lt.trajectory)

            # Adding deterministic nodes to the model so that we can access the trajectory in the Predictive object.
            [pyro.deterministic(k, v) for k, v in trajectory.items()]

            # if noise_model is not None:
            #     compiled_noise_model = compile_noise_model(
            #         noise_model, vars=set(trajectory.keys()), **noise_model_kwargs
            #     )
            #     # Adding noise to the model so that we can access the noisy trajectory in the Predictive object.
            #     compiled_noise_model(trajectory)

        # Sample from intervened model
        samples = pyro.infer.Predictive(
            wrapped_model, guide=self.guide, num_samples=self.num_samples
        )()
        # # Remove intervention events
        # self.model.remove_static_parameter_intervention_events()
        return samples


class solveOUU:
    """
    Solve the optimization under uncertainty problem. The core of this class is a wrapper around an appropriate SciPy optimization algorithm.
    """

    def __init__(
        self,
        x0: List[float],
        objfun: Callable,
        constraints: Tuple[Dict],
        minimizer_kwargs: Dict = dict(
            method="COBYLA",
            tol=1e-5,
            options={"disp": False, "maxiter": 10},
        ),
        optimizer_algorithm: str = "basinhopping",
        maxfeval: int = 100,
        maxiter: int = 100,
    ):
        self.x0 = np.squeeze(np.array([x0]))
        self.objfun = objfun
        self.constraints = constraints
        self.minimizer_kwargs = minimizer_kwargs.update(
            {"constraints": self.constraints}
        )
        self.optimizer_algorithm = optimizer_algorithm
        self.maxiter = maxiter
        self.maxfeval = maxfeval
        # self.kwargs = kwargs

    def solve(self):
        pbar = tqdm(total=self.maxfeval * (self.maxiter + 1))

        def update_progress(xk):
            pbar.update(1)

        # wrapper around SciPy optimizer(s).
        minimizer_kwargs = dict(
            constraints=self.constraints,
            method="COBYLA",
            tol=1e-5,
            callback=update_progress,
            options={"disp": False, "maxiter": self.maxfeval},
        )
        # take_step = RandomDisplacementBounds(self.u_bounds[0], self.u_bounds[1], stepsize=stepsize)
        # result = basinhopping(self._vrate, u_init, stepsize=stepsize, T=1.5,
        #                     niter=self.maxiter, minimizer_kwargs=minimizer_kwargs, take_step=take_step, interval=2)

        result = basinhopping(
            self.objfun,
            self.x0,
            stepsize=0.25,
            T=1.5,
            niter=self.maxiter,
            minimizer_kwargs=minimizer_kwargs,
            interval=2,
            disp=False,
        )

        return result