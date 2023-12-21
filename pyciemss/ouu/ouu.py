import numpy as np
from scipy.optimize import basinhopping
import contextlib
import pyro
import torch
from chirho.dynamical.handlers import (
    LogTrajectory,
)
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.interventional.ops import Intervention

from pyciemss.ouu.risk_measures import alpha_superquantile
from pyciemss.interruptions import (
    StaticParameterIntervention,
)

from typing import Any, Callable, Dict, List, Tuple
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
        interventions: Dict[torch.Tensor, Dict[str, Intervention]],
        qoi: Callable,
        end_time: float,
        logging_step_size: float,
        *,
        start_time: float = 0.0,
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
        self.solver_method = solver_method
        self.solver_options = solver_options
        print(start_time + logging_step_size, end_time, logging_step_size)
        self.timespan = torch.arange(start_time + logging_step_size, end_time, logging_step_size)

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
        x = np.atleast_1d(x)
        # Create intervention handlers
        static_parameter_intervention_handlers = []
        count = 0
        for time, param in self.interventions.items():
            static_parameter_intervention_handlers = static_parameter_intervention_handlers + [
                StaticParameterIntervention(time, dict([(param, torch.as_tensor(x[count]))]))
            ]
            count = count + 1

        def wrapped_model():
            with LogTrajectory(self.timespan) as lt:
                with TorchDiffEq(method=self.solver_method, options=self.solver_options):
                    with contextlib.ExitStack() as stack:
                        for handler in static_parameter_intervention_handlers:
                            stack.enter_context(handler)
                        self.model(torch.as_tensor(self.start_time), torch.as_tensor(self.end_time))

            trajectory = lt.trajectory
            [pyro.deterministic(f"{k}_state", v) for k, v in trajectory.items()]

            # Need to add observables to the trajectory, as well as add deterministic nodes to the model.
            trajectory_observables = self.model.observables(trajectory)
            [
                pyro.deterministic(f"{k}_observable", v)
                for k, v in trajectory_observables.items()
            ]
        
        # Sample from intervened model
        samples = pyro.infer.Predictive(
            wrapped_model, guide=self.guide, num_samples=self.num_samples
        )()
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