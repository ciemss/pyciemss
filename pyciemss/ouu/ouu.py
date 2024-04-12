import contextlib
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pyro
import torch
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.interventional.ops import Intervention
from scipy.optimize import basinhopping
from tqdm import tqdm

from pyciemss.integration_utils.intervention_builder import (
    combine_static_parameter_interventions,
)
from pyciemss.interruptions import (
    ParameterInterventionTracer,
    StaticParameterIntervention,
)
from pyciemss.ouu.risk_measures import alpha_superquantile


class RandomDisplacementBounds:
    """
    Callable to take random displacement step within bounds
    """

    def __init__(self, xmin, xmax, stepsize=None):
        self.xmin = xmin
        self.xmax = xmax
        if stepsize:
            self.stepsize = stepsize
        else:
            # stepsize is set to 30% of longest euclidean distance
            self.stepsize = 0.3 * np.linalg.norm(xmax - xmin)

    def __call__(self, x):
        if np.any(x - self.xmax >= 0.0):
            xnew = np.clip(
                x + np.random.uniform(-self.stepsize, 0, np.shape(x)),
                self.xmin,
                self.xmax,
            )
        elif np.any(x - self.xmin <= 0.0):
            xnew = np.clip(
                x + np.random.uniform(0, self.stepsize, np.shape(x)),
                self.xmin,
                self.xmax,
            )
        else:
            xnew = np.clip(
                x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x)),
                self.xmin,
                self.xmax,
            )
        return xnew


class computeRisk:
    """
    Implements necessary forward uncertainty propagation, quantity of interest and risk measure computation.
    """

    def __init__(
        self,
        model: Callable,
        interventions: Callable[[torch.Tensor], Dict[float, Dict[str, Intervention]]],
        qoi: Callable,
        end_time: float,
        logging_step_size: float,
        *,
        start_time: float = 0.0,
        risk_measure: Callable = lambda z: alpha_superquantile(z, alpha=0.95),
        num_samples: int = 1000,
        guide=None,
        fixed_static_parameter_interventions: Dict[
            torch.Tensor, Dict[str, Intervention]
        ] = {},
        solver_method: str = "dopri5",
        solver_options: Dict[str, Any] = {},
        u_bounds: np.ndarray = np.atleast_2d([[0], [1]]),
        risk_bound: float = 0.0,
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
        self.fixed_static_parameter_interventions = fixed_static_parameter_interventions
        self.solver_method = solver_method
        self.solver_options = solver_options
        self.logging_times = torch.arange(
            start_time + logging_step_size, end_time, logging_step_size
        )
        self.u_bounds = u_bounds
        self.risk_bound = risk_bound  # used for defining penalty
        warnings.simplefilter("always", UserWarning)

    def __call__(self, x):
        if np.any(x - self.u_bounds[0, :] < 0.0) or np.any(
            self.u_bounds[1, :] - x < 0.0
        ):
            warnings.warn(
                "Selected interventions are out of bounds. Will use a penalty instead of estimating risk."
            )
            risk_estimate = max(
                2 * self.risk_bound, 10.0
            )  # used as a penalty and the model is not run
        else:
            # Apply intervention and perform forward uncertainty propagation
            samples = self.propagate_uncertainty(x)
            # Compute quanity of interest
            sample_qoi = self.qoi(samples)
            # Estimate risk
            risk_estimate = self.risk_measure(sample_qoi)
        return risk_estimate

    def propagate_uncertainty(self, x):
        """
        Perform forward uncertainty propagation.
        """
        with pyro.poutine.seed(rng_seed=0):
            with torch.no_grad():
                x = np.atleast_1d(x)
                # Combine existing interventions with intervention being optimized
                static_parameter_interventions = combine_static_parameter_interventions(
                    [
                        deepcopy(self.fixed_static_parameter_interventions),
                        self.interventions(torch.from_numpy(x)),
                    ]
                )
                static_parameter_intervention_handlers = [
                    StaticParameterIntervention(
                        time, dict(**static_intervention_assignment)
                    )
                    for time, static_intervention_assignment in static_parameter_interventions.items()
                ]

                def wrapped_model():
                    with ParameterInterventionTracer():
                        with TorchDiffEq(
                            method=self.solver_method, options=self.solver_options
                        ):
                            with contextlib.ExitStack() as stack:
                                for handler in static_parameter_intervention_handlers:
                                    stack.enter_context(handler)
                                self.model(
                                    torch.as_tensor(self.start_time),
                                    torch.as_tensor(self.end_time),
                                    logging_times=self.logging_times,
                                    is_traced=True,
                                )

                # Sample from intervened model
                samples = pyro.infer.Predictive(
                    wrapped_model,
                    guide=self.guide,
                    num_samples=self.num_samples,
                    parallel=True,
                )()
        return samples


class solveOUU:
    """
    Solve the optimization under uncertainty problem.
    The core of this class is a wrapper around an appropriate SciPy optimization algorithm.
    """

    def __init__(
        self,
        x0: List[float],
        objfun: Callable,
        constraints: Tuple[Dict[str, object], Dict[str, object], Dict[str, object]],
        minimizer_kwargs: Dict = dict(
            method="COBYLA",
            tol=1e-5,
            options={"disp": False, "maxiter": 10},
        ),
        optimizer_algorithm: str = "basinhopping",
        maxfeval: int = 100,
        maxiter: int = 100,
        u_bounds: np.ndarray = np.atleast_2d([[0], [1]]),
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
        self.u_bounds = u_bounds
        # self.kwargs = kwargs

    def solve(self):
        pbar = tqdm(total=self.maxfeval * (self.maxiter + 1))

        def update_progress(xk):
            pbar.update(1)

        # wrapper around SciPy optimizer(s)
        # rhobeg is set to 10% of longest euclidean distance
        minimizer_kwargs = dict(
            constraints=self.constraints,
            method="COBYLA",
            tol=1e-5,
            callback=update_progress,
            options={
                "rhobeg": 0.1
                * np.linalg.norm(self.u_bounds[1, :] - self.u_bounds[0, :]),
                "disp": False,
                "maxiter": self.maxfeval,
                "catol": 1e-5,
            },
        )
        take_step = RandomDisplacementBounds(self.u_bounds[0, :], self.u_bounds[1, :])
        # result = basinhopping(self._vrate, u_init, stepsize=stepsize, T=1.5,
        #                     niter=self.maxiter, minimizer_kwargs=minimizer_kwargs, take_step=take_step, interval=2)

        result = basinhopping(
            self.objfun,
            self.x0,
            T=1.5,
            niter=self.maxiter,
            minimizer_kwargs=minimizer_kwargs,
            take_step=take_step,
            interval=2,
            disp=False,
        )

        return result
