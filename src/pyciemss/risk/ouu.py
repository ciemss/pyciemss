import numpy as np
from scipy.optimize import basinhopping, minimize

from pyciemss.interfaces import intervene
from pyro.infer import Predictive

from pyciemss.risk.risk_measures import alpha_superquantile

class RandomDisplacementBounds():
    '''
    Callable to take random displacement step within bounds
    '''
    def __init__(self, xmin, xmax, stepsize=0.25):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        return np.clip(x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x)), self.xmin, self.xmax)


class computeRisk():
    '''
    Implements necessary forward uncertainty propagation, quantity of interest and risk measure computation.
    '''
    def __init__(self,
                 model: callable,
                 intervention_fun: callable,
                 qoi: callable,
                 model_state: tuple,
                 tspan: np.ndarray,
                 risk_measure: callable = alpha_superquantile,
                 num_samples: int = 1000,
                 guide=None,
                ):
        self.model = model
        self.intervention_fun = intervention_fun
        self.qoi = qoi
        self.risk_measure = risk_measure
        self.num_samples = num_samples
        self.model_state = model_state
        self.tspan = tspan
        self.guide = guide


    # TODO: figure out a way to pass samples between the constraint and the optimization objective function so as not to do double the labor.
    def __call__(self, x):
        # Apply intervention, perform forward uncertainty propagation
        samples = self.propagate_uncertainty(x)

        # Compute quanity of interest
        sample_qoi = self.qoi(samples)

        # Compute risk measure
        return self.risk_measure(sample_qoi)


    def propagate_uncertainty(self, x):
        '''
        Perform forward uncertainty propagation.
        '''
        # Apply intervention to model
        intervened_model = intervene(self.model, self.intervention_fun(x))

        samples = Predictive(intervened_model, guide=self.guide, num_samples=self.num_samples)(self.model_state, self.tspan)

        return samples


class solveOUU():
    '''
    Solve the optimization under uncertainty problem. The core of this class is a wrapper around an appropriate SciPy optimization algorithm.
    '''
    def __init__(self,
                 x0: np.ndarray,
                 objfun: callable,
                 constraints: dict,
                 minimizer_kwargs: dict = dict(
                        method="COBYLA",
                        options={
                                 "disp": False,
                                },
                       ),
                 optimizer_algorithm: str = "basinhopping",
                 maxiter: int = 100,
                 **kwargs
                ):
        self.x0 = x0
        self.objfun = objfun
        self.constraints = constraints
        self.minimizer_kwargs = minimizer_kwargs.update({"constraints": self.constraints})
        self.optimizer_algorithm = optimizer_algorithm
        self.maxiter = maxiter

        self.kwargs = kwargs

    def solve(self):
        # Thin wrapper around SciPy optimizer(s).
        # Note: not sure that there is a cleaner way to specify the optimizer algorithm as this call must interface with the SciPy optimizer which is not consistent across algorithms.

        if self.optimizer_algorithm == "basinhopping":
            result = basinhopping(
                func=self.objfun,
                x0=self.x0,
                niter=self.maxiter,
                minimizer_kwargs=self.minimizer_kwargs,
                **self.kwargs
            )
        else:
            # TODO: extend so as not just to be a pass through to minimize
            result = minimize(
                fun=self.objfun,
                x0=self.x0,
                method=self.optimizer_algorithm,
                constraints=self.constraints,
                **self.kwargs
            )

        return result

    # TODO: implement logging callback for optimizer
    def _save(self):
        raise NotImplementedError
