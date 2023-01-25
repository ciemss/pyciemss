import numpy as np
from scipy.optimize import basinhopping

from causal_pyro.query.do_messenger import do
from pyro.infer import Predictive

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
                 risk_measure: callable,
                 num_samples: int,
                 model_state: tuple,
                 tspan: np.ndarray,
                 guide=None,
                 compartment: str = None,
                ):
        self.model = model
        self.intervention_fun = intervention_fun
        self.qoi = qoi
        self.risk_measure = risk_measure
        self.num_samples = num_samples
        self.model_state = model_state
        self.tspan = tspan
        self.guide = guide
        self.compartment = compartment


    def __call__(self, x):
        # Apply intervention to model
        intervened_model = do(self.model, self.intervention_fun(x))
        
        # Perform forward uncertainty propagation
        if self.guide is not None:
            samples = Predictive(intervened_model, guide=self.guide, num_samples=self.num_samples)(self.model_state, self.tspan)
        else:
            samples = Predictive(intervened_model, num_samples=self.num_samples)(self.model_state, self.tspan)
        
        # TODO: add generality for QoI dealing with multiple compartments
        if self.compartment is not None:
            samples = samples[self.compartment].detach().numpy()

        # Compute quanity of interest
        sample_qoi = self.qoi(samples)
        
        # Compute risk measure
        return self.risk_measure(sample_qoi)


class solveOUU():
    '''
    Solve the optimization under uncertainty problem. The core of this class is a wrapper around an appropriate SciPy optimization algorithm.
    '''
    def __init__(self,
                 x0: np.ndarray,
                 objfun: callable,
                 constraints: dict,
                 minimizer_kwargs: dict,
                 optimizer_algorithm: str = "basinhopping",
                 maxiter: int = 100,
                 **kwargs
                ):
        self.x0 = x0
        self.objfun = objfun
        self.constraints = constraints
        self.minimizer_kwargs = minimizer_kwargs
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
            pass # TODO: implement other optimizers

        return result
