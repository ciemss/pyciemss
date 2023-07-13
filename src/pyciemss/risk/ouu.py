import numpy as np
from scipy.optimize import basinhopping
from pyro.infer import Predictive
import pyro
# TODO: generalize to other models also
from pyciemss.PetriNetODE.events import LoggingEvent, StaticParameterInterventionEvent
from pyciemss.risk.risk_measures import alpha_superquantile
from typing import Iterable, Optional, Tuple, Union

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
                 interventions: Iterable[Tuple[float, str]],
                 qoi: callable,
                 tspan: np.ndarray,
                 risk_measure: callable = alpha_superquantile,
                 num_samples: int = 1000,
                 guide=None,
                 method="dopri5"):
        self.model = model
        self.interventions = interventions
        self.qoi = qoi
        self.risk_measure = risk_measure
        self.num_samples = num_samples
        self.tspan = tspan
        self.guide = guide
        logging_events = [LoggingEvent(timepoint) for timepoint in self.tspan]
        self.model.load_events(logging_events)
        self.method = method


    def __call__(self, x):
        # Apply intervention and perform forward uncertainty propagation
        samples = self.propagate_uncertainty(x)
        # Compute quanity of interest
        sample_qoi = self.qoi(samples)
        # Estimate risk measure
        return self.risk_measure(sample_qoi)


    def propagate_uncertainty(self, x):
        '''
        Perform forward uncertainty propagation.
        '''
        pyro.set_rng_seed(0)
        # TODO: generalize for more sophisticated interventions.
        x = np.atleast_1d(x)
        interventions = []
        count=0
        for k in self.interventions:
            interventions.append(StaticParameterInterventionEvent(k[0], k[1], x[count]))
            count=count+1
        # Apply intervention to model
        self.model.load_events(interventions)
        # Sample from intervened model
        samples = Predictive(self.model, guide=self.guide, num_samples=self.num_samples)(method=self.method)
        # Remove intervention events
        self.model.remove_static_parameter_intervention_events()
        return samples


class solveOUU():
    '''
    Solve the optimization under uncertainty problem. The core of this class is a wrapper around an appropriate SciPy optimization algorithm.
    '''
    def __init__(self,
                 x0: np.ndarray,
                 objfun: callable,
                 constraints: tuple,
                 minimizer_kwargs: dict = dict(
                        method="COBYLA",
                        tol=1e-5, options={'disp': False, 'maxiter':  10},
                       ),
                 optimizer_algorithm: str = "basinhopping",
                 maxfeval: int = 100,
                 maxiter: int = 100,
                 **kwargs
                ):
        self.x0 = np.squeeze(np.array([x0]))
        self.objfun = objfun
        self.constraints = constraints
        self.minimizer_kwargs = minimizer_kwargs.update({"constraints": self.constraints})
        self.optimizer_algorithm = optimizer_algorithm
        self.maxiter = maxiter
        self.maxfeval = maxfeval        
        # self.kwargs = kwargs

    def solve(self):
        # wrapper around SciPy optimizer(s).
        minimizer_kwargs = dict(constraints=self.constraints, method='COBYLA', 
                                tol=1e-5, options={'disp': False, 'maxiter':  self.maxfeval})
        # take_step = RandomDisplacementBounds(self.u_bounds[0], self.u_bounds[1], stepsize=stepsize)
        # result = basinhopping(self._vrate, u_init, stepsize=stepsize, T=1.5, 
        #                     niter=self.maxiter, minimizer_kwargs=minimizer_kwargs, take_step=take_step, interval=2)

        result = basinhopping(self.objfun, self.x0, stepsize=0.25, T=1.5, 
                          niter=self.maxiter, minimizer_kwargs=minimizer_kwargs, 
                          interval=2, disp=False) 

        return result