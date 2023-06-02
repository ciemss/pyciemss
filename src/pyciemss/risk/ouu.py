import numpy as np
from scipy.optimize import basinhopping, minimize
from pyro.infer import Predictive
# TODO: generalize below to other models also
# from pyciemss.PetriNetODE.events import LoggingEvent
from pyciemss.PetriNetODE.events import StartEvent, ObservationEvent, LoggingEvent, StaticParameterInterventionEvent
# from pyciemss.PetriNetODE.interfaces import intervene
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
                 interventions: list,
                #  sampling_fun: callable,
                 qoi: callable,
                #  model_state: tuple,
                 tspan: np.ndarray,
                 risk_measure: callable = alpha_superquantile,
                 num_samples: int = 1000,
                 guide=None,
                ):
        self.model = model    # Model is not required here if wrapped in intervention function
        self.interventions = interventions
        # self.sample = sampling_fun
        self.qoi = qoi
        self.risk_measure = risk_measure
        self.num_samples = num_samples
        # self.model_state = model_state
        self.tspan = tspan
        self.guide = guide
        logging_events = [LoggingEvent(timepoint) for timepoint in self.tspan]
        self.model.load_events(logging_events)


    # TODO: figure out a way to pass samples between the constraint and the optimization objective function so as not to do double the labor.
    def __call__(self, x):
        # Apply intervention, perform forward uncertainty propagation
        samples = self.propagate_uncertainty(x)
        # Compute quanity of interest
        sample_qoi = self.qoi(samples)
        # Estimate risk measure
        return self.risk_measure(sample_qoi)


    def propagate_uncertainty(self, x):
        '''
        Perform forward uncertainty propagation.
        '''
        # # Apply intervention to model
        # intervened_model = intervene(self.model, self.intervention_fun(x))
        # samples = Predictive(intervened_model, guide=self.guide, num_samples=self.num_samples)(self.model_state, self.tspan)
        # Apply intervention to model
        # intervened_model = intervene(self.model, self.intervention_fun(x))
        # intervened_model = self.intervention_fun(x)
        # samples = sample(intervened_model, timepoints=self.tspan, num_samples=self.num_samples, inferred_parameters=self.guide)
        # intervened_model.load_events(logging_events)
        # TODO: generalize for more sophisticated interventions.
        x = np.atleast_1d(x)
        interventions = []
        for count in range(len(self.interventions)):
            interventions.append(StaticParameterInterventionEvent(self.interventions[count][0], self.interventions[count][1], x[count]))
        # interventions = [StaticParameterInterventionEvent(intervention_param[0], intervention_param[1], x[count]) for count, intervention_param
        #                 in self.interventions]
        # new_petri = copy.deepcopy(petri)
        self.model.load_events(interventions)
        samples = Predictive(self.model, guide=self.guide, num_samples=self.num_samples)(method="dopri5")
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
    
    # # TODO: implement logging callback for optimizer
    # def _save(self):
    #     raise NotImplementedError
