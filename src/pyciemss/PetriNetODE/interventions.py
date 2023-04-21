import functools


def time_dependent_intervention_builder(name, f, tspan):
    '''
    `f(t)` is a function from time to intervention assignment.
    '''
    return {name + " %f" % (t): f(t) for t in tspan}


def state_dependent_intervention_builder(name, f, tspan):
    '''
    `f(x)` is a function from the value of the variable at trace address `name` to the intervention assigment.
    '''
    return {name + " %f" % (t): f for t in tspan}


def time_and_state_dependent_intervention_builder(name, f, tspan):
    '''
    'f(t, x)' is a function from time and the value of the variable at trace address 'name' to the intervention
    assignment.
    Note that the first argument of `f` must be time.
    '''
    # Partially evaluating `f` at `t` results in a collection of functions of only `x` indexed by time.
    # Causal pyro interprets interventions to callables `f` by calling `f` to what the value otherwise would have been.
    return {name + " %f" % (t): functools.partial(f, t) for t in tspan}


def constant_intervention_builder(name, intervention_assignment, tspan):
    '''
    `intervention_assignment` is a torch.tensor of the same size as the variable at `name`.
    '''
    return {name + " %f" % (t): intervention_assignment for t in tspan}


def parameter_intervention_builder(name, intervention_assignment):
    '''
    `intervention_assignment` is a torch.tensor of the same size as the variable at `name`.
    '''
    return {name: intervention_assignment}
