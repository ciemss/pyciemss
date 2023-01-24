def time_dependent_intervention(name, f, tspan):
    '''
    `f` is a function from time to intervention assignment.
    '''
    return {name + " %f" % (t): f(t) for t in tspan}

def state_dependent_intervention(name, f, tspan):
    '''
    `f` is a function from the value of the variable at trace address `name` to intervention assigment.
    '''
    return {name + " %f" % (t): f for t in tspan}

def constant_intervention(name, intervention_assignment, tspan):
    '''
    `intervention_assignment` is a torch.tensor of the same size as the variable at `name`.
    '''
    return {name + " %f" %(t): intervention_assignment for t in tspan}

def parameter_intervention(name, intervention_assignment):
    '''
    `intervention_assignment` is a torch.tensor of the same size as the variable at `name`.
    '''
    return {name: intervention_assignment}
    
