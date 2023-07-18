"""
This file contains functions to update the AMR model and change the parameters/distributions when debugging/offline testing.

@VigneshSella 07/17/2023
"""

import urllib.request, json


def update_AMR(SEIRD_model_url, SEIRD_model_path):
    """
    Update the AMR model from the URL in SEIRD_model_url and save it to SEIRD_model_path.
    """
    with urllib.request.urlopen(SEIRD_model_url) as url:
        json_object = json.load(url)
        with open(SEIRD_model_path, "w") as outfile:
            json.dump(json_object, outfile)

def change_model_parameters(filename, new_params):
    """
    Change the parameters in the AMR model, given by filename, to the new parameters in new_params. Where new_params is a 
    list of tuples (param, value).
        
        new params = [(param, value), (param, value)]
    """
    with open(filename, 'r') as f:
        model = json.load(f)
        # Change initial parameters
        for (param, value) in new_params:
            for idx in model["semantics"]["ode"]["parameters"]:
                if idx["id"] == param:
                    idx["value"] = value
    return model

def add_distribution(filename, new_params):
    """
    Add a distribution to the parameters in the AMR model, given by filename, to the new parameters in new_params. Where new_params is a
    list of tuples (param, new_distribution). The new distribution is given by the dict:
        new_distribution = {
            "type": "Uniform1",
            "parameters": {
              "minimum": 0.026,
              "maximum": 0.028
            }
        }
        
        new params = [(param, new_distribution), (param, new_distribution)]
    """
    with open(filename, 'r') as f:
        model = json.load(f)
        # Change initial parameters
        for (param, new_distribution) in new_params:
            for idx in model["semantics"]["ode"]["parameters"]:
                if idx["id"] == param:
                    idx["distribution"] = new_distribution
    return model