import json
import dill
from fastapi import FastAPI, Request, Response
from fastapi.logger import logger
from pydantic import BaseModel, Field
from typing import Type, Dict, List, Any, Iterable, Union
from networkx import Graph
from io import BytesIO
import torch

from pyciemss.ODE import interventions
from pyciemss.ODE.frontend import compile_pp, sample, infer_parameters, intervene
from pyciemss.utils import load as load_petrinet, add_state_indicies, get_tspan


class Timespan(BaseModel):
    start: int = Field(default=0)
    end: int = Field()
    step: int = Field(default=1)


class CompilePayload(BaseModel):
    petrinet: Dict[str, Any] = Field()
    priors: Dict[str, List[Any]] = Field()


class SamplePayload(BaseModel):
    ode_model: str = Field()
    num_samples: int = Field()
    initial_state: Iterable[Union[int, float]] = Field()
    tspan: Timespan = Field()


class InferPayload(BaseModel):
    ode_model: str = Field()
    num_iterations: int = Field()
    hidden_observations: List[str] = Field()
    data: Dict[str, Any] = Field()
    initial_state: Iterable[Union[int, float]] = Field()
    observed_tspan: Timespan = Field()


class InterventionPayload(BaseModel):
    ode_model: str = Field()
    intervention_class: str = Field()
    name: str = Field()
    assignment: Iterable[Union[int, float]] = Field()
    tspan: Timespan = Field()


# Tensor capable encoder
# WARNING: This is probably lossy. An alternative way of 
# encoding/serializing Tensors is recommended.
class TensorJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, torch.Tensor):
            return list(float(i) for i in o) 
        return super().default(o)


api = FastAPI(docs_url="/")

@api.post("/compile")
def compile_req(compile_payload: CompilePayload):
    petrinet = load_petrinet(compile_payload.petrinet)
    petrinet = add_state_indicies(petrinet)

    pyro_ode_model = compile_pp(
        petri_G=petrinet, 
        prior_json=compile_payload.priors
    )
    ser = dill.dumps(pyro_ode_model).hex()
    return {"ode_model": ser}
    
    
@api.post("/sample")
def sample_req(sample_payload: SamplePayload):
    ode_model = dill.loads(bytes.fromhex(sample_payload.ode_model))

    tspan = get_tspan(
        sample_payload.tspan.start, 
        sample_payload.tspan.end, 
        sample_payload.tspan.step
    )
    state = tuple(torch.as_tensor(s) for s in sample_payload.initial_state)
    prior_data_cube = sample(
        ode_model,
        sample_payload.num_samples,
        state,
        tspan,
    )
    return Response(
        content=json.dumps(prior_data_cube, cls=TensorJSONEncoder),
        headers={
            'content-type': 'application/json',
        }
    )


@api.post("/infer_parameters")
def infer_req(infer_payload: InferPayload):
    ode_model = dill.loads(bytes.fromhex(infer_payload.ode_model))

    initial_state = tuple(torch.as_tensor(s) for s in infer_payload.initial_state)
    observed_tspan = get_tspan(
        infer_payload.observed_tspan.start, 
        infer_payload.observed_tspan.end, 
        infer_payload.observed_tspan.step
    )
    data = {
        k: torch.tensor(v) if v is not None else None 
        for k, v in infer_payload.data.items()
    }
    
    inferred_params = infer_parameters(
        ode_model,
        infer_payload.num_iterations,
        infer_payload.hidden_observations,
        data,
        initial_state,
        observed_tspan
    )
    return {"representation": repr(inferred_params)}

    
@api.post("/intervene")
def intervene_req(intervention_payload: InterventionPayload):
    ode_model = dill.loads(bytes.fromhex(intervention_payload.ode_model))

    intervention_class = getattr(interventions, intervention_payload.intervention_class)
    tspan = get_tspan(
        intervention_payload.tspan.start, 
        intervention_payload.tspan.end, 
        intervention_payload.tspan.step
    )
    intervention = intervention_class(
        intervention_payload.name,
        intervention_payload.assignment,
        tspan,
    )

    intervened_ode_model = intervene(ode_model, intervention)
    logger.warn(intervened_ode_model)

    # TODO: Return something useful here

    return {}