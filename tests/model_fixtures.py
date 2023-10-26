import torch

PETRI_URLS = [
    # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/flux_typed.json", # No parameters or rate laws
    # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/flux_typed_aug.json", # No parameters or rate laws
    # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/ont_pop_vax.json", # No parameters or rate laws
    "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir.json",
    # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir_flux_span.json", # No parameters or rate laws
    # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir_typed.json",
    # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir_typed_aug.json", # No parameters or rate laws
]

REGNET_URLS = [
    # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/regnet/examples/lotka_volterra.json",
    # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/regnet/examples/syntax_edge_cases.json",
]

STOCKFLOW_URLS = [
    # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/stockflow/examples/sir.json"
]

MODEL_URLS = PETRI_URLS + REGNET_URLS + STOCKFLOW_URLS

# START_TIMES = [torch.tensor(0.0), torch.tensor(1.0), torch.tensor(2.0)]
# END_TIMES = [torch.tensor(3.0), torch.tensor(4.0), torch.tensor(5.0)]
START_TIMES = [torch.tensor(0.0)]
END_TIMES = [torch.tensor(3.0)]