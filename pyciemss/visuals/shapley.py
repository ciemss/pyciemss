import pandas as pd
from . import vega
from datetime import datetime
import itertools
import torch
import pyciemss
import random
import warnings

"""
Shapley values are a tool to explain the contribution of different variables to an outcome.
They were originally developed in economics, and have been applied to general machine learning.

In the original formulation, Shapley values are computed over coallitions of players to determine
what impact having/removing a player from from a coalition will have on a "payoff" function.  In
the context of pyciemss simulations, a "player" is a specific input variable value and the
"payoff" is another variables value.  So a model with two integer inputs ranging from 1-10 and an output
would have twenty possible players (10 for each variable) in 100 different coalitions.

The gist of interpreting Shapley values is they allow you to rank the importance of different
parameter values to the outcome.  They do not directly indicate the magnitude of the change,
but a variable with a higher shapley value will generally give a larger change than a variable
with a lower shapley value.More details on how to interpet Shapley values can be
found at: christophm.github.io/interpretable-ml-book/shapley.html.

This package includes methods or *estimating* shapley values on pyciemss models (exact calculation
is combintoric in the number of stats) and communicating the results of those estimations.  The
visualizations are based on those in the SHAP package (shap.readthedocs.io).


"""

# SHAP inspired plots (https://shap.readthedocs.io/en/latest/)


def shapley_decision_plot(deltas: pd.DataFrame, players, payout) -> vega.VegaSchema:
    """
    Generates a Shapley decision plot using the provided explainer output.

    Parameters:
    explainer_output (Any): The output from a Shapley explainer which contains the data to be visualized.

    Returns:
    vega.VegaSchema: A Vega schema object representing the Shapley decision plot.


    """

    schema = vega.load_schema("shapley_decision_plot.vg.json")

    fields = [f"{p[2]}_delta" for p in players]
    wide = deltas.groupby("individual")[fields].mean()

    narrow = wide.reset_index().melt("individual", value_name="shapley_value")
    combined = narrow.assign(expected_value=deltas[payout].mean())
    records = combined.to_dict(orient="records")

    # load heatmap data
    schema["data"] = vega.replace_named_with(schema["data"], "table", ["values"], records)
    return schema


def shapley_bar_chart(deltas: pd.DataFrame, players) -> vega.VegaSchema:
    """
    Generates a Shapley bar chart visualization using the provided explainer output.

    Parameters:
    explainer_output (dict): The output from a Shapley explainer, containing the Shapley values for different features.

    Returns:
    vega.VegaSchema: A Vega schema object representing the Shapley bar chart.
    """

    schema = vega.load_schema("shapley_bar_chart.vg.json")
    fields = [f"{p[2]}_delta" for p in players]
    records = [
        {"variable": " ".join(k.split("_")[:-1]), "Mean Shapley Value": v}
        for k, v in deltas[fields].mean().to_dict().items()
    ]

    # load heatmap data
    schema["data"] = vega.replace_named_with(schema["data"], "table", ["values"], records)
    return schema


def shapley_waterfall(deltas: pd.DataFrame, players, payout) -> vega.VegaSchema:
    """
    Generates a Shapley waterfall plot using the provided explainer output.

    Parameters:
    explainer_output (dict): The output from a Shapley explainer containing the Shapley values.

    Returns:
    vega.VegaSchema: A Vega schema object representing the Shapley waterfall plot.
    """

    schema = vega.load_schema("shapley_waterfall.vg.json")
    # json_data = process_explainer_output(explainer_output, return_mean_shapley=True)
    fields = [f"{p[2]}_delta" for p in players]
    wide = deltas.groupby("individual")[fields].mean()

    narrow = wide.reset_index().melt("individual", value_name="Mean Shapley Value")
    mean = narrow.drop(columns=["individual"]).groupby("variable").mean()
    combined = mean.assign(expected_value=deltas[payout].mean())
    records = combined.reset_index().to_dict(orient="records")

    # load heatmap data
    schema["data"] = vega.replace_named_with(schema["data"], "table", ["values"], records)
    return schema


# Compute Shapley Values ---------------------------
class Sampler:
    def __init__(self, target_inputs):
        self.target_inputs = target_inputs

    def __next__(self):
        return {var: torch.tensor(strategy()) for var, strategy in self.target_inputs}


def choose(values):
    return lambda: random.choice(values)


def random_discretized(*, n, center, step):
    """Create a sequence of n sample points centered on center the given step-size"""
    low = [*reversed([center - (step * i) for i in range(1, n + 1)])]
    high = [center + (step * i) for i in range(n + 1)]
    steps = low + high
    return lambda: random.choice(steps)


class TQDM_noop:
    """Provides a no-op replacement for the parts of TQDM used"""

    n = 0

    def update(self, *_, **__):
        pass

    def set_description(self, *_, **__):
        pass

    def close(self, *_, **__):
        pass


def neyman_method(
    calibrated_model,
    parameter_estimates,
    players,
    payout,
    *,
    end_time,
    logging_step_size,
    num_samples,
    start_time,
    budget=1000,
    silent=False,
):
    """
    calibrated_model -- Model that has been calibrated to an objective function
    parameter_estimates -- Full set of calibrated inputs
    players -- Tripples of (input var name, sampling strategy, output var name)
        - The "input var name" is the name of the variable in the model.
        - The "output var name" is the name of the variable in the pyciemss
          dataframe the corresponds to the input var name.
        - The "sampling strategy" is a function of zero variables (e.g., a thunk)
          that produce an single input value when invoked.
          The sampling strategy may randomly generate values or it may iterate over them,
          but it should always return exactly
          one value when invoked.  It is essentially "picking a player" and invoking the
          sampling strategy for each variable builds the coalition being tested.
    payout -- Which *single* output parameter is of interest?  This is a name of a variable in the output dataframe,
       similar tot he "output var name" in the players definition, but for a field corresponding to an output variable
       in the model.

    #TODO: Make 'payout' a function fo the result (not just a single column).  This could
    #       be done with a callable that combines several columns into one

    budget -- Number of seconds to keep sampling

    Returns an list of "observations".  Each observation is a timeseries dataframe representing a single simulation.
      It only includes the "player" variable values and payout.
    """

    if silent:
        progress = TQDM_noop()
    else:
        try:
            import tqdm

            progress = tqdm.tqdm(desc="Shapley sample 0", total=budget, unit="seconds")
            warnings.filterwarnings("ignore", category=tqdm.TqdmWarning)
        except ImportError:
            warnings.warn("Could not import tqdm, running as-if silent=True.")
            progress = TQDM_noop()

    input_gen = Sampler([p[:2] for p in players])
    output_names = [p[2] for p in players] + [payout]
    observations = []

    start = datetime.now()

    for i in itertools.count():
        test_intervention = {torch.tensor(0): next(input_gen)}
        progress.set_description(f"Shapley sample {i+1}")
        test_results = pyciemss.sample(
            calibrated_model,
            end_time,
            logging_step_size,
            num_samples,
            start_time=start_time,
            inferred_parameters=parameter_estimates,
            static_parameter_interventions=test_intervention,
        )

        observations.append(test_results["data"][output_names])

        elapse = (datetime.now() - start).total_seconds()
        progress.update(elapse - progress.n)
        if elapse > budget:
            break

    progress.close()
    progress.set_description(f"Stopped after {i} samples and {elapse} seconds")
    return observations


def average_marginal_contribution(baseline, observations, players, payout):
    # def znorm(s):
    #     return (s - s.mean()) / s.std()

    # input_names = [p[2] for p in players]

    deltas = [(obs - baseline[obs.columns]).assign(individual=i) for i, obs in enumerate(observations)]
    deltas = (
        pd.concat(observations)
        .reset_index(drop=False, names="time")
        .join(pd.concat(deltas).reset_index(drop=True), rsuffix="_delta")
    )
    # fine = {name: deltas.groupby(f"{name}_delta")[f"{payout}_delta"].mean() for name in input_names}
    # fine_norm = {name: znorm(value) for name, value in fine.items()}
    # coarse = {name: value.mean() for name, value in fine.items()}
    # coarse_norm = {name: value.mean() for name, value in fine_norm.items()}

    return deltas  # , fine, fine_norm, coarse, coarse_norm
