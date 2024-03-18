import random
from itertools import chain

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import pyciemss
from pyciemss.integration_utils.result_processing import convert_to_output_format
from pyciemss.visuals import plots, vega

import numpy as np
import pandas as pd
import pyro
import pytest
import torch

from pyciemss.compiled_dynamics import CompiledDynamics
from pyciemss.integration_utils.result_processing import convert_to_output_format
from pyciemss.interfaces import (
    sample,
)

import os
from collections.abc import Mapping
from typing import Any, Dict, Optional, TypeVar

import numpy as np
import pandas as pd
import torch

from pyciemss.integration_utils.intervention_builder import (
    param_value_objective,
    start_time_objective,
)
from pyciemss.ouu.qoi import obs_nday_average_qoi

T = TypeVar("T")

MODELS_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/models/"
PDE_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/pde-petri-amrs/petrinet/"
DATA_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/datasets/"


class ModelFixture:
    def __init__(
        self,
        url: str,
        important_parameter: Optional[str] = None,
        data_path: Optional[str] = None,
        data_mapping: Dict[str, str] = {},
        data_mapped_to_observable: bool = False,
        optimize_kwargs: Dict[str, Any] = None,
    ):
        self.url = url
        self.important_parameter = important_parameter
        self.data_path = data_path
        self.data_mapping = data_mapping
        self.data_mapped_to_observable = data_mapped_to_observable
        self.optimize_kwargs = optimize_kwargs


# See https://github.com/DARPA-ASKEM/Model-Representations/issues/62 for discussion of valid models.

PETRI_MODELS = [
    ModelFixture(
        os.path.join(MODELS_PATH, "SEIRHD_NPI_Type1_petrinet.json"),
        "gamma",
        os.path.join(DATA_PATH, "traditional.csv"),
        {"Infected": "I"},
        False,
    ),
    ModelFixture(
        os.path.join(MODELS_PATH, "SEIRHD_NPI_Type2_petrinet.json"),
        "kappa",
        os.path.join(DATA_PATH, "SIR_data_case_hosp.csv"),
        {"case": "infected", "hosp": "hospitalized"},
        True,
    ),
    ModelFixture(
        os.path.join(MODELS_PATH, "SEIRHD_with_reinfection01_petrinet.json"),
        "beta",
        os.path.join(DATA_PATH, "SIR_data_case_hosp.csv"),
        {"case": "infected", "hosp": "hospitalized"},
        True,
    ),
    ModelFixture(
        os.path.join(
            PDE_PATH, "examples/pde/advection/advection_backward_1_0.01_3.json"
        ),
        "u",
    ),
]

REGNET_MODELS = [
    ModelFixture(
        os.path.join(MODELS_PATH, "LV_rabbits_wolves_model02_regnet.json"), "beta"
    ),
    ModelFixture(
        os.path.join(MODELS_PATH, "LV_rabbits_wolves_model03_regnet.json"), "beta"
    ),
    # ModelFixture(os.path.join(MODELS_PATH, "LV_goat_chupacabra_regnet.json"), "beta"),
]

STOCKFLOW_MODELS = [
    ModelFixture(os.path.join(MODELS_PATH, "SEIRD_stockflow.json"), "p_cbeta"),
    ModelFixture(os.path.join(MODELS_PATH, "SEIRHDS_stockflow.json"), "p_cbeta"),
    ModelFixture(os.path.join(MODELS_PATH, "SEIRHD_stockflow.json"), "p_cbeta"),
]

optimize_kwargs_SIRstockflow_param = {
    "qoi": lambda x: obs_nday_average_qoi(x, ["I_state"], 1),
    "risk_bound": 300.0,
    "static_parameter_interventions": param_value_objective(
        param_name=["p_cbeta"],
        param_value=[lambda x: torch.tensor([x])],
        start_time=[torch.tensor(1.0)],
    ),
    "objfun": lambda x: np.abs(0.35 - x),
    "initial_guess_interventions": 0.15,
    "bounds_interventions": [[0.1], [0.5]],
}

optimize_kwargs_SIRstockflow_time = {
    "qoi": lambda x: obs_nday_average_qoi(x, ["I_state"], 1),
    "risk_bound": 300.0,
    "static_parameter_interventions": start_time_objective(
        param_name=["p_cbeta"],
        param_value=[torch.tensor([0.15])],
    ),
    "objfun": lambda x: -x,
    "initial_guess_interventions": 1.0,
    "bounds_interventions": [[0.0], [40.0]],
}

OPT_MODELS = [
    ModelFixture(
        os.path.join(MODELS_PATH, "SIR_stockflow.json"),
        important_parameter="p_cbeta",
        optimize_kwargs=optimize_kwargs_SIRstockflow_param,
    ),
    ModelFixture(
        os.path.join(MODELS_PATH, "SIR_stockflow.json"),
        important_parameter="p_cbeta",
        optimize_kwargs=optimize_kwargs_SIRstockflow_time,
    ),
]

MODELS = PETRI_MODELS + REGNET_MODELS + STOCKFLOW_MODELS

MODEL_URLS = [model.url for model in MODELS]

START_TIMES = [0.0]
END_TIMES = [40.0]

LOGGING_STEP_SIZES = [5.0]

NUM_SAMPLES = [2]
NON_POS_INTS = [
    3.5,
    -3,
    0,
    torch.tensor(3),
]  # bad candidates for num_samples/num_iterations

bad_data1 = {
    "Timestamp": {0: 1.1, 1: 2.2, 2: 3.3},
    "case": {0: 15.0, 1: "", 2: 20.0},
    "hosp": {0: 0.1, 1: 1.0, 2: 2.2},
}
bad_data2 = {
    "Timestamp": {0: 1.1, 1: 2.2, 2: 3.3},
    "case": {0: 15.0, 1: "apple", 2: 20.0},
    "hosp": {0: 0.1, 1: 1.0, 2: 2.2},
}
bad_data3 = {
    "Timestamp": {0: 1.1, 1: 2.2, 2: 3.3},
    "case": {0: 15.0, 1: " ", 2: 20.0},
    "hosp": {0: 0.1, 1: 1.0, 2: 2.2},
}
bad_data4 = {
    "Timestamp": {0: 1.1, 1: 2.2, 2: 3.3},
    "case": {0: 15.0, 1: None, 2: 20.0},
    "hosp": {0: 0.1, 1: 1.0, 2: 2.2},
}
bad_data5 = {
    "Timepoints": {0: 1.1, 1: 2.2, 2: 3.3},
    "case": {0: 15.0, 1: 18.0, 2: 20.0},
    "hosp": {0: 0.1, 1: 1.0, 2: 2.2},
}
BADLY_FORMATTED_DATAFRAMES = [
    pd.DataFrame(bad_data1),
    pd.DataFrame(bad_data2),
    pd.DataFrame(bad_data3),
    pd.DataFrame(bad_data4),
    pd.DataFrame(bad_data5),
]  # improperly formatted datasets
MAPPING_FOR_DATA_TESTS = {"case": "I", "hosp": "H"}


def check_keys_match(obj1: Dict[str, T], obj2: Dict[str, T]):
    assert set(obj1.keys()) == set(obj2.keys()), "Objects have different variables."
    return True


def check_states_match(traj1: Dict[str, torch.Tensor], traj2: Dict[str, torch.Tensor]):
    assert check_keys_match(traj1, traj2)

    for k, val in traj1.items():
        if val.ndim == 2:
            assert torch.allclose(
                traj2[k], traj1[k]
            ), f"Trajectories differ in state trajectory of variable {k}."

    return True


def check_states_match_in_all_but_values(
    traj1: Dict[str, torch.Tensor], traj2: Dict[str, torch.Tensor], state_ndim: int = 2
):
    assert check_keys_match(traj1, traj2)

    for k, val in traj1.items():
        if val.ndim == state_ndim:
            if not torch.allclose(traj2[k], traj1[k]):
                # early return, as we've already confirmed they're not identical
                return True

    assert False, "Trajectories have identical values."


def check_result_sizes(
    traj: Dict[str, torch.Tensor],
    start_time: float,
    end_time: float,
    logging_step_size: float,
    num_samples: int,
):
    for k, v in traj.items():
        assert isinstance(k, str)
        assert isinstance(v, torch.Tensor)

        num_timesteps = len(
            torch.arange(start_time + logging_step_size, end_time, logging_step_size)
        )
        if v.ndim == 2 and k == "model_weights":
            assert v.shape[0] == num_samples
        elif v.ndim == 2:
            assert v.shape == (num_samples, num_timesteps)
        else:
            assert v.shape == (num_samples,)

    return True


def check_is_state(state: torch.Tensor, value_type):
    assert isinstance(state, Mapping)

    assert all(isinstance(key, str) for key in state.keys())

    assert all(isinstance(value, value_type) for value in state.values())

    return True




def by_key_value(targets, key, value):
    for entry in targets:
        if entry[key] == value:
            return entry


def make_nice_labels(labels):
    """Utility for generally-nice-labels for testing purposes."""
    return {
        k: "_".join(k.split("_")[:-1])
        for k in labels
        if "_" in k and k not in ["sample_id", "timepoint_id", "timepoint_notional"]
    }


def create_distributions(logging_step_size=20, time_unit="twenty"):
    model_1_path = (
        "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration"
        "/main/data/models/SEIRHD_NPI_Type1_petrinet.json"
    )
    start_time = 0.0
    end_time = 100.0
    num_samples = 30
    sample = pyciemss.sample(
        model_1_path,
        end_time,
        logging_step_size,
        num_samples,
        time_unit=time_unit,
        start_time=start_time,
        solver_method="euler",
        solver_options={"step_size": 1e-2},
    )["unprocessed_result"]

    return convert_to_output_format(
        sample,
        # using same time point formula as in 'logging_times' from  pyciemms interfaces formula 'sample'
        timepoints=np.arange(
            start_time + logging_step_size, end_time, logging_step_size
        ),
        time_unit=time_unit,
    )



@pytest.mark.parametrize("model_fixture", MODELS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("num_samples", NUM_SAMPLES)
def test_sample_with_multiple_parameter_interventions(
    model_fixture,
    start_time,
    end_time,
    logging_step_size,
    num_samples,
    time_unit="twenty"
):

    model_url = model_fixture.url
    model = CompiledDynamics.load(model_url)

    important_parameter_name = model_fixture.important_parameter
    important_parameter = getattr(model, important_parameter_name)

    intervention_effect = 1.1
    intervention_time_0 = (end_time + start_time) / 4  # Quarter of the way through
    intervention_time_1 = (end_time + start_time) / 2  # Half way through

    intervention_0 = {
        important_parameter_name: important_parameter.detach() * intervention_effect
    }

    intervention_1 = {
        important_parameter_name: important_parameter.detach() / intervention_effect
    }

    model_args = [model_url, end_time, logging_step_size, num_samples]
    model_kwargs = {"start_time": start_time}

    with pyro.poutine.seed(rng_seed=0):
        intervened_result = sample(
            *model_args,
            **model_kwargs,
            static_parameter_interventions={
                intervention_time_0: intervention_0,
                intervention_time_1: intervention_1,
            },
        )["unprocessed_result"]
    distributions = convert_to_output_format(
        intervened_result,
        # using same time point formula as in 'logging_times' from  pyciemms interfaces formula 'sample'
        timepoints=np.arange(
            start_time + logging_step_size, end_time, logging_step_size
        ),
        time_unit=time_unit,
    )
    #{'parameter_intervention_value_gamma_0': tensor([0.3997, 0.3997]), 'parameter_intervention_value_gamma_1': tensor([0.3304, 0.3304])}
    intervention_data = {
        k: v for k, v in intervened_result.items() if "parameter_intervention_value" in k
    }
    #{'parameter_intervention_time_0': tensor([10., 10.]), 'parameter_intervention_time_1': tensor([20., 20.])}
    intervention_times = {
        k: v for k, v in intervened_result.items() if "parameter_intervention_time" in k
    }
    intervention_list = [(a[0], a[1].numpy()[0], b[0], b[1].numpy()[0]) for (a,b) in zip(intervention_data.items(), intervention_times.items())]
    base_markers_v = {}
    for intervention in  intervention_list:
        base_markers_v['_'.join(intervention[0].split('_')[3:])] = int(intervention[3])
    schema = plots.trajectories(
        distributions,
        base_markers_v=base_markers_v,
        base_markers_h={}
    )
    plots.save_schema(schema, "test_intervention.vg.json")
    return




class TestTrajectory:
    @staticmethod
    @pytest.fixture
    def distributions():
        return create_distributions(logging_step_size=1, time_unit="notional")

    @staticmethod
    @pytest.fixture
    def traces(distributions):
        return (
            distributions[distributions["sample_id"] == 0]
            .set_index("timepoint_notional")[["dead_observable_state", "I_state"]]
            .rename(
                columns={
                    "dead_observable_state": "dead_exemplar",
                    "I_state": "I_exemplar",
                }
            )
        )

    @staticmethod
    @pytest.fixture
    def observed_points(traces):
        return traces.iloc[::10]

    def test_base(self, distributions):
        #{'S': tensor(19340000.), 'I': tensor(3.9251), 'E': tensor(41.), 'R': tensor(1.), 'H': tensor(1.), 'D': tensor(1.)}
        schema = plots.trajectories(distributions)

        df = pd.DataFrame(vega.find_named(schema["data"], "distributions")["values"])
        assert {"trajectory", "timepoint", "lower", "upper"} == set(df.columns)

    @pytest.mark.parametrize("logging_step_size", [5, 1])
    @pytest.mark.parametrize("time_unit", ["five", None])
    @pytest.mark.parametrize("end_time", [80, 100])
    def test_timepoints(self, logging_step_size, time_unit, end_time):
        # distribution will create timepoint from logging_step_size, and start and end time
        new_distribution = create_distributions(
            logging_step_size=logging_step_size, time_unit=time_unit
        )
        label = "timepoint_unknown" if time_unit is None else f"timepoint_{time_unit}"
        # check if new time label is correct
        assert label in new_distribution.columns

        schema = plots.trajectories(new_distribution)
        df = pd.DataFrame(vega.find_named(schema["data"], "distributions")["values"])
        new_timepoints = [
            float(x) for x in np.arange(logging_step_size, end_time, logging_step_size)
        ]
        # check timepoint created match the input logging_step_size and start and end time
        assert df.timepoint[: len(new_timepoints)].tolist() == new_timepoints

    def test_markers(self, distributions):
        schema = plots.trajectories(
            distributions,
            base_markers_h={"Low marker": 10000000, "High marker": 20000000},
        )

        plots.save_schema(schema, "test_markers_schema.vg.json")
        df = pd.DataFrame(vega.find_named(schema["data"], "markers_h")["values"])
        assert {"axis_value", "label"} == set(df.columns)

    def test_rename(self, distributions):
        nice_labels = make_nice_labels(distributions.columns)

        schema = plots.trajectories(distributions, relabel=nice_labels)

        df = pd.DataFrame(vega.find_named(schema["data"], "distributions")["values"])

        kept_names = df["trajectory"].unique()
        for name in nice_labels.values():
            assert name in kept_names, f"Nice name '{name}' not found"

        for name in nice_labels.keys():
            assert name not in kept_names, "Bad name unexpectedly found"

    def test_keep(self, distributions):
        schema = plots.trajectories(distributions, keep=".*_observable.*")
        df = pd.DataFrame(vega.find_named(schema["data"], "distributions")["values"])

        kept = sorted(df["trajectory"].unique())
        assert [
            "dead_observable_state",
            "exposed_observable_state",
            "hospitalized_observable_state",
            "infected_observable_state",
        ] == kept, f"Keeping by regex failed.  Kept {kept}"

        keep_list = ["dead_observable_state", "exposed_observable_state"]
        schema = plots.trajectories(distributions, keep=keep_list)
        df = pd.DataFrame(vega.find_named(schema["data"], "distributions")["values"])
        assert keep_list == sorted(df["trajectory"].unique()), "Keeping by list"

        nice_labels = make_nice_labels(distributions.columns)
        schema = plots.trajectories(
            distributions,
            relabel=nice_labels,
            keep=keep_list,
        )
        df = pd.DataFrame(vega.find_named(schema["data"], "distributions")["values"])
        assert ["dead_observable", "exposed_observable"] == sorted(
            df["trajectory"].unique()
        ), "Rename after keeping"

    def test_keep_drop(self, distributions):
        assert (
            "H_state" in distributions.columns
        ), "Expected trajectory not found in pre-test"

        should_keep = [
            p
            for p in distributions.columns
            if "_state" not in p
            and p not in ["sample_id", "timepoint_id", "timepoint_notional"]
        ]
        should_drop = [p for p in distributions.columns if "_state" in p]

        assert len(should_keep) > 0, "Expected keep trajectories not found in pre-test"
        assert len(should_drop) > 0, "Expected drop trajectories not found in pre-test"

        schema = plots.trajectories(distributions, keep=".*_.*", drop=".*_state")
        df = pd.DataFrame(vega.find_named(schema["data"], "distributions")["values"])

        kept = df["trajectory"].unique()
        for name in should_keep:
            assert name in kept, f"Unexpectedly lost '{name}'"

        for name in should_drop:
            assert name not in kept, f"Unexpectedly kept '{name}'"

    def test_drop(self, distributions):
        assert (
            "H_state" in distributions.columns
        ), "Exepected trajectory not found in pre-test"

        should_drop = [p for p in distributions.columns if "_observable_state" in p]
        assert len(should_drop) > 0, "Exepected trajectory not found in pre-test"

        schema = plots.trajectories(distributions, drop=should_drop)
        df = pd.DataFrame(vega.find_named(schema["data"], "distributions")["values"])

        assert (
            "H_state" in df["trajectory"].unique()
        ), "Exepected trajectory not retained from list"
        assert (
            "D_state" in df["trajectory"].unique()
        ), "Exepected trajectory not retained from list"

        for t in should_drop:
            assert (
                t not in df["trajectory"]
            ), "Trajectory still present after drop from list"

        try:
            schema = plots.trajectories(distributions, drop="THIS IS NOT HERE")
        except Exception:
            assert False, "Error dropping non-existent trajectory"

        schema = plots.trajectories(distributions, drop=".*_observable")
        df = pd.DataFrame(vega.find_named(schema["data"], "distributions")["values"])
        assert (
            "E_state" in df["trajectory"].unique()
        ), "Exepected trajectory not retained from pattern"
        assert (
            "R_state" in df["trajectory"].unique()
        ), "Exepected trajectory not retained from pattern"
        assert (
            "hospitalized_observable_state" not in df["trajectory"].unique()
        ), "Trajectory still present after drop from pattern"

    def test_points(self, distributions, observed_points):
        schema = plots.trajectories(
            distributions,
            keep=".*_state",
            points=observed_points,
        )

        points = pd.DataFrame(vega.find_named(schema["data"], "points")["values"])

        assert len(points["trajectory"].unique()) == 2, "Unexpected number of exemplars"

        assert (
            len(points) == observed_points.count().sum()
        ), "Unexpected number of exemplar points"

    def test_traces(self, distributions, traces):
        schema = plots.trajectories(
            distributions,
            keep=".*_state",
            traces=traces,
        )

        shown_traces = pd.DataFrame(vega.find_named(schema["data"], "traces")["values"])
        plots.save_schema(schema, "_schema.vg.json")

        assert sorted(traces.columns.unique()) == sorted(
            shown_traces["trajectory"].unique()
        ), "Unexpected traces"

        for exemplar in shown_traces["trajectory"].unique():
            assert len(traces) == len(
                shown_traces[shown_traces["trajectory"] == exemplar]
            ), "Unexpected number of trace data points"


class TestHistograms:
    @staticmethod
    @pytest.fixture
    def simulation_result():
        model_1_path = (
            "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/"
            "main/data/models/SEIRHD_NPI_Type1_petrinet.json"
        )
        start_time = 0.0
        end_time = 100.0
        logging_step_size = 10.0
        num_samples = 3

        return pyciemss.sample(
            model_1_path,
            end_time,
            logging_step_size,
            num_samples,
            start_time=start_time,
            solver_method="euler",
            solver_options={"step_size": 0.1},
        )["unprocessed_result"]

    def test_histogram(self, simulation_result):
        hist, bins = plots.histogram_multi(
            D=simulation_result["D_state"], return_bins=True
        )

        bins = bins.reset_index()

        assert all(bins["bin0"].value_counts() == 1), "Duplicated bins found"
        assert all(bins["bin1"].value_counts() == 1), "Duplicated bins found"

        hist_data = pd.DataFrame(by_key_value(hist["data"], "name", "binned")["values"])
        assert all(bins == hist_data), "Bin count not as expected"

        assert (
            len(by_key_value(hist["data"], "name", "xref")["values"]) == 0
        ), "Missing xrefs"
        assert (
            len(by_key_value(hist["data"], "name", "yref")["values"]) == 0
        ), "Missing yrefs"

    def test_histogram_empty_refs(self, simulation_result):
        xrefs = []
        yrefs = []
        hist = plots.histogram_multi(
            D=simulation_result["D_state"],
            xrefs=xrefs,
            yrefs=yrefs,
            return_bins=False,
        )

        assert (
            len(by_key_value(hist["data"], "name", "xref")["values"]) == 0
        ), "xrefs found when not expected"
        assert (
            len(by_key_value(hist["data"], "name", "yref")["values"]) == 0
        ), "yrefs found when not expected"

    @pytest.mark.parametrize("num_refs", range(1, 20))
    def test_histogram_refs(self, num_refs, simulation_result):
        xrefs = [*range(num_refs)]
        yrefs = [*range(num_refs)]
        hist = plots.histogram_multi(
            D=simulation_result["D_state"],
            xrefs=xrefs,
            yrefs=yrefs,
            return_bins=False,
        )

        assert num_refs == len(
            by_key_value(hist["data"], "name", "xref")["values"]
        ), "Nonzero xrefs not as expected"
        assert num_refs == len(
            by_key_value(hist["data"], "name", "yref")["values"]
        ), "Nonzero yrefs not as expected"

        hist = plots.histogram_multi(
            D=simulation_result["D_state"],
            xrefs=xrefs,
            yrefs=[],
            return_bins=False,
        )

        assert num_refs == len(
            by_key_value(hist["data"], "name", "xref")["values"]
        ), "Nonzero xrefs not as expected when there are zero yrefs"
        assert (
            len(by_key_value(hist["data"], "name", "yref")["values"]) == 0
        ), "Zero yrefs not as expected when there are nonzero xrefs"

        hist = plots.histogram_multi(
            D=simulation_result["D_state"],
            xrefs=[],
            yrefs=yrefs,
            return_bins=False,
        )

        assert (
            len(by_key_value(hist["data"], "name", "xref")["values"]) == 0
        ), "Zero xrefs not as expected when there are nonzero yrefs"
        assert num_refs == len(
            by_key_value(hist["data"], "name", "yref")["values"]
        ), "Nonzero yrefs not as expected when there are zero xrefs"

    def test_histogram_multi(self, simulation_result):
        hist = plots.histogram_multi(
            D=simulation_result["D_state"],
            E=simulation_result["E_state"],
            H=simulation_result["R_state"],
        )
        data = pd.DataFrame(by_key_value(hist["data"], "name", "binned")["values"])
        assert set(data["label"].values) == {"D", "E", "H"}

        hist = plots.histogram_multi(
            D_state=simulation_result["D_state"],
            E_state=simulation_result["E_state"],
        )
        data = pd.DataFrame(by_key_value(hist["data"], "name", "binned")["values"])
        assert set(data["label"].values) == {"D_state", "E_state"}


class TestHeatmapScatter:
    def test_implicit_heatmap(self):
        df = pd.DataFrame(3 * np.random.random((100, 2)), columns=["test4", "test5"])
        schema = plots.heatmap_scatter(df, max_x_bins=4, max_y_bins=4)

        points = vega.find_named(schema["data"], "points")["values"]
        assert all(pd.DataFrame(points) == df), "Unexpected points values found"

    def test_explicit_heatmap(self):
        def create_fake_data():
            nx, ny = (10, 10)
            x = np.linspace(0, 10, nx)
            y, a = np.linspace(0, 10, ny, retstep=True)

            # create mesh data
            xv, yv = np.meshgrid(x, y)
            zz = xv**2 + yv**2

            # create scatter plot
            df = pd.DataFrame(
                10 * np.random.random((100, 2)), columns=["alpha", "gamma"]
            )
            return (xv, yv, zz), df

        mesh_data, scatter_data = create_fake_data()
        schema = plots.heatmap_scatter(scatter_data, mesh_data)

        points = vega.find_named(schema["data"], "points")["values"]
        assert all(
            pd.DataFrame(points) == scatter_data
        ), "Unexpected points values found"

        mesh = pd.DataFrame(vega.find_named(schema["data"], "mesh")["values"])
        assert mesh.size == 500, "Unexpected mesh representation size."
        assert all(mesh["__count"].isin(mesh_data[2].ravel())), "Unexpected count found"


class TestGraph:
    @staticmethod
    @pytest.fixture
    def test_graph():
        def rand_attributions():
            possible = "ABCD"
            return random.sample(possible, random.randint(1, len(possible)))

        def rand_label():
            possible = "TUVWXYZ"
            return random.randint(1, 10)
            return random.sample(possible, 1)[0]

        g = nx.generators.barabasi_albert_graph(5, 3)
        node_properties = {
            n: {"attribution": rand_attributions(), "label": rand_label()}
            for n in g.nodes()
        }

        edge_attributions = {e: {"attribution": rand_attributions()} for e in g.edges()}

        nx.set_node_attributes(g, node_properties)
        nx.set_edge_attributes(g, edge_attributions)
        return g

    def test_multigraph(self, test_graph):
        uncollapsed = plots.attributed_graph(test_graph)
        nodes = vega.find_named(uncollapsed["data"], "node-data")["values"]
        edges = vega.find_named(uncollapsed["data"], "link-data")["values"]
        assert len(test_graph.nodes) == len(nodes), "Nodes issue in conversion"
        assert len(test_graph.edges) == len(edges), "Edges issue in conversion"

        all_attributions = set(
            chain(*nx.get_node_attributes(test_graph, "attribution").values())
        )
        nx.set_node_attributes(test_graph, {0: {"attribution": all_attributions}})
        collapsed = plots.attributed_graph(test_graph, collapse_all=True)
        nodes = vega.find_named(collapsed["data"], "node-data")["values"]
        edges = vega.find_named(collapsed["data"], "link-data")["values"]
        assert len(test_graph.nodes) == len(
            nodes
        ), "Nodes issue in conversion (collapse-case)"
        assert len(test_graph.edges) == len(
            edges
        ), "Edges issue in conversion (collapse-case)"
        assert [["*all*"]] == [
            n["attribution"] for n in nodes if n["label"] == 0
        ], "All tag not found as expected"

    def test_springgraph(self, test_graph):
        schema = plots.spring_force_graph(test_graph, node_labels="label")
        nodes = vega.find_named(schema["data"], "node-data")["values"]
        edges = vega.find_named(schema["data"], "link-data")["values"]
        assert len(test_graph.nodes) == len(nodes), "Nodes issue in conversion"
        assert len(test_graph.edges) == len(edges), "Edges issue in conversion"

    def test_provided_layout(self, test_graph):
        pos = nx.fruchterman_reingold_layout(test_graph)
        schema = plots.spring_force_graph(test_graph, node_labels="label", layout=pos)

        nodes = vega.find_named(schema["data"], "node-data")["values"]
        edges = vega.find_named(schema["data"], "link-data")["values"]
        assert len(test_graph.nodes) == len(nodes), "Nodes issue in conversion"
        assert len(test_graph.edges) == len(edges), "Edges issue in conversion"

        for id, (x, y) in pos.items():
            n = [n for n in nodes if n["label"] == id][0]
            assert n["inputX"] == x, f"Layout lost for {id}"
            assert n["inputY"] == y, f"Layout lost for {id}"
