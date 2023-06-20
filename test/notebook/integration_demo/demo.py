# %%
import os
from pyciemss.PetriNetODE.interfaces import (
    load_and_sample_petri_model,
    load_and_calibrate_and_sample_petri_model,
)

# %%
MIRA_PATH = "test/models/april_ensemble_demo/"
NOTEBOOK_PATH = "notebook/integration_demo/"

filename = "BIOMD0000000955_template_model.json"
filename = os.path.join(MIRA_PATH, filename)

# %% [markdown]
# ## load_and_sample_petri_model

# %%
num_samples = 100
timepoints = [0.0, 1.0, 2.0, 3.0, 4.0]

# Run sampling
samples = load_and_sample_petri_model(
    filename, num_samples, timepoints=timepoints, add_uncertainty=True
)

# Save results
samples.to_csv(
    os.path.join(NOTEBOOK_PATH, "results_petri/sample_results.csv"), index=False
)

# %% [markdown]
# ## load_and_calibrate_and_sample_petri_model

# %%
# List of tuples of the form `(time, observation_dict)` where each `observation_dict` is of the form `{observable_name: observable_value}`.
# Once we get a data file format we're very happy to handle the csv -> this format processing.
# Note: As you can see here, not every variable must be observed at every timepoint.
data_path = os.path.join(NOTEBOOK_PATH, "data.csv")
num_samples = 100
timepoints = [0.0, 1.0, 2.0, 3.0, 4.0]

# Run the calibration and sampling
calibrated_samples = load_and_calibrate_and_sample_petri_model(
    filename,
    data_path,
    num_samples,
    timepoints=timepoints,
    add_uncertainty=True,
    verbose=True,
)

# Save results
calibrated_samples.to_csv(
    os.path.join(NOTEBOOK_PATH, "results_petri/calibrated_sample_results.csv"),
    index=False,
)

# %%
