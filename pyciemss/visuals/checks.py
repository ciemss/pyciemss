from dataclasses import dataclass
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from . import plots
from .plots import VegaSchema


@dataclass(repr=False)
class Result:
    """Results of a check.
    At a minimum, the combined check, the individual checks and
    a visual representation of the evidence. May also include additional
    information computed for that individual check.
    """

    status: bool
    checks: Dict[str, Any]
    schema: VegaSchema
    aligned: Optional[Any] = None
    bins: Optional[Any] = None

    def __repr__(self):
        always_display = ["status", "checks", "schema"]
        schema = "<missing>" if self.schema is None else "<present>"

        additional = [
            f for f in dir(self) if not f.startswith("_") and f not in always_display
        ]
        extras = f"; Additional Fields: {additional}" if len(additional) > 0 else ""
        return f"Result(status:{self.status}, checks:{self.checks}, schema:{schema}{extras})"


def contains(
    ref_lower: Number, ref_upper: Number, pct: Optional[float] = None
) -> Callable[[pd.DataFrame], bool]:
    """Check-generator function. Returns a function that performs a test.

    returns -- A function that takes a dataframe tests it against the bounds.

               If pct IS NOT SUPPLIED, the returned function needs
               a dataframe with bin-boundaries on the index.  It
               checks if ref_lower and ref_upper are within the
               distribution range.

               If pct IS SUPPLIED, the returned function checks takes
               a dataframe with  bin-boundaries on the index and a 'count'
               column.  It checks if the distribution between ref_lower
               and ref_upper is AT LEAST pct% of the total data.
    """

    def pct_test(bins: pd.DataFrame) -> bool:
        total = bins["count"].sum()
        level0 = bins.index.get_level_values(0)
        level1 = bins.index.get_level_values(1)
        subset = bins[(level0 >= ref_lower) & (level1 <= ref_upper)]
        covered = subset["count"].sum()

        return covered / total >= pct

    def simple_test(bins: pd.DataFrame) -> bool:
        level0 = bins.index.get_level_values(0)
        level1 = bins.index.get_level_values(1)

        data_lower = min(level0.min(), level1.min())
        data_upper = max(level0.max(), level1.max())
        return (data_lower <= ref_lower <= data_upper) and (
            data_lower <= ref_upper <= data_upper
        )

    if pct is not None:
        return pct_test
    else:
        return simple_test


def JS(max_acceptable: float, *, verbose: bool = False) -> Callable[[Any, Any], bool]:
    """Check-generator function. Returns a function that performs a test against jensen-shannon distance.

    max_acceptable -- Threshold for the returned check
    returns -- Returns a function that checks if JS distance of two lists of bin-counts is less than max_acceptable.
               Returned function takes two lists of bins-counts and returns a boolean result.  The signature
               is roughly (list[Number], list[Number]) -> bool.
    """

    def _inner(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        js = jensenshannon(a, b)
        if verbose:
            print(f"JS distance is {js}")
        return js <= max_acceptable

    return _inner


def check_distribution_range(
    distribution: pd.Series,
    lower: Number,
    upper: Number,
    *,
    label: Optional[str] = None,
    tests: Dict[str, Callable[[Union[pd.Series, Number, Number]], bool]] = {},
    combiner: Callable[[List[bool]], bool] = all,
    **kwargs: Dict[str, Any],
) -> Result:
    """
    Checks a single distribution against a lower- and upper-bound.

    distribution -- Distribution to check
    lower -- Lower bound to compare to the distribution
    upper -- Upper bound to compare to the distribution

    label -- Label to put on resulting plot
    tests -- Tests to make against the distribution
             (Typed as dict of label/value, but can be list of callables instead)
    combiner -- Combines the results of the test
    """
    if isinstance(tests, list):
        tests = dict(enumerate(tests))

    if label is None and distribution.name is not None:
        label = distribution.name
    else:
        label = ""

    combined_args = {**{label: distribution}, **kwargs}
    schema, bins = plots.histogram_multi(
        xrefs=[lower, upper], return_bins=True, **combined_args
    )

    checks = {label: test(bins) for label, test in tests.items()}
    status = combiner([*checks.values()])
    if not status:
        status_msg = f"Failed ({sum(checks.values())/len(checks.values()):.0%} passing)"
    else:
        status_msg = "Passed"

    schema = plots.set_title(schema, ["Distribution Check (Histogram)", status_msg])

    return Result(status, checks, schema, bins=bins)


def compare_distributions(
    subject: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    tests: Dict[str, Callable[[pd.DataFrame, pd.DataFrame], bool]] = {},
    combiner: Callable[[List[bool]], bool] = all,
    **kwargs,
) -> Result:
    """
    Compares two distributions.

    This function returns a histogram visualization of the two distributions
    and the result running passed checks against those two distributions.

    NOTE: As tests may be non-symetric, tests will be called with an aligned
          distribution built from the subject and the reference.  The subject
          distribution will be passed first as the first argument.
    """
    if isinstance(tests, list):
        tests = dict(enumerate(tests))

    schema, bins = plots.histogram_multi(
        Subject=subject, Reference=reference, return_bins=True, **kwargs
    )

    groups = dict([*bins.groupby("label")])
    subject_dist = (
        groups["Subject"].rename(columns={"count": "subject"}).drop(columns=["label"])
    )
    reference_dist = (
        groups["Reference"].rename(columns={"count": "ref"}).drop(columns=["label"])
    )

    aligned = subject_dist.join(reference_dist, how="outer").fillna(0)

    checks = {
        label: test(aligned["subject"].values, aligned["ref"].values)
        for label, test in tests.items()
    }
    status = combiner([*checks.values()])

    if not status:
        status_msg = f"Failed ({sum(checks.values())/len(checks.values()):.0%} passing)"
    else:
        status_msg = "Passed"

    schema = plots.set_title(schema, ["Distribution Comparison", status_msg])

    return Result(status, checks, schema, aligned=aligned)
