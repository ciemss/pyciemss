import pyciemss.workflow.vega as vega
import numpy as np

def contains(ref_lower, ref_upper, pct=None):
    """Check-generator function. Returns a function that performs a test.
    
    returns -- A functiont that takes a list of bins and tests the lower/upper bound against those bins.
               The signature of the returned function is (List[Number]) -> bool.
               
               If pct IS NOT SUPPLIED to the generator function, the returned function checks 
               if ref_lower and ref_upper are within the distribution range.
               
               If pct IS SUPPLIED, teh returned function checks that the precent of 
               distribution between ref_lower and ref_upper is AT LEAST pct% of the total data.
    """
    def pct_test(bins):
        total = bins["count"].sum()
        covered = bins[(bins["bin0"]>=ref_lower) & (bins["bin1"] <= ref_upper)]["count"].sum()

        return  covered/total >= pct
    
    def simple_test(bins):
        data_lower = min(bins["bin0"].min(), bins["bin1"].min())
        data_upper = max(bins["bin0"].max(), bins["bin1"].max())
        return ((data_lower <= ref_lower <= data_upper)
                and (data_lower <= ref_upper <= data_upper))

    if pct is not None:
        return pct_test
    else:
        return simple_test
    

def KL(max_acceptable):
    """Check-generator function. Returns a function that performs a test.
    
    max_acceptable -- Threshold for the returned check
    returns -- Returns a function that checks if KL/divergence of two lists of bin-counts is less than max_acceptable.
               Returned function takes two lists of bins-counts and returns a boolean result.  The signature
               is roughly (list[Number], list[Number]) -> bool.    
    """
    def KL(a, b):
        a = np.asarray(a, dtype=np.float32)+1
        b = np.asarray(b, dtype=np.float32)+1

        return np.sum(np.where(a != 0, a * np.log(a / b), 0)) <= max_acceptable
    return KL

        
def prior_predictive(posterior, lower, upper, *, label="posterior", tests=[], combiner=all, **kwargs):
    combined_args = {**{label: posterior}, **kwargs}
    schema, bins = vega.histogram_multi(xrefs=[lower, upper], 
                                        return_bins=True, 
                                        **combined_args)

    checks = [test(bins) for test in tests]
    
    status = combiner(checks)
    if not status:
        status = f"Failed ({sum(checks)/len(checks):.2%} passing)"
    else:
        status = "Passed"
    
    schema["title"]["text"] = ["Prior Predictive Test (Histogram)", status]
    
    return checks, schema


def posterior_predictive(posterior, data,  *, tests=[], combiner=all, **kwargs):
    schema, bins = vega.histogram_multi(Posterior=posterior, Reference=data,
                                  return_bins=True, **kwargs)
    
    groups = dict([*bins.groupby("label")])
    posterior_dist = groups["Posterior"]["count"].values
    reference_dist = groups["Reference"]["count"].values
    
    checks = [test(reference_dist, posterior_dist) for test in tests]
    status = combiner(checks)
    
    if not status:
        status = f"Failed ({sum(checks)/len(checks):.2%} passing)"
    else:
        status = "Passed"
    
    schema["title"]["text"] = ["Posterior Predictive Check (Histogram)", status]
    
    return checks, schema
    
    