from pyciemss.workflow import checks
from pyciemss.workflow import vega
from pathlib import Path
import unittest
import xarray as xr

_data_file = Path(__file__).parent/"data"/"ciemss_datacube.nc"


class TestCheck(unittest.TestCase):
    def setUp(self):
        def read_cube(file):
            ds = xr.open_mfdataset([file])
            real_data  = ds.to_dataframe().reset_index()
            real_data.rename(columns={'timesteps': 'time', 
                                      'experimental conditions': 'conditions', 
                                      'attributes': 'state_names', 
                                      '__xarray_dataarray_variable__': "state_values"}, 
                             inplace=True)
            return real_data

        raw_data = read_cube(_data_file)
        self.s30 = raw_data.loc[(raw_data['time']== 30) & (raw_data['state_names'] == "S")]
        self.r30 = raw_data.loc[(raw_data['time']== 30) & (raw_data['state_names'] == "R")]        
        self.i30 = raw_data.loc[(raw_data['time']== 30) & (raw_data['state_names'] == "I")]    
    
    def test_KL(self):
        pass
    
    def test_contains(self):
        _, bins = vega.histogram_multi(range=[*range(25)], return_bins=True)

        checker = checks.contains(3, 10)
        self.assertTrue(checker(bins), "In range")

        checker = checks.contains(-1, 10)
        self.assertFalse(checker(bins), "Out lower")
                         
        checker = checks.contains(3, 100)
        self.assertFalse(checker(bins), "Out upper")
                         
                         
        checker = checks.contains(-10, 40)
        self.assertFalse(checker(bins), "Out both")
                                 
        
    def test_contains_pct(self):
        _, bins = vega.histogram_multi(range=[*range(20)], return_bins=True, bins=20)

        checker = checks.contains(0, 20, 1)
        self.assertTrue(checker(bins), "Full range")

        checker = checks.contains(5, 15, .5)
        self.assertTrue(checker(bins), "Half middle")

        checker = checks.contains(15, 20, .25)
        self.assertTrue(checker(bins), ".25 upper")

        checker = checks.contains(0, 15, .75)
        self.assertTrue(checker(bins), ".75 lower")
        
    
    def test_prior_predictive(self):
        lower, upper = 70_000, 94_000
        result, schema = checks.prior_predictive(
                    self.s30, lower, upper,
                    label="s30",
                    tests=[checks.contains(lower, upper), 
                           checks.contains(lower, upper, .99)])

    def test_posterior_predictive(self):
        result, schema = checks.posterior_predictive(self.r30, self.i30, tests=[checks.KL(1)])
    
