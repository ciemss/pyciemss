import unittest
from mira.sources.askenet import model_from_url, model_from_json_file

class TestAMR(unittest.TestCase):
    """Test the AMR model"""
    def setUp(self):
        """Load the model"""
        self.SEIV_model_url = "https://raw.githubusercontent.com/indralab/mira/hackathon2/notebooks/evaluation_2023.07/eval_scenario3_base.json"

        self.model_from_url = model_from_url(self.SEIV_model_url)
        self.model_from_json_file = model_from_json_file("test/models/AMR_examples/eval_scenario3_base.json")

    def test_controlled_production(self):
        """Test controlled production"""
        expected_transitions = ['t1','t2','t3','t4']
        self.assertEqual(expected_transitions, [t.name for t in self.model_from_url.templates])
        self.assertEqual(expected_transitions, [t.name for t in self.model_from_json_file.templates])
