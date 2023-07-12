import pandas as pd
import unittest
import sympy
from sympytorch import SymPyModule
import mira
from pyciemss.PetriNetODE.interfaces import (
    setup_petri_model,
    sample,
    load_petri_model,
    load_and_sample_petri_model,
)
import torch
from mira.metamodel import (
    Concept,
    ControlledConversion,
    Initial,
    NaturalConversion,
    Parameter,
    TemplateModel,
)
from pyciemss.PetriNetODE.base import ScaledBetaNoisePetriNetODESystem
from pyciemss.utils import reparameterize


class TestRateLaw(unittest.TestCase):
    """Test the symbolic rate law."""

    def setUp(self):
        """Set up the test fixtures."""
        beta, gamma, S, I, R, total_population = sympy.symbols(
            "beta, gamma, susceptible_population, infected_population, recovered_population, total_population"
        )

        susceptible = Concept(
            name="susceptible_population", identifiers={"ido": "0000514"}
        )
        infected = Concept(
            name="infected_population", identifiers={"ido": "0000573"}
        )  # http://purl.obolibrary.org/obo/IDO_0000573
        recovered = Concept(name="recovered_population", identifiers={"ido": "0000592"})
        total_pop = 100000

        S_to_I = ControlledConversion(
            controller=infected,
            subject=susceptible,
            outcome=infected,
            rate_law=beta * S * I / (S + I + R),
        )
        I_to_R = NaturalConversion(
            subject=infected, outcome=recovered, rate_law=gamma * I
        )
        self.tm = TemplateModel(
            templates=[S_to_I, I_to_R],
            parameters={
                "beta": Parameter(name="beta", value=0.55),  # transmission rate
                "gamma": Parameter(name="gamma", value=0.2),  # recovery rate
            },
            initials={
                "susceptible_population": (
                    Initial(concept=susceptible, value=(total_pop - 1))
                ),
                "infected_population": (Initial(concept=infected, value=1)),
                "recovered_population": (Initial(concept=recovered, value=0)),
            },
        )

        compiled_sir = ScaledBetaNoisePetriNetODESystem(
            mira.modeling.Model(self.tm), compile_rate_law_p=True
        )
        uncompiled_sir = ScaledBetaNoisePetriNetODESystem(
            mira.modeling.Model(self.tm), compile_rate_law_p=False
        )

        self.start_state = {k: v.value for k, v in self.tm.initials.items()}
        symbolic_derivs = {}

        symbolic_derivs["infected_population"] = beta * S * I / (S + I + R) - gamma * I
        symbolic_derivs["recovered_population"] = gamma * I
        symbolic_derivs["susceptible_population"] = -beta * S * I / (S + I + R)

        self.numeric_deriv = SymPyModule(expressions=list(symbolic_derivs.values()))
        self.nsamples = 5
        self.timepoints = [1.0, 2.0, 3.0]

        self.compiled_sir = setup_petri_model(
            compiled_sir, 0.0, start_state=self.start_state
        )
        self.uncompiled_sir = setup_petri_model(
            uncompiled_sir, 0.0, start_state=self.start_state
        )

    def test_rate_law_compilation(self):
        """Test that the rate law can be compiled correctly."""
        self.uncompiled_sir.param_prior()
        self.compiled_sir.param_prior()
        compiled_trajectories = sample(
            self.compiled_sir, self.timepoints, self.nsamples
        )
        for i in range(self.nsamples):
            uncompiled_sir = reparameterize(
                self.uncompiled_sir,
                {
                    "beta": compiled_trajectories["beta"][i],
                    "gamma": compiled_trajectories["gamma"][i],
                },
            )
            uncompiled_trajectories = sample(uncompiled_sir, self.timepoints, 1)
            for state_variable in compiled_trajectories:
                if "_sol" in state_variable:
                    self.assertTrue(
                        torch.allclose(
                            compiled_trajectories[state_variable][i],
                            uncompiled_trajectories[state_variable][0],
                            atol=1e-4,
                        ),
                        f"Compiled {state_variable} trajectory {i}: {compiled_trajectories[state_variable][i]}\n"
                        f"Uncompiled {state_variable} trajectory: {uncompiled_trajectories[state_variable][0]}",
                    )

    def test_extract_sympy(self):
        """Test that the sympy expression can be extracted from the rate law."""
        for template in self.tm.templates:
            rate_law = template.rate_law
            expected_rate_law = sympy.sympify(
                str(rate_law), locals={str(x): x for x in rate_law.free_symbols}
            )
            self.assertEqual(str(expected_rate_law), str(rate_law))
            self.assertEqual(
                str(expected_rate_law), str(self.compiled_sir.extract_sympy(rate_law))
            )
            self.assertEqual(
                expected_rate_law, self.compiled_sir.extract_sympy(rate_law)
            )

    def test_symbolic_flux_to_numeric_flux(self):
        """Test that the symbolic flux can be converted to a numeric flux."""
        self.compiled_sir.param_prior()
        expected_deriv = self.numeric_deriv(
            beta=0.5,
            gamma=0.2,
            susceptible_population=99999.0,
            infected_population=1.0,
            recovered_population=0.0,
        )
        actual_deriv = self.compiled_sir.compiled_rate_law(
            beta=0.5,
            gamma=0.2,
            susceptible_population=99999.0,
            infected_population=1.0,
            recovered_population=0.0,
        )
        self.assertTrue(
            torch.allclose(expected_deriv, actual_deriv, atol=1e-3),
            f"Expected deriv: {expected_deriv}\n" f"Actual deriv {actual_deriv}",
        )

    def test_time_varying_parameter_rate_law(self):
        """Test that the rate law can be compiled correctly."""
        url = "https://raw.githubusercontent.com/indralab/mira/56bf4c0d77919142684c8cbfb3521b7bf4470888/notebooks/hackathon_2023.07/scenario1_c.json"
        scenario1_c = load_petri_model(url, compile_rate_law_p=True)
        expected_rate_law_str = "kappa*(beta_nc + (beta_c - beta_nc)/(1 + exp(-k_2*(-t + t_1))) + (-beta_c + beta_s)/(1 + exp(-k_1*(-t + t_0))))"
        expected_rate_law_symbolic = sympy.sympify(expected_rate_law_str)
        param_vals = dict(
            kappa=1.0,
            beta_nc=0.5,
            beta_c=0.6,
            beta_s=0.4,
            k_1=0.1,
            k_2=0.1,
            t_0=0.0,
            t_1=1.0,
        )
        expected_rate_law_mod = SymPyModule(expressions=[expected_rate_law_symbolic])
        expected_rate_law_values = expected_rate_law_mod(
            **param_vals, **dict(t=torch.tensor(0.2))
        )
        actual_rate_law_symbolic = scenario1_c.extract_sympy(
            scenario1_c.G.template_model.templates[0].rate_law
        )
        actual_rate_law_mod = SymPyModule(expressions=[actual_rate_law_symbolic])
        actual_rate_law_values1 = actual_rate_law_mod(
            **param_vals, **dict(t=torch.tensor(23.0))
        )
        actual_rate_law_values2 = actual_rate_law_mod(
            **param_vals, **dict(t=torch.tensor(0.2))
        )

        self.assertFalse(
            torch.allclose(
                expected_rate_law_values, actual_rate_law_values1, atol=1e-3
            ),
            f"Expected rate law value: {expected_rate_law_values}\n"
            f"Actual rate law value {actual_rate_law_values1}",
        )

        self.assertTrue(
            torch.allclose(
                expected_rate_law_values, actual_rate_law_values2, atol=1e-3
            ),
            f"Expected rate law value: {expected_rate_law_values}\n"
            f"Actual rate law value {actual_rate_law_values2}",
        )

    def test_askem_model_representation(self):
        """Test that the rate law can be compiled correctly."""
        url = "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir_typed.json"
        samples = load_and_sample_petri_model(
            url,
            num_samples=self.nsamples,
            timepoints=self.timepoints,
            compile_rate_law_p=True,
        )
        self.assertIsInstance(samples, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
