# Model Kitchen Utils

# Load dependencies, MIRA modeling tools ##############################################
import sympy
from copy import deepcopy as _d
from mira.metamodel import *
from mira.modeling import Model #, Author
from mira.modeling.askenet.petrinet import AskeNetPetriNetModel
import jsonschema
import itertools as itt
from tqdm.auto import tqdm
from collections import defaultdict
import requests
from sympy import IndexedBase, Indexed

# Define model and AMR sanity checks ##################################################
def sanity_check_tm(tm: TemplateModel):
    assert tm.templates
    all_concept_names = set(tm.get_concepts_name_map())
    all_parameter_names = set(tm.parameters)
    all_symbols = (
        all_concept_names
        | all_parameter_names
        | ({tm.time.name} if tm.time else set())
    )
    for template in tm.templates:
        assert template.rate_law
        symbols = template.rate_law.args[0].free_symbols
        for symbol in symbols:
            assert symbol.name in all_symbols, f"missing symbol: {symbol.name}"
    all_initial_names = {init.concept.name for init in tm.initials.values()}
    for concept in all_concept_names:
        assert concept in all_initial_names

def sanity_check_amr(amr_json):
    import requests

    assert "schema" in amr_json
    schema_json = requests.get(amr_json["schema"]).json()
    jsonschema.validate(schema_json, amr_json)

# Define units ######################################################################
person_units = lambda: Unit(expression=sympy.Symbol('person'))
virus_units = lambda: Unit(expression=sympy.Symbol('virus'))
virus_per_gram_units = lambda: Unit(expression=sympy.Symbol('virus')/sympy.Symbol('gram'))
day_units = lambda: Unit(expression=sympy.Symbol('day'))
per_day_units = lambda: Unit(expression=1/sympy.Symbol('day'))
dimensionless_units = lambda: Unit(expression=sympy.Integer('1'))
gram_units = lambda: Unit(expression=sympy.Symbol('gram'))
per_day_per_person_units = lambda: Unit(expression=1/(sympy.Symbol('day')*sympy.Symbol('person')))

# Define BASE_CONCEPTS, BASE_INITIALS, BASE_PARAMETERS #############################
def base_ingredients(total_population_value, I0, E0):
    
    BASE_CONCEPTS = {
        "S": Concept(
            name="S", units=person_units(), identifiers={"ido": "0000514"}
        ),
        "E": Concept(
            name="E", units=person_units(), identifiers={"apollosv": "0000154"}
        ),
        "I": Concept(
            name="I", units=person_units(), identifiers={"ido": "0000511"}
        ),
        "R": Concept(
            name="R", units=person_units(), identifiers={"ido": "0000592"}
        ),
        "H": Concept(
            name="H", units=person_units(), identifiers={"ido": "0000511"}, 
                    context={"property": "ncit:C25179"}
        ),
        "D": Concept(
            name="D", units=person_units(), identifiers={"ncit": "C28554"}
        ),
    }
    
    BASE_PARAMETERS = {
        'total_population': Parameter(name='total_population', value=total_population_value, units=person_units()),
        'beta': Parameter(name='beta', value=0.4, units=per_day_units(),
                           distribution=Distribution(type='Uniform1',
                                                     parameters={
                                                         'minimum': 0.05,
                                                         'maximum': 0.8
                                                     })),
        'delta': Parameter(name='delta', value=0.25, units=per_day_units()),
        'gamma': Parameter(name='gamma', value=0.2, units=per_day_units(),
                           distribution=Distribution(type='Uniform1',
                                                     parameters={
                                                         'minimum': 0.1,
                                                         'maximum': 0.5
                                                     })),
        'death': Parameter(name='death', value=0.007, units=per_day_units(), # death rate of infectious population
                           distribution=Distribution(type='Uniform1',
                                                     parameters={
                                                         'minimum': 0.001,
                                                         'maximum': 0.01
                                                     })),
        'hosp': Parameter(name='hosp', value=0.1, units=per_day_units(), 
                           distribution=Distribution(type='Uniform1', 
                                                     parameters={
                                                         'minimum': 0.005,
                                                         'maximum': 0.2
                                                     })),
        'los': Parameter(name='los', value=5, units=day_units()),
        'death_hosp' = Parameter(name='death', value=0.07, units=per_day_units(), # death rate of hospitalized individuals
                           distribution=Distribution(type='Uniform1', 
                                                     parameters={
                                                          'minimum': 0.01,
                                                          'maximum': 0.1
                                                     }))
    }
    
    BASE_INITIALS = {
        "S": Initial(concept=Concept(name="S"), value=total_population_value - (E0 + I0)),
        "E": Initial(concept=Concept(name="E"), value=E0),
        "I": Initial(concept=Concept(name="I"), value=I0),
        "R": Initial(concept=Concept(name="R"), value=0),
        "H": Initial(concept=Concept(name="H"), value=0),
        "D": Initial(concept=Concept(name="D"), value=0),
    }
    
    observables = {}

    return BASE_CONCEPTS, BASE_PARAMETERS, BASE_INITIALS, observables

def make_SEIRD_model(total_population_value, I0, E0, model_name):

    S, E, I, R, D, total_population, beta, delta, gamma, death = \
        sympy.symbols('S E I R D total_population beta delta gamma death')
    
    t1 = ControlledConversion(subject=BASE_CONCEPTS['S'],
                              outcome=BASE_CONCEPTS['E'],
                              controller=BASE_CONCEPTS['I'],
                              rate_law=S*I*beta / total_population)
    t2 = NaturalConversion(subject=BASE_CONCEPTS['E'],
                           outcome=BASE_CONCEPTS['I'],
                           rate_law=delta*E)
    t3 = NaturalConversion(subject=BASE_CONCEPTS['I'],
                           outcome=BASE_CONCEPTS['R'],
                           rate_law=gamma*(1 - death)*I)
    t4 = NaturalConversion(subject=BASE_CONCEPTS['I'],
                           outcome=BASE_CONCEPTS['D'],
                           rate_law=gamma*death*I)

    templates = [t1, t2, t3, t4]
    tm = TemplateModel(
        templates=templates,
        parameters=BASE_PARAMETERS,
        initials=BASE_INITIALS,
        time=Time(name='t', units=day_units()),
        observables=observables,
        annotations=Annotations(name=MODEL_NAME)
    )
    
    sanity_check_tm(tm)
    return tm
        
















