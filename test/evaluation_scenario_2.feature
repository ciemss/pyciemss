Feature: Evaluation_scenario 2
    Reproduce the result in page 9 of the supplementary methods section of the SIDARTHE publication

    Scenario: Unit test 1
        Given initial conditions
        And parameters
        And initial state
        And SIDARTHE model
    
    	When simulating the model for 100 days

	    Then peak of infection is around day 47

    Scenario: Unit test 1 with MIRA
        Given MIRA SIDARTHE model
        And MIRA initial state

        When simulating the model for 100 days

        Then peak of infection is around day 47
 
    Scenario: Unit test 2
        Given initial conditions
        And parameters
        And SIDARTHE model
        And interventions

        When applying all interventions
	    And simulating the intervened model for 100 days

        Then peak of infection is around day 50
        And percent infected is around 0.2%
