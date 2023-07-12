<?xml version="1.0" encoding="UTF-8"?>
<!-- generated with COPASI 4.29 (Build 228) (http://www.copasi.org) at 2020-09-20T11:06:31Z -->
<?oxygen RNGSchema="http://www.copasi.org/static/schema/CopasiML.rng" type="xml"?>
<COPASI xmlns="http://www.copasi.org/static/schema" versionMajor="4" versionMinor="29" versionDevel="228" copasiSourcesModified="0">
  <ListOfFunctions>
    <Function key="Function_13" name="Mass action (irreversible)" type="MassAction" reversible="false">
      <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
   <rdf:Description rdf:about="#Function_13">
   <CopasiMT:is rdf:resource="urn:miriam:obo.sbo:SBO:0000163" />
   </rdf:Description>
   </rdf:RDF>
      </MiriamAnnotation>
      <Comment>
        <body xmlns="http://www.w3.org/1999/xhtml">
<b>Mass action rate law for irreversible reactions</b>
<p>
Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does not include any reverse process that creates the reactants from the products. The change of a product quantity is proportional to the quantity of one reactant.
</p>
</body>
      </Comment>
      <Expression>
        k1*PRODUCT&lt;substrate_i>
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_80" name="k1" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_81" name="substrate" order="1" role="substrate"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_40" name="Rate Law for Susceptible_to_Infected" type="UserDefined" reversible="false">
      <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Function_40">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:42:52Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        S*(alpha*I + beta*D + gamma*A + delta*R)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_264" name="S" order="0" role="substrate"/>
        <ParameterDescription key="FunctionParameter_263" name="alpha" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_262" name="I" order="2" role="product"/>
        <ParameterDescription key="FunctionParameter_261" name="beta" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_250" name="D" order="4" role="modifier"/>
        <ParameterDescription key="FunctionParameter_265" name="gamma" order="5" role="constant"/>
        <ParameterDescription key="FunctionParameter_266" name="A" order="6" role="modifier"/>
        <ParameterDescription key="FunctionParameter_267" name="delta" order="7" role="constant"/>
        <ParameterDescription key="FunctionParameter_268" name="R" order="8" role="modifier"/>
      </ListOfParameterDescriptions>
    </Function>
  </ListOfFunctions>
  <Model key="Model_1" name="Giordano2020 - SIDARTHE model of COVID-19 spread in Italy" simulationType="time" timeUnit="d" volumeUnit="1" areaUnit="1" lengthUnit="1" quantityUnit="#" type="deterministic" avogadroConstant="6.0221408570000002e+23">
    <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:vCard="http://www.w3.org/2001/vcard-rdf/3.0#">
  <rdf:Description rdf:about="#Model_1">
    <bqbiol:hasProperty rdf:resource="urn:miriam:doid:DOID:0080600"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:ido:0000503"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:mamo:MAMO_0000028"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:mamo:MAMO_0000046"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:mamo:MAMO_0000206"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:mamo:MAMO_0000207"/>
    <bqbiol:hasTaxon rdf:resource="urn:miriam:taxonomy:2697049"/>
    <bqbiol:hasTaxon rdf:resource="urn:miriam:taxonomy:9606"/>
    <dcterms:bibliographicCitation>
      <rdf:Description>
        <CopasiMT:isDescribedBy rdf:resource="urn:miriam:pubmed:32322102"/>
      </rdf:Description>
    </dcterms:bibliographicCitation>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:17:26Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <dcterms:creator>
      <rdf:Description>
        <vCard:EMAIL>ktiwari@ebi.ac.uk</vCard:EMAIL>
        <vCard:N>
          <rdf:Description>
            <vCard:Family>Tiwari</vCard:Family>
            <vCard:Given>Krishna</vCard:Given>
          </rdf:Description>
        </vCard:N>
        <vCard:ORG>
          <rdf:Description>
            <vCard:Orgname>European Bioinformatics Institute, European Molecular Biology Laboratory</vCard:Orgname>
          </rdf:Description>
        </vCard:ORG>
      </rdf:Description>
    </dcterms:creator>
    <dcterms:creator>
      <rdf:Description>
        <vCard:EMAIL>kramachandran@ebi.ac.uk</vCard:EMAIL>
        <vCard:N>
          <rdf:Description>
            <vCard:Family>Ramachandran</vCard:Family>
            <vCard:Given>Kausthubh</vCard:Given>
          </rdf:Description>
        </vCard:N>
        <vCard:ORG>
          <rdf:Description>
            <vCard:Orgname>European Bioinformatics Institute, European Molecular Biology Laboratory</vCard:Orgname>
          </rdf:Description>
        </vCard:ORG>
      </rdf:Description>
    </dcterms:creator>
    <dcterms:creator>
      <rdf:Description>
        <vCard:EMAIL>sheriff@ebi.ac.uk</vCard:EMAIL>
        <vCard:N>
          <rdf:Description>
            <vCard:Family>Sheriff</vCard:Family>
            <vCard:Given>Rahuman</vCard:Given>
          </rdf:Description>
        </vCard:N>
        <vCard:ORG>
          <rdf:Description>
            <vCard:Orgname>European Bioinformatics Institute, European Molecular Biology Laboratory</vCard:Orgname>
          </rdf:Description>
        </vCard:ORG>
      </rdf:Description>
    </dcterms:creator>
    <dcterms:modified>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-17T02:10:46</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:modified>
  </rdf:Description>
</rdf:RDF>

    </MiriamAnnotation>
    <Comment>
      This is a model created on COPASI 4.27 (Build 217) which reproduces the Figures 2b, 2d, 3b, 3d, 4b, 4d in the article - https://www.nature.com/articles/s41591-020-0883-7

To reproduce Fig 2b and 2d, set Event_trigger_Fig3b = 0, Event_trigger_Fig3d = 0, Event_trigger_Fig4b = 0, Event_trigger_Fig4d = 0, epsilon_modifer = 1, alpha_modifer = 1 and run for t = 45 d (for Fig 2b) and t = 350 days (for Fig 2d).
Set alpha_modifier = 0 for the remaining 4 cases
To reproduce Fig 3b, set Event_trigger_Fig3b = 1, Event_trigger_Fig3d = 0, Event_trigger_Fig4b = 0, Event_trigger_Fig4d = 0, epsilon_modifer = 1 and run for t = 350 days.
To reproduce Fig 3d, set Event_trigger_Fig3b = 0, Event_trigger_Fig3d = 1, Event_trigger_Fig4b = 0, Event_trigger_Fig4d = 0, epsilon_modifer = 1 and run for t = 350 days.
To reproduce Fig 4b, set Event_trigger_Fig3b = 0, Event_trigger_Fig3d = 0, Event_trigger_Fig4b = 1, Event_trigger_Fig4d = 0, epsilon_modifer = 0 and run for t = 350 days.
To reproduce Fig 4d, set Event_trigger_Fig3b = 0, Event_trigger_Fig3d = 0, Event_trigger_Fig4b = 0, Event_trigger_Fig4d = 1, epsilon_modifer = 0 and run for t = 350 days.



Modelling the COVID-19 epidemic and implementation of population-wide interventions in Italy
Giulia Giordano, Franco Blanchini, Raffaele Bruno, Patrizio Colaneri, Alessandro Di Filippo, Angela Di Matteo and Marta Colaneri
Journal - Nature Medicine
DOI - 10.1038/s41591-020-0883-7

In Italy, 128,948 confirmed cases and 15,887 deaths of people who tested positive for SARS-CoV-2 were registered as of 5 April 2020. Ending the global SARS-CoV-2 pandemic requires implementation of multiple population-wide strategies, including social distancing, testing and contact tracing. We propose a new model that predicts the course of the epidemic to help plan an effective control strategy. The model considers eight stages of infection: susceptible (S), infected (I), diagnosed (D), ailing (A), recognized (R), threatened (T), healed (H) and extinct (E), collectively termed SIDARTHE. Our SIDARTHE model discriminates between infected individuals depending on whether they have been diagnosed and on the severity of their symptoms. The distinction between diagnosed and non-diagnosed individuals is important because the former are typically isolated and hence less likely to spread the infection. This delineation also helps to explain misperceptions of the case fatality rate and of the epidemic spread. We compare simulation results with real data on the COVID-19 epidemic in Italy, and we model possible scenarios of implementation of countermeasures. Our results demonstrate that restrictive social-distancing measures will need to be combined with widespread testing and contact tracing to end the ongoing COVID-19 pandemic.
    </Comment>
    <ListOfCompartments>
      <Compartment key="Compartment_0" name="Italy" simulationType="fixed" dimensionality="3" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Compartment_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:20:49Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C16761" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Compartment>
    </ListOfCompartments>
    <ListOfMetabolites>
      <Metabolite key="Metabolite_0" name="Susceptible" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:bqbiol="http://biomodels.net/biology-qualifiers/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_0">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ido:0000468" />
    <bqbiol:hasProperty rdf:resource="urn:miriam:ido:0000659" />
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:20:55Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000514" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_1" name="Infected" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:bqbiol="http://biomodels.net/biology-qualifiers/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_1">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C113725" />
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:20:58Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000569" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Comment>
          Infected ND AS - non-diagnosed, asymptomatic (I)
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_2" name="Diagnosed" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:bqbiol="http://biomodels.net/biology-qualifiers/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_2">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C15220" />
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C3833" />
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:20:59Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Comment>
          Infected D AS - diagnosed, asymptomatic (D)
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_3" name="Ailing" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:bqbiol="http://biomodels.net/biology-qualifiers/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_3">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C113725" />
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:21:01Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000573" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Comment>
          Infected ND S - non-diagnosed, symptomatic (A)
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_4" name="Recognized" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:bqbiol="http://biomodels.net/biology-qualifiers/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_4">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C25587" />
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:21:02Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000573" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Comment>
          Infected D S - diagnosed, symptomatic (R)
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_5" name="Threatened" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:bqbiol="http://biomodels.net/biology-qualifiers/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_5">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C15220" />
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C25467" />
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:21:03Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000573" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Comment>
          Infected D IC - diagnosed, life-threatening symptoms (T)
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_6" name="Healed" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:bqbiol="http://biomodels.net/biology-qualifiers/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_6">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ido:0000621" />
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:21:04Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_7" name="Extinct" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_7">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:21:06Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C28554" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
    </ListOfMetabolites>
    <ListOfModelValues>
      <ModelValue key="ModelValue_0" name="alpha" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:28:04Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          1/(#*d)
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_1" name="beta" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_1">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:28:09Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          1/(#*d)
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_2" name="gamma" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_2">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:28:10Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          1/(#*d)
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_3" name="delta" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_3">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:28:12Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          1/(#*d)
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_4" name="epsilon" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_4">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:28:14Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          1/d
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_5" name="theta" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_5">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:28:27Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          1/d
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_6" name="zeta" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_6">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:28:43Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          1/d
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_7" name="eta" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_7">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:28:44Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          1/d
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_8" name="mu" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_8">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:28:53Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          1/d
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_9" name="nu" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_9">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:28:54Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          1/d
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_10" name="tau" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_10">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:29:10Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          1/d
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_11" name="lambda" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_11">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:29:17Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          1/d
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_12" name="kappa" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_12">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:29:19Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          1/d
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_13" name="rho" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_13">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:29:27Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          1/d
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_14" name="sigma" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_14">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:29:29Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          1/d
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_15" name="xi" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_15">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:30:01Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          1/d
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_16" name="Event_trigger_Fig3b" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_16">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-17T20:28:24Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          dimensionless
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_17" name="Event_trigger_Fig3d" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_17">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-17T20:32:36Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          dimensionless
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_18" name="Event_trigger_Fig4b" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_18">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-17T20:38:20Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          dimensionless
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_19" name="Event_trigger_Fig4d" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_19">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-17T20:38:22Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          dimensionless
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_20" name="epsilon_modifier" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_20">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-17T21:29:48Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          dimensionless
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_21" name="alpha_modifier" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_21">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-28T06:37:18Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          dimensionless
        </Unit>
      </ModelValue>
    </ListOfModelValues>
    <ListOfReactions>
      <Reaction key="Reaction_0" name="Susceptible_to_Infected" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:22:19Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:doid:DOID:0080600" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_0" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_1" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_2" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_3" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_4" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_5390" name="beta" value="0.011"/>
          <Constant key="Parameter_5389" name="gamma" value="0.456"/>
          <Constant key="Parameter_5388" name="alpha" value="0.57"/>
          <Constant key="Parameter_5387" name="delta" value="0.011"/>
        </ListOfConstants>
        <KineticLaw function="Function_40" unitType="ConcentrationPerTime" scalingCompartment="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_264">
              <SourceParameter reference="Metabolite_0"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_263">
              <SourceParameter reference="ModelValue_0"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_262">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_261">
              <SourceParameter reference="ModelValue_1"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_250">
              <SourceParameter reference="Metabolite_2"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_265">
              <SourceParameter reference="ModelValue_2"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_266">
              <SourceParameter reference="Metabolite_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_267">
              <SourceParameter reference="ModelValue_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_268">
              <SourceParameter reference="Metabolite_4"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_1" name="Infected_to_Diagnosed" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_1">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:22:42Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:doid:DOID:0080600" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C15220" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_1" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_2" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5386" name="k1" value="0.171"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="ConcentrationPerTime" scalingCompartment="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_4"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_2" name="Infected_to_Ailing" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_2">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:22:43Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:doid:DOID:0080600" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C168447" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_1" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_3" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5385" name="k1" value="0.125"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="ConcentrationPerTime" scalingCompartment="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_6"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_3" name="Infected_to_Healed" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_3">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:22:43Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C25746" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_1" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_6" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5384" name="k1" value="0.034"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="ConcentrationPerTime" scalingCompartment="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_11"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_4" name="Diagnosed_to_Recognized" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_4">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:22:44Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:doid:DOID:0080600" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C15220" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C168447" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_2" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_4" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5383" name="k1" value="0.125"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="ConcentrationPerTime" scalingCompartment="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_7"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_2"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_5" name="Diagnosed_to_Healed" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_5">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:22:44Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C25746" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_2" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_6" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5382" name="k1" value="0.034"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="ConcentrationPerTime" scalingCompartment="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_13"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_2"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_6" name="Ailing_to_Recognised" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_6">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:22:47Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:doid:DOID:0080600" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C15220" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_3" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_4" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5381" name="k1" value="0.371"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="ConcentrationPerTime" scalingCompartment="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_5"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_3"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_7" name="Ailing_to_Healed" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_7">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:22:48Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C25746" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_3" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_6" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5380" name="k1" value="0.017"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="ConcentrationPerTime" scalingCompartment="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_12"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_3"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_8" name="Ailing_to_Threatened" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_8">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:22:49Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:doid:DOID:0080600" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:meddra:10049438" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_3" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_5" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5379" name="k1" value="0.017"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="ConcentrationPerTime" scalingCompartment="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_8"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_3"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_9" name="Recognised_to_Threatened" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_9">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:22:51Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:doid:DOID:0080600" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:meddra:10049438" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_4" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_5" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5378" name="k1" value="0.027"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="ConcentrationPerTime" scalingCompartment="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_9"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_4"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_10" name="Recognised_to_Healed" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_10">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:26:43Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C25746" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_4" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_6" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5377" name="k1" value="0.017"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="ConcentrationPerTime" scalingCompartment="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_15"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_4"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_11" name="Threatened_to_Extinct" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_11">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:26:45Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:doid:DOID:0080600" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C28554" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_5" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_7" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5376" name="k1" value="0.01"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="ConcentrationPerTime" scalingCompartment="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_10"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_5"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_12" name="Threatened_to_Healed" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_12">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-17T19:59:33Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C25746" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_5" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_6" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5375" name="k1" value="0.017"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="ConcentrationPerTime" scalingCompartment="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_14"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_5"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
    </ListOfReactions>
    <ListOfEvents>
      <Event key="Event_0" name="Day_4" delayAssignment="true" fireAtInitialTime="0" persistentTrigger="0">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:57:15Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Reference=Time> > 4
        </TriggerExpression>
        <DelayExpression>
          0
        </DelayExpression>
        <ListOfAssignments>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[alpha]" targetKey="ModelValue_0">
            <Expression>
              0.422
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[beta]" targetKey="ModelValue_1">
            <Expression>
              0.0057
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[gamma]" targetKey="ModelValue_2">
            <Expression>
              0.285
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[delta]" targetKey="ModelValue_3">
            <Expression>
              0.0057
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_1" name="Day_12" delayAssignment="true" fireAtInitialTime="0" persistentTrigger="0">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_1">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:57:31Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Reference=Time> > 12
        </TriggerExpression>
        <DelayExpression>
          0
        </DelayExpression>
        <ListOfAssignments>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[epsilon]" targetKey="ModelValue_4">
            <Expression>
              0.143
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_2" name="Day_22" delayAssignment="true" fireAtInitialTime="0" persistentTrigger="0">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_2">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T21:57:44Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Reference=Time> > 22
        </TriggerExpression>
        <DelayExpression>
          0
        </DelayExpression>
        <ListOfAssignments>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[alpha]" targetKey="ModelValue_0">
            <Expression>
              0.360
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[beta]" targetKey="ModelValue_1">
            <Expression>
              0.005
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[gamma]" targetKey="ModelValue_2">
            <Expression>
              0.200
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[delta]" targetKey="ModelValue_3">
            <Expression>
              0.005
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[mu]" targetKey="ModelValue_8">
            <Expression>
              0.008
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[nu]" targetKey="ModelValue_9">
            <Expression>
              0.015
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[lambda]" targetKey="ModelValue_11">
            <Expression>
              0.08
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[rho]" targetKey="ModelValue_13">
            <Expression>
              0.017
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[kappa]" targetKey="ModelValue_12">
            <Expression>
              0.017
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[xi]" targetKey="ModelValue_15">
            <Expression>
              0.017
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[sigma]" targetKey="ModelValue_14">
            <Expression>
              0.017
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[zeta]" targetKey="ModelValue_6">
            <Expression>
              0.034
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[eta]" targetKey="ModelValue_7">
            <Expression>
              0.034
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_3" name="Day_28" delayAssignment="true" fireAtInitialTime="0" persistentTrigger="0">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_3">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T22:04:41Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Reference=Time> > 28
        </TriggerExpression>
        <DelayExpression>
          0
        </DelayExpression>
        <ListOfAssignments>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[alpha]" targetKey="ModelValue_0">
            <Expression>
              0.210
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[gamma]" targetKey="ModelValue_2">
            <Expression>
              0.110
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_4" name="Day_38" delayAssignment="true" fireAtInitialTime="0" persistentTrigger="0">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_4">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-13T22:05:17Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Reference=Time> > 38
        </TriggerExpression>
        <DelayExpression>
          0
        </DelayExpression>
        <ListOfAssignments>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[rho]" targetKey="ModelValue_13">
            <Expression>
              0.020
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[xi]" targetKey="ModelValue_15">
            <Expression>
              0.020
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[sigma]" targetKey="ModelValue_14">
            <Expression>
              0.010
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[zeta]" targetKey="ModelValue_6">
            <Expression>
              0.025
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[epsilon]" targetKey="ModelValue_4">
            <Expression>
              0.200
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[eta]" targetKey="ModelValue_7">
            <Expression>
              0.025
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[kappa]" targetKey="ModelValue_12">
            <Expression>
              0.020
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_5" name="Day_50" delayAssignment="true" fireAtInitialTime="0" persistentTrigger="0">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_5">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-07-17T21:17:15Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Reference=Time> > 50
        </TriggerExpression>
        <DelayExpression>
          0
        </DelayExpression>
        <ListOfAssignments>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[alpha]" targetKey="ModelValue_0">
            <Expression>
              0.209*&lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[alpha_modifier],Reference=InitialValue>+0.252*&lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[Event_trigger_Fig3b],Reference=InitialValue>+0.105*&lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[Event_trigger_Fig3d],Reference=InitialValue>+0.420*&lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[Event_trigger_Fig4d],Reference=InitialValue>
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[epsilon]" targetKey="ModelValue_4">
            <Expression>
              0.200*&lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[epsilon_modifier],Reference=InitialValue>+0.400*&lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[Event_trigger_Fig4b],Reference=InitialValue>+0.600*&lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[Event_trigger_Fig4d],Reference=InitialValue>
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
    </ListOfEvents>
    <ListOfModelParameterSets activeSet="ModelParameterSet_1">
      <ModelParameterSet key="ModelParameterSet_1" name="Initial State">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelParameterSet_1">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-09-20T11:02:35Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ModelParameterGroup cn="String=Initial Time" type="Group">
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy" value="0" type="Model" simulationType="time"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Compartment Sizes" type="Group">
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy]" value="1" type="Compartment" simulationType="fixed"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Species Values" type="Group">
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy],Vector=Metabolites[Susceptible]" value="0.99999629999999995" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy],Vector=Metabolites[Infected]" value="3.3333333299999999e-06" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy],Vector=Metabolites[Diagnosed]" value="3.33333333e-07" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy],Vector=Metabolites[Ailing]" value="1.6666666599999999e-08" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy],Vector=Metabolites[Recognized]" value="3.33333333e-08" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy],Vector=Metabolites[Threatened]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy],Vector=Metabolites[Healed]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy],Vector=Metabolites[Extinct]" value="0" type="Species" simulationType="reactions"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Global Quantities" type="Group">
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[alpha]" value="0.56999999999999995" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[beta]" value="0.010999999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[gamma]" value="0.45600000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[delta]" value="0.010999999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[epsilon]" value="0.17100000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[theta]" value="0.371" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[zeta]" value="0.125" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[eta]" value="0.125" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[mu]" value="0.017000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[nu]" value="0.027" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[tau]" value="0.01" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[lambda]" value="0.034000000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[kappa]" value="0.017000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[rho]" value="0.034000000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[sigma]" value="0.017000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[xi]" value="0.017000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[Event_trigger_Fig3b]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[Event_trigger_Fig3d]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[Event_trigger_Fig4b]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[Event_trigger_Fig4d]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[epsilon_modifier]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[alpha_modifier]" value="1" type="ModelValue" simulationType="fixed"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Kinetic Parameters" type="Group">
          <ModelParameterGroup cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Susceptible_to_Infected]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Susceptible_to_Infected],ParameterGroup=Parameters,Parameter=beta" value="0.010999999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[beta],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Susceptible_to_Infected],ParameterGroup=Parameters,Parameter=gamma" value="0.45600000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[gamma],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Susceptible_to_Infected],ParameterGroup=Parameters,Parameter=alpha" value="0.56999999999999995" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[alpha],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Susceptible_to_Infected],ParameterGroup=Parameters,Parameter=delta" value="0.010999999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[delta],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Infected_to_Diagnosed]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Infected_to_Diagnosed],ParameterGroup=Parameters,Parameter=k1" value="0.17100000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[epsilon],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Infected_to_Ailing]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Infected_to_Ailing],ParameterGroup=Parameters,Parameter=k1" value="0.125" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[zeta],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Infected_to_Healed]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Infected_to_Healed],ParameterGroup=Parameters,Parameter=k1" value="0.034000000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[lambda],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Diagnosed_to_Recognized]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Diagnosed_to_Recognized],ParameterGroup=Parameters,Parameter=k1" value="0.125" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[eta],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Diagnosed_to_Healed]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Diagnosed_to_Healed],ParameterGroup=Parameters,Parameter=k1" value="0.034000000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[rho],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Ailing_to_Recognised]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Ailing_to_Recognised],ParameterGroup=Parameters,Parameter=k1" value="0.371" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[theta],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Ailing_to_Healed]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Ailing_to_Healed],ParameterGroup=Parameters,Parameter=k1" value="0.017000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[kappa],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Ailing_to_Threatened]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Ailing_to_Threatened],ParameterGroup=Parameters,Parameter=k1" value="0.017000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[mu],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Recognised_to_Threatened]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Recognised_to_Threatened],ParameterGroup=Parameters,Parameter=k1" value="0.027" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[nu],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Recognised_to_Healed]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Recognised_to_Healed],ParameterGroup=Parameters,Parameter=k1" value="0.017000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[xi],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Threatened_to_Extinct]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Threatened_to_Extinct],ParameterGroup=Parameters,Parameter=k1" value="0.01" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[tau],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Threatened_to_Healed]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Reactions[Threatened_to_Healed],ParameterGroup=Parameters,Parameter=k1" value="0.017000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Values[sigma],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
        </ModelParameterGroup>
      </ModelParameterSet>
    </ListOfModelParameterSets>
    <StateTemplate>
      <StateTemplateVariable objectReference="Model_1"/>
      <StateTemplateVariable objectReference="Metabolite_6"/>
      <StateTemplateVariable objectReference="Metabolite_1"/>
      <StateTemplateVariable objectReference="Metabolite_4"/>
      <StateTemplateVariable objectReference="Metabolite_5"/>
      <StateTemplateVariable objectReference="Metabolite_3"/>
      <StateTemplateVariable objectReference="Metabolite_2"/>
      <StateTemplateVariable objectReference="Metabolite_7"/>
      <StateTemplateVariable objectReference="Metabolite_0"/>
      <StateTemplateVariable objectReference="Compartment_0"/>
      <StateTemplateVariable objectReference="ModelValue_0"/>
      <StateTemplateVariable objectReference="ModelValue_1"/>
      <StateTemplateVariable objectReference="ModelValue_2"/>
      <StateTemplateVariable objectReference="ModelValue_3"/>
      <StateTemplateVariable objectReference="ModelValue_4"/>
      <StateTemplateVariable objectReference="ModelValue_5"/>
      <StateTemplateVariable objectReference="ModelValue_6"/>
      <StateTemplateVariable objectReference="ModelValue_7"/>
      <StateTemplateVariable objectReference="ModelValue_8"/>
      <StateTemplateVariable objectReference="ModelValue_9"/>
      <StateTemplateVariable objectReference="ModelValue_10"/>
      <StateTemplateVariable objectReference="ModelValue_11"/>
      <StateTemplateVariable objectReference="ModelValue_12"/>
      <StateTemplateVariable objectReference="ModelValue_13"/>
      <StateTemplateVariable objectReference="ModelValue_14"/>
      <StateTemplateVariable objectReference="ModelValue_15"/>
      <StateTemplateVariable objectReference="ModelValue_16"/>
      <StateTemplateVariable objectReference="ModelValue_17"/>
      <StateTemplateVariable objectReference="ModelValue_18"/>
      <StateTemplateVariable objectReference="ModelValue_19"/>
      <StateTemplateVariable objectReference="ModelValue_20"/>
      <StateTemplateVariable objectReference="ModelValue_21"/>
    </StateTemplate>
    <InitialState type="initialState">
      0 0 3.3333333299999999e-06 3.33333333e-08 0 1.6666666599999999e-08 3.33333333e-07 0 0.99999629999999995 1 0.56999999999999995 0.010999999999999999 0.45600000000000002 0.010999999999999999 0.17100000000000001 0.371 0.125 0.125 0.017000000000000001 0.027 0.01 0.034000000000000002 0.017000000000000001 0.034000000000000002 0.017000000000000001 0.017000000000000001 0 0 0 0 1 1 
    </InitialState>
  </Model>
  <ListOfTasks>
    <Task key="Task_15" name="Steady-State" type="steadyState" scheduled="false" updateModel="false">
      <Report reference="Report_11" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="JacobianRequested" type="bool" value="1"/>
        <Parameter name="StabilityAnalysisRequested" type="bool" value="1"/>
      </Problem>
      <Method name="Enhanced Newton" type="EnhancedNewton">
        <Parameter name="Resolution" type="unsignedFloat" value="1.0000000000000001e-09"/>
        <Parameter name="Derivation Factor" type="unsignedFloat" value="0.001"/>
        <Parameter name="Use Newton" type="bool" value="1"/>
        <Parameter name="Use Integration" type="bool" value="1"/>
        <Parameter name="Use Back Integration" type="bool" value="0"/>
        <Parameter name="Accept Negative Concentrations" type="bool" value="0"/>
        <Parameter name="Iteration Limit" type="unsignedInteger" value="50"/>
        <Parameter name="Maximum duration for forward integration" type="unsignedFloat" value="1000000000"/>
        <Parameter name="Maximum duration for backward integration" type="unsignedFloat" value="1000000"/>
        <Parameter name="Target Criterion" type="string" value="Distance and Rate"/>
      </Method>
    </Task>
    <Task key="Task_16" name="Time-Course" type="timeCourse" scheduled="false" updateModel="false">
      <Report reference="Report_21" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="AutomaticStepSize" type="bool" value="0"/>
        <Parameter name="StepNumber" type="unsignedInteger" value="36000"/>
        <Parameter name="StepSize" type="float" value="0.01"/>
        <Parameter name="Duration" type="float" value="360"/>
        <Parameter name="TimeSeriesRequested" type="bool" value="1"/>
        <Parameter name="OutputStartTime" type="float" value="0"/>
        <Parameter name="Output Event" type="bool" value="0"/>
        <Parameter name="Start in Steady State" type="bool" value="0"/>
        <Parameter name="Use Values" type="bool" value="0"/>
        <Parameter name="Values" type="string" value=""/>
      </Problem>
      <Method name="Deterministic (LSODA)" type="Deterministic(LSODA)">
        <Parameter name="Integrate Reduced Model" type="bool" value="0"/>
        <Parameter name="Relative Tolerance" type="unsignedFloat" value="9.9999999999999995e-07"/>
        <Parameter name="Absolute Tolerance" type="unsignedFloat" value="9.9999999999999998e-13"/>
        <Parameter name="Max Internal Steps" type="unsignedInteger" value="100000"/>
        <Parameter name="Max Internal Step Size" type="unsignedFloat" value="0"/>
      </Method>
    </Task>
    <Task key="Task_17" name="Scan" type="scan" scheduled="false" updateModel="false">
      <Problem>
        <Parameter name="Subtask" type="unsignedInteger" value="1"/>
        <ParameterGroup name="ScanItems">
        </ParameterGroup>
        <Parameter name="Output in subtask" type="bool" value="1"/>
        <Parameter name="Adjust initial conditions" type="bool" value="0"/>
        <Parameter name="Continue on Error" type="bool" value="0"/>
      </Problem>
      <Method name="Scan Framework" type="ScanFramework">
      </Method>
    </Task>
    <Task key="Task_18" name="Elementary Flux Modes" type="fluxMode" scheduled="false" updateModel="false">
      <Report reference="Report_12" target="" append="1" confirmOverwrite="1"/>
      <Problem>
      </Problem>
      <Method name="EFM Algorithm" type="EFMAlgorithm">
      </Method>
    </Task>
    <Task key="Task_19" name="Optimization" type="optimization" scheduled="false" updateModel="false">
      <Report reference="Report_13" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="Subtask" type="cn" value="CN=Root,Vector=TaskList[Steady-State]"/>
        <ParameterText name="ObjectiveExpression" type="expression">
          
        </ParameterText>
        <Parameter name="Maximize" type="bool" value="0"/>
        <Parameter name="Randomize Start Values" type="bool" value="0"/>
        <Parameter name="Calculate Statistics" type="bool" value="1"/>
        <ParameterGroup name="OptimizationItemList">
        </ParameterGroup>
        <ParameterGroup name="OptimizationConstraintList">
        </ParameterGroup>
      </Problem>
      <Method name="Random Search" type="RandomSearch">
        <Parameter name="Log Verbosity" type="unsignedInteger" value="0"/>
        <Parameter name="Number of Iterations" type="unsignedInteger" value="100000"/>
        <Parameter name="Random Number Generator" type="unsignedInteger" value="1"/>
        <Parameter name="Seed" type="unsignedInteger" value="0"/>
      </Method>
    </Task>
    <Task key="Task_20" name="Parameter Estimation" type="parameterFitting" scheduled="false" updateModel="false">
      <Report reference="Report_14" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="Maximize" type="bool" value="0"/>
        <Parameter name="Randomize Start Values" type="bool" value="0"/>
        <Parameter name="Calculate Statistics" type="bool" value="1"/>
        <ParameterGroup name="OptimizationItemList">
        </ParameterGroup>
        <ParameterGroup name="OptimizationConstraintList">
        </ParameterGroup>
        <Parameter name="Steady-State" type="cn" value="CN=Root,Vector=TaskList[Steady-State]"/>
        <Parameter name="Time-Course" type="cn" value="CN=Root,Vector=TaskList[Time-Course]"/>
        <Parameter name="Create Parameter Sets" type="bool" value="0"/>
        <Parameter name="Use Time Sens" type="bool" value="0"/>
        <Parameter name="Time-Sens" type="cn" value=""/>
        <ParameterGroup name="Experiment Set">
        </ParameterGroup>
        <ParameterGroup name="Validation Set">
          <Parameter name="Weight" type="unsignedFloat" value="1"/>
          <Parameter name="Threshold" type="unsignedInteger" value="5"/>
        </ParameterGroup>
      </Problem>
      <Method name="Evolutionary Programming" type="EvolutionaryProgram">
        <Parameter name="Log Verbosity" type="unsignedInteger" value="0"/>
        <Parameter name="Number of Generations" type="unsignedInteger" value="200"/>
        <Parameter name="Population Size" type="unsignedInteger" value="20"/>
        <Parameter name="Random Number Generator" type="unsignedInteger" value="1"/>
        <Parameter name="Seed" type="unsignedInteger" value="0"/>
        <Parameter name="Stop after # Stalled Generations" type="unsignedInteger" value="0"/>
      </Method>
    </Task>
    <Task key="Task_21" name="Metabolic Control Analysis" type="metabolicControlAnalysis" scheduled="false" updateModel="false">
      <Report reference="Report_15" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="Steady-State" type="key" value="Task_15"/>
      </Problem>
      <Method name="MCA Method (Reder)" type="MCAMethod(Reder)">
        <Parameter name="Modulation Factor" type="unsignedFloat" value="1.0000000000000001e-09"/>
        <Parameter name="Use Reder" type="bool" value="1"/>
        <Parameter name="Use Smallbone" type="bool" value="1"/>
      </Method>
    </Task>
    <Task key="Task_22" name="Lyapunov Exponents" type="lyapunovExponents" scheduled="false" updateModel="false">
      <Report reference="Report_16" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="ExponentNumber" type="unsignedInteger" value="3"/>
        <Parameter name="DivergenceRequested" type="bool" value="1"/>
        <Parameter name="TransientTime" type="float" value="0"/>
      </Problem>
      <Method name="Wolf Method" type="WolfMethod">
        <Parameter name="Orthonormalization Interval" type="unsignedFloat" value="1"/>
        <Parameter name="Overall time" type="unsignedFloat" value="1000"/>
        <Parameter name="Relative Tolerance" type="unsignedFloat" value="9.9999999999999995e-07"/>
        <Parameter name="Absolute Tolerance" type="unsignedFloat" value="9.9999999999999998e-13"/>
        <Parameter name="Max Internal Steps" type="unsignedInteger" value="10000"/>
      </Method>
    </Task>
    <Task key="Task_23" name="Time Scale Separation Analysis" type="timeScaleSeparationAnalysis" scheduled="false" updateModel="false">
      <Report reference="Report_17" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="StepNumber" type="unsignedInteger" value="100"/>
        <Parameter name="StepSize" type="float" value="0.01"/>
        <Parameter name="Duration" type="float" value="1"/>
        <Parameter name="TimeSeriesRequested" type="bool" value="1"/>
        <Parameter name="OutputStartTime" type="float" value="0"/>
      </Problem>
      <Method name="ILDM (LSODA,Deuflhard)" type="TimeScaleSeparation(ILDM,Deuflhard)">
        <Parameter name="Deuflhard Tolerance" type="unsignedFloat" value="0.0001"/>
      </Method>
    </Task>
    <Task key="Task_24" name="Sensitivities" type="sensitivities" scheduled="false" updateModel="false">
      <Report reference="Report_18" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="SubtaskType" type="unsignedInteger" value="1"/>
        <ParameterGroup name="TargetFunctions">
          <Parameter name="SingleObject" type="cn" value=""/>
          <Parameter name="ObjectListType" type="unsignedInteger" value="7"/>
        </ParameterGroup>
        <ParameterGroup name="ListOfVariables">
          <ParameterGroup name="Variables">
            <Parameter name="SingleObject" type="cn" value=""/>
            <Parameter name="ObjectListType" type="unsignedInteger" value="41"/>
          </ParameterGroup>
          <ParameterGroup name="Variables">
            <Parameter name="SingleObject" type="cn" value=""/>
            <Parameter name="ObjectListType" type="unsignedInteger" value="0"/>
          </ParameterGroup>
        </ParameterGroup>
      </Problem>
      <Method name="Sensitivities Method" type="SensitivitiesMethod">
        <Parameter name="Delta factor" type="unsignedFloat" value="0.001"/>
        <Parameter name="Delta minimum" type="unsignedFloat" value="9.9999999999999998e-13"/>
      </Method>
    </Task>
    <Task key="Task_25" name="Moieties" type="moieties" scheduled="false" updateModel="false">
      <Report reference="Report_19" target="" append="1" confirmOverwrite="1"/>
      <Problem>
      </Problem>
      <Method name="Householder Reduction" type="Householder">
      </Method>
    </Task>
    <Task key="Task_26" name="Cross Section" type="crosssection" scheduled="false" updateModel="false">
      <Problem>
        <Parameter name="AutomaticStepSize" type="bool" value="0"/>
        <Parameter name="StepNumber" type="unsignedInteger" value="100"/>
        <Parameter name="StepSize" type="float" value="0.01"/>
        <Parameter name="Duration" type="float" value="1"/>
        <Parameter name="TimeSeriesRequested" type="bool" value="1"/>
        <Parameter name="OutputStartTime" type="float" value="0"/>
        <Parameter name="Output Event" type="bool" value="0"/>
        <Parameter name="Start in Steady State" type="bool" value="0"/>
        <Parameter name="Use Values" type="bool" value="0"/>
        <Parameter name="Values" type="string" value=""/>
        <Parameter name="LimitCrossings" type="bool" value="0"/>
        <Parameter name="NumCrossingsLimit" type="unsignedInteger" value="0"/>
        <Parameter name="LimitOutTime" type="bool" value="0"/>
        <Parameter name="LimitOutCrossings" type="bool" value="0"/>
        <Parameter name="PositiveDirection" type="bool" value="1"/>
        <Parameter name="NumOutCrossingsLimit" type="unsignedInteger" value="0"/>
        <Parameter name="LimitUntilConvergence" type="bool" value="0"/>
        <Parameter name="ConvergenceTolerance" type="float" value="9.9999999999999995e-07"/>
        <Parameter name="Threshold" type="float" value="0"/>
        <Parameter name="DelayOutputUntilConvergence" type="bool" value="0"/>
        <Parameter name="OutputConvergenceTolerance" type="float" value="9.9999999999999995e-07"/>
        <ParameterText name="TriggerExpression" type="expression">
          
        </ParameterText>
        <Parameter name="SingleVariable" type="cn" value=""/>
      </Problem>
      <Method name="Deterministic (LSODA)" type="Deterministic(LSODA)">
        <Parameter name="Integrate Reduced Model" type="bool" value="0"/>
        <Parameter name="Relative Tolerance" type="unsignedFloat" value="9.9999999999999995e-07"/>
        <Parameter name="Absolute Tolerance" type="unsignedFloat" value="9.9999999999999998e-13"/>
        <Parameter name="Max Internal Steps" type="unsignedInteger" value="100000"/>
        <Parameter name="Max Internal Step Size" type="unsignedFloat" value="0"/>
      </Method>
    </Task>
    <Task key="Task_27" name="Linear Noise Approximation" type="linearNoiseApproximation" scheduled="false" updateModel="false">
      <Report reference="Report_20" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="Steady-State" type="key" value="Task_15"/>
      </Problem>
      <Method name="Linear Noise Approximation" type="LinearNoiseApproximation">
      </Method>
    </Task>
    <Task key="Task_28" name="Time-Course Sensitivities" type="timeSensitivities" scheduled="false" updateModel="false">
      <Problem>
        <Parameter name="AutomaticStepSize" type="bool" value="0"/>
        <Parameter name="StepNumber" type="unsignedInteger" value="100"/>
        <Parameter name="StepSize" type="float" value="0.01"/>
        <Parameter name="Duration" type="float" value="1"/>
        <Parameter name="TimeSeriesRequested" type="bool" value="1"/>
        <Parameter name="OutputStartTime" type="float" value="0"/>
        <Parameter name="Output Event" type="bool" value="0"/>
        <Parameter name="Start in Steady State" type="bool" value="0"/>
        <Parameter name="Use Values" type="bool" value="0"/>
        <Parameter name="Values" type="string" value=""/>
        <ParameterGroup name="ListOfParameters">
        </ParameterGroup>
        <ParameterGroup name="ListOfTargets">
        </ParameterGroup>
      </Problem>
      <Method name="LSODA Sensitivities" type="Sensitivities(LSODA)">
        <Parameter name="Integrate Reduced Model" type="bool" value="0"/>
        <Parameter name="Relative Tolerance" type="unsignedFloat" value="9.9999999999999995e-07"/>
        <Parameter name="Absolute Tolerance" type="unsignedFloat" value="9.9999999999999998e-13"/>
        <Parameter name="Max Internal Steps" type="unsignedInteger" value="10000"/>
        <Parameter name="Max Internal Step Size" type="unsignedFloat" value="0"/>
      </Method>
    </Task>
  </ListOfTasks>
  <ListOfReports>
    <Report key="Report_11" name="Steady-State" taskType="steadyState" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Footer>
        <Object cn="CN=Root,Vector=TaskList[Steady-State]"/>
      </Footer>
    </Report>
    <Report key="Report_12" name="Elementary Flux Modes" taskType="fluxMode" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Footer>
        <Object cn="CN=Root,Vector=TaskList[Elementary Flux Modes],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_13" name="Optimization" taskType="optimization" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Object=Description"/>
        <Object cn="String=\[Function Evaluations\]"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="String=\[Best Value\]"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="String=\[Best Parameters\]"/>
      </Header>
      <Body>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Problem=Optimization,Reference=Function Evaluations"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Problem=Optimization,Reference=Best Value"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Problem=Optimization,Reference=Best Parameters"/>
      </Body>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_14" name="Parameter Estimation" taskType="parameterFitting" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Object=Description"/>
        <Object cn="String=\[Function Evaluations\]"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="String=\[Best Value\]"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="String=\[Best Parameters\]"/>
      </Header>
      <Body>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Problem=Parameter Estimation,Reference=Function Evaluations"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Problem=Parameter Estimation,Reference=Best Value"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Problem=Parameter Estimation,Reference=Best Parameters"/>
      </Body>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_15" name="Metabolic Control Analysis" taskType="metabolicControlAnalysis" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Metabolic Control Analysis],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Metabolic Control Analysis],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_16" name="Lyapunov Exponents" taskType="lyapunovExponents" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Lyapunov Exponents],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Lyapunov Exponents],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_17" name="Time Scale Separation Analysis" taskType="timeScaleSeparationAnalysis" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Time Scale Separation Analysis],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Time Scale Separation Analysis],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_18" name="Sensitivities" taskType="sensitivities" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Sensitivities],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Sensitivities],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_19" name="Moieties" taskType="moieties" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Moieties],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Moieties],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_20" name="Linear Noise Approximation" taskType="linearNoiseApproximation" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Linear Noise Approximation],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Linear Noise Approximation],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_21" name="Time-Course" taskType="timeCourse" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Time-Course],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="CN=Root,Vector=TaskList[Time-Course],Object=Result"/>
      </Footer>
    </Report>
  </ListOfReports>
  <ListOfPlots>
    <PlotSpecification name="Concentrations, Volumes, and Global Quantity Values" type="Plot2D" active="1" taskTypes="">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="[Infected]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy],Vector=Metabolites[Infected],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[Diagnosed]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy],Vector=Metabolites[Diagnosed],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[Ailing]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy],Vector=Metabolites[Ailing],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[Recognized]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy],Vector=Metabolites[Recognized],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[Threatened]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Giordano2020 - SIDARTHE model of COVID-19 spread in Italy,Vector=Compartments[Italy],Vector=Metabolites[Threatened],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
  </ListOfPlots>
  <GUI>
  </GUI>
  <ListOfUnitDefinitions>
    <UnitDefinition key="Unit_5" name="second" symbol="s">
      <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Unit_4">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-09-20T11:02:17Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        s
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_15" name="dimensionless" symbol="1">
      <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Unit_14">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-09-20T11:02:17Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        1
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_17" name="item" symbol="#">
      <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Unit_16">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-09-20T11:02:17Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        #
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_69" name="day" symbol="d">
      <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Unit_68">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-09-20T11:02:17Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        86400*s
      </Expression>
    </UnitDefinition>
  </ListOfUnitDefinitions>
</COPASI>
