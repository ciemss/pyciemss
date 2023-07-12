<?xml version="1.0" encoding="UTF-8"?>
<!-- generated with COPASI 4.29 (Build 228) (http://www.copasi.org) at 2020-08-20T07:55:25Z -->
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
    <Function key="Function_41" name="Rate Law for Susceptible_to_Exposed" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Function_41">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T18:36:53Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

      </MiriamAnnotation>
      <Expression>
        S*beta*(I + l_a*A + l*H)/N
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_306" name="S" order="0" role="substrate"/>
        <ParameterDescription key="FunctionParameter_299" name="beta" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_292" name="I" order="2" role="modifier"/>
        <ParameterDescription key="FunctionParameter_298" name="l_a" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_307" name="A" order="4" role="modifier"/>
        <ParameterDescription key="FunctionParameter_309" name="l" order="5" role="constant"/>
        <ParameterDescription key="FunctionParameter_311" name="H" order="6" role="modifier"/>
        <ParameterDescription key="FunctionParameter_313" name="N" order="7" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_43" name="Rate Law for Cumulative_cases" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Function_43">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T18:41:10Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

      </MiriamAnnotation>
      <Expression>
        constant*E
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_310" name="constant" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_305" name="E" order="1" role="modifier"/>
      </ListOfParameterDescriptions>
    </Function>
  </ListOfFunctions>
  <Model key="Model_1" name="Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19" simulationType="time" timeUnit="d" volumeUnit="1" areaUnit="1" lengthUnit="1" quantityUnit="#" type="deterministic" avogadroConstant="6.0221408570000002e+23">
    <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:vCard="http://www.w3.org/2001/vcard-rdf/3.0#">
  <rdf:Description rdf:about="#Model_1">
    <bqbiol:hasProperty rdf:resource="urn:miriam:mamo:MAMO_0000028"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:mamo:MAMO_0000046"/>
    <bqbiol:hasTaxon rdf:resource="urn:miriam:taxonomy:2697049"/>
    <bqbiol:hasTaxon rdf:resource="urn:miriam:taxonomy:9606"/>
    <dcterms:bibliographicCitation>
      <rdf:Description>
        <CopasiMT:isDescribedBy rdf:resource="urn:miriam:pubmed:32735581"/>
      </rdf:Description>
    </dcterms:bibliographicCitation>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:24:29Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
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
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:doid:DOID:0080600"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000503"/>
  </rdf:Description>
</rdf:RDF>

    </MiriamAnnotation>
    <Comment>
      This paper proposes a dynamic model to describe and forecast the dynamics of the coronavirus disease COVID-19 transmission. The model is based on an approach previously used to describe the Middle East Respiratory Syndrome (MERS) epidemic. This methodology is used to describe the COVID-19 dynamics in six countries where the pandemic is widely spread, namely China, Italy, Spain, France, Germany, and the USA. For this purpose, data from the European Centre for Disease Prevention and Control (ECDC) are adopted. It is shown how the model can be used to forecast new infection cases and new deceased and how the uncertainties associated to this prediction can be quantified. This approach has the advantage of being relatively simple, grouping in few mathematical parameters the many conditions which affect the spreading of the disease. On the other hand, it requires previous data from the disease transmission in the country, being better suited for regions where the epidemic is not at a very early stage. With the estimated parameters at hand, one can use the model to predict the evolution of the disease, which in turn enables authorities to plan their actions. Moreover, one key advantage is the straightforward interpretation of these parameters and their influence over the evolution of the disease, which enables altering some of them, so that one can evaluate the effect of public policy, such as social distancing. The results presented for the selected countries confirm the accuracy to perform predictions.
    </Comment>
    <ListOfCompartments>
      <Compartment key="Compartment_0" name="Country" simulationType="fixed" dimensionality="3" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Compartment_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:24:40Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C16428"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C16592"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C16636"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C16761"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C17152"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C17234"/>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
      </Compartment>
    </ListOfCompartments>
    <ListOfMetabolites>
      <Metabolite key="Metabolite_0" name="Susceptible" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:24:55Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000514"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133"/>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <InitialExpression>
          0.9*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Total_pop],Reference=InitialValue>
        </InitialExpression>
      </Metabolite>
      <Metabolite key="Metabolite_1" name="Exposed" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_1">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:24:56Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000514"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000597"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133"/>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <InitialExpression>
          50*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country],Vector=Metabolites[Infectious],Reference=InitialConcentration>
        </InitialExpression>
      </Metabolite>
      <Metabolite key="Metabolite_2" name="Infectious" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_2">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:24:59Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133"/>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <InitialExpression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Initial_infected_pop],Reference=InitialValue>
        </InitialExpression>
      </Metabolite>
      <Metabolite key="Metabolite_3" name="Asymptomatic" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_3">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:25:00Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000569"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133"/>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <InitialExpression>
          3*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country],Vector=Metabolites[Infectious],Reference=InitialConcentration>
        </InitialExpression>
      </Metabolite>
      <Metabolite key="Metabolite_4" name="Hospitalized" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_4">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:25:00Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C168447"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C25179"/>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_5" name="Recovered" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_5">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:25:01Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000621"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133"/>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_6" name="Deceased" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_6">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:25:03Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C28554"/>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_8" name="Cumulative_Cases" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_8">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:25:29Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000480"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133"/>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
      </Metabolite>
    </ListOfMetabolites>
    <ListOfModelValues>
      <ModelValue key="ModelValue_0" name="1_Trigger_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_0">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T13:29:54Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_1" name="2_Trigger_Italy" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_1">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T13:30:07Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_2" name="3_Trigger_Spain" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_2">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T13:30:13Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_3" name="4_Trigger_France" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_3">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T13:30:18Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_4" name="5_Trigger_Germany" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_4">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T13:30:24Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_5" name="6_Trigger_USA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_5">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T13:30:54Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_6" name="beta_1" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_6">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:31:32Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[1_Trigger_China],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1_China],Reference=Value>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[2_Trigger_Italy],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1_Italy],Reference=Value>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[3_Trigger_Spain],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1_Spain],Reference=Value>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[4_Trigger_France],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1_France],Reference=Value>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[5_Trigger_Germany],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1_Germany],Reference=Value>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[6_Trigger_USA],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1_USA],Reference=Value>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_8" name="l_1" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_8">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:31:42Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[1_Trigger_China],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1_China],Reference=Value>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[2_Trigger_Italy],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1_Italy],Reference=Value>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[3_Trigger_Spain],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1_Spain],Reference=Value>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[4_Trigger_France],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1_France],Reference=Value>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[5_Trigger_Germany],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1_Germany],Reference=Value>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[6_Trigger_USA],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1_USA],Reference=Value>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_10" name="l_a_1" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_10">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:31:47Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[1_Trigger_China],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1_China],Reference=Value>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[2_Trigger_Italy],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1_Italy],Reference=Value>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[3_Trigger_Spain],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1_Spain],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[4_Trigger_France],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1_France],Reference=Value>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[5_Trigger_Germany],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1_Germany],Reference=Value>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[6_Trigger_USA],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1_USA],Reference=Value>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_12" name="kappa" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_12">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:31:54Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[1_Trigger_China],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa_China],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[2_Trigger_Italy],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa_Italy],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[3_Trigger_Spain],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa_Spain],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[4_Trigger_France],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa_France],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[5_Trigger_Germany],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa_Germany],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[6_Trigger_USA],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa_USA],Reference=InitialValue>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_13" name="rho" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_13">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:31:59Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[1_Trigger_China],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[rho_China],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[2_Trigger_Italy],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[rho_Italy],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[3_Trigger_Spain],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[rho_Spain],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[4_Trigger_France],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[rho_France],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[5_Trigger_Germany],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[rho_Germany],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[6_Trigger_USA],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[rho_USA],Reference=InitialValue>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_14" name="gamma_a" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_14">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:32:03Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[1_Trigger_China],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_a_China],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[2_Trigger_Italy],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_a_Italy],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[3_Trigger_Spain],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_a_Spain],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[4_Trigger_France],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_a_France],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[5_Trigger_Germany],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_a_Germany],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[6_Trigger_USA],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_a_USA],Reference=InitialValue>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_15" name="gamma_i" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_15">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:32:08Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[1_Trigger_China],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_i_China],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[2_Trigger_Italy],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_i_Italy],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[3_Trigger_Spain],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_i_Spain],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[4_Trigger_France],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_i_France],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[5_Trigger_Germany],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_i_Germany],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[6_Trigger_USA],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_i_USA],Reference=InitialValue>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_16" name="gamma_r" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_16">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:32:13Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[1_Trigger_China],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_r_China],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[2_Trigger_Italy],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_r_Italy],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[3_Trigger_Spain],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_r_Spain],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[4_Trigger_France],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_r_France],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[5_Trigger_Germany],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_r_Germany],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[6_Trigger_USA],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_r_USA],Reference=InitialValue>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_17" name="mu" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_17">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:32:15Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[1_Trigger_China],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[mu_China],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[2_Trigger_Italy],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[mu_Italy],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[3_Trigger_Spain],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[mu_Spain],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[4_Trigger_France],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[mu_France],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[5_Trigger_Germany],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[mu_Germany],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[6_Trigger_USA],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[mu_USA],Reference=InitialValue>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_18" name="delta_A" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_18">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:32:19Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[1_Trigger_China],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_A_China],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[2_Trigger_Italy],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_A_Italy],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[3_Trigger_Spain],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_A_Spain],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[4_Trigger_France],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_A_Spain],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[4_Trigger_France],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_A_France],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[5_Trigger_Germany],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_A_Germany],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[6_Trigger_USA],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_A_USA],Reference=InitialValue>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_19" name="delta_H" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_19">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:32:24Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[1_Trigger_China],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_H_China],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[2_Trigger_Italy],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_H_Italy],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[3_Trigger_Spain],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_H_Spain],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[4_Trigger_France],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_H_France],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[5_Trigger_Germany],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_H_Germany],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[6_Trigger_USA],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_H_USA],Reference=InitialValue>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_20" name="delta_I" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_20">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:32:29Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[1_Trigger_China],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_I_China],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[2_Trigger_Italy],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_I_Italy],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[3_Trigger_Spain],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_I_Spain],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[4_Trigger_France],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_I_France],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[5_Trigger_Germany],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_I_Germany],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[6_Trigger_USA],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_I_USA],Reference=InitialValue>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_21" name="Initial_infected_pop" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_21">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:33:21Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[1_Trigger_China],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Initial_infected_pop_China],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[2_Trigger_Italy],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Initial_infected_pop_Italy],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[3_Trigger_Spain],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Initial_infected_pop_Spain],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[4_Trigger_France],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Initial_infected_pop_France],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[5_Trigger_Germany],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Initial_infected_pop_Germany],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[6_Trigger_USA],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Initial_infected_pop_USA],Reference=InitialValue>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_22" name="kappa(rho)" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_22">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:33:58Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[rho],Reference=InitialValue>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_23" name="kappa(1-rho)" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_23">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:34:04Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa],Reference=InitialValue>*(1-&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[rho],Reference=InitialValue>)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_24" name="mu(delta_A)" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_24">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:34:35Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[mu],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_A],Reference=InitialValue>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_25" name="mu(1-delta_A)" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_25">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:34:40Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[mu],Reference=InitialValue>*(1-&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_A],Reference=InitialValue>)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_26" name="Time_threshold" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_26">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:34:46Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[1_Trigger_China],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Time_threshold_China],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[2_Trigger_Italy],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Time_threshold_Italy],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[3_Trigger_Spain],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Time_threshold_Spain],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[4_Trigger_France],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Time_threshold_France],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[5_Trigger_Germany],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Time_threshold_Germany],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[6_Trigger_USA],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Time_threshold_USA],Reference=InitialValue>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_38" name="beta_1_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_38">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:25:05Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_37" name="beta_2_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_37">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:25:12Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_36" name="l_1_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_36">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:25:15Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_35" name="l_2_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_35">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:25:30Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_34" name="l_a_1_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_34">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:25:33Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_33" name="l_a_2_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_33">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:25:54Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_32" name="kappa_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_32">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:25:57Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_31" name="rho_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_31">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:25:59Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_30" name="gamma_a_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_30">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:26:04Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_29" name="gamma_i_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_29">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:26:09Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_28" name="gamma_r_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_28">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:26:13Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_27" name="mu_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_27">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:26:15Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_39" name="delta_A_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_39">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:26:20Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_40" name="delta_H_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_40">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:26:32Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_41" name="delta_I_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_41">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:26:41Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_42" name="Initial_infected_pop_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_42">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:26:53Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_43" name="Time_threshold_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_43">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:27:02Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_44" name="beta_1_Italy" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_44">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:27:20Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_45" name="beta_2_Italy" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_45">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:27:52Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_46" name="l_1_Italy" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_46">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:27:54Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_47" name="l_2_Italy" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_47">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:27:57Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_48" name="l_a_1_Italy" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_48">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:28:02Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_49" name="l_a_2_Italy" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_49">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:28:09Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_50" name="kappa_Italy" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_50">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:28:14Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_51" name="rho_Italy" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_51">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:28:20Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_52" name="gamma_a_Italy" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_52">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:28:23Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_53" name="gamma_i_Italy" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_53">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:28:33Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_54" name="mu_Italy" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_54">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:28:50Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_55" name="delta_A_Italy" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_55">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:28:54Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_56" name="delta_H_Italy" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_56">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:29:07Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_57" name="delta_I_Italy" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_57">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:29:16Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_58" name="Initial_infected_pop_Italy" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_58">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:29:35Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_59" name="Time_threshold_Italy" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_59">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:29:42Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_60" name="beta_1_Spain" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_60">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:30:01Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_61" name="beta_2_Spain" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_61">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:30:06Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_62" name="l_1_Spain" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_62">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:30:13Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_63" name="l_2_Spain" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_63">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:30:15Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_64" name="l_a_1_Spain" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_64">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:30:22Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_65" name="l_a_2_Spain" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_65">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:30:27Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_66" name="kappa_Spain" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_66">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:30:30Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_67" name="rho_Spain" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_67">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:30:31Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_68" name="gamma_a_Spain" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_68">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:30:36Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_69" name="gamma_i_Spain" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_69">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:30:40Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_70" name="gamma_r_Spain" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_70">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:30:43Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_71" name="mu_Spain" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_71">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:30:48Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_72" name="delta_A_Spain" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_72">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:30:51Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_73" name="delta_H_Spain" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_73">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:31:01Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_74" name="delta_I_Spain" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_74">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:31:06Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_75" name="Initial_infected_pop_Spain" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_75">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:31:17Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_76" name="Time_threshold_Spain" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_76">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:31:27Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_77" name="beta_1_France" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_77">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:32:08Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_78" name="beta_2_France" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_78">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:32:16Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_79" name="l_1_France" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_79">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:32:20Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_80" name="l_2_France" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_80">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:32:23Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_81" name="l_a_1_France" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_81">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:32:27Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_82" name="l_a_2_France" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_82">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:32:32Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_83" name="kappa_France" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_83">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:32:35Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_84" name="rho_France" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_84">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:32:38Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_85" name="gamma_a_France" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_85">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:32:42Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_86" name="gamma_i_France" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_86">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:32:49Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_87" name="gamma_r_France" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_87">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:32:52Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_88" name="mu_France" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_88">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:33:00Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_89" name="delta_A_France" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_89">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:33:05Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_90" name="delta_H_France" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_90">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:33:08Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_91" name="delta_I_France" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_91">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:33:13Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_92" name="Initial_infected_pop_France" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_92">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:33:20Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_93" name="Time_threshold_France" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_93">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:33:33Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_94" name="beta_1_Germany" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_94">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:34:10Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_95" name="beta_2_Germany" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_95">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:34:15Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_96" name="l_1_Germany" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_96">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:34:18Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_97" name="l_2_Germany" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_97">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:34:20Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_98" name="l_a_1_Germany" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_98">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:34:25Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_99" name="l_a_2_Germany" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_99">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:34:30Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_100" name="kappa_Germany" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_100">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:34:36Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_101" name="rho_Germany" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_101">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:34:39Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_102" name="gamma_a_Germany" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_102">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:34:42Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_103" name="gamma_i_Germany" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_103">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:34:46Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_104" name="gamma_r_Germany" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_104">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:34:50Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_105" name="mu_Germany" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_105">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:34:52Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_106" name="delta_A_Germany" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_106">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:34:56Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_107" name="delta_H_Germany" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_107">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T17:34:59Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_108" name="delta_I_Germany" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_108">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:35:02Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_109" name="Initial_infected_pop_Germany" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_109">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:35:23Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_110" name="Time_threshold_Germany" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_110">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:35:33Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_111" name="beta_1_USA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_111">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:35:41Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_112" name="beta_2_USA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_112">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:35:45Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_113" name="l_1_USA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_113">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:35:50Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_114" name="l_2_USA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_114">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:35:54Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_115" name="l_a_1_USA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_115">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:36:00Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_116" name="l_a_2_USA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_116">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:36:09Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_117" name="kappa_USA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_117">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:36:10Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_118" name="rho_USA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_118">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:36:13Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_119" name="gamma_a_USA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_119">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:36:20Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_120" name="gamma_i_USA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_120">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:36:24Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_121" name="gamma_r_USA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_121">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:36:30Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_122" name="mu_USA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_122">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:36:34Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_123" name="delta_A_USA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_123">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:36:37Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_124" name="delta_H_USA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_124">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:36:42Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_125" name="delta_I_USA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_125">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:36:47Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_126" name="Initial_infected_pop_USA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_126">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:36:58Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_127" name="Time_threshold_USA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_127">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T17:37:08Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_128" name="gamma_r_Italy" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_128">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T18:00:11Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_129" name="Total_pop" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_129">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T18:16:23Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[1_Trigger_China],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Total_pop_China],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[2_Trigger_Italy],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Total_pop_Italy],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[3_Trigger_Spain],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Total_pop_Spain],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[4_Trigger_France],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Total_pop_France],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[5_Trigger_Germany],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Total_pop_Germany],Reference=InitialValue>+&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[6_Trigger_USA],Reference=InitialValue>*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Total_pop_USA],Reference=InitialValue>
        </Expression>
        <Unit>
          #
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_130" name="Total_pop_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_130">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T18:16:29Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Unit>
          #
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_131" name="Total_pop_Italy" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_131">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T18:16:35Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Unit>
          #
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_132" name="Total_pop_Spain" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_132">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T18:16:48Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Unit>
          #
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_133" name="Total_pop_France" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_133">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T18:16:54Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Unit>
          #
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_134" name="Total_pop_Germany" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_134">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T18:16:58Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Unit>
          #
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_135" name="Total_pop_USA" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_135">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T18:17:02Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Unit>
          #
        </Unit>
      </ModelValue>
    </ListOfModelValues>
    <ListOfReactions>
      <Reaction key="Reaction_0" name="Susceptible_to_Exposed" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:27:39Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000514"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000597"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133"/>
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
          <Constant key="Parameter_5404" name="l_a" value="0.1"/>
          <Constant key="Parameter_5412" name="beta" value="1"/>
          <Constant key="Parameter_8248" name="l" value="0.1"/>
          <Constant key="Parameter_4999" name="N" value="0.1"/>
        </ListOfConstants>
        <KineticLaw function="Function_41" unitType="Default" scalingCompartment="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_306">
              <SourceParameter reference="Metabolite_0"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_299">
              <SourceParameter reference="ModelValue_6"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_292">
              <SourceParameter reference="Metabolite_2"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_298">
              <SourceParameter reference="ModelValue_10"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_307">
              <SourceParameter reference="Metabolite_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_309">
              <SourceParameter reference="ModelValue_8"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_311">
              <SourceParameter reference="Metabolite_4"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_313">
              <SourceParameter reference="ModelValue_129"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_1" name="Exposed_to_Infectious" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_1">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:27:43Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C128320"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C168447"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133"/>
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
          <Constant key="Parameter_5403" name="k1" value="0.1"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_22"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_2" name="Exposed_to_Asymptomatic" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_2">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:27:43Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C128320"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133"/>
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
          <Constant key="Parameter_5410" name="k1" value="0.1"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_23"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_3" name="Infectious_to_Hospitalized" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_3">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:27:44Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C25179"/>
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
          <Constant key="Parameter_5405" name="k1" value="0.1"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_14"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_2"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_4" name="Infectious_to_Recovered" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_4">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:27:51Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C25746"/>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_2" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_5" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5407" name="k1" value="0.1"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_15"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_2"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_5" name="Infectious_to_Deceased" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_5">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:27:52Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C28554"/>
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
          <Constant key="Parameter_5409" name="k1" value="0.1"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_20"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_2"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_6" name="Asymptomatic_to_Recovered" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_6">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:27:53Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C25746"/>
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
          <Constant key="Parameter_5413" name="k1" value="0.1"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_25"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_3"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_7" name="Asymptomatic_to_Deceased" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_7">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:27:53Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C28554"/>
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
          <Constant key="Parameter_5415" name="k1" value="0.1"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_24"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_3"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_8" name="Hospitalized_to_Deceased" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_8">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:27:55Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C28554"/>
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
          <Constant key="Parameter_5416" name="k1" value="0.1"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_19"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_4"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_9" name="Hospitalized_to_Recovered" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_9">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:27:57Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C25746"/>
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
          <Constant key="Parameter_5408" name="k1" value="0.1"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_16"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_4"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_10" name="Cumulative cases" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_10">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T13:27:57Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000480"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133"/>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_8" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_1" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_8564" name="constant" value="1"/>
        </ListOfConstants>
        <KineticLaw function="Function_43" unitType="Default" scalingCompartment="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_310">
              <SourceParameter reference="ModelValue_22"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_305">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
    </ListOfReactions>
    <ListOfEvents>
      <Event key="Event_0" name="event" delayAssignment="true" fireAtInitialTime="0" persistentTrigger="0">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-19T18:20:38Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Reference=Time> >= &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Time_threshold],Reference=InitialValue>
        </TriggerExpression>
        <DelayExpression>
          0
        </DelayExpression>
        <ListOfAssignments>
          <Assignment target="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1_China]" targetKey="ModelValue_38">
            <Expression>
              &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_2_China],Reference=Value>
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1_France]" targetKey="ModelValue_77">
            <Expression>
              &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_2_France],Reference=Value>
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1_Germany]" targetKey="ModelValue_94">
            <Expression>
              &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_2_Germany],Reference=Value>
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1_Italy]" targetKey="ModelValue_44">
            <Expression>
              &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_2_Italy],Reference=Value>
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1_Spain]" targetKey="ModelValue_60">
            <Expression>
              &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_2_Spain],Reference=Value>
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1_USA]" targetKey="ModelValue_111">
            <Expression>
              &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_2_USA],Reference=Value>
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1_China]" targetKey="ModelValue_36">
            <Expression>
              &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_2_China],Reference=Value>
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1_France]" targetKey="ModelValue_79">
            <Expression>
              &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_2_France],Reference=Value>
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1_Germany]" targetKey="ModelValue_96">
            <Expression>
              &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_2_Germany],Reference=Value>
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1_Italy]" targetKey="ModelValue_46">
            <Expression>
              &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_2_Italy],Reference=Value>
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1_Spain]" targetKey="ModelValue_62">
            <Expression>
              &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_2_Spain],Reference=Value>
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1_USA]" targetKey="ModelValue_113">
            <Expression>
              &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_2_USA],Reference=Value>
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1_China]" targetKey="ModelValue_34">
            <Expression>
              &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_2_China],Reference=Value>
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1_France]" targetKey="ModelValue_81">
            <Expression>
              &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_2_France],Reference=Value>
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1_Germany]" targetKey="ModelValue_98">
            <Expression>
              &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_2_Germany],Reference=Value>
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1_Italy]" targetKey="ModelValue_48">
            <Expression>
              &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_2_Italy],Reference=Value>
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1_Spain]" targetKey="ModelValue_64">
            <Expression>
              &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_2_Spain],Reference=Value>
            </Expression>
          </Assignment>
          <Assignment target="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1_USA]" targetKey="ModelValue_115">
            <Expression>
              &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_2_USA],Reference=Value>
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
<dcterms:W3CDTF>2020-08-19T13:24:29Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ModelParameterGroup cn="String=Initial Time" type="Group">
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19" value="0" type="Model" simulationType="time"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Compartment Sizes" type="Group">
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country]" value="1" type="Compartment" simulationType="fixed"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Species Values" type="Group">
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country],Vector=Metabolites[Susceptible]" value="9900000" type="Species" simulationType="reactions">
            <InitialExpression>
              0.9*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Total_pop],Reference=InitialValue>
            </InitialExpression>
          </ModelParameter>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country],Vector=Metabolites[Exposed]" value="36600" type="Species" simulationType="reactions">
            <InitialExpression>
              50*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country],Vector=Metabolites[Infectious],Reference=InitialConcentration>
            </InitialExpression>
          </ModelParameter>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country],Vector=Metabolites[Infectious]" value="732" type="Species" simulationType="reactions">
            <InitialExpression>
              &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Initial_infected_pop],Reference=InitialValue>
            </InitialExpression>
          </ModelParameter>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country],Vector=Metabolites[Asymptomatic]" value="2196" type="Species" simulationType="reactions">
            <InitialExpression>
              3*&lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country],Vector=Metabolites[Infectious],Reference=InitialConcentration>
            </InitialExpression>
          </ModelParameter>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country],Vector=Metabolites[Hospitalized]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country],Vector=Metabolites[Recovered]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country],Vector=Metabolites[Deceased]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country],Vector=Metabolites[Cumulative_Cases]" value="0" type="Species" simulationType="reactions"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Global Quantities" type="Group">
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[1_Trigger_China]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[2_Trigger_Italy]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[3_Trigger_Spain]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[4_Trigger_France]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[5_Trigger_Germany]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[6_Trigger_USA]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1]" value="0.33400000000000002" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1]" value="0.67300000000000004" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1]" value="8" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa]" value="0.44" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[rho]" value="0.052999999999999999" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_a]" value="0.503" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_i]" value="0.26300000000000001" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_r]" value="0.14099999999999999" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[mu]" value="1.6399999999999999" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_A]" value="0" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_H]" value="0.0080000000000000002" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_I]" value="0.0030000000000000001" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Initial_infected_pop]" value="732" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa(rho)]" value="0.02332" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa(1-rho)]" value="0.41667999999999999" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[mu(delta_A)]" value="0" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[mu(1-delta_A)]" value="1.6399999999999999" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Time_threshold]" value="18" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1_China]" value="0.33400000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_2_China]" value="0.14000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1_China]" value="0.67300000000000004" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_2_China]" value="0.13500000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1_China]" value="8" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_2_China]" value="8" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa_China]" value="0.44" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[rho_China]" value="0.052999999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_a_China]" value="0.503" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_i_China]" value="0.26300000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_r_China]" value="0.14099999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[mu_China]" value="1.6399999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_A_China]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_H_China]" value="0.0080000000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_I_China]" value="0.0030000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Initial_infected_pop_China]" value="732" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Time_threshold_China]" value="18" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1_Italy]" value="0.189" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_2_Italy]" value="0.081000000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1_Italy]" value="8" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_2_Italy]" value="8" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1_Italy]" value="0.64900000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_2_Italy]" value="0.64900000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa_Italy]" value="0.28399999999999997" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[rho_Italy]" value="0.27000000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_a_Italy]" value="0.224" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_i_Italy]" value="0.040000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[mu_Italy]" value="0.14599999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_A_Italy]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_H_Italy]" value="0.023" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_I_Italy]" value="0.023" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Initial_infected_pop_Italy]" value="648" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Time_threshold_Italy]" value="30" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1_Spain]" value="0.38200000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_2_Spain]" value="0.16" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1_Spain]" value="7.6900000000000004" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_2_Spain]" value="6.4900000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1_Spain]" value="3.8999999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_2_Spain]" value="3.8999999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa_Spain]" value="0.36199999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[rho_Spain]" value="0.10199999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_a_Spain]" value="0.11600000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_i_Spain]" value="0.063" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_r_Spain]" value="0.28100000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[mu_Spain]" value="1.03" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_A_Spain]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_H_Spain]" value="0.019" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_I_Spain]" value="0.016" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Initial_infected_pop_Spain]" value="500" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Time_threshold_Spain]" value="23" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1_France]" value="0.29799999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_2_France]" value="0.129" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1_France]" value="8" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_2_France]" value="8" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1_France]" value="8" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_2_France]" value="8" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa_France]" value="0.309" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[rho_France]" value="0.033000000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_a_France]" value="0.29999999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_i_France]" value="0.02" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_r_France]" value="0.13100000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[mu_France]" value="1.53" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_A_France]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_H_France]" value="0.029000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_I_France]" value="0.017999999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Initial_infected_pop_France]" value="575" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Time_threshold_France]" value="26" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1_Germany]" value="0.13500000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_2_Germany]" value="0.055" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1_Germany]" value="4.7999999999999998" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_2_Germany]" value="1.1299999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1_Germany]" value="4.9000000000000004" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_2_Germany]" value="4.9000000000000004" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa_Germany]" value="0.57799999999999996" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[rho_Germany]" value="0.021000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_a_Germany]" value="0.54200000000000004" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_i_Germany]" value="0.050000000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_r_Germany]" value="0.035999999999999997" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[mu_Germany]" value="0.30199999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_A_Germany]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_H_Germany]" value="0.0030000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_I_Germany]" value="0.002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Initial_infected_pop_Germany]" value="735" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Time_threshold_Germany]" value="24" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1_USA]" value="0.30299999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_2_USA]" value="0.13" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1_USA]" value="0.85099999999999998" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_2_USA]" value="0.85099999999999998" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1_USA]" value="4.0899999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_2_USA]" value="0.81999999999999995" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa_USA]" value="1.3300000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[rho_USA]" value="1.01" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_a_USA]" value="0.055" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_i_USA]" value="0.29599999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_r_USA]" value="0.017999999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[mu_USA]" value="0.82799999999999996" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_A_USA]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_H_USA]" value="0.00029" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_I_USA]" value="0.023" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Initial_infected_pop_USA]" value="576" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Time_threshold_USA]" value="51" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_r_Italy]" value="0.23999999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Total_pop]" value="11000000" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Total_pop_China]" value="11000000" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Total_pop_Italy]" value="60400000" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Total_pop_Spain]" value="46900000" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Total_pop_France]" value="67000000" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Total_pop_Germany]" value="83000000" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Total_pop_USA]" value="328200000" type="ModelValue" simulationType="fixed"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Kinetic Parameters" type="Group">
          <ModelParameterGroup cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Susceptible_to_Exposed]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Susceptible_to_Exposed],ParameterGroup=Parameters,Parameter=l_a" value="8" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_a_1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Susceptible_to_Exposed],ParameterGroup=Parameters,Parameter=beta" value="0.33400000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[beta_1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Susceptible_to_Exposed],ParameterGroup=Parameters,Parameter=l" value="0.67300000000000004" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[l_1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Susceptible_to_Exposed],ParameterGroup=Parameters,Parameter=N" value="11000000" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[Total_pop],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Exposed_to_Infectious]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Exposed_to_Infectious],ParameterGroup=Parameters,Parameter=k1" value="0.02332" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa(rho)],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Exposed_to_Asymptomatic]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Exposed_to_Asymptomatic],ParameterGroup=Parameters,Parameter=k1" value="0.41667999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa(1-rho)],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Infectious_to_Hospitalized]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Infectious_to_Hospitalized],ParameterGroup=Parameters,Parameter=k1" value="0.503" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_a],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Infectious_to_Recovered]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Infectious_to_Recovered],ParameterGroup=Parameters,Parameter=k1" value="0.26300000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_i],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Infectious_to_Deceased]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Infectious_to_Deceased],ParameterGroup=Parameters,Parameter=k1" value="0.0030000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_I],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Asymptomatic_to_Recovered]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Asymptomatic_to_Recovered],ParameterGroup=Parameters,Parameter=k1" value="1.6399999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[mu(1-delta_A)],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Asymptomatic_to_Deceased]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Asymptomatic_to_Deceased],ParameterGroup=Parameters,Parameter=k1" value="0" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[mu(delta_A)],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Hospitalized_to_Deceased]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Hospitalized_to_Deceased],ParameterGroup=Parameters,Parameter=k1" value="0.0080000000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[delta_H],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Hospitalized_to_Recovered]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Hospitalized_to_Recovered],ParameterGroup=Parameters,Parameter=k1" value="0.14099999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[gamma_r],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Cumulative cases]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Reactions[Cumulative cases],ParameterGroup=Parameters,Parameter=constant" value="0.02332" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Values[kappa(rho)],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
        </ModelParameterGroup>
      </ModelParameterSet>
    </ListOfModelParameterSets>
    <StateTemplate>
      <StateTemplateVariable objectReference="Model_1"/>
      <StateTemplateVariable objectReference="Metabolite_2"/>
      <StateTemplateVariable objectReference="Metabolite_3"/>
      <StateTemplateVariable objectReference="Metabolite_4"/>
      <StateTemplateVariable objectReference="Metabolite_1"/>
      <StateTemplateVariable objectReference="Metabolite_5"/>
      <StateTemplateVariable objectReference="Metabolite_8"/>
      <StateTemplateVariable objectReference="Metabolite_0"/>
      <StateTemplateVariable objectReference="Metabolite_6"/>
      <StateTemplateVariable objectReference="ModelValue_6"/>
      <StateTemplateVariable objectReference="ModelValue_8"/>
      <StateTemplateVariable objectReference="ModelValue_10"/>
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
      <StateTemplateVariable objectReference="ModelValue_22"/>
      <StateTemplateVariable objectReference="ModelValue_23"/>
      <StateTemplateVariable objectReference="ModelValue_24"/>
      <StateTemplateVariable objectReference="ModelValue_25"/>
      <StateTemplateVariable objectReference="ModelValue_26"/>
      <StateTemplateVariable objectReference="ModelValue_129"/>
      <StateTemplateVariable objectReference="Compartment_0"/>
      <StateTemplateVariable objectReference="ModelValue_0"/>
      <StateTemplateVariable objectReference="ModelValue_1"/>
      <StateTemplateVariable objectReference="ModelValue_2"/>
      <StateTemplateVariable objectReference="ModelValue_3"/>
      <StateTemplateVariable objectReference="ModelValue_4"/>
      <StateTemplateVariable objectReference="ModelValue_5"/>
      <StateTemplateVariable objectReference="ModelValue_38"/>
      <StateTemplateVariable objectReference="ModelValue_37"/>
      <StateTemplateVariable objectReference="ModelValue_36"/>
      <StateTemplateVariable objectReference="ModelValue_35"/>
      <StateTemplateVariable objectReference="ModelValue_34"/>
      <StateTemplateVariable objectReference="ModelValue_33"/>
      <StateTemplateVariable objectReference="ModelValue_32"/>
      <StateTemplateVariable objectReference="ModelValue_31"/>
      <StateTemplateVariable objectReference="ModelValue_30"/>
      <StateTemplateVariable objectReference="ModelValue_29"/>
      <StateTemplateVariable objectReference="ModelValue_28"/>
      <StateTemplateVariable objectReference="ModelValue_27"/>
      <StateTemplateVariable objectReference="ModelValue_39"/>
      <StateTemplateVariable objectReference="ModelValue_40"/>
      <StateTemplateVariable objectReference="ModelValue_41"/>
      <StateTemplateVariable objectReference="ModelValue_42"/>
      <StateTemplateVariable objectReference="ModelValue_43"/>
      <StateTemplateVariable objectReference="ModelValue_44"/>
      <StateTemplateVariable objectReference="ModelValue_45"/>
      <StateTemplateVariable objectReference="ModelValue_46"/>
      <StateTemplateVariable objectReference="ModelValue_47"/>
      <StateTemplateVariable objectReference="ModelValue_48"/>
      <StateTemplateVariable objectReference="ModelValue_49"/>
      <StateTemplateVariable objectReference="ModelValue_50"/>
      <StateTemplateVariable objectReference="ModelValue_51"/>
      <StateTemplateVariable objectReference="ModelValue_52"/>
      <StateTemplateVariable objectReference="ModelValue_53"/>
      <StateTemplateVariable objectReference="ModelValue_54"/>
      <StateTemplateVariable objectReference="ModelValue_55"/>
      <StateTemplateVariable objectReference="ModelValue_56"/>
      <StateTemplateVariable objectReference="ModelValue_57"/>
      <StateTemplateVariable objectReference="ModelValue_58"/>
      <StateTemplateVariable objectReference="ModelValue_59"/>
      <StateTemplateVariable objectReference="ModelValue_60"/>
      <StateTemplateVariable objectReference="ModelValue_61"/>
      <StateTemplateVariable objectReference="ModelValue_62"/>
      <StateTemplateVariable objectReference="ModelValue_63"/>
      <StateTemplateVariable objectReference="ModelValue_64"/>
      <StateTemplateVariable objectReference="ModelValue_65"/>
      <StateTemplateVariable objectReference="ModelValue_66"/>
      <StateTemplateVariable objectReference="ModelValue_67"/>
      <StateTemplateVariable objectReference="ModelValue_68"/>
      <StateTemplateVariable objectReference="ModelValue_69"/>
      <StateTemplateVariable objectReference="ModelValue_70"/>
      <StateTemplateVariable objectReference="ModelValue_71"/>
      <StateTemplateVariable objectReference="ModelValue_72"/>
      <StateTemplateVariable objectReference="ModelValue_73"/>
      <StateTemplateVariable objectReference="ModelValue_74"/>
      <StateTemplateVariable objectReference="ModelValue_75"/>
      <StateTemplateVariable objectReference="ModelValue_76"/>
      <StateTemplateVariable objectReference="ModelValue_77"/>
      <StateTemplateVariable objectReference="ModelValue_78"/>
      <StateTemplateVariable objectReference="ModelValue_79"/>
      <StateTemplateVariable objectReference="ModelValue_80"/>
      <StateTemplateVariable objectReference="ModelValue_81"/>
      <StateTemplateVariable objectReference="ModelValue_82"/>
      <StateTemplateVariable objectReference="ModelValue_83"/>
      <StateTemplateVariable objectReference="ModelValue_84"/>
      <StateTemplateVariable objectReference="ModelValue_85"/>
      <StateTemplateVariable objectReference="ModelValue_86"/>
      <StateTemplateVariable objectReference="ModelValue_87"/>
      <StateTemplateVariable objectReference="ModelValue_88"/>
      <StateTemplateVariable objectReference="ModelValue_89"/>
      <StateTemplateVariable objectReference="ModelValue_90"/>
      <StateTemplateVariable objectReference="ModelValue_91"/>
      <StateTemplateVariable objectReference="ModelValue_92"/>
      <StateTemplateVariable objectReference="ModelValue_93"/>
      <StateTemplateVariable objectReference="ModelValue_94"/>
      <StateTemplateVariable objectReference="ModelValue_95"/>
      <StateTemplateVariable objectReference="ModelValue_96"/>
      <StateTemplateVariable objectReference="ModelValue_97"/>
      <StateTemplateVariable objectReference="ModelValue_98"/>
      <StateTemplateVariable objectReference="ModelValue_99"/>
      <StateTemplateVariable objectReference="ModelValue_100"/>
      <StateTemplateVariable objectReference="ModelValue_101"/>
      <StateTemplateVariable objectReference="ModelValue_102"/>
      <StateTemplateVariable objectReference="ModelValue_103"/>
      <StateTemplateVariable objectReference="ModelValue_104"/>
      <StateTemplateVariable objectReference="ModelValue_105"/>
      <StateTemplateVariable objectReference="ModelValue_106"/>
      <StateTemplateVariable objectReference="ModelValue_107"/>
      <StateTemplateVariable objectReference="ModelValue_108"/>
      <StateTemplateVariable objectReference="ModelValue_109"/>
      <StateTemplateVariable objectReference="ModelValue_110"/>
      <StateTemplateVariable objectReference="ModelValue_111"/>
      <StateTemplateVariable objectReference="ModelValue_112"/>
      <StateTemplateVariable objectReference="ModelValue_113"/>
      <StateTemplateVariable objectReference="ModelValue_114"/>
      <StateTemplateVariable objectReference="ModelValue_115"/>
      <StateTemplateVariable objectReference="ModelValue_116"/>
      <StateTemplateVariable objectReference="ModelValue_117"/>
      <StateTemplateVariable objectReference="ModelValue_118"/>
      <StateTemplateVariable objectReference="ModelValue_119"/>
      <StateTemplateVariable objectReference="ModelValue_120"/>
      <StateTemplateVariable objectReference="ModelValue_121"/>
      <StateTemplateVariable objectReference="ModelValue_122"/>
      <StateTemplateVariable objectReference="ModelValue_123"/>
      <StateTemplateVariable objectReference="ModelValue_124"/>
      <StateTemplateVariable objectReference="ModelValue_125"/>
      <StateTemplateVariable objectReference="ModelValue_126"/>
      <StateTemplateVariable objectReference="ModelValue_127"/>
      <StateTemplateVariable objectReference="ModelValue_128"/>
      <StateTemplateVariable objectReference="ModelValue_130"/>
      <StateTemplateVariable objectReference="ModelValue_131"/>
      <StateTemplateVariable objectReference="ModelValue_132"/>
      <StateTemplateVariable objectReference="ModelValue_133"/>
      <StateTemplateVariable objectReference="ModelValue_134"/>
      <StateTemplateVariable objectReference="ModelValue_135"/>
    </StateTemplate>
    <InitialState type="initialState">
      0 732 2196 0 36600 0 0 9900000 0 0.33400000000000002 0.67300000000000004 8 0.44 0.052999999999999999 0.503 0.26300000000000001 0.14099999999999999 1.6399999999999999 0 0.0080000000000000002 0.0030000000000000001 732 0.02332 0.41667999999999999 0 1.6399999999999999 18 11000000 1 1 0 0 0 0 0 0.33400000000000002 0.14000000000000001 0.67300000000000004 0.13500000000000001 8 8 0.44 0.052999999999999999 0.503 0.26300000000000001 0.14099999999999999 1.6399999999999999 0 0.0080000000000000002 0.0030000000000000001 732 18 0.189 0.081000000000000003 8 8 0.64900000000000002 0.64900000000000002 0.28399999999999997 0.27000000000000002 0.224 0.040000000000000001 0.14599999999999999 0 0.023 0.023 648 30 0.38200000000000001 0.16 7.6900000000000004 6.4900000000000002 3.8999999999999999 3.8999999999999999 0.36199999999999999 0.10199999999999999 0.11600000000000001 0.063 0.28100000000000003 1.03 0 0.019 0.016 500 23 0.29799999999999999 0.129 8 8 8 8 0.309 0.033000000000000002 0.29999999999999999 0.02 0.13100000000000001 1.53 0 0.029000000000000001 0.017999999999999999 575 26 0.13500000000000001 0.055 4.7999999999999998 1.1299999999999999 4.9000000000000004 4.9000000000000004 0.57799999999999996 0.021000000000000001 0.54200000000000004 0.050000000000000003 0.035999999999999997 0.30199999999999999 0 0.0030000000000000001 0.002 735 24 0.30299999999999999 0.13 0.85099999999999998 0.85099999999999998 4.0899999999999999 0.81999999999999995 1.3300000000000001 1.01 0.055 0.29599999999999999 0.017999999999999999 0.82799999999999996 0 0.00029 0.023 576 51 0.23999999999999999 11000000 60400000 46900000 67000000 83000000 328200000 
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
      <Report reference="Report_12" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="AutomaticStepSize" type="bool" value="0"/>
        <Parameter name="StepNumber" type="unsignedInteger" value="1020"/>
        <Parameter name="StepSize" type="float" value="0.078431372550000003"/>
        <Parameter name="Duration" type="float" value="80"/>
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
      <Report reference="Report_13" target="" append="1" confirmOverwrite="1"/>
      <Problem>
      </Problem>
      <Method name="EFM Algorithm" type="EFMAlgorithm">
      </Method>
    </Task>
    <Task key="Task_19" name="Optimization" type="optimization" scheduled="false" updateModel="false">
      <Report reference="Report_14" target="" append="1" confirmOverwrite="1"/>
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
      <Report reference="Report_15" target="" append="1" confirmOverwrite="1"/>
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
      <Report reference="Report_16" target="" append="1" confirmOverwrite="1"/>
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
      <Report reference="Report_17" target="" append="1" confirmOverwrite="1"/>
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
      <Report reference="Report_18" target="" append="1" confirmOverwrite="1"/>
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
      <Report reference="Report_19" target="" append="1" confirmOverwrite="1"/>
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
      <Report reference="Report_20" target="" append="1" confirmOverwrite="1"/>
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
      <Report reference="Report_21" target="" append="1" confirmOverwrite="1"/>
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
    <Report key="Report_12" name="Time-Course" taskType="timeCourse" separator="&#x09;" precision="6">
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
    <Report key="Report_13" name="Elementary Flux Modes" taskType="fluxMode" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Footer>
        <Object cn="CN=Root,Vector=TaskList[Elementary Flux Modes],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_14" name="Optimization" taskType="optimization" separator="&#x09;" precision="6">
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
    <Report key="Report_15" name="Parameter Estimation" taskType="parameterFitting" separator="&#x09;" precision="6">
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
    <Report key="Report_16" name="Metabolic Control Analysis" taskType="metabolicControlAnalysis" separator="&#x09;" precision="6">
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
    <Report key="Report_17" name="Lyapunov Exponents" taskType="lyapunovExponents" separator="&#x09;" precision="6">
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
    <Report key="Report_18" name="Time Scale Separation Analysis" taskType="timeScaleSeparationAnalysis" separator="&#x09;" precision="6">
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
    <Report key="Report_19" name="Sensitivities" taskType="sensitivities" separator="&#x09;" precision="6">
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
    <Report key="Report_20" name="Moieties" taskType="moieties" separator="&#x09;" precision="6">
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
    <Report key="Report_21" name="Linear Noise Approximation" taskType="linearNoiseApproximation" separator="&#x09;" precision="6">
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
  </ListOfReports>
  <ListOfPlots>
    <PlotSpecification name="Concentrations, Volumes, and Global Quantity Values" type="Plot2D" active="1" taskTypes="">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="[D]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country],Vector=Metabolites[Deceased],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
    <PlotSpecification name="Concentrations, Volumes, and Global Quantity Values_1" type="Plot2D" active="1" taskTypes="">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="[Cumulative_Cases]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Paiva2020 - SEIAHRD model of transmission dynamics of COVID-19,Vector=Compartments[Country],Vector=Metabolites[Cumulative_Cases],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
  </ListOfPlots>
  <GUI>
  </GUI>
  <SBMLReference file="Submission files/Paiva2020.xml">
    <SBMLMap SBMLid="Asymptomatic" COPASIkey="Metabolite_3"/>
    <SBMLMap SBMLid="Asymptomatic_to_Deceased" COPASIkey="Reaction_7"/>
    <SBMLMap SBMLid="Asymptomatic_to_Recovered" COPASIkey="Reaction_6"/>
    <SBMLMap SBMLid="Country" COPASIkey="Compartment_0"/>
    <SBMLMap SBMLid="Cumulative_Cases" COPASIkey="Metabolite_8"/>
    <SBMLMap SBMLid="Cumulative_cases" COPASIkey="Reaction_10"/>
    <SBMLMap SBMLid="Deceased" COPASIkey="Metabolite_6"/>
    <SBMLMap SBMLid="Exposed" COPASIkey="Metabolite_1"/>
    <SBMLMap SBMLid="Exposed_to_Asymptomatic" COPASIkey="Reaction_2"/>
    <SBMLMap SBMLid="Exposed_to_Infectious" COPASIkey="Reaction_1"/>
    <SBMLMap SBMLid="Hospitalized" COPASIkey="Metabolite_4"/>
    <SBMLMap SBMLid="Hospitalized_to_Deceased" COPASIkey="Reaction_8"/>
    <SBMLMap SBMLid="Hospitalized_to_Recovered" COPASIkey="Reaction_9"/>
    <SBMLMap SBMLid="Infectious" COPASIkey="Metabolite_2"/>
    <SBMLMap SBMLid="Infectious_to_Deceased" COPASIkey="Reaction_5"/>
    <SBMLMap SBMLid="Infectious_to_Hospitalized" COPASIkey="Reaction_3"/>
    <SBMLMap SBMLid="Infectious_to_Recovered" COPASIkey="Reaction_4"/>
    <SBMLMap SBMLid="Initial_infected_pop" COPASIkey="ModelValue_21"/>
    <SBMLMap SBMLid="Initial_infected_pop_China" COPASIkey="ModelValue_42"/>
    <SBMLMap SBMLid="Initial_infected_pop_France" COPASIkey="ModelValue_92"/>
    <SBMLMap SBMLid="Initial_infected_pop_Germany" COPASIkey="ModelValue_109"/>
    <SBMLMap SBMLid="Initial_infected_pop_Italy" COPASIkey="ModelValue_58"/>
    <SBMLMap SBMLid="Initial_infected_pop_Spain" COPASIkey="ModelValue_75"/>
    <SBMLMap SBMLid="Initial_infected_pop_USA" COPASIkey="ModelValue_126"/>
    <SBMLMap SBMLid="Rate_Law_for_Cumulative_cases" COPASIkey="Function_43"/>
    <SBMLMap SBMLid="Rate_Law_for_Susceptible_to_Exposed" COPASIkey="Function_41"/>
    <SBMLMap SBMLid="Recovered" COPASIkey="Metabolite_5"/>
    <SBMLMap SBMLid="Susceptible" COPASIkey="Metabolite_0"/>
    <SBMLMap SBMLid="Susceptible_to_Exposed" COPASIkey="Reaction_0"/>
    <SBMLMap SBMLid="Time_threshold" COPASIkey="ModelValue_26"/>
    <SBMLMap SBMLid="Time_threshold_China" COPASIkey="ModelValue_43"/>
    <SBMLMap SBMLid="Time_threshold_France" COPASIkey="ModelValue_93"/>
    <SBMLMap SBMLid="Time_threshold_Germany" COPASIkey="ModelValue_110"/>
    <SBMLMap SBMLid="Time_threshold_Italy" COPASIkey="ModelValue_59"/>
    <SBMLMap SBMLid="Time_threshold_Spain" COPASIkey="ModelValue_76"/>
    <SBMLMap SBMLid="Time_threshold_USA" COPASIkey="ModelValue_127"/>
    <SBMLMap SBMLid="Total_pop" COPASIkey="ModelValue_129"/>
    <SBMLMap SBMLid="Total_pop_China" COPASIkey="ModelValue_130"/>
    <SBMLMap SBMLid="Total_pop_France" COPASIkey="ModelValue_133"/>
    <SBMLMap SBMLid="Total_pop_Germany" COPASIkey="ModelValue_134"/>
    <SBMLMap SBMLid="Total_pop_Italy" COPASIkey="ModelValue_131"/>
    <SBMLMap SBMLid="Total_pop_Spain" COPASIkey="ModelValue_132"/>
    <SBMLMap SBMLid="Total_pop_USA" COPASIkey="ModelValue_135"/>
    <SBMLMap SBMLid="_1_Trigger_China" COPASIkey="ModelValue_0"/>
    <SBMLMap SBMLid="_2_Trigger_Italy" COPASIkey="ModelValue_1"/>
    <SBMLMap SBMLid="_3_Trigger_Spain" COPASIkey="ModelValue_2"/>
    <SBMLMap SBMLid="_4_Trigger_France" COPASIkey="ModelValue_3"/>
    <SBMLMap SBMLid="_5_Trigger_Germany" COPASIkey="ModelValue_4"/>
    <SBMLMap SBMLid="_6_Trigger_USA" COPASIkey="ModelValue_5"/>
    <SBMLMap SBMLid="beta_1" COPASIkey="ModelValue_6"/>
    <SBMLMap SBMLid="beta_1_China" COPASIkey="ModelValue_38"/>
    <SBMLMap SBMLid="beta_1_France" COPASIkey="ModelValue_77"/>
    <SBMLMap SBMLid="beta_1_Germany" COPASIkey="ModelValue_94"/>
    <SBMLMap SBMLid="beta_1_Italy" COPASIkey="ModelValue_44"/>
    <SBMLMap SBMLid="beta_1_Spain" COPASIkey="ModelValue_60"/>
    <SBMLMap SBMLid="beta_1_USA" COPASIkey="ModelValue_111"/>
    <SBMLMap SBMLid="beta_2_China" COPASIkey="ModelValue_37"/>
    <SBMLMap SBMLid="beta_2_France" COPASIkey="ModelValue_78"/>
    <SBMLMap SBMLid="beta_2_Germany" COPASIkey="ModelValue_95"/>
    <SBMLMap SBMLid="beta_2_Italy" COPASIkey="ModelValue_45"/>
    <SBMLMap SBMLid="beta_2_Spain" COPASIkey="ModelValue_61"/>
    <SBMLMap SBMLid="beta_2_USA" COPASIkey="ModelValue_112"/>
    <SBMLMap SBMLid="delta_A" COPASIkey="ModelValue_18"/>
    <SBMLMap SBMLid="delta_A_China" COPASIkey="ModelValue_39"/>
    <SBMLMap SBMLid="delta_A_France" COPASIkey="ModelValue_89"/>
    <SBMLMap SBMLid="delta_A_Germany" COPASIkey="ModelValue_106"/>
    <SBMLMap SBMLid="delta_A_Italy" COPASIkey="ModelValue_55"/>
    <SBMLMap SBMLid="delta_A_Spain" COPASIkey="ModelValue_72"/>
    <SBMLMap SBMLid="delta_A_USA" COPASIkey="ModelValue_123"/>
    <SBMLMap SBMLid="delta_H" COPASIkey="ModelValue_19"/>
    <SBMLMap SBMLid="delta_H_China" COPASIkey="ModelValue_40"/>
    <SBMLMap SBMLid="delta_H_France" COPASIkey="ModelValue_90"/>
    <SBMLMap SBMLid="delta_H_Germany" COPASIkey="ModelValue_107"/>
    <SBMLMap SBMLid="delta_H_Italy" COPASIkey="ModelValue_56"/>
    <SBMLMap SBMLid="delta_H_Spain" COPASIkey="ModelValue_73"/>
    <SBMLMap SBMLid="delta_H_USA" COPASIkey="ModelValue_124"/>
    <SBMLMap SBMLid="delta_I" COPASIkey="ModelValue_20"/>
    <SBMLMap SBMLid="delta_I_China" COPASIkey="ModelValue_41"/>
    <SBMLMap SBMLid="delta_I_France" COPASIkey="ModelValue_91"/>
    <SBMLMap SBMLid="delta_I_Germany" COPASIkey="ModelValue_108"/>
    <SBMLMap SBMLid="delta_I_Italy" COPASIkey="ModelValue_57"/>
    <SBMLMap SBMLid="delta_I_Spain" COPASIkey="ModelValue_74"/>
    <SBMLMap SBMLid="delta_I_USA" COPASIkey="ModelValue_125"/>
    <SBMLMap SBMLid="event_0" COPASIkey="Event_0"/>
    <SBMLMap SBMLid="gamma_a" COPASIkey="ModelValue_14"/>
    <SBMLMap SBMLid="gamma_a_China" COPASIkey="ModelValue_30"/>
    <SBMLMap SBMLid="gamma_a_France" COPASIkey="ModelValue_85"/>
    <SBMLMap SBMLid="gamma_a_Germany" COPASIkey="ModelValue_102"/>
    <SBMLMap SBMLid="gamma_a_Italy" COPASIkey="ModelValue_52"/>
    <SBMLMap SBMLid="gamma_a_Spain" COPASIkey="ModelValue_68"/>
    <SBMLMap SBMLid="gamma_a_USA" COPASIkey="ModelValue_119"/>
    <SBMLMap SBMLid="gamma_i" COPASIkey="ModelValue_15"/>
    <SBMLMap SBMLid="gamma_i_China" COPASIkey="ModelValue_29"/>
    <SBMLMap SBMLid="gamma_i_France" COPASIkey="ModelValue_86"/>
    <SBMLMap SBMLid="gamma_i_Germany" COPASIkey="ModelValue_103"/>
    <SBMLMap SBMLid="gamma_i_Italy" COPASIkey="ModelValue_53"/>
    <SBMLMap SBMLid="gamma_i_Spain" COPASIkey="ModelValue_69"/>
    <SBMLMap SBMLid="gamma_i_USA" COPASIkey="ModelValue_120"/>
    <SBMLMap SBMLid="gamma_r" COPASIkey="ModelValue_16"/>
    <SBMLMap SBMLid="gamma_r_China" COPASIkey="ModelValue_28"/>
    <SBMLMap SBMLid="gamma_r_France" COPASIkey="ModelValue_87"/>
    <SBMLMap SBMLid="gamma_r_Germany" COPASIkey="ModelValue_104"/>
    <SBMLMap SBMLid="gamma_r_Italy" COPASIkey="ModelValue_128"/>
    <SBMLMap SBMLid="gamma_r_Spain" COPASIkey="ModelValue_70"/>
    <SBMLMap SBMLid="gamma_r_USA" COPASIkey="ModelValue_121"/>
    <SBMLMap SBMLid="kappa" COPASIkey="ModelValue_12"/>
    <SBMLMap SBMLid="kappa_1_rho" COPASIkey="ModelValue_23"/>
    <SBMLMap SBMLid="kappa_China" COPASIkey="ModelValue_32"/>
    <SBMLMap SBMLid="kappa_France" COPASIkey="ModelValue_83"/>
    <SBMLMap SBMLid="kappa_Germany" COPASIkey="ModelValue_100"/>
    <SBMLMap SBMLid="kappa_Italy" COPASIkey="ModelValue_50"/>
    <SBMLMap SBMLid="kappa_Spain" COPASIkey="ModelValue_66"/>
    <SBMLMap SBMLid="kappa_USA" COPASIkey="ModelValue_117"/>
    <SBMLMap SBMLid="kappa_rho" COPASIkey="ModelValue_22"/>
    <SBMLMap SBMLid="l_1" COPASIkey="ModelValue_8"/>
    <SBMLMap SBMLid="l_1_China" COPASIkey="ModelValue_36"/>
    <SBMLMap SBMLid="l_1_France" COPASIkey="ModelValue_79"/>
    <SBMLMap SBMLid="l_1_Germany" COPASIkey="ModelValue_96"/>
    <SBMLMap SBMLid="l_1_Italy" COPASIkey="ModelValue_46"/>
    <SBMLMap SBMLid="l_1_Spain" COPASIkey="ModelValue_62"/>
    <SBMLMap SBMLid="l_1_USA" COPASIkey="ModelValue_113"/>
    <SBMLMap SBMLid="l_2_China" COPASIkey="ModelValue_35"/>
    <SBMLMap SBMLid="l_2_France" COPASIkey="ModelValue_80"/>
    <SBMLMap SBMLid="l_2_Germany" COPASIkey="ModelValue_97"/>
    <SBMLMap SBMLid="l_2_Italy" COPASIkey="ModelValue_47"/>
    <SBMLMap SBMLid="l_2_Spain" COPASIkey="ModelValue_63"/>
    <SBMLMap SBMLid="l_2_USA" COPASIkey="ModelValue_114"/>
    <SBMLMap SBMLid="l_a_1" COPASIkey="ModelValue_10"/>
    <SBMLMap SBMLid="l_a_1_China" COPASIkey="ModelValue_34"/>
    <SBMLMap SBMLid="l_a_1_France" COPASIkey="ModelValue_81"/>
    <SBMLMap SBMLid="l_a_1_Germany" COPASIkey="ModelValue_98"/>
    <SBMLMap SBMLid="l_a_1_Italy" COPASIkey="ModelValue_48"/>
    <SBMLMap SBMLid="l_a_1_Spain" COPASIkey="ModelValue_64"/>
    <SBMLMap SBMLid="l_a_1_USA" COPASIkey="ModelValue_115"/>
    <SBMLMap SBMLid="l_a_2_China" COPASIkey="ModelValue_33"/>
    <SBMLMap SBMLid="l_a_2_France" COPASIkey="ModelValue_82"/>
    <SBMLMap SBMLid="l_a_2_Germany" COPASIkey="ModelValue_99"/>
    <SBMLMap SBMLid="l_a_2_Italy" COPASIkey="ModelValue_49"/>
    <SBMLMap SBMLid="l_a_2_Spain" COPASIkey="ModelValue_65"/>
    <SBMLMap SBMLid="l_a_2_USA" COPASIkey="ModelValue_116"/>
    <SBMLMap SBMLid="mu" COPASIkey="ModelValue_17"/>
    <SBMLMap SBMLid="mu_1_delta_A" COPASIkey="ModelValue_25"/>
    <SBMLMap SBMLid="mu_China" COPASIkey="ModelValue_27"/>
    <SBMLMap SBMLid="mu_France" COPASIkey="ModelValue_88"/>
    <SBMLMap SBMLid="mu_Germany" COPASIkey="ModelValue_105"/>
    <SBMLMap SBMLid="mu_Italy" COPASIkey="ModelValue_54"/>
    <SBMLMap SBMLid="mu_Spain" COPASIkey="ModelValue_71"/>
    <SBMLMap SBMLid="mu_USA" COPASIkey="ModelValue_122"/>
    <SBMLMap SBMLid="mu_delta_A" COPASIkey="ModelValue_24"/>
    <SBMLMap SBMLid="rho" COPASIkey="ModelValue_13"/>
    <SBMLMap SBMLid="rho_China" COPASIkey="ModelValue_31"/>
    <SBMLMap SBMLid="rho_France" COPASIkey="ModelValue_84"/>
    <SBMLMap SBMLid="rho_Germany" COPASIkey="ModelValue_101"/>
    <SBMLMap SBMLid="rho_Italy" COPASIkey="ModelValue_51"/>
    <SBMLMap SBMLid="rho_Spain" COPASIkey="ModelValue_67"/>
    <SBMLMap SBMLid="rho_USA" COPASIkey="ModelValue_118"/>
  </SBMLReference>
  <ListOfUnitDefinitions>
    <UnitDefinition key="Unit_5" name="second" symbol="s">
      <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Unit_4">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-19T13:24:17Z</dcterms:W3CDTF>
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
<dcterms:W3CDTF>2020-08-19T13:24:17Z</dcterms:W3CDTF>
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
<dcterms:W3CDTF>2020-08-19T13:24:17Z</dcterms:W3CDTF>
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
<dcterms:W3CDTF>2020-08-19T13:24:17Z</dcterms:W3CDTF>
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
