<?xml version="1.0" encoding="UTF-8"?>
<!-- generated with COPASI 4.27 (Build 217) (http://www.copasi.org) at 2020-05-13T16:32:47Z -->
<?oxygen RNGSchema="http://www.copasi.org/static/schema/CopasiML.rng" type="xml"?>
<COPASI xmlns="http://www.copasi.org/static/schema" versionMajor="4" versionMinor="27" versionDevel="217" copasiSourcesModified="0">
  <ListOfFunctions>
    <Function key="Function_6" name="Constant flux (irreversible)" type="PreDefined" reversible="false">
      <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Function_6">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-05-13T15:46:46Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        v
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_49" name="v" order="0" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_13" name="Mass action (irreversible)" type="MassAction" reversible="false">
      <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
   <rdf:Description rdf:about="#Function_13">
   <CopasiMT:is rdf:resource="urn:miriam:obo.sbo:SBO:0000041" />
   </rdf:Description>
   </rdf:RDF>
      </MiriamAnnotation>
      <Comment>
        <body xmlns="http://www.w3.org/1999/xhtml">
<b>Mass action rate law for first order irreversible reactions</b>
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
    <Function key="Function_40" name="human birth rate" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Function_40">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:44:14Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        Psi_h*N_h
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_264" name="Psi_h" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_263" name="N_h" order="1" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_41" name="Mosquito birth rate" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Function_41">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:44:31Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        Psi_v*N_v
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_261" name="Psi_v" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_262" name="N_v" order="1" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
  </ListOfFunctions>
  <Model key="Model_1" name="Chitnis2008 - Mathematical model of malaria transmission" simulationType="time" timeUnit="s" volumeUnit="ml" areaUnit="m²" lengthUnit="m" quantityUnit="mmol" type="deterministic" avogadroConstant="6.0221408570000002e+23">
    <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:vCard="http://www.w3.org/2001/vcard-rdf/3.0#">
  <rdf:Description rdf:about="#Model_1">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C34797"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C71292"/>
    <bqbiol:hasTaxon rdf:resource="urn:miriam:taxonomy:9606"/>
    <dcterms:bibliographicCitation>
      <rdf:Description>
        <CopasiMT:isDescribedBy rdf:resource="urn:miriam:pubmed:18293044"/>
      </rdf:Description>
    </dcterms:bibliographicCitation>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:31:04Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <dcterms:creator>
      <rdf:Description>
        <vCard:EMAIL>mroberts@ebi.ac.uk</vCard:EMAIL>
        <vCard:N>
          <rdf:Description>
            <vCard:Family>Roberts</vCard:Family>
            <vCard:Given>Matthew Grant</vCard:Given>
          </rdf:Description>
        </vCard:N>
        <vCard:ORG>
          <rdf:Description>
            <vCard:Orgname>EMBL-EBI</vCard:Orgname>
          </rdf:Description>
        </vCard:ORG>
      </rdf:Description>
    </dcterms:creator>
    <dcterms:creator>
      <rdf:Description>
        <vCard:EMAIL>tiwarik@babraham.ac.uk</vCard:EMAIL>
        <vCard:N>
          <rdf:Description>
            <vCard:Family>Tiwari</vCard:Family>
            <vCard:Given>Krishna</vCard:Given>
          </rdf:Description>
        </vCard:N>
        <vCard:ORG>
          <rdf:Description>
            <vCard:Orgname>Babraham Institute</vCard:Orgname>
          </rdf:Description>
        </vCard:ORG>
      </rdf:Description>
    </dcterms:creator>
    <CopasiMT:hasPart rdf:resource="urn:miriam:efo:0001068"/>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C17214"/>
  </rdf:Description>
</rdf:RDF>

    </MiriamAnnotation>
    <ListOfCompartments>
      <Compartment key="Compartment_0" name="Human" simulationType="fixed" dimensionality="3" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Compartment_0">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C48694"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:taxonomy:9606"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-05-13T15:46:48Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
      </Compartment>
      <Compartment key="Compartment_1" name="Mosquito" simulationType="fixed" dimensionality="3" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Compartment_1">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C14198"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-05-13T15:46:48Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
      </Compartment>
    </ListOfCompartments>
    <ListOfMetabolites>
      <Metabolite key="Metabolite_0" name="Susceptible" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_0">
    <bqbiol:hasProperty rdf:resource="urn:miriam:omit:0005461"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:30:35Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <InitialExpression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=InitialValue> == 0,600,500)
        </InitialExpression>
      </Metabolite>
      <Metabolite key="Metabolite_1" name="Exposed" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_1">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C156623"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:taxonomy:9606"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:44:49Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <InitialExpression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=InitialValue> == 0,20,10)
        </InitialExpression>
      </Metabolite>
      <Metabolite key="Metabolite_2" name="Infected" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_2">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C128320"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:taxonomy:9606"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:29:45Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <InitialExpression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=InitialValue> == 0,3,30)
        </InitialExpression>
      </Metabolite>
      <Metabolite key="Metabolite_3" name="Recovered" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_3">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C25746"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:30:31Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_4" name="Susceptible" simulationType="reactions" compartment="Compartment_1" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_4">
    <bqbiol:hasProperty rdf:resource="urn:miriam:omit:0005461"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:11:21Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <InitialExpression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=InitialValue> == 0,2400,4000)
        </InitialExpression>
      </Metabolite>
      <Metabolite key="Metabolite_5" name="Exposed" simulationType="reactions" compartment="Compartment_1" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_5">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncim:C123547"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C17214"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:29:20Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <InitialExpression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=InitialValue> == 0,30,100)
        </InitialExpression>
      </Metabolite>
      <Metabolite key="Metabolite_6" name="Infected" simulationType="reactions" compartment="Compartment_1" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_6">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C123547"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C128320"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:30:08Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <InitialExpression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=InitialValue> == 0,5,50)
        </InitialExpression>
      </Metabolite>
    </ListOfMetabolites>
    <ListOfModelValues>
      <ModelValue key="ModelValue_0" name="Capital_lambda_h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:54:14Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=Value> == 0,0.041,0.033)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_1" name="Psi_h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_1">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:54:57Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=Value> == 0,5.5e-5,1.1e-4)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_2" name="Psi_v" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_2">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:55:24Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=Value> == 0,0.13,0.13)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_3" name="sigma_v" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_3">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:56:11Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=Value> == 0,0.33,0.5)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_4" name="sigma_h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_4">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:56:06Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=Value> == 0,4.3,19)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_5" name="Beta_hv" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_5">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:57:17Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=Value> == 0,0.022,0.022)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_6" name="Beta_vh" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_6">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:57:36Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=Value> == 0,0.24,0.48)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_7" name="Beta_tilde_vh" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_7">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:04:08Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=Value> == 0,0.024,0.048)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_8" name="v_h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_8">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:58:01Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=Value> == 0,0.1,0.1)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_9" name="v_v" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_9">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:58:23Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=Value> == 0,0.083,0.091)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_10" name="gamma_h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_10">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:58:58Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=Value> == 0,0.0035,0.0035)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_11" name="delta_h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_11">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:59:28Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=Value> == 0,1.8e-5,9e-5)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_12" name="rho_h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_12">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:59:58Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=Value> == 0,2.7e-3,5.5e-4)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_13" name="u_1h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_13">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:00:36Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=Value> == 0,8.8e-6,1.6e-5)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_14" name="u_2h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_14">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:01:12Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=Value> == 0,2e-7,3e-7)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_15" name="u_1v" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_15">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:01:34Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=Value> == 0,0.033,0.033)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_16" name="u_2v" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_16">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:01:54Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=Value> == 0,4e-5,2e-5)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_17" name="N_h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_17">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:44:59Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human],Vector=Metabolites[Susceptible],Reference=Concentration>+&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human],Vector=Metabolites[Exposed],Reference=Concentration>+&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human],Vector=Metabolites[Infected],Reference=Concentration>+&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human],Vector=Metabolites[Recovered],Reference=Concentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_18" name="N_v" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_18">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:45:35Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito],Vector=Metabolites[Susceptible],Reference=Concentration>+&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito],Vector=Metabolites[Exposed],Reference=Concentration>+&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito],Vector=Metabolites[Infected],Reference=Concentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_19" name="f_h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_19">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:46:01Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[u_1h],Reference=Value>+&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[u_2h],Reference=Value>*&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[N_h],Reference=Value>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_20" name="f_v" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_20">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:46:27Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[u_1v],Reference=Value>+&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[u_2v],Reference=Value>*&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[N_v],Reference=Value>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_21" name="lambda_h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_21">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:47:20Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[sigma_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[N_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[sigma_h],Reference=Value>/(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[sigma_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[N_v],Reference=Value>+&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[sigma_h],Reference=Value>*&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[N_h],Reference=Value>)*&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Beta_hv],Reference=Value>*(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito],Vector=Metabolites[Infected],Reference=Concentration>/&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[N_v],Reference=Value>)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_22" name="lambda_v" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_22">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:49:24Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[sigma_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[sigma_h],Reference=Value>*&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[N_h],Reference=Value>/(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[sigma_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[N_v],Reference=Value>+&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[sigma_h],Reference=Value>*&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[N_h],Reference=Value>)*(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Beta_vh],Reference=Value>*(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human],Vector=Metabolites[Infected],Reference=Concentration>/&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[N_h],Reference=Value>)+&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Beta_tilde_vh],Reference=Value>*(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human],Vector=Metabolites[Recovered],Reference=Concentration>/&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[N_h],Reference=Value>))
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_23" name="Baseline_dummy_variable" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_23">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T14:36:30Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Comment>
          Low == 0, High==1
        </Comment>
      </ModelValue>
    </ListOfModelValues>
    <ListOfReactions>
      <Reaction key="Reaction_0" name="Human Immigration" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_0">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C85498"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:09:36Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_0" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5390" name="v" value="0.041"/>
        </ListOfConstants>
        <KineticLaw function="Function_6" unitType="Default" scalingCompartment="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_49">
              <SourceParameter reference="ModelValue_0"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_1" name="Human Birth" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_1">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C25155"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:09:59Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_0" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5389" name="Psi_h" value="5.5e-05"/>
          <Constant key="Parameter_5388" name="N_h" value="623"/>
        </ListOfConstants>
        <KineticLaw function="Function_40" unitType="Default" scalingCompartment="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_264">
              <SourceParameter reference="ModelValue_1"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_263">
              <SourceParameter reference="ModelValue_17"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_2" name="Human Loss of Immunity" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_2">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C16710"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C25559"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:10:39Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_3" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_0" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5387" name="k1" value="0.0027"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human]">
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
      <Reaction key="Reaction_3" name="Human Infection" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_3">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C128320"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:11:23Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_0" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_1" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5386" name="k1" value="4.48219e-05"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_21"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_0"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_4" name="Human Susceptible natural death" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_4">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:11:45Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:hasPart rdf:resource="urn:miriam:omit:0017353"/>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_0" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfConstants>
          <Constant key="Parameter_5385" name="k1" value="0.0001334"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_19"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_0"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_5" name="Human disease progression" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_5">
    <bqbiol:hasProperty rdf:resource="urn:miriam:omit:0018647"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:12:06Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
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
          <Constant key="Parameter_5384" name="k1" value="0.1"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_8"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_6" name="Human Exposed natural death" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_6">
    <bqbiol:hasProperty rdf:resource="urn:miriam:omit:0017353"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:12:28Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_1" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfConstants>
          <Constant key="Parameter_5383" name="k1" value="0.0001334"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_19"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_7" name="Human Recovery" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_7">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C25746"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:12:48Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_2" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_3" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5382" name="k1" value="0.0035"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_10"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_2"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_8" name="Human disease-induced death" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_8">
    <bqbiol:hasProperty rdf:resource="urn:miriam:omit:0017353"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:13:44Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_2" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfConstants>
          <Constant key="Parameter_5381" name="k1" value="1.8e-05"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_11"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_2"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_9" name="Human Infectious natural death" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_9">
    <bqbiol:hasProperty rdf:resource="urn:miriam:omit:0017353"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:14:17Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_2" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfConstants>
          <Constant key="Parameter_5380" name="k1" value="0.0001334"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_19"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_2"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_10" name="Human Recovered natural death" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_10">
    <bqbiol:hasProperty rdf:resource="urn:miriam:omit:0017353"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:14:32Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_3" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfConstants>
          <Constant key="Parameter_5379" name="k1" value="0.0001334"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_19"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_3"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_11" name="Mosquito birth" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_11">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C25155"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:15:25Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_4" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5378" name="Psi_v" value="0.13"/>
          <Constant key="Parameter_5377" name="N_v" value="2435"/>
        </ListOfConstants>
        <KineticLaw function="Function_41" unitType="Default" scalingCompartment="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_261">
              <SourceParameter reference="ModelValue_2"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_262">
              <SourceParameter reference="ModelValue_18"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_12" name="Mosquito infection" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_12">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C128320"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:15:52Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
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
          <Constant key="Parameter_5376" name="k1" value="0.00029338"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_22"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_4"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_13" name="Mosquito Susceptible death" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_13">
    <bqbiol:hasProperty rdf:resource="urn:miriam:omit:0017353"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:16:26Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_4" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfConstants>
          <Constant key="Parameter_5375" name="k1" value="0.1304"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_20"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_4"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_14" name="Mosquito disease progression" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_14">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C17747"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:16:38Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
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
          <Constant key="Parameter_5374" name="k1" value="0.083"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_9"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_5"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_15" name="Mosquito Exposed death" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_15">
    <bqbiol:hasProperty rdf:resource="urn:miriam:omit:0017353"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:17:11Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_5" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfConstants>
          <Constant key="Parameter_5373" name="k1" value="0.1304"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_20"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_5"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_16" name="Mosquito Infected death" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_16">
    <bqbiol:hasProperty rdf:resource="urn:miriam:omit:0017353"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T15:17:21Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_6" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfConstants>
          <Constant key="Parameter_5372" name="k1" value="0.1304"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_20"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_6"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
    </ListOfReactions>
    <ListOfModelParameterSets activeSet="ModelParameterSet_1">
      <ModelParameterSet key="ModelParameterSet_1" name="Initial State">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelParameterSet_1">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-05-13T15:46:48Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ModelParameterGroup cn="String=Initial Time" type="Group">
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission" value="0" type="Model" simulationType="time"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Compartment Sizes" type="Group">
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human]" value="1" type="Compartment" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito]" value="1" type="Compartment" simulationType="fixed"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Species Values" type="Group">
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human],Vector=Metabolites[Susceptible]" value="3.6132845142e+23" type="Species" simulationType="reactions">
            <InitialExpression>
              if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=InitialValue> == 0,600,500)
            </InitialExpression>
          </ModelParameter>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human],Vector=Metabolites[Exposed]" value="1.2044281714000001e+22" type="Species" simulationType="reactions">
            <InitialExpression>
              if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=InitialValue> == 0,20,10)
            </InitialExpression>
          </ModelParameter>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human],Vector=Metabolites[Infected]" value="1.8066422570999999e+21" type="Species" simulationType="reactions">
            <InitialExpression>
              if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=InitialValue> == 0,3,30)
            </InitialExpression>
          </ModelParameter>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human],Vector=Metabolites[Recovered]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito],Vector=Metabolites[Susceptible]" value="1.44531380568e+24" type="Species" simulationType="reactions">
            <InitialExpression>
              if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=InitialValue> == 0,2400,4000)
            </InitialExpression>
          </ModelParameter>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito],Vector=Metabolites[Exposed]" value="1.8066422571000001e+22" type="Species" simulationType="reactions">
            <InitialExpression>
              if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=InitialValue> == 0,30,100)
            </InitialExpression>
          </ModelParameter>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito],Vector=Metabolites[Infected]" value="3.0110704285000002e+21" type="Species" simulationType="reactions">
            <InitialExpression>
              if(&lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable],Reference=InitialValue> == 0,5,50)
            </InitialExpression>
          </ModelParameter>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Global Quantities" type="Group">
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Capital_lambda_h]" value="0.041000000000000002" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Psi_h]" value="5.5000000000000002e-05" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Psi_v]" value="0.13" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[sigma_v]" value="0.33000000000000002" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[sigma_h]" value="4.2999999999999998" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Beta_hv]" value="0.021999999999999999" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Beta_vh]" value="0.23999999999999999" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Beta_tilde_vh]" value="0.024" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[v_h]" value="0.10000000000000001" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[v_v]" value="0.083000000000000004" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[gamma_h]" value="0.0035000000000000001" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[delta_h]" value="1.8e-05" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[rho_h]" value="0.0027000000000000001" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[u_1h]" value="8.8000000000000004e-06" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[u_2h]" value="1.9999999999999999e-07" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[u_1v]" value="0.033000000000000002" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[u_2v]" value="4.0000000000000003e-05" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[N_h]" value="623" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[N_v]" value="2435" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[f_h]" value="0.00013339999999999999" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[f_v]" value="0.13040000000000002" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[lambda_h]" value="4.4821892633060062e-05" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[lambda_v]" value="0.00029337966087093851" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Baseline_dummy_variable]" value="0" type="ModelValue" simulationType="fixed"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Kinetic Parameters" type="Group">
          <ModelParameterGroup cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human Immigration]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human Immigration],ParameterGroup=Parameters,Parameter=v" value="0.041000000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Capital_lambda_h],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human Birth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human Birth],ParameterGroup=Parameters,Parameter=Psi_h" value="5.5000000000000002e-05" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Psi_h],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human Birth],ParameterGroup=Parameters,Parameter=N_h" value="623" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[N_h],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human Loss of Immunity]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human Loss of Immunity],ParameterGroup=Parameters,Parameter=k1" value="0.0027000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[rho_h],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human Infection]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human Infection],ParameterGroup=Parameters,Parameter=k1" value="4.4821892633060062e-05" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[lambda_h],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human Susceptible natural death]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human Susceptible natural death],ParameterGroup=Parameters,Parameter=k1" value="0.00013339999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[f_h],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human disease progression]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human disease progression],ParameterGroup=Parameters,Parameter=k1" value="0.10000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[v_h],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human Exposed natural death]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human Exposed natural death],ParameterGroup=Parameters,Parameter=k1" value="0.00013339999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[f_h],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human Recovery]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human Recovery],ParameterGroup=Parameters,Parameter=k1" value="0.0035000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[gamma_h],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human disease-induced death]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human disease-induced death],ParameterGroup=Parameters,Parameter=k1" value="1.8e-05" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[delta_h],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human Infectious natural death]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human Infectious natural death],ParameterGroup=Parameters,Parameter=k1" value="0.00013339999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[f_h],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human Recovered natural death]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Human Recovered natural death],ParameterGroup=Parameters,Parameter=k1" value="0.00013339999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[f_h],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Mosquito birth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Mosquito birth],ParameterGroup=Parameters,Parameter=Psi_v" value="0.13" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[Psi_v],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Mosquito birth],ParameterGroup=Parameters,Parameter=N_v" value="2435" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[N_v],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Mosquito infection]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Mosquito infection],ParameterGroup=Parameters,Parameter=k1" value="0.00029337966087093851" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[lambda_v],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Mosquito Susceptible death]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Mosquito Susceptible death],ParameterGroup=Parameters,Parameter=k1" value="0.13040000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[f_v],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Mosquito disease progression]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Mosquito disease progression],ParameterGroup=Parameters,Parameter=k1" value="0.083000000000000004" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[v_v],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Mosquito Exposed death]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Mosquito Exposed death],ParameterGroup=Parameters,Parameter=k1" value="0.13040000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[f_v],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Mosquito Infected death]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Reactions[Mosquito Infected death],ParameterGroup=Parameters,Parameter=k1" value="0.13040000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Values[f_v],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
        </ModelParameterGroup>
      </ModelParameterSet>
    </ListOfModelParameterSets>
    <StateTemplate>
      <StateTemplateVariable objectReference="Model_1"/>
      <StateTemplateVariable objectReference="Metabolite_0"/>
      <StateTemplateVariable objectReference="Metabolite_2"/>
      <StateTemplateVariable objectReference="Metabolite_4"/>
      <StateTemplateVariable objectReference="Metabolite_5"/>
      <StateTemplateVariable objectReference="Metabolite_1"/>
      <StateTemplateVariable objectReference="Metabolite_3"/>
      <StateTemplateVariable objectReference="Metabolite_6"/>
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
      <StateTemplateVariable objectReference="ModelValue_22"/>
      <StateTemplateVariable objectReference="Compartment_0"/>
      <StateTemplateVariable objectReference="Compartment_1"/>
      <StateTemplateVariable objectReference="ModelValue_23"/>
    </StateTemplate>
    <InitialState type="initialState">
      0 3.6132845142e+23 1.8066422570999999e+21 1.44531380568e+24 1.8066422571000001e+22 1.2044281714000001e+22 0 3.0110704285000002e+21 0.041000000000000002 5.5000000000000002e-05 0.13 0.33000000000000002 4.2999999999999998 0.021999999999999999 0.23999999999999999 0.024 0.10000000000000001 0.083000000000000004 0.0035000000000000001 1.8e-05 0.0027000000000000001 8.8000000000000004e-06 1.9999999999999999e-07 0.033000000000000002 4.0000000000000003e-05 623 2435 0.00013339999999999999 0.13040000000000002 4.4821892633060062e-05 0.00029337966087093851 1 1 0 
    </InitialState>
  </Model>
  <ListOfTasks>
    <Task key="Task_15" name="Steady-State" type="steadyState" scheduled="false" updateModel="false">
      <Report reference="Report_10" target="" append="1" confirmOverwrite="1"/>
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
      </Method>
    </Task>
    <Task key="Task_16" name="Time-Course" type="timeCourse" scheduled="false" updateModel="false">
      <Problem>
        <Parameter name="AutomaticStepSize" type="bool" value="0"/>
        <Parameter name="StepNumber" type="unsignedInteger" value="10000"/>
        <Parameter name="StepSize" type="float" value="1"/>
        <Parameter name="Duration" type="float" value="10000"/>
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
        <Parameter name="Max Internal Steps" type="unsignedInteger" value="10000"/>
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
      <Report reference="Report_11" target="" append="1" confirmOverwrite="1"/>
      <Problem>
      </Problem>
      <Method name="EFM Algorithm" type="EFMAlgorithm">
      </Method>
    </Task>
    <Task key="Task_19" name="Optimization" type="optimization" scheduled="false" updateModel="false">
      <Report reference="Report_12" target="" append="1" confirmOverwrite="1"/>
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
      <Report reference="Report_13" target="" append="1" confirmOverwrite="1"/>
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
      <Report reference="Report_14" target="" append="1" confirmOverwrite="1"/>
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
      <Report reference="Report_15" target="" append="1" confirmOverwrite="1"/>
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
      <Report reference="Report_16" target="" append="1" confirmOverwrite="1"/>
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
      <Report reference="Report_17" target="" append="1" confirmOverwrite="1"/>
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
        <Parameter name="Max Internal Steps" type="unsignedInteger" value="10000"/>
        <Parameter name="Max Internal Step Size" type="unsignedFloat" value="0"/>
      </Method>
    </Task>
    <Task key="Task_27" name="Linear Noise Approximation" type="linearNoiseApproximation" scheduled="false" updateModel="false">
      <Report reference="Report_18" target="" append="1" confirmOverwrite="1"/>
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
    <Report key="Report_10" name="Steady-State" taskType="steadyState" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Footer>
        <Object cn="CN=Root,Vector=TaskList[Steady-State]"/>
      </Footer>
    </Report>
    <Report key="Report_11" name="Elementary Flux Modes" taskType="fluxMode" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Footer>
        <Object cn="CN=Root,Vector=TaskList[Elementary Flux Modes],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_12" name="Optimization" taskType="optimization" separator="&#x09;" precision="6">
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
    <Report key="Report_13" name="Parameter Estimation" taskType="parameterFitting" separator="&#x09;" precision="6">
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
    <Report key="Report_14" name="Metabolic Control Analysis" taskType="metabolicControlAnalysis" separator="&#x09;" precision="6">
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
    <Report key="Report_15" name="Lyapunov Exponents" taskType="lyapunovExponents" separator="&#x09;" precision="6">
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
    <Report key="Report_16" name="Time Scale Separation Analysis" taskType="timeScaleSeparationAnalysis" separator="&#x09;" precision="6">
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
    <Report key="Report_17" name="Sensitivities" taskType="sensitivities" separator="&#x09;" precision="6">
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
    <Report key="Report_18" name="Linear Noise Approximation" taskType="linearNoiseApproximation" separator="&#x09;" precision="6">
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
  </ListOfReports>
  <ListOfPlots>
    <PlotSpecification name="Figure 2 top left" type="Plot2D" active="1" taskTypes="">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="[Exposed{Human}]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human],Vector=Metabolites[Exposed],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[Infected{Human}]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human],Vector=Metabolites[Infected],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[Recovered]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human],Vector=Metabolites[Recovered],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
    <PlotSpecification name="Figure 2 top right" type="Plot2D" active="1" taskTypes="">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="[Susceptible{Human}]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human],Vector=Metabolites[Susceptible],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
    <PlotSpecification name="Figure 2 bottom left" type="Plot2D" active="1" taskTypes="">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="[Exposed{Mosquito}]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito],Vector=Metabolites[Exposed],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[Infected{Mosquito}]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito],Vector=Metabolites[Infected],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
    <PlotSpecification name="Figure 2 bottom right" type="Plot2D" active="1" taskTypes="">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="[Susceptible{Mosquito}]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito],Vector=Metabolites[Susceptible],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
    <PlotSpecification name="Figure 3 top" type="Plot2D" active="1" taskTypes="">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="[Exposed{Human}]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human],Vector=Metabolites[Exposed],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[Infected{Human}]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human],Vector=Metabolites[Infected],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[Recovered]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human],Vector=Metabolites[Recovered],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[Susceptible{Human}]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Human],Vector=Metabolites[Susceptible],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
    <PlotSpecification name="Figure 3 bottom" type="Plot2D" active="1" taskTypes="">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="[Exposed{Mosquito}]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito],Vector=Metabolites[Exposed],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[Infected{Mosquito}]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito],Vector=Metabolites[Infected],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[Susceptible{Mosquito}]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2008 - Mathematical model of malaria transmission,Vector=Compartments[Mosquito],Vector=Metabolites[Susceptible],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
  </ListOfPlots>
  <GUI>
  </GUI>
  <SBMLReference file="Chitnis2008.xml">
    <SBMLMap SBMLid="Baseline_dummy_variable" COPASIkey="ModelValue_23"/>
    <SBMLMap SBMLid="Beta_hv" COPASIkey="ModelValue_5"/>
    <SBMLMap SBMLid="Beta_tilde_vh" COPASIkey="ModelValue_7"/>
    <SBMLMap SBMLid="Beta_vh" COPASIkey="ModelValue_6"/>
    <SBMLMap SBMLid="Capital_lambda_h" COPASIkey="ModelValue_0"/>
    <SBMLMap SBMLid="Constant_flux__irreversible" COPASIkey="Function_6"/>
    <SBMLMap SBMLid="Exposed_Human" COPASIkey="Metabolite_1"/>
    <SBMLMap SBMLid="Exposed_Mosquito" COPASIkey="Metabolite_5"/>
    <SBMLMap SBMLid="Human" COPASIkey="Compartment_0"/>
    <SBMLMap SBMLid="Human_Birth" COPASIkey="Reaction_1"/>
    <SBMLMap SBMLid="Human_Exposed_natural_death" COPASIkey="Reaction_6"/>
    <SBMLMap SBMLid="Human_Immigration" COPASIkey="Reaction_0"/>
    <SBMLMap SBMLid="Human_Infection" COPASIkey="Reaction_3"/>
    <SBMLMap SBMLid="Human_Infectious_natural_death" COPASIkey="Reaction_9"/>
    <SBMLMap SBMLid="Human_Loss_of_Immunity" COPASIkey="Reaction_2"/>
    <SBMLMap SBMLid="Human_Recovered_natural_death" COPASIkey="Reaction_10"/>
    <SBMLMap SBMLid="Human_Recovery" COPASIkey="Reaction_7"/>
    <SBMLMap SBMLid="Human_Susceptible_natural_death" COPASIkey="Reaction_4"/>
    <SBMLMap SBMLid="Human_disease_induced_death" COPASIkey="Reaction_8"/>
    <SBMLMap SBMLid="Human_disease_progression" COPASIkey="Reaction_5"/>
    <SBMLMap SBMLid="Infected_Human" COPASIkey="Metabolite_2"/>
    <SBMLMap SBMLid="Infected_Mosquito" COPASIkey="Metabolite_6"/>
    <SBMLMap SBMLid="Mosquito" COPASIkey="Compartment_1"/>
    <SBMLMap SBMLid="Mosquito_Exposed_death" COPASIkey="Reaction_15"/>
    <SBMLMap SBMLid="Mosquito_Infected_death" COPASIkey="Reaction_16"/>
    <SBMLMap SBMLid="Mosquito_Susceptible_death" COPASIkey="Reaction_13"/>
    <SBMLMap SBMLid="Mosquito_birth" COPASIkey="Reaction_11"/>
    <SBMLMap SBMLid="Mosquito_birth_rate" COPASIkey="Function_41"/>
    <SBMLMap SBMLid="Mosquito_disease_progression" COPASIkey="Reaction_14"/>
    <SBMLMap SBMLid="Mosquito_infection" COPASIkey="Reaction_12"/>
    <SBMLMap SBMLid="N_h" COPASIkey="ModelValue_17"/>
    <SBMLMap SBMLid="N_v" COPASIkey="ModelValue_18"/>
    <SBMLMap SBMLid="Psi_h" COPASIkey="ModelValue_1"/>
    <SBMLMap SBMLid="Psi_v" COPASIkey="ModelValue_2"/>
    <SBMLMap SBMLid="Recovered" COPASIkey="Metabolite_3"/>
    <SBMLMap SBMLid="Susceptible_Human" COPASIkey="Metabolite_0"/>
    <SBMLMap SBMLid="Susceptible_Mosquito" COPASIkey="Metabolite_4"/>
    <SBMLMap SBMLid="delta_h" COPASIkey="ModelValue_11"/>
    <SBMLMap SBMLid="f_h" COPASIkey="ModelValue_19"/>
    <SBMLMap SBMLid="f_v" COPASIkey="ModelValue_20"/>
    <SBMLMap SBMLid="gamma_h" COPASIkey="ModelValue_10"/>
    <SBMLMap SBMLid="human_birth_rate" COPASIkey="Function_40"/>
    <SBMLMap SBMLid="lambda_h" COPASIkey="ModelValue_21"/>
    <SBMLMap SBMLid="lambda_v" COPASIkey="ModelValue_22"/>
    <SBMLMap SBMLid="rho_h" COPASIkey="ModelValue_12"/>
    <SBMLMap SBMLid="sigma_h" COPASIkey="ModelValue_4"/>
    <SBMLMap SBMLid="sigma_v" COPASIkey="ModelValue_3"/>
    <SBMLMap SBMLid="u_1h" COPASIkey="ModelValue_13"/>
    <SBMLMap SBMLid="u_1v" COPASIkey="ModelValue_15"/>
    <SBMLMap SBMLid="u_2h" COPASIkey="ModelValue_14"/>
    <SBMLMap SBMLid="u_2v" COPASIkey="ModelValue_16"/>
    <SBMLMap SBMLid="v_h" COPASIkey="ModelValue_8"/>
    <SBMLMap SBMLid="v_v" COPASIkey="ModelValue_9"/>
  </SBMLReference>
  <ListOfUnitDefinitions>
    <UnitDefinition key="Unit_1" name="meter" symbol="m">
      <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Unit_0">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-05-13T15:46:46Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        m
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_5" name="second" symbol="s">
      <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Unit_4">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-05-13T15:46:46Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        s
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_13" name="Avogadro" symbol="Avogadro">
      <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Unit_12">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-05-13T15:46:46Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        Avogadro
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
<dcterms:W3CDTF>2020-05-13T15:46:46Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        #
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_35" name="liter" symbol="l">
      <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Unit_34">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-05-13T15:46:46Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        0.001*m^3
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_41" name="mole" symbol="mol">
      <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Unit_40">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-05-13T15:46:46Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        Avogadro*#
      </Expression>
    </UnitDefinition>
  </ListOfUnitDefinitions>
</COPASI>
