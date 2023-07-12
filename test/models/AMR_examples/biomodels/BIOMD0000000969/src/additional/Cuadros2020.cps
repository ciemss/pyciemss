<?xml version="1.0" encoding="UTF-8"?>
<!-- generated with COPASI 4.29 (Build 228) (http://www.copasi.org) at 2020-10-27T06:49:25Z -->
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
    <Function key="Function_40" name="Rate Law for reaction" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Function_40">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T08:08:40Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        epsilon*lambda_1*S1*(I1/N1)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_264" name="epsilon" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_263" name="lambda_1" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_262" name="S1" order="2" role="substrate"/>
        <ParameterDescription key="FunctionParameter_261" name="I1" order="3" role="product"/>
        <ParameterDescription key="FunctionParameter_250" name="N1" order="4" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_41" name="Rate Law for reaction_8" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Function_41">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T08:09:58Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        S2*(phi*(I1/N1) + epsilon*lambda_2*(I2/N2))
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_269" name="S2" order="0" role="substrate"/>
        <ParameterDescription key="FunctionParameter_268" name="phi" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_267" name="I1" order="2" role="modifier"/>
        <ParameterDescription key="FunctionParameter_266" name="N1" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_265" name="epsilon" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_270" name="lambda_2" order="5" role="constant"/>
        <ParameterDescription key="FunctionParameter_271" name="I2" order="6" role="product"/>
        <ParameterDescription key="FunctionParameter_272" name="N2" order="7" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_42" name="Rate Law for reaction_16" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Function_42">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T08:12:01Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        S3*(tau*(I2/N2) + epsilon*lambda_3*(I3/N3))
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_280" name="tau" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_279" name="I2" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_278" name="N2" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_277" name="epsilon" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_276" name="lambda_3" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_275" name="I3" order="5" role="product"/>
        <ParameterDescription key="FunctionParameter_274" name="N3" order="6" role="constant"/>
        <ParameterDescription key="FunctionParameter_273" name="S3" order="7" role="substrate"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_43" name="Rate Law for reaction_24" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Function_43">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T08:14:08Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        S4*(gamma*(I2/N2) + alpha*(I3/N3)+epsilon*lambda_4*(I4/N4))
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_288" name="S4" order="0" role="substrate"/>
        <ParameterDescription key="FunctionParameter_287" name="gamma" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_286" name="I2" order="2" role="modifier"/>
        <ParameterDescription key="FunctionParameter_285" name="N2" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_284" name="alpha" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_283" name="I3" order="5" role="modifier"/>
        <ParameterDescription key="FunctionParameter_282" name="N3" order="6" role="constant"/>
        <ParameterDescription key="FunctionParameter_281" name="epsilon" order="7" role="constant"/>
        <ParameterDescription key="FunctionParameter_289" name="lambda_4" order="8" role="constant"/>
        <ParameterDescription key="FunctionParameter_290" name="I4" order="9" role="product"/>
        <ParameterDescription key="FunctionParameter_291" name="N4" order="10" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
  </ListOfFunctions>
  <Model key="Model_1" name="Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio" simulationType="time" timeUnit="d" volumeUnit="1" areaUnit="1" lengthUnit="1" quantityUnit="#" type="deterministic" avogadroConstant="6.0221408570000002e+23">
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
        <CopasiMT:isDescribedBy rdf:resource="urn:miriam:pubmed:32736312"/>
      </rdf:Description>
    </dcterms:bibliographicCitation>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:13:57Z</dcterms:W3CDTF>
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
            <vCard:Orgname>EMBL-EBI</vCard:Orgname>
          </rdf:Description>
        </vCard:ORG>
      </rdf:Description>
    </dcterms:creator>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:doid:DOID:0080600"/>
  </rdf:Description>
</rdf:RDF>

    </MiriamAnnotation>
    <Comment>
      The role of geospatial disparities in the dynamics of the COVID-19 pandemic is poorly understood. We developed a spatially-explicit mathematical model to simulate transmission dynamics of COVID-19 disease infection in relation with the uneven distribution of the healthcare capacity in Ohio, U.S. The results showed substantial spatial variation in the spread of the disease, with localized areas showing marked differences in disease attack rates. Higher COVID-19 attack rates experienced in some highly connected and urbanized areas (274 cases per 100,000 people) could substantially impact the critical health care response of these areas regardless of their potentially high healthcare capacity compared to more rural and less connected counterparts (85 cases per 100,000). Accounting for the spatially uneven disease diffusion linked to the geographical distribution of the critical care resources is essential in designing effective prevention and control programmes aimed at reducing the impact of COVID-19 pandemic.
    </Comment>
    <ListOfCompartments>
      <Compartment key="Compartment_0" name="Counties_with_airports" simulationType="fixed" dimensionality="3" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:bqbiol="http://biomodels.net/biology-qualifiers/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Compartment_0">
    <bqbiol:hasProperty rdf:resource="urn:miriam:omit:0027506" />
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:14:30Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C111076" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C43482" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Compartment>
      <Compartment key="Compartment_1" name="Counties_neighbouring_counties_with_airports" simulationType="fixed" dimensionality="3" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:bqbiol="http://biomodels.net/biology-qualifiers/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Compartment_1">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C25633" />
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:15:01Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C111076" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C43482" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Compartment>
      <Compartment key="Compartment_2" name="Counties_with_highways" simulationType="fixed" dimensionality="3" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Compartment_2">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:15:09Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C111076" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C43482" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Compartment>
      <Compartment key="Compartment_3" name="Low_risk_counties" simulationType="fixed" dimensionality="3" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Compartment_3">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:15:16Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C111076" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C43482" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Compartment>
    </ListOfCompartments>
    <ListOfMetabolites>
      <Metabolite key="Metabolite_0" name="Susceptible_Counties_with_airports" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:22:37Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ido:0000514" />
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_1" name="Infected_Counties_with_airports" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_1">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:22:51Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_2" name="Hospitalised_Counties_with_airports" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_2">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:23:00Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C25179" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_3" name="ICU_Counties_with_airports" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_3">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:23:03Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C25179" />
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C53511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_4" name="Deceased_Counties_with_airports" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_4">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:23:04Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C28554" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_5" name="Recovered_Counties_with_airports" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_5">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:23:05Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ido:0000621" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_6" name="Discharged_Counties_with_airports" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_6">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:23:06Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C154475" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_7" name="Susceptible_Counties_neighbouring_counties_with_airports" simulationType="reactions" compartment="Compartment_1" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_7">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:23:08Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ido:0000514" />
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_8" name="Infected_Counties_neighbouring_counties_with_airports" simulationType="reactions" compartment="Compartment_1" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_8">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:23:27Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_9" name="Hospitalised_Counties_neighbouring_counties_with_airports" simulationType="reactions" compartment="Compartment_1" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_9">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:23:28Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C25179" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_10" name="ICU_Counties_neighbouring_counties_with_airports" simulationType="reactions" compartment="Compartment_1" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_10">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:23:29Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C25179" />
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C53511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_11" name="Deceased_Counties_neighbouring_counties_with_airports" simulationType="reactions" compartment="Compartment_1" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_11">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:23:29Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C28554" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_12" name="Recovered_Counties_neighbouring_counties_with_airports" simulationType="reactions" compartment="Compartment_1" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_12">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:23:42Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ido:0000621" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_13" name="Discharged_Counties_neighbouring_counties_with_airports" simulationType="reactions" compartment="Compartment_1" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_13">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:23:44Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C154475" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_14" name="Susceptible_Counties_with_highways" simulationType="reactions" compartment="Compartment_2" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_14">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:23:52Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ido:0000514" />
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_15" name="Infected_Counties_with_highways" simulationType="reactions" compartment="Compartment_2" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_15">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:24:05Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_16" name="Hospitalised_Counties_with_highways" simulationType="reactions" compartment="Compartment_2" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_16">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:24:08Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C25179" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_17" name="ICU_Counties_with_highways" simulationType="reactions" compartment="Compartment_2" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_17">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:24:12Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C25179" />
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C53511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_18" name="Deceased_Counties_with_highways" simulationType="reactions" compartment="Compartment_2" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_18">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:24:14Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C28554" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_19" name="Recovered_Counties_with_highways" simulationType="reactions" compartment="Compartment_2" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_19">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:24:15Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ido:0000621" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_20" name="Discharged_Counties_with_highways" simulationType="reactions" compartment="Compartment_2" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_20">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:24:17Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C154475" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_21" name="Susceptible_Low_risk_counties" simulationType="reactions" compartment="Compartment_3" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_21">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:24:21Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ido:0000514" />
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_22" name="Infected_Low_risk_counties" simulationType="reactions" compartment="Compartment_3" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_22">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:24:25Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_23" name="Hospitalised_Low_risk_counties" simulationType="reactions" compartment="Compartment_3" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_23">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:24:27Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C25179" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_24" name="ICU_Low_risk_counties" simulationType="reactions" compartment="Compartment_3" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_24">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:24:28Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C25179" />
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C53511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_25" name="Deceased_Low_risk_counties" simulationType="reactions" compartment="Compartment_3" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_25">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:24:28Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C28554" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_26" name="Recovered_Low_risk_counties" simulationType="reactions" compartment="Compartment_3" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_26">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:24:30Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ido:0000621" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_27" name="Discharged_Low_risk_counties" simulationType="reactions" compartment="Compartment_3" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_27">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:28:46Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C154475" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
    </ListOfMetabolites>
    <ListOfModelValues>
      <ModelValue key="ModelValue_0" name="lambda_Counties_with_airports" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:38:21Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_1" name="lambda_Counties_neighbouring_counties_with_airports" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_1">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:38:22Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_2" name="lambda_Counties_with_highways" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_2">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-09-11T07:38:24Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_3" name="lambda_Low_risk_counties" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_3">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-09-11T07:38:25Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_4" name="epsilon" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_4">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:38:34Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_5" name="delta" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_5">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:38:38Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_6" name="eta_Counties_with_airports" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_6">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:38:45Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_7" name="eta_Counties_neighbouring_counties_with_airports" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_7">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:38:46Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_8" name="eta_Counties_with_highways" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_8">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:38:47Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_9" name="eta_Low_risk_counties" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_9">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:38:48Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_10" name="omega_Counties_with_airports" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_10">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:38:58Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_11" name="omega_Counties_neighbouring_counties_with_airports" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_11">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:38:59Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_12" name="omega_Counties_with_highways" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_12">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:39:01Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_13" name="omega_Low_risk_counties" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_13">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:39:05Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_14" name="xi" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_14">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:39:53Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_15" name="sigma" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_15">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:39:55Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_16" name="mu_Counties_with_airports" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_16">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:40:00Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_17" name="mu_Counties_neighbouring_counties_with_airports" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_17">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:40:02Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_18" name="mu_Counties_with_highways" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_18">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:40:03Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_19" name="mu_Low_risk_counties" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_19">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:40:07Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_20" name="psi" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_20">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:40:08Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_21" name="phi" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_21">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:40:16Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_22" name="tau" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_22">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:40:18Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_23" name="gamma" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_23">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:40:19Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_24" name="alpha" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_24">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:40:23Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_25" name="Population_Counties_with_airports" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_25">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:42:14Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[ICU_Counties_with_airports],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[Hospitalised_Counties_with_airports],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[Infected_Counties_with_airports],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[Deceased_Counties_with_airports],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[Recovered_Counties_with_airports],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[Discharged_Counties_with_airports],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[Susceptible_Counties_with_airports],Reference=InitialConcentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_26" name="Population_Counties_neighbouring_counties_with_airports" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_26">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:42:16Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[ICU_Counties_neighbouring_counties_with_airports],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[Hospitalised_Counties_neighbouring_counties_with_airports],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[Infected_Counties_neighbouring_counties_with_airports],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[Deceased_Counties_neighbouring_counties_with_airports],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[Recovered_Counties_neighbouring_counties_with_airports],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[Discharged_Counties_neighbouring_counties_with_airports],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[Susceptible_Counties_neighbouring_counties_with_airports],Reference=InitialConcentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_27" name="Population_Counties_with_highways" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_27">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:42:17Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[ICU_Counties_with_highways],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[Hospitalised_Counties_with_highways],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[Infected_Counties_with_highways],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[Deceased_Counties_with_highways],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[Recovered_Counties_with_highways],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[Discharged_Counties_with_highways],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[Susceptible_Counties_with_highways],Reference=InitialConcentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_28" name="Population_Low_risk_counties" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_28">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:42:19Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[ICU_Low_risk_counties],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[Hospitalised_Low_risk_counties],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[Infected_Low_risk_counties],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[Deceased_Low_risk_counties],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[Recovered_Low_risk_counties],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[Discharged_Low_risk_counties],Reference=InitialConcentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[Susceptible_Low_risk_counties],Reference=InitialConcentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_29" name="Cumulative_cases_Counties_with_airports" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_29">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-10-14T09:50:38Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[Infected_Counties_with_airports],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[Recovered_Counties_with_airports],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[Discharged_Counties_with_airports],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[Deceased_Counties_with_airports],Reference=Concentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_30" name="Cumulative_cases_Counties_neighbouring_counties_with_airports" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_30">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-10-14T09:50:41Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[Deceased_Counties_neighbouring_counties_with_airports],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[Recovered_Counties_neighbouring_counties_with_airports],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[Discharged_Counties_neighbouring_counties_with_airports],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[Infected_Counties_neighbouring_counties_with_airports],Reference=Concentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_31" name="Cumulative_cases_Counties_with_highways" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_31">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-10-14T09:50:42Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[Deceased_Counties_with_highways],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[Recovered_Counties_with_highways],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[Discharged_Counties_with_highways],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[Infected_Counties_with_highways],Reference=Concentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_32" name="Cumulative_cases_Low_risk_counties" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_32">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-10-14T09:50:43Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[Deceased_Low_risk_counties],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[Recovered_Low_risk_counties],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[Discharged_Low_risk_counties],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[Infected_Low_risk_counties],Reference=Concentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_33" name="Total_cumulative_cases" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_33">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-10-14T11:49:31Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Cumulative_cases_Counties_with_airports],Reference=Value>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Cumulative_cases_Counties_neighbouring_counties_with_airports],Reference=Value>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Cumulative_cases_Counties_with_highways],Reference=Value>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Cumulative_cases_Low_risk_counties],Reference=Value>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_34" name="Total_hospitalisations" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_34">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-10-14T11:51:21Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[Deceased_Counties_with_airports],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[Discharged_Counties_neighbouring_counties_with_airports],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[Deceased_Counties_neighbouring_counties_with_airports],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[Deceased_Counties_with_highways],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[Deceased_Low_risk_counties],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[Discharged_Counties_with_airports],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[Discharged_Counties_with_highways],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[Discharged_Low_risk_counties],Reference=Concentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_35" name="Total_deaths" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_35">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-10-14T11:55:01Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[Deceased_Counties_with_airports],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[Deceased_Counties_neighbouring_counties_with_airports],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[Deceased_Counties_with_highways],Reference=Concentration>+&lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[Deceased_Low_risk_counties],Reference=Concentration>
        </Expression>
      </ModelValue>
    </ListOfModelValues>
    <ListOfReactions>
      <Reaction key="Reaction_0" name="Susceptible_to_Infected_Counties_with_airports" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:37:03Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C128320" />
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
        <ListOfConstants>
          <Constant key="Parameter_5390" name="lambda_1" value="0.41"/>
          <Constant key="Parameter_5389" name="epsilon" value="1"/>
          <Constant key="Parameter_5388" name="N1" value="4.05291e+06"/>
        </ListOfConstants>
        <KineticLaw function="Function_40" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_264">
              <SourceParameter reference="ModelValue_4"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_263">
              <SourceParameter reference="ModelValue_0"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_262">
              <SourceParameter reference="Metabolite_0"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_261">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_250">
              <SourceParameter reference="ModelValue_25"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_1" name="Infected_to_Hospitalised_Counties_with_airports" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_1">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:37:16Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C25179" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
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
          <Constant key="Parameter_5387" name="k1" value="0.05"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports]">
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
      <Reaction key="Reaction_2" name="Infected_to_Deceased_Counties_with_airports" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_2">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:37:22Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C28554" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_1" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_4" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5386" name="k1" value="0.01"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_20"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_3" name="Infected_to_Recovered_Counties_with_airports" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_3">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:38:13Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ido:0000621" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C71133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_1" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_5" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5385" name="k1" value="0.229885"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_5"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_4" name="Hospitalised_to_ICU_Counties_with_airports" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_4">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:38:32Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C171454" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
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
          <Constant key="Parameter_5384" name="k1" value="0.04"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports]">
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
      <Reaction key="Reaction_5" name="Hospitalised_to_Discharged_Counties_with_airports" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_5">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:38:42Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:omit:0011345" />
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
          <Constant key="Parameter_5383" name="k1" value="0.08"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports]">
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
      <Reaction key="Reaction_6" name="ICU_to_Hospitalised_Counties_with_airports" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_6">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:38:50Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C94226" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_3" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_2" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5382" name="k1" value="0.06"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_14"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_3"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_7" name="ICU_to_Deceased_Counties_with_airports" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_7">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T06:39:31Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C28554" />
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
          <Constant key="Parameter_5381" name="k1" value="0.22"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_16"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_3"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_8" name="Susceptible_to_Infected_Counties_neighbouring_counties_with_airports" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_8">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:33:56Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C128320" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_7" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_8" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_1" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_5380" name="N1" value="4.05291e+06"/>
          <Constant key="Parameter_5379" name="epsilon" value="1"/>
          <Constant key="Parameter_5378" name="lambda_2" value="0.34"/>
          <Constant key="Parameter_5377" name="phi" value="0.04"/>
          <Constant key="Parameter_5376" name="N2" value="4.4317e+06"/>
        </ListOfConstants>
        <KineticLaw function="Function_41" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_269">
              <SourceParameter reference="Metabolite_7"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_268">
              <SourceParameter reference="ModelValue_21"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_267">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_266">
              <SourceParameter reference="ModelValue_25"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_265">
              <SourceParameter reference="ModelValue_4"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_270">
              <SourceParameter reference="ModelValue_1"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_271">
              <SourceParameter reference="Metabolite_8"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_272">
              <SourceParameter reference="ModelValue_26"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_9" name="Infected_to_Hospitalised_Counties_neighbouring_counties_with_airports" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_9">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:34:20Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C25179" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_8" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_9" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5375" name="k1" value="0.07"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_7"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_8"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_10" name="Infected_to_Deceased_Counties_neighbouring_counties_with_airports" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_10">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:34:25Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C28554" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_8" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_11" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5374" name="k1" value="0.01"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_20"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_8"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_11" name="Infected_to_Recovered_Counties_neighbouring_counties_with_airports" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_11">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:34:37Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ido:0000621" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_8" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_12" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5373" name="k1" value="0.229885"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_5"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_8"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_12" name="Hospitalised_to_ICU_Counties_neighbouring_counties_with_airports" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_12">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:34:42Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C171454" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_9" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_10" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5372" name="k1" value="0.06"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_11"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_9"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_13" name="Hospitalised_to_Discharged_Counties_neighbouring_counties_with_airports" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_13">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:34:53Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:omit:0011345" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_9" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_13" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5371" name="k1" value="0.08"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_15"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_9"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_14" name="ICU_to_Hospitalised_Counties_neighbouring_counties_with_airports" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_14">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:34:57Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C94226" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_10" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_9" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5370" name="k1" value="0.06"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_14"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_10"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_15" name="ICU_to_Deceased_Counties_neighbouring_counties_with_airports" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_15">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:35:08Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C28554" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_10" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_11" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5369" name="k1" value="0.17"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_17"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_10"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_16" name="Susceptible_to_Infected_Counties_with_highways" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_16">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:35:23Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C128320" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_14" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_15" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_8" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_5368" name="N2" value="4.4317e+06"/>
          <Constant key="Parameter_5367" name="epsilon" value="1"/>
          <Constant key="Parameter_5366" name="lambda_3" value="0.23"/>
          <Constant key="Parameter_5365" name="tau" value="0.08"/>
          <Constant key="Parameter_5364" name="N3" value="1.81107e+06"/>
        </ListOfConstants>
        <KineticLaw function="Function_42" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_280">
              <SourceParameter reference="ModelValue_22"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_279">
              <SourceParameter reference="Metabolite_8"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_278">
              <SourceParameter reference="ModelValue_26"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_277">
              <SourceParameter reference="ModelValue_4"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_276">
              <SourceParameter reference="ModelValue_2"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_275">
              <SourceParameter reference="Metabolite_15"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_274">
              <SourceParameter reference="ModelValue_27"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_273">
              <SourceParameter reference="Metabolite_14"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_17" name="Infected_to_Hospitalised_Counties_with_highways" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_17">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:35:41Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C25179" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_15" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_16" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5363" name="k1" value="0.07"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_8"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_15"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_18" name="Infected_to_Deceased_Counties_with_highways" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_18">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:35:44Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C28554" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_15" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_18" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5687" name="k1" value="0.01"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_20"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_15"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_19" name="Infected_to_Recovered_Counties_with_highways" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_19">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:35:52Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ido:0000621" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_15" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_19" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5688" name="k1" value="0.229885"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_5"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_15"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_20" name="Hospitalised_to_ICU_Counties_with_highways" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_20">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:35:58Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C171454" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_16" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_17" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5686" name="k1" value="0.04"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_12"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_16"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_21" name="Hospitalised_to_Discharged_Counties_with_highways" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_21">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:36:12Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:omit:0011345" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_16" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_20" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5689" name="k1" value="0.08"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_15"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_16"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_22" name="ICU_to_Hospitalised_Counties_with_highways" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_22">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:36:16Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C94226" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_17" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_16" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5716" name="k1" value="0.06"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_14"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_17"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_23" name="ICU_to_Deceased_Counties_with_highways" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_23">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:36:23Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C28554" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_17" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_18" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5717" name="k1" value="0.05"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_18"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_17"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_24" name="Susceptible_to_Infected_Low_risk_counties" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_24">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:36:34Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C128320" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_21" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_22" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_8" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_15" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_5715" name="N2" value="4.4317e+06"/>
          <Constant key="Parameter_5718" name="alpha" value="0.03"/>
          <Constant key="Parameter_5700" name="gamma" value="0.02"/>
          <Constant key="Parameter_5701" name="epsilon" value="1"/>
          <Constant key="Parameter_5699" name="lambda_4" value="0.13"/>
          <Constant key="Parameter_5702" name="N3" value="1.81107e+06"/>
          <Constant key="Parameter_5362" name="N4" value="1.26995e+06"/>
        </ListOfConstants>
        <KineticLaw function="Function_43" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_288">
              <SourceParameter reference="Metabolite_21"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_287">
              <SourceParameter reference="ModelValue_23"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_286">
              <SourceParameter reference="Metabolite_8"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_285">
              <SourceParameter reference="ModelValue_26"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_284">
              <SourceParameter reference="ModelValue_24"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_283">
              <SourceParameter reference="Metabolite_15"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_282">
              <SourceParameter reference="ModelValue_27"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_281">
              <SourceParameter reference="ModelValue_4"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_289">
              <SourceParameter reference="ModelValue_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_290">
              <SourceParameter reference="Metabolite_22"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_291">
              <SourceParameter reference="ModelValue_28"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_25" name="Infected_to_Hospitalised_Low_risk_counties" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_25">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:37:10Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C25179" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_22" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_23" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5361" name="k1" value="0.14"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_9"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_22"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_26" name="Infected_to_Deceased_Low_risk_counties" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_26">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:37:22Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C28554" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_22" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_25" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5360" name="k1" value="0.01"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_20"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_22"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_27" name="Infected_to_Recovered_Low_risk_counties" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_27">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:37:24Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ido:0000621" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_22" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_26" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5359" name="k1" value="0.229885"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_5"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_22"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_28" name="Hospitalised_to_ICU_Low_risk_counties" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_28">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:37:31Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C171454" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_23" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_24" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5358" name="k1" value="0.07"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_13"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_23"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_29" name="Hospitalised_to_Discharged_Low_risk_counties" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_29">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:37:37Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:omit:0011345" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_23" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_27" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5357" name="k1" value="0.08"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_15"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_23"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_30" name="ICU_to_Hospitalised_Low_risk_counties" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_30">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:37:52Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C94226" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_24" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_23" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5356" name="k1" value="0.06"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_14"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_24"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_31" name="ICU_to_Deceased_Low_risk_counties" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_31">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-09-11T07:37:57Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is rdf:resource="urn:miriam:ncit:C28554" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_24" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_25" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_5355" name="k1" value="0.25"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_19"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_24"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
    </ListOfReactions>
    <ListOfEvents>
      <Event key="Event_0" name="Lockdown" delayAssignment="true" fireAtInitialTime="0" persistentTrigger="0">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-10-14T10:40:41Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Reference=Time> > 22
        </TriggerExpression>
        <DelayExpression>
          0
        </DelayExpression>
        <ListOfAssignments>
          <Assignment target="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[epsilon]" targetKey="ModelValue_4">
            <Expression>
              0.75
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
<dcterms:W3CDTF>2020-10-27T06:45:13Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ModelParameterGroup cn="String=Initial Time" type="Group">
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio" value="0" type="Model" simulationType="time"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Compartment Sizes" type="Group">
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports]" value="1" type="Compartment" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports]" value="1" type="Compartment" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways]" value="1" type="Compartment" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties]" value="1" type="Compartment" simulationType="fixed"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Species Values" type="Group">
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[Susceptible_Counties_with_airports]" value="4052876" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[Infected_Counties_with_airports]" value="30" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[Hospitalised_Counties_with_airports]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[ICU_Counties_with_airports]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[Deceased_Counties_with_airports]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[Recovered_Counties_with_airports]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_airports],Vector=Metabolites[Discharged_Counties_with_airports]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[Susceptible_Counties_neighbouring_counties_with_airports]" value="4431673" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[Infected_Counties_neighbouring_counties_with_airports]" value="26" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[Hospitalised_Counties_neighbouring_counties_with_airports]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[ICU_Counties_neighbouring_counties_with_airports]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[Deceased_Counties_neighbouring_counties_with_airports]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[Recovered_Counties_neighbouring_counties_with_airports]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_neighbouring_counties_with_airports],Vector=Metabolites[Discharged_Counties_neighbouring_counties_with_airports]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[Susceptible_Counties_with_highways]" value="1811059" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[Infected_Counties_with_highways]" value="10" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[Hospitalised_Counties_with_highways]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[ICU_Counties_with_highways]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[Deceased_Counties_with_highways]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[Recovered_Counties_with_highways]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Counties_with_highways],Vector=Metabolites[Discharged_Counties_with_highways]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[Susceptible_Low_risk_counties]" value="1269942" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[Infected_Low_risk_counties]" value="6" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[Hospitalised_Low_risk_counties]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[ICU_Low_risk_counties]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[Deceased_Low_risk_counties]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[Recovered_Low_risk_counties]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Compartments[Low_risk_counties],Vector=Metabolites[Discharged_Low_risk_counties]" value="0" type="Species" simulationType="reactions"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Global Quantities" type="Group">
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[lambda_Counties_with_airports]" value="0.40999999999999998" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[lambda_Counties_neighbouring_counties_with_airports]" value="0.34000000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[lambda_Counties_with_highways]" value="0.23000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[lambda_Low_risk_counties]" value="0.13" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[epsilon]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[delta]" value="0.22988500000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[eta_Counties_with_airports]" value="0.050000000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[eta_Counties_neighbouring_counties_with_airports]" value="0.070000000000000007" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[eta_Counties_with_highways]" value="0.070000000000000007" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[eta_Low_risk_counties]" value="0.14000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[omega_Counties_with_airports]" value="0.040000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[omega_Counties_neighbouring_counties_with_airports]" value="0.059999999999999998" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[omega_Counties_with_highways]" value="0.040000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[omega_Low_risk_counties]" value="0.070000000000000007" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[xi]" value="0.059999999999999998" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[sigma]" value="0.080000000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[mu_Counties_with_airports]" value="0.22" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[mu_Counties_neighbouring_counties_with_airports]" value="0.17000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[mu_Counties_with_highways]" value="0.050000000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[mu_Low_risk_counties]" value="0.25" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[psi]" value="0.01" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[phi]" value="0.040000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[tau]" value="0.080000000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[gamma]" value="0.02" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[alpha]" value="0.029999999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Population_Counties_with_airports]" value="4052906" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Population_Counties_neighbouring_counties_with_airports]" value="4431699" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Population_Counties_with_highways]" value="1811069" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Population_Low_risk_counties]" value="1269948" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Cumulative_cases_Counties_with_airports]" value="30" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Cumulative_cases_Counties_neighbouring_counties_with_airports]" value="26" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Cumulative_cases_Counties_with_highways]" value="10" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Cumulative_cases_Low_risk_counties]" value="6" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Total_cumulative_cases]" value="72" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Total_hospitalisations]" value="0" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Total_deaths]" value="0" type="ModelValue" simulationType="assignment"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Kinetic Parameters" type="Group">
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Counties_with_airports]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Counties_with_airports],ParameterGroup=Parameters,Parameter=lambda_1" value="0.40999999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[lambda_Counties_with_airports],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Counties_with_airports],ParameterGroup=Parameters,Parameter=epsilon" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[epsilon],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Counties_with_airports],ParameterGroup=Parameters,Parameter=N1" value="4052906" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Population_Counties_with_airports],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Hospitalised_Counties_with_airports]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Hospitalised_Counties_with_airports],ParameterGroup=Parameters,Parameter=k1" value="0.050000000000000003" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[eta_Counties_with_airports],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Deceased_Counties_with_airports]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Deceased_Counties_with_airports],ParameterGroup=Parameters,Parameter=k1" value="0.01" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[psi],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Recovered_Counties_with_airports]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Recovered_Counties_with_airports],ParameterGroup=Parameters,Parameter=k1" value="0.22988500000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[delta],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Hospitalised_to_ICU_Counties_with_airports]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Hospitalised_to_ICU_Counties_with_airports],ParameterGroup=Parameters,Parameter=k1" value="0.040000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[omega_Counties_with_airports],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Hospitalised_to_Discharged_Counties_with_airports]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Hospitalised_to_Discharged_Counties_with_airports],ParameterGroup=Parameters,Parameter=k1" value="0.080000000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[sigma],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[ICU_to_Hospitalised_Counties_with_airports]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[ICU_to_Hospitalised_Counties_with_airports],ParameterGroup=Parameters,Parameter=k1" value="0.059999999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[xi],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[ICU_to_Deceased_Counties_with_airports]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[ICU_to_Deceased_Counties_with_airports],ParameterGroup=Parameters,Parameter=k1" value="0.22" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[mu_Counties_with_airports],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Counties_neighbouring_counties_with_airports]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Counties_neighbouring_counties_with_airports],ParameterGroup=Parameters,Parameter=N1" value="4052906" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Population_Counties_with_airports],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Counties_neighbouring_counties_with_airports],ParameterGroup=Parameters,Parameter=epsilon" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[epsilon],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Counties_neighbouring_counties_with_airports],ParameterGroup=Parameters,Parameter=lambda_2" value="0.34000000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[lambda_Counties_neighbouring_counties_with_airports],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Counties_neighbouring_counties_with_airports],ParameterGroup=Parameters,Parameter=phi" value="0.040000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[phi],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Counties_neighbouring_counties_with_airports],ParameterGroup=Parameters,Parameter=N2" value="4431699" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Population_Counties_neighbouring_counties_with_airports],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Hospitalised_Counties_neighbouring_counties_with_airports]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Hospitalised_Counties_neighbouring_counties_with_airports],ParameterGroup=Parameters,Parameter=k1" value="0.070000000000000007" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[eta_Counties_neighbouring_counties_with_airports],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Deceased_Counties_neighbouring_counties_with_airports]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Deceased_Counties_neighbouring_counties_with_airports],ParameterGroup=Parameters,Parameter=k1" value="0.01" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[psi],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Recovered_Counties_neighbouring_counties_with_airports]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Recovered_Counties_neighbouring_counties_with_airports],ParameterGroup=Parameters,Parameter=k1" value="0.22988500000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[delta],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Hospitalised_to_ICU_Counties_neighbouring_counties_with_airports]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Hospitalised_to_ICU_Counties_neighbouring_counties_with_airports],ParameterGroup=Parameters,Parameter=k1" value="0.059999999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[omega_Counties_neighbouring_counties_with_airports],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Hospitalised_to_Discharged_Counties_neighbouring_counties_with_airports]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Hospitalised_to_Discharged_Counties_neighbouring_counties_with_airports],ParameterGroup=Parameters,Parameter=k1" value="0.080000000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[sigma],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[ICU_to_Hospitalised_Counties_neighbouring_counties_with_airports]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[ICU_to_Hospitalised_Counties_neighbouring_counties_with_airports],ParameterGroup=Parameters,Parameter=k1" value="0.059999999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[xi],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[ICU_to_Deceased_Counties_neighbouring_counties_with_airports]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[ICU_to_Deceased_Counties_neighbouring_counties_with_airports],ParameterGroup=Parameters,Parameter=k1" value="0.17000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[mu_Counties_neighbouring_counties_with_airports],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Counties_with_highways]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Counties_with_highways],ParameterGroup=Parameters,Parameter=N2" value="4431699" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Population_Counties_neighbouring_counties_with_airports],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Counties_with_highways],ParameterGroup=Parameters,Parameter=epsilon" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[epsilon],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Counties_with_highways],ParameterGroup=Parameters,Parameter=lambda_3" value="0.23000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[lambda_Counties_with_highways],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Counties_with_highways],ParameterGroup=Parameters,Parameter=tau" value="0.080000000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[tau],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Counties_with_highways],ParameterGroup=Parameters,Parameter=N3" value="1811069" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Population_Counties_with_highways],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Hospitalised_Counties_with_highways]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Hospitalised_Counties_with_highways],ParameterGroup=Parameters,Parameter=k1" value="0.070000000000000007" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[eta_Counties_with_highways],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Deceased_Counties_with_highways]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Deceased_Counties_with_highways],ParameterGroup=Parameters,Parameter=k1" value="0.01" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[psi],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Recovered_Counties_with_highways]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Recovered_Counties_with_highways],ParameterGroup=Parameters,Parameter=k1" value="0.22988500000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[delta],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Hospitalised_to_ICU_Counties_with_highways]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Hospitalised_to_ICU_Counties_with_highways],ParameterGroup=Parameters,Parameter=k1" value="0.040000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[omega_Counties_with_highways],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Hospitalised_to_Discharged_Counties_with_highways]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Hospitalised_to_Discharged_Counties_with_highways],ParameterGroup=Parameters,Parameter=k1" value="0.080000000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[sigma],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[ICU_to_Hospitalised_Counties_with_highways]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[ICU_to_Hospitalised_Counties_with_highways],ParameterGroup=Parameters,Parameter=k1" value="0.059999999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[xi],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[ICU_to_Deceased_Counties_with_highways]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[ICU_to_Deceased_Counties_with_highways],ParameterGroup=Parameters,Parameter=k1" value="0.050000000000000003" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[mu_Counties_with_highways],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Low_risk_counties]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Low_risk_counties],ParameterGroup=Parameters,Parameter=N2" value="4431699" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Population_Counties_neighbouring_counties_with_airports],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Low_risk_counties],ParameterGroup=Parameters,Parameter=alpha" value="0.029999999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[alpha],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Low_risk_counties],ParameterGroup=Parameters,Parameter=gamma" value="0.02" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[gamma],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Low_risk_counties],ParameterGroup=Parameters,Parameter=epsilon" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[epsilon],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Low_risk_counties],ParameterGroup=Parameters,Parameter=lambda_4" value="0.13" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[lambda_Low_risk_counties],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Low_risk_counties],ParameterGroup=Parameters,Parameter=N3" value="1811069" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Population_Counties_with_highways],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Susceptible_to_Infected_Low_risk_counties],ParameterGroup=Parameters,Parameter=N4" value="1269948" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Population_Low_risk_counties],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Hospitalised_Low_risk_counties]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Hospitalised_Low_risk_counties],ParameterGroup=Parameters,Parameter=k1" value="0.14000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[eta_Low_risk_counties],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Deceased_Low_risk_counties]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Deceased_Low_risk_counties],ParameterGroup=Parameters,Parameter=k1" value="0.01" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[psi],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Recovered_Low_risk_counties]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Infected_to_Recovered_Low_risk_counties],ParameterGroup=Parameters,Parameter=k1" value="0.22988500000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[delta],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Hospitalised_to_ICU_Low_risk_counties]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Hospitalised_to_ICU_Low_risk_counties],ParameterGroup=Parameters,Parameter=k1" value="0.070000000000000007" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[omega_Low_risk_counties],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Hospitalised_to_Discharged_Low_risk_counties]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[Hospitalised_to_Discharged_Low_risk_counties],ParameterGroup=Parameters,Parameter=k1" value="0.080000000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[sigma],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[ICU_to_Hospitalised_Low_risk_counties]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[ICU_to_Hospitalised_Low_risk_counties],ParameterGroup=Parameters,Parameter=k1" value="0.059999999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[xi],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[ICU_to_Deceased_Low_risk_counties]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Reactions[ICU_to_Deceased_Low_risk_counties],ParameterGroup=Parameters,Parameter=k1" value="0.25" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[mu_Low_risk_counties],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
        </ModelParameterGroup>
      </ModelParameterSet>
    </ListOfModelParameterSets>
    <StateTemplate>
      <StateTemplateVariable objectReference="Model_1"/>
      <StateTemplateVariable objectReference="Metabolite_1"/>
      <StateTemplateVariable objectReference="Metabolite_8"/>
      <StateTemplateVariable objectReference="Metabolite_15"/>
      <StateTemplateVariable objectReference="Metabolite_22"/>
      <StateTemplateVariable objectReference="Metabolite_9"/>
      <StateTemplateVariable objectReference="Metabolite_2"/>
      <StateTemplateVariable objectReference="Metabolite_16"/>
      <StateTemplateVariable objectReference="Metabolite_23"/>
      <StateTemplateVariable objectReference="Metabolite_10"/>
      <StateTemplateVariable objectReference="Metabolite_17"/>
      <StateTemplateVariable objectReference="Metabolite_3"/>
      <StateTemplateVariable objectReference="Metabolite_24"/>
      <StateTemplateVariable objectReference="Metabolite_4"/>
      <StateTemplateVariable objectReference="Metabolite_18"/>
      <StateTemplateVariable objectReference="Metabolite_11"/>
      <StateTemplateVariable objectReference="Metabolite_25"/>
      <StateTemplateVariable objectReference="Metabolite_7"/>
      <StateTemplateVariable objectReference="Metabolite_19"/>
      <StateTemplateVariable objectReference="Metabolite_21"/>
      <StateTemplateVariable objectReference="Metabolite_0"/>
      <StateTemplateVariable objectReference="Metabolite_13"/>
      <StateTemplateVariable objectReference="Metabolite_20"/>
      <StateTemplateVariable objectReference="Metabolite_6"/>
      <StateTemplateVariable objectReference="Metabolite_27"/>
      <StateTemplateVariable objectReference="Metabolite_5"/>
      <StateTemplateVariable objectReference="Metabolite_12"/>
      <StateTemplateVariable objectReference="Metabolite_26"/>
      <StateTemplateVariable objectReference="Metabolite_14"/>
      <StateTemplateVariable objectReference="ModelValue_25"/>
      <StateTemplateVariable objectReference="ModelValue_26"/>
      <StateTemplateVariable objectReference="ModelValue_27"/>
      <StateTemplateVariable objectReference="ModelValue_28"/>
      <StateTemplateVariable objectReference="ModelValue_29"/>
      <StateTemplateVariable objectReference="ModelValue_30"/>
      <StateTemplateVariable objectReference="ModelValue_31"/>
      <StateTemplateVariable objectReference="ModelValue_32"/>
      <StateTemplateVariable objectReference="ModelValue_33"/>
      <StateTemplateVariable objectReference="ModelValue_34"/>
      <StateTemplateVariable objectReference="ModelValue_35"/>
      <StateTemplateVariable objectReference="Compartment_0"/>
      <StateTemplateVariable objectReference="Compartment_1"/>
      <StateTemplateVariable objectReference="Compartment_2"/>
      <StateTemplateVariable objectReference="Compartment_3"/>
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
      <StateTemplateVariable objectReference="ModelValue_23"/>
      <StateTemplateVariable objectReference="ModelValue_24"/>
    </StateTemplate>
    <InitialState type="initialState">
      0 30 26 10 6 0 0 0 0 0 0 0 0 0 0 0 0 4431673 0 1269942 4052876 0 0 0 0 0 0 0 1811059 4052906 4431699 1811069 1269948 30 26 10 6 72 0 0 1 1 1 1 0.40999999999999998 0.34000000000000002 0.23000000000000001 0.13 1 0.22988500000000001 0.050000000000000003 0.070000000000000007 0.070000000000000007 0.14000000000000001 0.040000000000000001 0.059999999999999998 0.040000000000000001 0.070000000000000007 0.059999999999999998 0.080000000000000002 0.22 0.17000000000000001 0.050000000000000003 0.25 0.01 0.040000000000000001 0.080000000000000002 0.02 0.029999999999999999 
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
        <Parameter name="StepNumber" type="unsignedInteger" value="75"/>
        <Parameter name="StepSize" type="float" value="1"/>
        <Parameter name="Duration" type="float" value="75"/>
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
    <PlotSpecification name="plot_1" type="Plot2D" active="1" taskTypes="">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="Values[Cumulative_cases]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Total_cumulative_cases],Reference=Value"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
    <PlotSpecification name="plot_3" type="Plot2D" active="1" taskTypes="">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="Values[Total_deaths]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[Total_deaths],Reference=Value"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
  </ListOfPlots>
  <GUI>
    <ListOfSliders>
      <Slider key="Slider_0" associatedEntityKey="Task_16" objectCN="CN=Root,Model=Cuadros2020 - SIHRD spatiotemporal model of COVID-19 transmission in Ohio,Vector=Values[epsilon],Reference=InitialValue" objectType="float" objectValue="1" minValue="0.027" maxValue="2.7" tickNumber="1000" tickFactor="100" scaling="logarithmic"/>
    </ListOfSliders>
  </GUI>
  <SBMLReference file="Cuadros2020.xml">
    <SBMLMap SBMLid="Counties_neighbouring_counties_with_airports" COPASIkey="Compartment_1"/>
    <SBMLMap SBMLid="Counties_with_airports" COPASIkey="Compartment_0"/>
    <SBMLMap SBMLid="Counties_with_highways" COPASIkey="Compartment_2"/>
    <SBMLMap SBMLid="Cumulative_cases_Counties_neighbouring_counties_with_airports" COPASIkey="ModelValue_30"/>
    <SBMLMap SBMLid="Cumulative_cases_Counties_with_airports" COPASIkey="ModelValue_29"/>
    <SBMLMap SBMLid="Cumulative_cases_Counties_with_highways" COPASIkey="ModelValue_31"/>
    <SBMLMap SBMLid="Cumulative_cases_Low_risk_counties" COPASIkey="ModelValue_32"/>
    <SBMLMap SBMLid="Deceased_Counties_neighbouring_counties_with_airports" COPASIkey="Metabolite_11"/>
    <SBMLMap SBMLid="Deceased_Counties_with_airports" COPASIkey="Metabolite_4"/>
    <SBMLMap SBMLid="Deceased_Counties_with_highways" COPASIkey="Metabolite_18"/>
    <SBMLMap SBMLid="Deceased_Low_risk_counties" COPASIkey="Metabolite_25"/>
    <SBMLMap SBMLid="Discharged_Counties_neighbouring_counties_with_airports" COPASIkey="Metabolite_13"/>
    <SBMLMap SBMLid="Discharged_Counties_with_airports" COPASIkey="Metabolite_6"/>
    <SBMLMap SBMLid="Discharged_Counties_with_highways" COPASIkey="Metabolite_20"/>
    <SBMLMap SBMLid="Discharged_Low_risk_counties" COPASIkey="Metabolite_27"/>
    <SBMLMap SBMLid="Hospitalised_Counties_neighbouring_counties_with_airports" COPASIkey="Metabolite_9"/>
    <SBMLMap SBMLid="Hospitalised_Counties_with_airports" COPASIkey="Metabolite_2"/>
    <SBMLMap SBMLid="Hospitalised_Counties_with_highways" COPASIkey="Metabolite_16"/>
    <SBMLMap SBMLid="Hospitalised_Low_risk_counties" COPASIkey="Metabolite_23"/>
    <SBMLMap SBMLid="Hospitalised_to_Discharged_Counties_neighbouring_counties_with_airports" COPASIkey="Reaction_13"/>
    <SBMLMap SBMLid="Hospitalised_to_Discharged_Counties_with_airports" COPASIkey="Reaction_5"/>
    <SBMLMap SBMLid="Hospitalised_to_Discharged_Counties_with_highways" COPASIkey="Reaction_21"/>
    <SBMLMap SBMLid="Hospitalised_to_Discharged_Low_risk_counties" COPASIkey="Reaction_29"/>
    <SBMLMap SBMLid="Hospitalised_to_ICU_Counties_neighbouring_counties_with_airports" COPASIkey="Reaction_12"/>
    <SBMLMap SBMLid="Hospitalised_to_ICU_Counties_with_airports" COPASIkey="Reaction_4"/>
    <SBMLMap SBMLid="Hospitalised_to_ICU_Counties_with_highways" COPASIkey="Reaction_20"/>
    <SBMLMap SBMLid="Hospitalised_to_ICU_Low_risk_counties" COPASIkey="Reaction_28"/>
    <SBMLMap SBMLid="ICU_Counties_neighbouring_counties_with_airports" COPASIkey="Metabolite_10"/>
    <SBMLMap SBMLid="ICU_Counties_with_airports" COPASIkey="Metabolite_3"/>
    <SBMLMap SBMLid="ICU_Counties_with_highways" COPASIkey="Metabolite_17"/>
    <SBMLMap SBMLid="ICU_Low_risk_counties" COPASIkey="Metabolite_24"/>
    <SBMLMap SBMLid="ICU_to_Deceased_Counties_neighbouring_counties_with_airports" COPASIkey="Reaction_15"/>
    <SBMLMap SBMLid="ICU_to_Deceased_Counties_with_airports" COPASIkey="Reaction_7"/>
    <SBMLMap SBMLid="ICU_to_Deceased_Counties_with_highways" COPASIkey="Reaction_23"/>
    <SBMLMap SBMLid="ICU_to_Deceased_Low_risk_counties" COPASIkey="Reaction_31"/>
    <SBMLMap SBMLid="ICU_to_Hospitalised_Counties_neighbouring_counties_with_airports" COPASIkey="Reaction_14"/>
    <SBMLMap SBMLid="ICU_to_Hospitalised_Counties_with_airports" COPASIkey="Reaction_6"/>
    <SBMLMap SBMLid="ICU_to_Hospitalised_Counties_with_highways" COPASIkey="Reaction_22"/>
    <SBMLMap SBMLid="ICU_to_Hospitalised_Low_risk_counties" COPASIkey="Reaction_30"/>
    <SBMLMap SBMLid="Infected_Counties_neighbouring_counties_with_airports" COPASIkey="Metabolite_8"/>
    <SBMLMap SBMLid="Infected_Counties_with_airports" COPASIkey="Metabolite_1"/>
    <SBMLMap SBMLid="Infected_Counties_with_highways" COPASIkey="Metabolite_15"/>
    <SBMLMap SBMLid="Infected_Low_risk_counties" COPASIkey="Metabolite_22"/>
    <SBMLMap SBMLid="Infected_to_Deceased_Counties_neighbouring_counties_with_airports" COPASIkey="Reaction_10"/>
    <SBMLMap SBMLid="Infected_to_Deceased_Counties_with_airports" COPASIkey="Reaction_2"/>
    <SBMLMap SBMLid="Infected_to_Deceased_Counties_with_highways" COPASIkey="Reaction_18"/>
    <SBMLMap SBMLid="Infected_to_Deceased_Low_risk_counties" COPASIkey="Reaction_26"/>
    <SBMLMap SBMLid="Infected_to_Hospitalised_Counties_neighbouring_counties_with_airports" COPASIkey="Reaction_9"/>
    <SBMLMap SBMLid="Infected_to_Hospitalised_Counties_with_airports" COPASIkey="Reaction_1"/>
    <SBMLMap SBMLid="Infected_to_Hospitalised_Counties_with_highways" COPASIkey="Reaction_17"/>
    <SBMLMap SBMLid="Infected_to_Hospitalised_Low_risk_counties" COPASIkey="Reaction_25"/>
    <SBMLMap SBMLid="Infected_to_Recovered_Counties_neighbouring_counties_with_airports" COPASIkey="Reaction_11"/>
    <SBMLMap SBMLid="Infected_to_Recovered_Counties_with_airports" COPASIkey="Reaction_3"/>
    <SBMLMap SBMLid="Infected_to_Recovered_Counties_with_highways" COPASIkey="Reaction_19"/>
    <SBMLMap SBMLid="Infected_to_Recovered_Low_risk_counties" COPASIkey="Reaction_27"/>
    <SBMLMap SBMLid="Lockdown_0" COPASIkey="Event_0"/>
    <SBMLMap SBMLid="Low_risk_counties" COPASIkey="Compartment_3"/>
    <SBMLMap SBMLid="Population_Counties_neighbouring_counties_with_airports" COPASIkey="ModelValue_26"/>
    <SBMLMap SBMLid="Population_Counties_with_airports" COPASIkey="ModelValue_25"/>
    <SBMLMap SBMLid="Population_Counties_with_highways" COPASIkey="ModelValue_27"/>
    <SBMLMap SBMLid="Population_Low_risk_counties" COPASIkey="ModelValue_28"/>
    <SBMLMap SBMLid="Rate_Law_for_reaction" COPASIkey="Function_40"/>
    <SBMLMap SBMLid="Rate_Law_for_reaction_16" COPASIkey="Function_42"/>
    <SBMLMap SBMLid="Rate_Law_for_reaction_24" COPASIkey="Function_43"/>
    <SBMLMap SBMLid="Rate_Law_for_reaction_8" COPASIkey="Function_41"/>
    <SBMLMap SBMLid="Recovered_Counties_neighbouring_counties_with_airports" COPASIkey="Metabolite_12"/>
    <SBMLMap SBMLid="Recovered_Counties_with_airports" COPASIkey="Metabolite_5"/>
    <SBMLMap SBMLid="Recovered_Counties_with_highways" COPASIkey="Metabolite_19"/>
    <SBMLMap SBMLid="Recovered_Low_risk_counties" COPASIkey="Metabolite_26"/>
    <SBMLMap SBMLid="Susceptible_Counties_neighbouring_counties_with_airports" COPASIkey="Metabolite_7"/>
    <SBMLMap SBMLid="Susceptible_Counties_with_airports" COPASIkey="Metabolite_0"/>
    <SBMLMap SBMLid="Susceptible_Counties_with_highways" COPASIkey="Metabolite_14"/>
    <SBMLMap SBMLid="Susceptible_Low_risk_counties" COPASIkey="Metabolite_21"/>
    <SBMLMap SBMLid="Susceptible_to_Infected_Counties_neighbouring_counties_with_airports" COPASIkey="Reaction_8"/>
    <SBMLMap SBMLid="Susceptible_to_Infected_Counties_with_airports" COPASIkey="Reaction_0"/>
    <SBMLMap SBMLid="Susceptible_to_Infected_Counties_with_highways" COPASIkey="Reaction_16"/>
    <SBMLMap SBMLid="Susceptible_to_Infected_Low_risk_counties" COPASIkey="Reaction_24"/>
    <SBMLMap SBMLid="Total_cumulative_cases" COPASIkey="ModelValue_33"/>
    <SBMLMap SBMLid="Total_deaths" COPASIkey="ModelValue_35"/>
    <SBMLMap SBMLid="Total_hospitalisations" COPASIkey="ModelValue_34"/>
    <SBMLMap SBMLid="alpha" COPASIkey="ModelValue_24"/>
    <SBMLMap SBMLid="delta" COPASIkey="ModelValue_5"/>
    <SBMLMap SBMLid="epsilon" COPASIkey="ModelValue_4"/>
    <SBMLMap SBMLid="eta_Counties_neighbouring_counties_with_airports" COPASIkey="ModelValue_7"/>
    <SBMLMap SBMLid="eta_Counties_with_airports" COPASIkey="ModelValue_6"/>
    <SBMLMap SBMLid="eta_Counties_with_highways" COPASIkey="ModelValue_8"/>
    <SBMLMap SBMLid="eta_Low_risk_counties" COPASIkey="ModelValue_9"/>
    <SBMLMap SBMLid="gamma" COPASIkey="ModelValue_23"/>
    <SBMLMap SBMLid="lambda_Counties_neighbouring_counties_with_airports" COPASIkey="ModelValue_1"/>
    <SBMLMap SBMLid="lambda_Counties_with_airports" COPASIkey="ModelValue_0"/>
    <SBMLMap SBMLid="lambda_Counties_with_highways" COPASIkey="ModelValue_2"/>
    <SBMLMap SBMLid="lambda_Low_risk_counties" COPASIkey="ModelValue_3"/>
    <SBMLMap SBMLid="mu_Counties_neighbouring_counties_with_airports" COPASIkey="ModelValue_17"/>
    <SBMLMap SBMLid="mu_Counties_with_airports" COPASIkey="ModelValue_16"/>
    <SBMLMap SBMLid="mu_Counties_with_highways" COPASIkey="ModelValue_18"/>
    <SBMLMap SBMLid="mu_Low_risk_counties" COPASIkey="ModelValue_19"/>
    <SBMLMap SBMLid="omega_Counties_neighbouring_counties_with_airports" COPASIkey="ModelValue_11"/>
    <SBMLMap SBMLid="omega_Counties_with_airports" COPASIkey="ModelValue_10"/>
    <SBMLMap SBMLid="omega_Counties_with_highways" COPASIkey="ModelValue_12"/>
    <SBMLMap SBMLid="omega_Low_risk_counties" COPASIkey="ModelValue_13"/>
    <SBMLMap SBMLid="phi" COPASIkey="ModelValue_21"/>
    <SBMLMap SBMLid="psi" COPASIkey="ModelValue_20"/>
    <SBMLMap SBMLid="sigma" COPASIkey="ModelValue_15"/>
    <SBMLMap SBMLid="tau" COPASIkey="ModelValue_22"/>
    <SBMLMap SBMLid="xi" COPASIkey="ModelValue_14"/>
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
<dcterms:W3CDTF>2020-10-27T06:45:08Z</dcterms:W3CDTF>
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
<dcterms:W3CDTF>2020-10-27T06:45:08Z</dcterms:W3CDTF>
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
<dcterms:W3CDTF>2020-10-27T06:45:08Z</dcterms:W3CDTF>
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
<dcterms:W3CDTF>2020-10-27T06:45:08Z</dcterms:W3CDTF>
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
