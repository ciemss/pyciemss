<?xml version="1.0" encoding="UTF-8"?>
<!-- generated with COPASI 4.29 (Build 228) (http://www.copasi.org) at 2020-08-26T14:33:58Z -->
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
    <Function key="Function_40" name="Rate Law for R1" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Function_40">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T09:17:34Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        alpha*S*U/N
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_264" name="alpha" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_263" name="S" order="1" role="substrate"/>
        <ParameterDescription key="FunctionParameter_262" name="U" order="2" role="product"/>
        <ParameterDescription key="FunctionParameter_261" name="N" order="3" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
  </ListOfFunctions>
  <Model key="Model_1" name="Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan, Hubei, and China" simulationType="time" timeUnit="d" volumeUnit="1" areaUnit="1" lengthUnit="1" quantityUnit="#" type="deterministic" avogadroConstant="6.0221408570000002e+23">
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
        <CopasiMT:isDescribedBy rdf:resource="urn:miriam:pubmed:32219006"/>
      </rdf:Description>
    </dcterms:bibliographicCitation>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:14:39Z</dcterms:W3CDTF>
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
      Background - The coronavirus disease 2019 (COVID-19) is rapidly spreading in China and more than 30 countries over last two months. COVID-19 has multiple characteristics distinct from other infectious diseases, including high infectivity during incubation, time delay between real dynamics and daily observed number of confirmed cases, and the intervention effects of implemented quarantine and control measures. Methods - We develop a Susceptible, Un-quanrantined infected, Quarantined infected, Confirmed infected (SUQC) model to characterize the dynamics of COVID-19 and explicitly parameterize the intervention effects of control measures, which is more suitable for analysis than other existing epidemic models. Results - The SUQC model is applied to the daily released data of the confirmed infections to analyze the outbreak of COVID-19 in Wuhan, Hubei (excluding Wuhan), China (excluding Hubei) and four first-tier cities of China. We found that, before January 30, 2020, all these regions except Beijing had a reproductive number R > 1, and after January 30, all regions had a reproductive number R lesser than 1, indicating that the quarantine and control measures are effective in preventing the spread of COVID-19. The confirmation rate of Wuhan estimated by our model is 0.0643, substantially lower than that of Hubei excluding Wuhan (0.1914), and that of China excluding Hubei (0.2189), but it jumps to 0.3229 after February 12 when clinical evidence was adopted in new diagnosis guidelines. The number of unquarantined infected cases in Wuhan on February 12, 2020 is estimated to be 3,509 and declines to 334 on February 21, 2020. After fitting the model with data as of February 21, 2020, we predict that the end time of COVID-19 in Wuhan and Hubei is around late March, around mid March for China excluding Hubei, and before early March 2020 for the four tier-one cities. A total of 80,511 individuals are estimated to be infected in China, among which 49,510 are from Wuhan, 17,679 from Hubei (excluding Wuhan), and the rest 13,322 from other regions of China (excluding Hubei). Note that the estimates are from a deterministic ODE model and should be interpreted with some uncertainty. Conclusions - We suggest that rigorous quarantine and control measures should be kept before early March in Beijing, Shanghai, Guangzhou and Shenzhen, and before late March in Hubei. The model can also be useful to predict the trend of epidemic and provide quantitative guide for other countries at high risk of outbreak, such as South Korea, Japan, Italy and Iran.
    </Comment>
    <ListOfCompartments>
      <Compartment key="Compartment_0" name="China" simulationType="fixed" dimensionality="3" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Compartment_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:14:57Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C16428" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C25632" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Compartment>
    </ListOfCompartments>
    <ListOfMetabolites>
      <Metabolite key="Metabolite_0" name="Susceptible" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:15:02Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000514" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <InitialExpression>
          &lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Total_Pop],Reference=InitialValue>
        </InitialExpression>
      </Metabolite>
      <Metabolite key="Metabolite_1" name="Unquarantined_Infected" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_1">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:15:04Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Comment>
          Wuhan - initial values
Stage I - 258
Stage II - 15270
Stage III - 4000

Hubei - initial values
Stage I - 270
Stage II - 5700

China - initial values
Stage I - 291 (Set model initial time to -30. Keep it at 0 for everything else)
Stage II - 2800
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_2" name="Quarantined_Infected" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_2">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:15:06Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C25549" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Comment>
          Wuhan - initial values
Stage I - 0
Stage II - 0
Stage III - 5000

Hubei - initial values
Stage I - 0
Stage II - 1500

China - initial values
Stage I - 0 (Set model initial time to -30 for Stage I alone. Keep it at 0 for everything else)
Stage II - 2000
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_3" name="Confirmed_Infected" simulationType="reactions" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_3">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:15:07Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C25297" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C72159" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Comment>
          Wuhan - initial values
Stage I - 258
Stage II - 2000
Stage III - 36000

Hubei - initial values
Stage I - 0
Stage II - 1600

China - initial values
Stage I - 0 (Set model initial time to -30. Keep it at 0 for everything else)
Stage II - 4000
        </Comment>
      </Metabolite>
      <Metabolite key="Metabolite_4" name="Cumulative_Infected" simulationType="assignment" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_4">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:15:08Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ido:0000511" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Compartments[China],Vector=Metabolites[Confirmed_Infected],Reference=Concentration>+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Compartments[China],Vector=Metabolites[Quarantined_Infected],Reference=Concentration>+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Compartments[China],Vector=Metabolites[Unquarantined_Infected],Reference=Concentration>
        </Expression>
      </Metabolite>
    </ListOfMetabolites>
    <ListOfModelValues>
      <ModelValue key="ModelValue_0" name="Trigger_Stage_I" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:18:00Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          dimensionless
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_1" name="Trigger_Stage_II" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_1">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:18:02Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          dimensionless
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_2" name="Trigger_Stage_III" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_2">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:18:06Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          dimensionless
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_3" name="Trigger_Wuhan" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_3">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:18:16Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          dimensionless
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_4" name="Trigger_Hubei" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_4">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:18:46Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          dimensionless
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_5" name="Trigger_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_5">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:18:54Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Unit>
          dimensionless
        </Unit>
      </ModelValue>
      <ModelValue key="ModelValue_6" name="R" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_6">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:19:34Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Wuhan],Reference=InitialValue>*(&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_I],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[R_Stage_I_Wuhan],Reference=InitialValue>+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_II],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[R_Stage_II_Wuhan],Reference=InitialValue>+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_III],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[R_Stage_III_Wuhan],Reference=InitialValue>)+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Hubei],Reference=InitialValue>*(&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_I],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[R_Stage_I_Hubei],Reference=InitialValue>+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_II],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[R_Stage_II_Hubei],Reference=InitialValue>)+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_China],Reference=InitialValue>*(&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_I],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[R_Stage_I_China],Reference=InitialValue>+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_II],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[R_Stage_II_China],Reference=InitialValue>)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_7" name="gamma_1" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_7">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:19:38Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Wuhan],Reference=InitialValue>*(&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_I],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_1_Stage_I_Wuhan],Reference=InitialValue>+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_II],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_1_Stage_II_Wuhan],Reference=InitialValue>+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_III],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_1_Stage_III_Wuhan],Reference=InitialValue>)+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Hubei],Reference=InitialValue>*(&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_I],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_1_Stage_I_Hubei],Reference=InitialValue>+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_II],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_1_Stage_II_Hubei],Reference=InitialValue>)+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_China],Reference=InitialValue>*(&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_I],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_1_Stage_I_China],Reference=InitialValue>+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_II],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_1_Stage_II_China],Reference=InitialValue>)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_8" name="gamma_2" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_8">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:19:41Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Wuhan],Reference=InitialValue>*(&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_I],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_2_Stage_I_Wuhan],Reference=InitialValue>+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_II],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_2_Stage_II_Wuhan],Reference=InitialValue>+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_III],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_2_Stage_III_Wuhan],Reference=InitialValue>)+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Hubei],Reference=InitialValue>*(&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_I],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_2_Stage_I_Hubei],Reference=InitialValue>+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_II],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_2_Stage_II_Hubei],Reference=InitialValue>)+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_China],Reference=InitialValue>*(&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_I],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_2_Stage_I_China],Reference=InitialValue>+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_II],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_2_Stage_II_China],Reference=InitialValue>)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_9" name="sigma" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_9">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T10:19:43Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_10" name="alpha" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_10">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:20:05Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[R],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_1],Reference=InitialValue>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_11" name="beta" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_11">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:20:06Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_2],Reference=InitialValue>+(1-&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_2],Reference=InitialValue>)*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[sigma],Reference=InitialValue>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_12" name="Total_Pop" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_12">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:22:44Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Wuhan],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Total_Pop_Wuhan],Reference=InitialValue>+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Hubei],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Total_Pop_Hubei],Reference=InitialValue>+&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_China],Reference=InitialValue>*&lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Total_Pop_China],Reference=InitialValue>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_13" name="Total_Pop_Wuhan" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_13">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T10:23:01Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_14" name="Total_Pop_Hubei" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_14">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T10:23:32Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_15" name="Total_Pop_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_15">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T10:23:39Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_16" name="R_Stage_I_Wuhan" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_16">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T10:24:00Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_17" name="gamma_1_Stage_I_Wuhan" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_17">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T10:24:05Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_18" name="gamma_2_Stage_I_Wuhan" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_18">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T10:24:10Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_19" name="R_Stage_II_Wuhan" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_19">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T10:24:18Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_20" name="gamma_1_Stage_II_Wuhan" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_20">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T10:24:23Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_21" name="gamma_2_Stage_II_Wuhan" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_21">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T10:24:27Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_22" name="R_Stage_III_Wuhan" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_22">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T10:24:35Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_23" name="gamma_1_Stage_III_Wuhan" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_23">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T10:24:40Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_24" name="gamma_2_Stage_III_Wuhan" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_24">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T10:24:47Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_25" name="R_Stage_I_Hubei" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_25">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T12:14:42Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_26" name="gamma_1_Stage_I_Hubei" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_26">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T12:14:51Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_27" name="gamma_2_Stage_I_Hubei" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_27">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T12:14:56Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_28" name="R_Stage_II_Hubei" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_28">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T12:15:05Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_29" name="gamma_1_Stage_II_Hubei" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_29">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T12:15:11Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_30" name="gamma_2_Stage_II_Hubei" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_30">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T12:15:15Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_31" name="R_Stage_I_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_31">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T12:42:27Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_32" name="gamma_1_Stage_I_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_32">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T12:42:30Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_33" name="gamma_2_Stage_I_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_33">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T12:42:37Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_34" name="R_Stage_II_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_34">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T12:42:44Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_35" name="gamma_1_Stage_II_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_35">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T12:42:49Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_36" name="gamma_2_Stage_II_China" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelValue_36">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-08-26T12:42:54Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
    </ListOfModelValues>
    <ListOfReactions>
      <Reaction key="Reaction_0" name="Susceptible_to_Unquarantined" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:15:35Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C128320" />
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
          <Constant key="Parameter_5390" name="N" value="9.01e+06"/>
          <Constant key="Parameter_5389" name="alpha" value="0.29668"/>
        </ListOfConstants>
        <KineticLaw function="Function_40" unitType="Default" scalingCompartment="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Compartments[China]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_264">
              <SourceParameter reference="ModelValue_10"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_263">
              <SourceParameter reference="Metabolite_0"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_262">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_261">
              <SourceParameter reference="ModelValue_12"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_1" name="Unquarantined_to_Quarantined" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_1">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:15:37Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C25458" />
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
          <Constant key="Parameter_5388" name="k1" value="0.063"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Compartments[China]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_80">
              <SourceParameter reference="ModelValue_7"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="Metabolite_1"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_2" name="Quarantined_to_Confirmed" reversible="false" fast="false" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_2">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-08-26T10:15:38Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C171133" />
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C25458" />
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
          <Constant key="Parameter_5387" name="k1" value="0.05"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Compartments[China]">
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
<dcterms:W3CDTF>2020-08-26T14:33:50Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ModelParameterGroup cn="String=Initial Time" type="Group">
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China" value="0" type="Model" simulationType="time"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Compartment Sizes" type="Group">
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Compartments[China]" value="1" type="Compartment" simulationType="fixed"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Species Values" type="Group">
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Compartments[China],Vector=Metabolites[Susceptible]" value="9010000" type="Species" simulationType="reactions">
            <InitialExpression>
              &lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Total_Pop],Reference=InitialValue>
            </InitialExpression>
          </ModelParameter>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Compartments[China],Vector=Metabolites[Unquarantined_Infected]" value="258" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Compartments[China],Vector=Metabolites[Quarantined_Infected]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Compartments[China],Vector=Metabolites[Confirmed_Infected]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Compartments[China],Vector=Metabolites[Cumulative_Infected]" value="258" type="Species" simulationType="assignment"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Global Quantities" type="Group">
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_I]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_II]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Stage_III]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Wuhan]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_Hubei]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Trigger_China]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[R]" value="4.7092000000000001" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_1]" value="0.063" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_2]" value="0.050000000000000003" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[sigma]" value="0" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[alpha]" value="0.29667959999999999" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[beta]" value="0.050000000000000003" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Total_Pop]" value="9010000" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Total_Pop_Wuhan]" value="9010000" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Total_Pop_Hubei]" value="48000000" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Total_Pop_China]" value="1335000000" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[R_Stage_I_Wuhan]" value="4.7092000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_1_Stage_I_Wuhan]" value="0.063" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_2_Stage_I_Wuhan]" value="0.050000000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[R_Stage_II_Wuhan]" value="0.75749999999999995" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_1_Stage_II_Wuhan]" value="0.39169999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_2_Stage_II_Wuhan]" value="0.064299999999999996" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[R_Stage_III_Wuhan]" value="0.47970000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_1_Stage_III_Wuhan]" value="0.61850000000000005" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_2_Stage_III_Wuhan]" value="0.32200000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[R_Stage_I_Hubei]" value="5.9340000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_1_Stage_I_Hubei]" value="0.050000000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_2_Stage_I_Hubei]" value="0.050000000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[R_Stage_II_Hubei]" value="0.6079" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_1_Stage_II_Hubei]" value="0.48799999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_2_Stage_II_Hubei]" value="0.19139999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[R_Stage_I_China]" value="1.5283" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_1_Stage_I_China]" value="0.19409999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_2_Stage_I_China]" value="0.050000000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[R_Stage_II_China]" value="0.57530000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_1_Stage_II_China]" value="0.51570000000000005" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_2_Stage_II_China]" value="0.21890000000000001" type="ModelValue" simulationType="fixed"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Kinetic Parameters" type="Group">
          <ModelParameterGroup cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Reactions[Susceptible_to_Unquarantined]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Reactions[Susceptible_to_Unquarantined],ParameterGroup=Parameters,Parameter=N" value="9010000" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[Total_Pop],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Reactions[Susceptible_to_Unquarantined],ParameterGroup=Parameters,Parameter=alpha" value="0.29667959999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[alpha],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Reactions[Unquarantined_to_Quarantined]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Reactions[Unquarantined_to_Quarantined],ParameterGroup=Parameters,Parameter=k1" value="0.063" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[gamma_1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Reactions[Quarantined_to_Confirmed]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Reactions[Quarantined_to_Confirmed],ParameterGroup=Parameters,Parameter=k1" value="0.050000000000000003" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Values[beta],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
        </ModelParameterGroup>
      </ModelParameterSet>
    </ListOfModelParameterSets>
    <StateTemplate>
      <StateTemplateVariable objectReference="Model_1"/>
      <StateTemplateVariable objectReference="Metabolite_1"/>
      <StateTemplateVariable objectReference="Metabolite_2"/>
      <StateTemplateVariable objectReference="Metabolite_0"/>
      <StateTemplateVariable objectReference="Metabolite_3"/>
      <StateTemplateVariable objectReference="Metabolite_4"/>
      <StateTemplateVariable objectReference="ModelValue_6"/>
      <StateTemplateVariable objectReference="ModelValue_7"/>
      <StateTemplateVariable objectReference="ModelValue_8"/>
      <StateTemplateVariable objectReference="ModelValue_10"/>
      <StateTemplateVariable objectReference="ModelValue_11"/>
      <StateTemplateVariable objectReference="ModelValue_12"/>
      <StateTemplateVariable objectReference="Compartment_0"/>
      <StateTemplateVariable objectReference="ModelValue_0"/>
      <StateTemplateVariable objectReference="ModelValue_1"/>
      <StateTemplateVariable objectReference="ModelValue_2"/>
      <StateTemplateVariable objectReference="ModelValue_3"/>
      <StateTemplateVariable objectReference="ModelValue_4"/>
      <StateTemplateVariable objectReference="ModelValue_5"/>
      <StateTemplateVariable objectReference="ModelValue_9"/>
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
      <StateTemplateVariable objectReference="ModelValue_27"/>
      <StateTemplateVariable objectReference="ModelValue_28"/>
      <StateTemplateVariable objectReference="ModelValue_29"/>
      <StateTemplateVariable objectReference="ModelValue_30"/>
      <StateTemplateVariable objectReference="ModelValue_31"/>
      <StateTemplateVariable objectReference="ModelValue_32"/>
      <StateTemplateVariable objectReference="ModelValue_33"/>
      <StateTemplateVariable objectReference="ModelValue_34"/>
      <StateTemplateVariable objectReference="ModelValue_35"/>
      <StateTemplateVariable objectReference="ModelValue_36"/>
    </StateTemplate>
    <InitialState type="initialState">
      0 258 0 9010000 0 258 4.7092000000000001 0.063 0.050000000000000003 0.29667959999999999 0.050000000000000003 9010000 1 1 0 0 1 0 0 0 9010000 48000000 1335000000 4.7092000000000001 0.063 0.050000000000000003 0.75749999999999995 0.39169999999999999 0.064299999999999996 0.47970000000000002 0.61850000000000005 0.32200000000000001 5.9340000000000002 0.050000000000000003 0.050000000000000003 0.6079 0.48799999999999999 0.19139999999999999 1.5283 0.19409999999999999 0.050000000000000003 0.57530000000000003 0.51570000000000005 0.21890000000000001 
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
        <Parameter name="StepNumber" type="unsignedInteger" value="1050"/>
        <Parameter name="StepSize" type="float" value="0.47619047619999999"/>
        <Parameter name="Duration" type="float" value="500"/>
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
        <PlotItem name="[Unquarantined_Infected]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Compartments[China],Vector=Metabolites[Unquarantined_Infected],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[Quarantined_Infected]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Compartments[China],Vector=Metabolites[Quarantined_Infected],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[Confirmed_Infected]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Compartments[China],Vector=Metabolites[Confirmed_Infected],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[Cumulative_Infected]" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Zhao2020 - SUQC model of COVID-19 transmission dynamics in Wuhan\, Hubei\, and China,Vector=Compartments[China],Vector=Metabolites[Cumulative_Infected],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
  </ListOfPlots>
  <GUI>
  </GUI>
  <SBMLReference file="Zhao2020.xml">
    <SBMLMap SBMLid="China" COPASIkey="Compartment_0"/>
    <SBMLMap SBMLid="Confirmed_Infected" COPASIkey="Metabolite_3"/>
    <SBMLMap SBMLid="Cumulative_Infected" COPASIkey="Metabolite_4"/>
    <SBMLMap SBMLid="Quarantined_Infected" COPASIkey="Metabolite_2"/>
    <SBMLMap SBMLid="Quarantined_to_Confirmed" COPASIkey="Reaction_2"/>
    <SBMLMap SBMLid="R" COPASIkey="ModelValue_6"/>
    <SBMLMap SBMLid="R_Stage_III_Wuhan" COPASIkey="ModelValue_22"/>
    <SBMLMap SBMLid="R_Stage_II_China" COPASIkey="ModelValue_34"/>
    <SBMLMap SBMLid="R_Stage_II_Hubei" COPASIkey="ModelValue_28"/>
    <SBMLMap SBMLid="R_Stage_II_Wuhan" COPASIkey="ModelValue_19"/>
    <SBMLMap SBMLid="R_Stage_I_China" COPASIkey="ModelValue_31"/>
    <SBMLMap SBMLid="R_Stage_I_Hubei" COPASIkey="ModelValue_25"/>
    <SBMLMap SBMLid="R_Stage_I_Wuhan" COPASIkey="ModelValue_16"/>
    <SBMLMap SBMLid="Rate_Law_for_R1" COPASIkey="Function_40"/>
    <SBMLMap SBMLid="Susceptible" COPASIkey="Metabolite_0"/>
    <SBMLMap SBMLid="Susceptible_to_Unquarantined" COPASIkey="Reaction_0"/>
    <SBMLMap SBMLid="Total_Pop" COPASIkey="ModelValue_12"/>
    <SBMLMap SBMLid="Total_Pop_China" COPASIkey="ModelValue_15"/>
    <SBMLMap SBMLid="Total_Pop_Hubei" COPASIkey="ModelValue_14"/>
    <SBMLMap SBMLid="Total_Pop_Wuhan" COPASIkey="ModelValue_13"/>
    <SBMLMap SBMLid="Trigger_China" COPASIkey="ModelValue_5"/>
    <SBMLMap SBMLid="Trigger_Hubei" COPASIkey="ModelValue_4"/>
    <SBMLMap SBMLid="Trigger_Stage_I" COPASIkey="ModelValue_0"/>
    <SBMLMap SBMLid="Trigger_Stage_II" COPASIkey="ModelValue_1"/>
    <SBMLMap SBMLid="Trigger_Stage_III" COPASIkey="ModelValue_2"/>
    <SBMLMap SBMLid="Trigger_Wuhan" COPASIkey="ModelValue_3"/>
    <SBMLMap SBMLid="Unquarantined_Infected" COPASIkey="Metabolite_1"/>
    <SBMLMap SBMLid="Unquarantined_to_Quarantined" COPASIkey="Reaction_1"/>
    <SBMLMap SBMLid="alpha" COPASIkey="ModelValue_10"/>
    <SBMLMap SBMLid="beta" COPASIkey="ModelValue_11"/>
    <SBMLMap SBMLid="gamma_1" COPASIkey="ModelValue_7"/>
    <SBMLMap SBMLid="gamma_1_Stage_III_Wuhan" COPASIkey="ModelValue_23"/>
    <SBMLMap SBMLid="gamma_1_Stage_II_China" COPASIkey="ModelValue_35"/>
    <SBMLMap SBMLid="gamma_1_Stage_II_Hubei" COPASIkey="ModelValue_29"/>
    <SBMLMap SBMLid="gamma_1_Stage_II_Wuhan" COPASIkey="ModelValue_20"/>
    <SBMLMap SBMLid="gamma_1_Stage_I_China" COPASIkey="ModelValue_32"/>
    <SBMLMap SBMLid="gamma_1_Stage_I_Hubei" COPASIkey="ModelValue_26"/>
    <SBMLMap SBMLid="gamma_1_Stage_I_Wuhan" COPASIkey="ModelValue_17"/>
    <SBMLMap SBMLid="gamma_2" COPASIkey="ModelValue_8"/>
    <SBMLMap SBMLid="gamma_2_Stage_III_Wuhan" COPASIkey="ModelValue_24"/>
    <SBMLMap SBMLid="gamma_2_Stage_II_China" COPASIkey="ModelValue_36"/>
    <SBMLMap SBMLid="gamma_2_Stage_II_Hubei" COPASIkey="ModelValue_30"/>
    <SBMLMap SBMLid="gamma_2_Stage_II_Wuhan" COPASIkey="ModelValue_21"/>
    <SBMLMap SBMLid="gamma_2_Stage_I_China" COPASIkey="ModelValue_33"/>
    <SBMLMap SBMLid="gamma_2_Stage_I_Hubei" COPASIkey="ModelValue_27"/>
    <SBMLMap SBMLid="gamma_2_Stage_I_Wuhan" COPASIkey="ModelValue_18"/>
    <SBMLMap SBMLid="sigma" COPASIkey="ModelValue_9"/>
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
<dcterms:W3CDTF>2020-08-26T14:33:45Z</dcterms:W3CDTF>
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
<dcterms:W3CDTF>2020-08-26T14:33:45Z</dcterms:W3CDTF>
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
<dcterms:W3CDTF>2020-08-26T14:33:45Z</dcterms:W3CDTF>
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
<dcterms:W3CDTF>2020-08-26T14:33:45Z</dcterms:W3CDTF>
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
