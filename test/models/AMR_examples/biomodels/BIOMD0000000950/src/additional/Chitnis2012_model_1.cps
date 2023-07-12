<?xml version="1.0" encoding="UTF-8"?>
<!-- generated with COPASI 4.27 (Build 217) (http://www.copasi.org) at 2020-05-13T17:21:19Z -->
<?oxygen RNGSchema="http://www.copasi.org/static/schema/CopasiML.rng" type="xml"?>
<COPASI xmlns="http://www.copasi.org/static/schema" versionMajor="4" versionMinor="27" versionDevel="217" copasiSourcesModified="0">
  <Model key="Model_1" name="Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1)" simulationType="time" timeUnit="s" volumeUnit="ml" areaUnit="m²" lengthUnit="m" quantityUnit="mmol" type="deterministic" avogadroConstant="6.0221408570000002e+23">
    <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:vCard="http://www.w3.org/2001/vcard-rdf/3.0#">
  <rdf:Description rdf:about="#Model_1">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C128419"/>
    <dcterms:bibliographicCitation>
      <rdf:Description>
        <CopasiMT:isDescribedBy rdf:resource="urn:miriam:pubmed:23098257"/>
      </rdf:Description>
    </dcterms:bibliographicCitation>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T14:14:40Z</dcterms:W3CDTF>
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
        <vCard:EMAIL>tiwarik@ebi.ac.uk</vCard:EMAIL>
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
    <CopasiMT:isVersionOf rdf:resource="urn:miriam:ncit:C17214"/>
    <CopasiMT:occursIn rdf:resource="urn:miriam:omit:0003748"/>
  </rdf:Description>
</rdf:RDF>

    </MiriamAnnotation>
    <ListOfCompartments>
      <Compartment key="Compartment_0" name="Cattle" simulationType="fixed" dimensionality="3" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Compartment_0">
    <bqbiol:hasProperty rdf:resource="urn:miriam:omit:0003748"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-05-13T16:52:58Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
      </Compartment>
      <Compartment key="Compartment_1" name="Vector" simulationType="fixed" dimensionality="3" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Compartment_1">
    <bqbiol:hasProperty rdf:resource="urn:miriam:omit:0004757"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2020-05-13T16:52:58Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
      </Compartment>
    </ListOfCompartments>
    <ListOfMetabolites>
      <Metabolite key="Metabolite_0" name="S_h" simulationType="ode" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_0">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C66819"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:omit:0003748"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T14:26:51Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[u_h],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[C0],Reference=Value>-&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[lambda_h],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[S_h],Reference=Concentration>-&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[u_h],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[S_h],Reference=Concentration>
        </Expression>
        <InitialExpression>
          if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=InitialValue> == 0,1000,1000)
        </InitialExpression>
      </Metabolite>
      <Metabolite key="Metabolite_1" name="A_h" simulationType="ode" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_1">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C128320"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C3833"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T14:27:49Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[theta_h],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[lambda_h],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[S_h],Reference=Concentration>-(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[u_h],Reference=Value>+&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[gamma_tilde_h],Reference=Value>)*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[A_h],Reference=Concentration>
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_2" name="I_h" simulationType="ode" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_2">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ido:0000460"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:omit:0003748"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T14:32:08Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          (1-&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[theta_h],Reference=Value>)*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[lambda_h],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[S_h],Reference=Concentration>-(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[u_h],Reference=Value>+&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[gamma_h],Reference=Value>+&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[delta_h],Reference=Value>)*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[I_h],Reference=Concentration>
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_3" name="R_h" simulationType="ode" compartment="Compartment_0" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_3">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C49498"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:omit:0003748"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T14:39:22Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[gamma_h],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[I_h],Reference=Concentration>+&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[gamma_tilde_h],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[A_h],Reference=Concentration>-&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[u_h],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[R_h],Reference=Concentration>
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_4" name="S_v" simulationType="ode" compartment="Compartment_1" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_4">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ncit:C66819"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:omit:0004757"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T14:40:17Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          (&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[N_v],Reference=Value>-&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[psi_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Vector],Vector=Metabolites[I_v],Reference=Concentration>)/&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[N_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[u_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[M0],Reference=Value>-&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[lambda_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Vector],Vector=Metabolites[S_v],Reference=Concentration>-&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[u_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Vector],Vector=Metabolites[S_v],Reference=Concentration>
        </Expression>
        <InitialExpression>
          if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=InitialValue> == 0,19999,3999)
        </InitialExpression>
      </Metabolite>
      <Metabolite key="Metabolite_5" name="E_v" simulationType="ode" compartment="Compartment_1" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_5">
    <bqbiol:hasProperty rdf:resource="urn:miriam:omit:0003748"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:pato:PATO:0002425"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T14:48:04Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[lambda_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Vector],Vector=Metabolites[S_v],Reference=Concentration>-(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[u_v],Reference=Value>+&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[v_v],Reference=Value>)*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Vector],Vector=Metabolites[E_v],Reference=Concentration>
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_6" name="I_v" simulationType="ode" compartment="Compartment_1" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_6">
    <bqbiol:hasProperty rdf:resource="urn:miriam:ido:0000460"/>
    <bqbiol:hasProperty rdf:resource="urn:miriam:omit:0004757"/>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T14:48:49Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[psi_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Vector],Vector=Metabolites[I_v],Reference=Concentration>/&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[N_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[u_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[M0],Reference=Value>+&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[v_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Vector],Vector=Metabolites[E_v],Reference=Concentration>-&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[u_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Vector],Vector=Metabolites[I_v],Reference=Concentration>
        </Expression>
      </Metabolite>
    </ListOfMetabolites>
    <ListOfModelValues>
      <ModelValue key="ModelValue_0" name="u_h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T14:19:10Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=Value> == 0,1/2190,1/2190)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_1" name="u_v" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_1">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T14:19:31Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=Value> == 0,1/20,1/14)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_2" name="psi_v" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_2">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T15:10:08Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=Value> == 0,0.1,0.1)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_3" name="theta_h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_3">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T15:10:35Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=Value> == 0,0.4,0.4)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_4" name="sigma_v" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_4">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T15:11:22Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=Value> == 0,0.33,0.25)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_5" name="sigma_h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_5">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T15:11:44Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=Value> == 0,19,19)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_6" name="beta_hv" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_6">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T14:56:12Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=Value> == 0,0.21,0.21)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_7" name="beta_vh" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_7">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T15:27:41Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=Value> == 0,0.7,0.7)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_8" name="beta_tilde_vh" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_8">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T15:30:29Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=Value> == 0,0.3,0.3)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_9" name="v_v" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_9">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T14:20:36Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=Value> == 0,1/14,1/14)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_10" name="gamma_h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_10">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T14:20:47Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=Value> == 0,1/4,1/4)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_11" name="gamma_tilde_h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_11">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T14:21:24Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=Value> == 0,1/4,1/4)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_12" name="delta_h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_12">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T15:35:11Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=Value> == 0,0.1,0.1)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_13" name="gamma_e" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_13">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T15:35:39Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=Value> == 0,0.2,0.1)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_14" name="M0" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_14">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T15:36:03Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=Value> == 0,20000,4000)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_15" name="C0" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_15">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T15:36:24Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=Value> == 0,1000,1000)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_16" name="N_v" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_16">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T14:42:03Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Vector],Vector=Metabolites[S_v],Reference=Concentration>+&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Vector],Vector=Metabolites[E_v],Reference=Concentration>+&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Vector],Vector=Metabolites[I_v],Reference=Concentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_17" name="lambda_h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_17">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T14:54:12Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[sigma_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[N_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[sigma_h],Reference=Value>/(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[sigma_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[N_v],Reference=Value>+&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[sigma_h],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[N_h],Reference=Value>)*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[beta_hv],Reference=Value>*(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Vector],Vector=Metabolites[I_v],Reference=Concentration>/&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[N_v],Reference=Value>)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_18" name="lambda_v" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_18">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T14:56:18Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[sigma_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[sigma_h],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[N_h],Reference=Value>/(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[sigma_v],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[N_v],Reference=Value>+&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[sigma_h],Reference=Value>*&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[N_h],Reference=Value>)*(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[beta_vh],Reference=Value>*(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[I_h],Reference=Concentration>/&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[N_h],Reference=Value>)+&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[beta_tilde_vh],Reference=Value>*(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[A_h],Reference=Concentration>/&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[N_h],Reference=Value>))
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_19" name="N_h" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_19">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T14:53:29Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[S_h],Reference=Concentration>+&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[I_h],Reference=Concentration>+&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[A_h],Reference=Concentration>+&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[R_h],Reference=Concentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_20" name="Total_Infected_Cattle" simulationType="assignment" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_20">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T15:02:19Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[A_h],Reference=Concentration>+&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[I_h],Reference=Concentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_21" name="Season_Dummy_Variable" simulationType="fixed" addNoise="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_21">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T15:27:30Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
    </ListOfModelValues>
    <ListOfModelParameterSets activeSet="ModelParameterSet_1">
      <ModelParameterSet key="ModelParameterSet_1" name="Initial State">
        <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#ModelParameterSet_1">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-05-13T16:52:58Z</dcterms:W3CDTF>
</rdf:Description>
</dcterms:created>
</rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ModelParameterGroup cn="String=Initial Time" type="Group">
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1)" value="0" type="Model" simulationType="time"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Compartment Sizes" type="Group">
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle]" value="1" type="Compartment" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Vector]" value="1" type="Compartment" simulationType="fixed"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Species Values" type="Group">
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[S_h]" value="6.0221408570000002e+23" type="Species" simulationType="ode">
            <InitialExpression>
              if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=InitialValue> == 0,1000,1000)
            </InitialExpression>
          </ModelParameter>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[A_h]" value="0" type="Species" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[I_h]" value="0" type="Species" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[R_h]" value="0" type="Species" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Vector],Vector=Metabolites[S_v]" value="1.2043679499914301e+25" type="Species" simulationType="ode">
            <InitialExpression>
              if(&lt;CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable],Reference=InitialValue> == 0,19999,3999)
            </InitialExpression>
          </ModelParameter>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Vector],Vector=Metabolites[E_v]" value="0" type="Species" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Vector],Vector=Metabolites[I_v]" value="6.0221408570000002e+20" type="Species" simulationType="ode"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Global Quantities" type="Group">
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[u_h]" value="0.00045662100456621003" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[u_v]" value="0.050000000000000003" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[psi_v]" value="0.10000000000000001" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[theta_h]" value="0.40000000000000002" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[sigma_v]" value="0.33000000000000002" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[sigma_h]" value="19" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[beta_hv]" value="0.20999999999999999" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[beta_vh]" value="0.69999999999999996" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[beta_tilde_vh]" value="0.29999999999999999" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[v_v]" value="0.071428571428571425" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[gamma_h]" value="0.25" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[gamma_tilde_h]" value="0.25" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[delta_h]" value="0.10000000000000001" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[gamma_e]" value="0.20000000000000001" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[M0]" value="20000" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[C0]" value="1000" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[N_v]" value="20000" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[lambda_h]" value="5.143359375e-05" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[lambda_v]" value="0" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[N_h]" value="1000" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Total_Infected_Cattle]" value="0" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Season_Dummy_Variable]" value="0" type="ModelValue" simulationType="fixed"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Kinetic Parameters" type="Group">
        </ModelParameterGroup>
      </ModelParameterSet>
    </ListOfModelParameterSets>
    <StateTemplate>
      <StateTemplateVariable objectReference="Model_1"/>
      <StateTemplateVariable objectReference="Metabolite_0"/>
      <StateTemplateVariable objectReference="Metabolite_1"/>
      <StateTemplateVariable objectReference="Metabolite_2"/>
      <StateTemplateVariable objectReference="Metabolite_3"/>
      <StateTemplateVariable objectReference="Metabolite_4"/>
      <StateTemplateVariable objectReference="Metabolite_5"/>
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
      <StateTemplateVariable objectReference="Compartment_0"/>
      <StateTemplateVariable objectReference="Compartment_1"/>
      <StateTemplateVariable objectReference="ModelValue_21"/>
    </StateTemplate>
    <InitialState type="initialState">
      0 6.0221408570000002e+23 0 0 0 1.2043679499914301e+25 0 6.0221408570000002e+20 0.00045662100456621003 0.050000000000000003 0.10000000000000001 0.40000000000000002 0.33000000000000002 19 0.20999999999999999 0.69999999999999996 0.29999999999999999 0.071428571428571425 0.25 0.25 0.10000000000000001 0.20000000000000001 20000 1000 20000 5.143359375e-05 0 1000 0 1 1 0 
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
        <Parameter name="StepNumber" type="unsignedInteger" value="36500"/>
        <Parameter name="StepSize" type="float" value="0.10000000000000001"/>
        <Parameter name="Duration" type="float" value="3650"/>
        <Parameter name="TimeSeriesRequested" type="bool" value="0"/>
        <Parameter name="OutputStartTime" type="float" value="0.10000000000000001"/>
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
    <PlotSpecification name="Figure 3,4,6 a" type="Plot2D" active="1" taskTypes="">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="Values[Total_Infected_Cattle]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[Total_Infected_Cattle],Reference=Value"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[A_h]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[A_h],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[I_h]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[I_h],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
    <PlotSpecification name="Figure 3,4,6 b" type="Plot2D" active="1" taskTypes="">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="[E_v]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Vector],Vector=Metabolites[E_v],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[I_v]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Vector],Vector=Metabolites[I_v],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
    <PlotSpecification name="figure 5" type="Plot2D" active="1" taskTypes="">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="Values[N_h]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Values[N_h],Reference=Value"/>
          </ListOfChannels>
        </PlotItem>
        <PlotItem name="[R_h]|Time" type="Curve2D">
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Chitnis2012 - Model Rift Valley Fever transmission between cattle and mosquitoes (Model 1),Vector=Compartments[Cattle],Vector=Metabolites[R_h],Reference=Concentration"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
  </ListOfPlots>
  <GUI>
  </GUI>
  <ListOfUnitDefinitions>
    <UnitDefinition key="Unit_1" name="meter" symbol="m">
      <MiriamAnnotation>
<rdf:RDF
xmlns:dcterms="http://purl.org/dc/terms/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="#Unit_0">
<dcterms:created>
<rdf:Description>
<dcterms:W3CDTF>2020-05-13T16:52:56Z</dcterms:W3CDTF>
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
<dcterms:W3CDTF>2020-05-13T16:52:56Z</dcterms:W3CDTF>
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
<dcterms:W3CDTF>2020-05-13T16:52:56Z</dcterms:W3CDTF>
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
<dcterms:W3CDTF>2020-05-13T16:52:56Z</dcterms:W3CDTF>
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
<dcterms:W3CDTF>2020-05-13T16:52:56Z</dcterms:W3CDTF>
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
<dcterms:W3CDTF>2020-05-13T16:52:56Z</dcterms:W3CDTF>
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
