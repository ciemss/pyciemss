<?xml version="1.0" encoding="UTF-8"?>
<!-- generated with COPASI 4.22 (Build 170) (http://www.copasi.org) at 2018-05-23 09:24:32 UTC -->
<?oxygen RNGSchema="http://www.copasi.org/static/schema/CopasiML.rng" type="xml"?>
<COPASI xmlns="http://www.copasi.org/static/schema" versionMajor="4" versionMinor="22" versionDevel="170" copasiSourcesModified="0">
  <Model key="Model_0" name="Mukandavire2009 - Model for HIV-Malaria co-infection" simulationType="time" timeUnit="s" volumeUnit="ml" areaUnit="m²" lengthUnit="m" quantityUnit="mmol" type="deterministic" avogadroConstant="6.0221408570000002e+23">
    <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:vCard="http://www.w3.org/2001/vcard-rdf/3.0#">
  <rdf:Description rdf:about="#Model_0">
    <dcterms:bibliographicCitation>
      <rdf:Description>
        <CopasiMT:isDescribedBy rdf:resource="urn:miriam:pubmed:19364156"/>
      </rdf:Description>
    </dcterms:bibliographicCitation>
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T16:51:57Z</dcterms:W3CDTF>
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
    <CopasiMT:hasPart rdf:resource="urn:miriam:efo:0000764"/>
    <CopasiMT:hasPart rdf:resource="urn:miriam:efo:0001068"/>
  </rdf:Description>
</rdf:RDF>

    </MiriamAnnotation>
    <ListOfCompartments>
      <Compartment key="Compartment_0" name="Human" simulationType="fixed" dimensionality="3">
      </Compartment>
      <Compartment key="Compartment_1" name="Vector" simulationType="fixed" dimensionality="3">
      </Compartment>
    </ListOfCompartments>
    <ListOfMetabolites>
      <Metabolite key="Metabolite_0" name="S_H" simulationType="ode" compartment="Compartment_0">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T09:09:31Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[Capital_lambda_H],Reference=Value>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[phi1],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_M],Reference=Concentration>-&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_H],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[S_H],Reference=Concentration>-&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_M],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[S_H],Reference=Concentration>-&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[u_H],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[S_H],Reference=Concentration>
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_1" name="E_M" simulationType="ode" compartment="Compartment_0">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_1">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T17:28:45Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_M],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[S_H],Reference=Concentration>-&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_H],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[E_M],Reference=Concentration>-(&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[gamma_H],Reference=Value>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[u_H],Reference=Value>)*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[E_M],Reference=Concentration>
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_2" name="I_M" simulationType="ode" compartment="Compartment_0">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_2">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T17:37:52Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[gamma_H],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[E_M],Reference=Concentration>-&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[sigma],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_H],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_M],Reference=Concentration>-(&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[u_H],Reference=Value>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[delta_M],Reference=Value>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[phi1],Reference=Value>)*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_M],Reference=Concentration>
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_3" name="I_H" simulationType="ode" compartment="Compartment_0">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_3">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T17:32:47Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_H],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[S_H],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[phi2],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_H\,M],Reference=Concentration>-&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[nu],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_M],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_H],Reference=Concentration>-(&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[u_H],Reference=Value>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[kappa],Reference=Value>)*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_H],Reference=Concentration>
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_4" name="E_H,M" simulationType="ode" compartment="Compartment_0">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_4">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T17:24:39Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_H],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[E_M],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[nu],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_M],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_H],Reference=Concentration>-(&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[epsilon],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_H],Reference=Value>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[u_H],Reference=Value>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[kappa],Reference=Value>)*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[E_H\,M],Reference=Concentration>
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_5" name="I_H,M" simulationType="ode" compartment="Compartment_0">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_5">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T17:34:25Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[sigma],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_H],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_M],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[epsilon],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[gamma_H],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[E_H\,M],Reference=Concentration>-(&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[u_H],Reference=Value>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[tau],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[delta_M],Reference=Value>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[phi2],Reference=Value>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[xi],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[kappa],Reference=Value>)*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_H\,M],Reference=Concentration>
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_6" name="A_H" simulationType="ode" compartment="Compartment_0">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_6">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T17:15:46Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[kappa],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_H],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[phi3],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[A_H\,M],Reference=Concentration>-&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[nu],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_M],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[A_H],Reference=Concentration>-(&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[u_H],Reference=Value>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[delta_H],Reference=Value>)*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[A_H],Reference=Concentration>
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_7" name="E_A,M" simulationType="ode" compartment="Compartment_0">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_7">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T17:22:32Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[nu],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_M],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[A_H],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[kappa],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[eta_HM],Reference=Value>-(&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[epsilon],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[gamma_H],Reference=Value>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[u_H],Reference=Value>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[delta_H],Reference=Value>)*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[E_A\,M],Reference=Concentration>
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_8" name="A_H,M" simulationType="ode" compartment="Compartment_0">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_8">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T17:20:22Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[xi],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[kappa],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_H\,M],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[epsilon],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[gamma_H],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[E_A\,M],Reference=Concentration>-(&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[u_H],Reference=Value>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[phi3],Reference=Value>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[tau],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[delta_M],Reference=Value>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[psi],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[delta_H],Reference=Value>)*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[A_H\,M],Reference=Concentration>
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_9" name="S_V" simulationType="ode" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_9">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T09:10:47Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[Capital_lambda_V],Reference=Value>-&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_V],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Vector],Vector=Metabolites[S_V],Reference=Concentration>-&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[u_V],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Vector],Vector=Metabolites[S_V],Reference=Concentration>
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_10" name="E_V" simulationType="ode" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_10">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T17:31:43Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_V],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Vector],Vector=Metabolites[S_V],Reference=Concentration>-(&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[gamma_V],Reference=Value>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[u_V],Reference=Value>)*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Vector],Vector=Metabolites[E_V],Reference=Concentration>
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_11" name="I_V" simulationType="ode" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_11">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T17:39:16Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[gamma_V],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Vector],Vector=Metabolites[E_V],Reference=Concentration>-&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[u_V],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Vector],Vector=Metabolites[I_V],Reference=Concentration>
        </Expression>
      </Metabolite>
    </ListOfMetabolites>
    <ListOfModelValues>
      <ModelValue key="ModelValue_0" name="Capital_lambda_H" simulationType="fixed">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_0">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T09:19:34Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_1" name="Capital_lambda_V" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_2" name="u_H" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_3" name="u_V" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_4" name="delta_H" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_5" name="delta_M" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_6" name="Beta_H" simulationType="fixed">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_6">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T09:14:29Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_7" name="Beta_M" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_8" name="Beta_V" simulationType="fixed">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_8">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T09:15:09Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Comment>
          Value: (0,1)
        </Comment>
      </ModelValue>
      <ModelValue key="ModelValue_9" name="b_M" simulationType="fixed">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_9">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T09:17:46Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Comment>
          value: (0.25,1)
        </Comment>
      </ModelValue>
      <ModelValue key="ModelValue_10" name="eta_A" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_11" name="eta_HM" simulationType="fixed">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_11">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T10:12:06Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_12" name="xi" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_13" name="theta_HM" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_14" name="sigma" simulationType="fixed">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_14">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T09:16:20Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_15" name="tau" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_16" name="epsilon" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_17" name="nu" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_18" name="psi" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_19" name="phi1" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_20" name="phi2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_21" name="phi3" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_22" name="eta_v" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_23" name="theta_V" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_24" name="kappa" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_25" name="gamma_H" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_26" name="gamma_V" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_27" name="N_H" simulationType="assignment">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_27">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T17:08:39Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[S_H],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[E_M],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_M],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_H],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[E_H\,M],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_H\,M],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[A_H],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[E_A\,M],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[A_H\,M],Reference=Concentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_28" name="N_V" simulationType="assignment">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_28">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T17:10:06Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Vector],Vector=Metabolites[S_V],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Vector],Vector=Metabolites[E_V],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Vector],Vector=Metabolites[I_V],Reference=Concentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_29" name="lambda_H" simulationType="assignment">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_29">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T17:10:39Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[Beta_H],Reference=Value>*(&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_H],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[eta_HM],Reference=Value>*(&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[E_H\,M],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[theta_HM],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_M],Reference=Concentration>)+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[eta_A],Reference=Value>*(&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[A_H],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[eta_HM],Reference=Value>*(&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[E_A\,M],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[theta_HM],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[A_H\,M],Reference=Concentration>)))/&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[N_H],Reference=Value>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_30" name="lambda_M" simulationType="assignment">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_30">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T17:13:36Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[Beta_M],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[b_M],Reference=Value>*(&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Vector],Vector=Metabolites[I_V],Reference=Concentration>/&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[N_H],Reference=Value>)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_31" name="lambda_V" simulationType="assignment">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_31">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-22T17:14:18Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[Beta_V],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[b_M],Reference=Value>*((&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_M],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[eta_v],Reference=Value>*(&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_H\,M],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[theta_V],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[A_H\,M],Reference=Concentration>))/&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[N_H],Reference=Value>)
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_32" name="New_HIV_cases" simulationType="ode">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_32">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T09:32:29Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_H],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[S_H],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[sigma],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_H],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_M],Reference=Concentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_33" name="New_Malaria_Cases" simulationType="ode">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_33">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T09:47:31Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[gamma_H],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[E_M],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[epsilon],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[gamma_H],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[E_H\,M],Reference=Concentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_34" name="New_Co-infection_Cases" simulationType="ode">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_34">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T10:00:55Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[sigma],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_H],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_M],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[epsilon],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[gamma_H],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[E_H\,M],Reference=Concentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_35" name="Mortality_Malaria" simulationType="ode">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_35">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T10:08:28Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[delta_M],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_M],Reference=Concentration>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[tau],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[delta_M],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_H\,M],Reference=Concentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_36" name="Mortality_HIV" simulationType="ode">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_36">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T10:09:33Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[delta_H],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[A_H],Reference=Concentration>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_37" name="Mortality_Mixed" simulationType="ode">
        <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_37">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-05-23T10:15:18Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

        </MiriamAnnotation>
        <Expression>
          (&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[psi],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[delta_H],Reference=Value>+&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[tau],Reference=Value>*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[delta_M],Reference=Value>)*&lt;CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[A_H\,M],Reference=Concentration>
        </Expression>
      </ModelValue>
    </ListOfModelValues>
    <ListOfModelParameterSets activeSet="ModelParameterSet_0">
      <ModelParameterSet key="ModelParameterSet_0" name="Initial State">
        <ModelParameterGroup cn="String=Initial Time" type="Group">
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection" value="0" type="Model" simulationType="time"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Compartment Sizes" type="Group">
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human]" value="1" type="Compartment" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Vector]" value="1" type="Compartment" simulationType="fixed"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Species Values" type="Group">
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[S_H]" value="6.0221408569999997e+24" type="Species" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[E_M]" value="9.0332112854999999e+22" type="Species" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_M]" value="1.5055352142500001e+22" type="Species" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_H]" value="3.0110704285000002e+21" type="Species" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[E_H\,M]" value="1.8066422570999999e+21" type="Species" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[I_H\,M]" value="6.0221408570000002e+20" type="Species" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[A_H]" value="0" type="Species" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[E_A\,M]" value="0" type="Species" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Human],Vector=Metabolites[A_H\,M]" value="0" type="Species" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Vector],Vector=Metabolites[S_V]" value="1.5055352142500002e+25" type="Species" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Vector],Vector=Metabolites[E_V]" value="3.0110704285000001e+23" type="Species" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Compartments[Vector],Vector=Metabolites[I_V]" value="3.0110704285000002e+22" type="Species" simulationType="ode"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Global Quantities" type="Group">
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[Capital_lambda_H]" value="0.050000000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[Capital_lambda_V]" value="6" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[u_H]" value="3.8999999999999999e-05" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[u_V]" value="0.1429" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[delta_H]" value="0.00091299999999999997" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[delta_M]" value="0.00034539999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[Beta_H]" value="0.00069999999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[Beta_M]" value="0.83330000000000004" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[Beta_V]" value="0.90000000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[b_M]" value="0.25" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[eta_A]" value="1.3999999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[eta_HM]" value="1.5" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[xi]" value="1.002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[theta_HM]" value="1.002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[sigma]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[tau]" value="1.0009999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[epsilon]" value="1.02" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[nu]" value="1.002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[psi]" value="1.002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[phi1]" value="0.0055599999999999998" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[phi2]" value="0.002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[phi3]" value="0.00050000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[eta_v]" value="1.5" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[theta_V]" value="1.5" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[kappa]" value="0.00054799999999999998" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[gamma_H]" value="0.083330000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[gamma_V]" value="0.10000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[N_H]" value="10184" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[N_V]" value="25550" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_H]" value="3.2357128829536529e-06" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_M]" value="0.001022805380989788" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[lambda_V]" value="0.00058547721916732127" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[New_HIV_cases]" value="0" type="ModelValue" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[New_Malaria_Cases]" value="0" type="ModelValue" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[New_Co-infection_Cases]" value="0" type="ModelValue" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[Mortality_Malaria]" value="0" type="ModelValue" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[Mortality_HIV]" value="0" type="ModelValue" simulationType="ode"/>
          <ModelParameter cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[Mortality_Mixed]" value="0" type="ModelValue" simulationType="ode"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Kinetic Parameters" type="Group">
        </ModelParameterGroup>
      </ModelParameterSet>
    </ListOfModelParameterSets>
    <StateTemplate>
      <StateTemplateVariable objectReference="Model_0"/>
      <StateTemplateVariable objectReference="ModelValue_32"/>
      <StateTemplateVariable objectReference="ModelValue_33"/>
      <StateTemplateVariable objectReference="ModelValue_34"/>
      <StateTemplateVariable objectReference="ModelValue_35"/>
      <StateTemplateVariable objectReference="ModelValue_36"/>
      <StateTemplateVariable objectReference="ModelValue_37"/>
      <StateTemplateVariable objectReference="Metabolite_0"/>
      <StateTemplateVariable objectReference="Metabolite_1"/>
      <StateTemplateVariable objectReference="Metabolite_2"/>
      <StateTemplateVariable objectReference="Metabolite_3"/>
      <StateTemplateVariable objectReference="Metabolite_4"/>
      <StateTemplateVariable objectReference="Metabolite_5"/>
      <StateTemplateVariable objectReference="Metabolite_6"/>
      <StateTemplateVariable objectReference="Metabolite_7"/>
      <StateTemplateVariable objectReference="Metabolite_8"/>
      <StateTemplateVariable objectReference="Metabolite_9"/>
      <StateTemplateVariable objectReference="Metabolite_10"/>
      <StateTemplateVariable objectReference="Metabolite_11"/>
      <StateTemplateVariable objectReference="ModelValue_27"/>
      <StateTemplateVariable objectReference="ModelValue_28"/>
      <StateTemplateVariable objectReference="ModelValue_29"/>
      <StateTemplateVariable objectReference="ModelValue_30"/>
      <StateTemplateVariable objectReference="ModelValue_31"/>
      <StateTemplateVariable objectReference="Compartment_0"/>
      <StateTemplateVariable objectReference="Compartment_1"/>
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
      <StateTemplateVariable objectReference="ModelValue_25"/>
      <StateTemplateVariable objectReference="ModelValue_26"/>
    </StateTemplate>
    <InitialState type="initialState">
      0 0 0 0 0 0 0 6.0221408569999997e+24 9.0332112854999999e+22 1.5055352142500001e+22 3.0110704285000002e+21 1.8066422570999999e+21 6.0221408570000002e+20 0 0 0 1.5055352142500002e+25 3.0110704285000001e+23 3.0110704285000002e+22 10184 25550 3.2357128829536529e-06 0.001022805380989788 0.00058547721916732127 1 1 0.050000000000000003 6 3.8999999999999999e-05 0.1429 0.00091299999999999997 0.00034539999999999999 0.00069999999999999999 0.83330000000000004 0.90000000000000002 0.25 1.3999999999999999 1.5 1.002 1.002 1 1.0009999999999999 1.02 1.002 1.002 0.0055599999999999998 0.002 0.00050000000000000001 1.5 1.5 0.00054799999999999998 0.083330000000000001 0.10000000000000001 
    </InitialState>
  </Model>
  <ListOfTasks>
    <Task key="Task_12" name="Steady-State" type="steadyState" scheduled="false" updateModel="false">
      <Report reference="Report_8" target="" append="1" confirmOverwrite="1"/>
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
    <Task key="Task_11" name="Time-Course" type="timeCourse" scheduled="false" updateModel="false">
      <Problem>
        <Parameter name="AutomaticStepSize" type="bool" value="0"/>
        <Parameter name="StepNumber" type="unsignedInteger" value="1500"/>
        <Parameter name="StepSize" type="float" value="1"/>
        <Parameter name="Duration" type="float" value="1500"/>
        <Parameter name="TimeSeriesRequested" type="bool" value="1"/>
        <Parameter name="OutputStartTime" type="float" value="0"/>
        <Parameter name="Output Event" type="bool" value="0"/>
        <Parameter name="Start in Steady State" type="bool" value="0"/>
      </Problem>
      <Method name="Deterministic (LSODA)" type="Deterministic(LSODA)">
        <Parameter name="Integrate Reduced Model" type="bool" value="0"/>
        <Parameter name="Relative Tolerance" type="unsignedFloat" value="9.9999999999999995e-07"/>
        <Parameter name="Absolute Tolerance" type="unsignedFloat" value="9.9999999999999998e-13"/>
        <Parameter name="Max Internal Steps" type="unsignedInteger" value="10000"/>
        <Parameter name="Max Internal Step Size" type="unsignedFloat" value="0"/>
      </Method>
    </Task>
    <Task key="Task_10" name="Scan" type="scan" scheduled="false" updateModel="false">
      <Problem>
        <Parameter name="Subtask" type="unsignedInteger" value="1"/>
        <ParameterGroup name="ScanItems">
          <ParameterGroup name="ScanItem">
            <Parameter name="Number of steps" type="unsignedInteger" value="2"/>
            <Parameter name="Type" type="unsignedInteger" value="1"/>
            <Parameter name="Object" type="cn" value="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[sigma],Reference=InitialValue"/>
            <Parameter name="Minimum" type="float" value="0"/>
            <Parameter name="Maximum" type="float" value="1"/>
            <Parameter name="log" type="bool" value="0"/>
          </ParameterGroup>
          <ParameterGroup name="ScanItem">
            <Parameter name="Number of steps" type="unsignedInteger" value="1"/>
            <Parameter name="Type" type="unsignedInteger" value="1"/>
            <Parameter name="Object" type="cn" value="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[nu],Reference=InitialValue"/>
            <Parameter name="Minimum" type="float" value="1.002"/>
            <Parameter name="Maximum" type="float" value="1.002"/>
            <Parameter name="log" type="bool" value="0"/>
          </ParameterGroup>
          <ParameterGroup name="ScanItem">
            <Parameter name="Number of steps" type="unsignedInteger" value="1"/>
            <Parameter name="Type" type="unsignedInteger" value="1"/>
            <Parameter name="Object" type="cn" value="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[psi],Reference=InitialValue"/>
            <Parameter name="Minimum" type="float" value="1.002"/>
            <Parameter name="Maximum" type="float" value="1.002"/>
            <Parameter name="log" type="bool" value="0"/>
          </ParameterGroup>
        </ParameterGroup>
        <Parameter name="Output in subtask" type="bool" value="1"/>
        <Parameter name="Adjust initial conditions" type="bool" value="0"/>
      </Problem>
      <Method name="Scan Framework" type="ScanFramework">
      </Method>
    </Task>
    <Task key="Task_9" name="Elementary Flux Modes" type="fluxMode" scheduled="false" updateModel="false">
      <Report reference="Report_7" target="" append="1" confirmOverwrite="1"/>
      <Problem>
      </Problem>
      <Method name="EFM Algorithm" type="EFMAlgorithm">
      </Method>
    </Task>
    <Task key="Task_8" name="Optimization" type="optimization" scheduled="false" updateModel="false">
      <Report reference="Report_6" target="" append="1" confirmOverwrite="1"/>
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
        <Parameter name="Number of Iterations" type="unsignedInteger" value="100000"/>
        <Parameter name="Random Number Generator" type="unsignedInteger" value="1"/>
        <Parameter name="Seed" type="unsignedInteger" value="0"/>
      </Method>
    </Task>
    <Task key="Task_7" name="Parameter Estimation" type="parameterFitting" scheduled="false" updateModel="false">
      <Report reference="Report_5" target="" append="1" confirmOverwrite="1"/>
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
          <Parameter name="Threshold" type="unsignedInteger" value="5"/>
          <Parameter name="Weight" type="unsignedFloat" value="1"/>
        </ParameterGroup>
      </Problem>
      <Method name="Evolutionary Programming" type="EvolutionaryProgram">
        <Parameter name="Number of Generations" type="unsignedInteger" value="200"/>
        <Parameter name="Population Size" type="unsignedInteger" value="20"/>
        <Parameter name="Random Number Generator" type="unsignedInteger" value="1"/>
        <Parameter name="Seed" type="unsignedInteger" value="0"/>
      </Method>
    </Task>
    <Task key="Task_6" name="Metabolic Control Analysis" type="metabolicControlAnalysis" scheduled="false" updateModel="false">
      <Report reference="Report_4" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="Steady-State" type="key" value="Task_12"/>
      </Problem>
      <Method name="MCA Method (Reder)" type="MCAMethod(Reder)">
        <Parameter name="Modulation Factor" type="unsignedFloat" value="1.0000000000000001e-09"/>
        <Parameter name="Use Reder" type="bool" value="1"/>
        <Parameter name="Use Smallbone" type="bool" value="1"/>
      </Method>
    </Task>
    <Task key="Task_5" name="Lyapunov Exponents" type="lyapunovExponents" scheduled="false" updateModel="false">
      <Report reference="Report_3" target="" append="1" confirmOverwrite="1"/>
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
    <Task key="Task_4" name="Time Scale Separation Analysis" type="timeScaleSeparationAnalysis" scheduled="false" updateModel="false">
      <Report reference="Report_2" target="" append="1" confirmOverwrite="1"/>
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
    <Task key="Task_3" name="Sensitivities" type="sensitivities" scheduled="false" updateModel="false">
      <Report reference="Report_1" target="" append="1" confirmOverwrite="1"/>
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
    <Task key="Task_2" name="Moieties" type="moieties" scheduled="false" updateModel="false">
      <Problem>
      </Problem>
      <Method name="Householder Reduction" type="Householder">
      </Method>
    </Task>
    <Task key="Task_1" name="Cross Section" type="crosssection" scheduled="false" updateModel="false">
      <Problem>
        <Parameter name="AutomaticStepSize" type="bool" value="0"/>
        <Parameter name="StepNumber" type="unsignedInteger" value="100"/>
        <Parameter name="StepSize" type="float" value="0.01"/>
        <Parameter name="Duration" type="float" value="1"/>
        <Parameter name="TimeSeriesRequested" type="bool" value="1"/>
        <Parameter name="OutputStartTime" type="float" value="0"/>
        <Parameter name="Output Event" type="bool" value="0"/>
        <Parameter name="Start in Steady State" type="bool" value="0"/>
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
    <Task key="Task_13" name="Linear Noise Approximation" type="linearNoiseApproximation" scheduled="false" updateModel="false">
      <Report reference="Report_0" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="Steady-State" type="key" value="Task_12"/>
      </Problem>
      <Method name="Linear Noise Approximation" type="LinearNoiseApproximation">
      </Method>
    </Task>
  </ListOfTasks>
  <ListOfReports>
    <Report key="Report_8" name="Steady-State" taskType="steadyState" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Footer>
        <Object cn="CN=Root,Vector=TaskList[Steady-State]"/>
      </Footer>
    </Report>
    <Report key="Report_7" name="Elementary Flux Modes" taskType="fluxMode" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Footer>
        <Object cn="CN=Root,Vector=TaskList[Elementary Flux Modes],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_6" name="Optimization" taskType="optimization" separator="&#x09;" precision="6">
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
    <Report key="Report_5" name="Parameter Estimation" taskType="parameterFitting" separator="&#x09;" precision="6">
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
    <Report key="Report_4" name="Metabolic Control Analysis" taskType="metabolicControlAnalysis" separator="&#x09;" precision="6">
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
    <Report key="Report_3" name="Lyapunov Exponents" taskType="lyapunovExponents" separator="&#x09;" precision="6">
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
    <Report key="Report_2" name="Time Scale Separation Analysis" taskType="timeScaleSeparationAnalysis" separator="&#x09;" precision="6">
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
    <Report key="Report_1" name="Sensitivities" taskType="sensitivities" separator="&#x09;" precision="6">
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
    <Report key="Report_0" name="Linear Noise Approximation" taskType="linearNoiseApproximation" separator="&#x09;" precision="6">
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
    <PlotSpecification name="Figure 7/8 A" type="Plot2D" active="1">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="Values[quantity]|Time" type="Curve2D">
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[New_HIV_cases],Reference=Value"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
    <PlotSpecification name="Figure 7/8 B" type="Plot2D" active="1">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="Values[New_Malaria_Cases]|Time" type="Curve2D">
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[New_Malaria_Cases],Reference=Value"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
    <PlotSpecification name="Figure 7/8 C" type="Plot2D" active="1">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="Values[New_Co-infection_Cases]|Time" type="Curve2D">
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[New_Co-infection_Cases],Reference=Value"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
    <PlotSpecification name="Figure 9 A" type="Plot2D" active="1">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="Values[Mortality_Malaria]|Time" type="Curve2D">
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[Mortality_Malaria],Reference=Value"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
    <PlotSpecification name="Figure 9 B" type="Plot2D" active="1">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="Values[Mortality_HIV]|Time" type="Curve2D">
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[Mortality_HIV],Reference=Value"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
    <PlotSpecification name="Figure 9 C" type="Plot2D" active="1">
      <Parameter name="log X" type="bool" value="0"/>
      <Parameter name="log Y" type="bool" value="0"/>
      <ListOfPlotItems>
        <PlotItem name="Values[Mortality_Mixed]|Time" type="Curve2D">
          <Parameter name="Color" type="string" value="auto"/>
          <Parameter name="Line subtype" type="unsignedInteger" value="0"/>
          <Parameter name="Line type" type="unsignedInteger" value="0"/>
          <Parameter name="Line width" type="unsignedFloat" value="1"/>
          <Parameter name="Recording Activity" type="string" value="during"/>
          <Parameter name="Symbol subtype" type="unsignedInteger" value="0"/>
          <ListOfChannels>
            <ChannelSpec cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Reference=Time"/>
            <ChannelSpec cn="CN=Root,Model=Mukandavire2009 - Model for HIV-Malaria co-infection,Vector=Values[Mortality_Mixed],Reference=Value"/>
          </ListOfChannels>
        </PlotItem>
      </ListOfPlotItems>
    </PlotSpecification>
  </ListOfPlots>
  <GUI>
  </GUI>
  <ListOfUnitDefinitions>
    <UnitDefinition key="Unit_0" name="meter" symbol="m">
      <Expression>
        m
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_2" name="second" symbol="s">
      <Expression>
        s
      </Expression>
    </UnitDefinition>
  </ListOfUnitDefinitions>
</COPASI>
